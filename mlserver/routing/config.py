from __future__ import annotations

import json
import os
import re

from enum import Enum
from typing import Any, List, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field

from .errors import RoutingConfigError

ROUTE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

PathLike = Union[str, "os.PathLike[str]"]


class RouteStrategy(str, Enum):
    LATEST = "latest"
    PINNED = "pinned"
    WEIGHTED = "weighted"


class RouteTarget(BaseModel):
    model_config = ConfigDict(frozen=True, protected_namespaces=())

    model: str
    version: Optional[str] = None
    weight: int = Field(default=1, ge=0)


class RouteSpec(BaseModel):
    model_config = ConfigDict(frozen=True, protected_namespaces=())

    name: str
    strategy: RouteStrategy = RouteStrategy.LATEST
    targets: List[RouteTarget]

    @property
    def model(self) -> str:
        return self.targets[0].model


class RoutingConfig(list):
    """A route list which also exposes the parsed route sequence explicitly."""

    def __init__(
        self,
        routes: Sequence[RouteSpec],
        config_path: Optional[str] = None,
    ):
        super().__init__(routes)
        self.routes: List[RouteSpec] = list(routes)
        self.config_path = config_path


def _settings_value(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name)
    return None


def _as_path(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, os.PathLike):
        raw_value = os.fspath(value)
        if isinstance(raw_value, str):
            return raw_value
    return None


def _resolve_config_path(
    path: Any,
    model_repository_root: Any = None,
    settings: Any = None,
    config_path: Any = None,
    routing_config_path: Any = None,
) -> str:
    setting_candidate = settings if settings is not None else path
    settings_path = _settings_value(setting_candidate, "routing_config_path")
    settings_root = _settings_value(setting_candidate, "model_repository_root")

    raw_path = (
        _as_path(routing_config_path)
        or _as_path(config_path)
        or _as_path(path if not _looks_like_settings(path) else None)
        or _as_path(settings_path)
    )
    if raw_path is None:
        raise RoutingConfigError("Routing configuration path is not set")

    root = (
        _as_path(model_repository_root)
        if not _looks_like_settings(model_repository_root)
        else _settings_value(model_repository_root, "model_repository_root")
    )
    if root is None:
        root = _as_path(settings_root)

    if not os.path.isabs(raw_path) and root:
        raw_path = os.path.join(root, raw_path)

    return os.path.abspath(raw_path)


def _looks_like_settings(obj: Any) -> bool:
    return (
        obj is not None
        and not isinstance(obj, (str, bytes, os.PathLike))
        and hasattr(obj, "routing_config_path")
    )


def _config_error(config_path: str, reason: str) -> RoutingConfigError:
    return RoutingConfigError(
        f"Invalid routing configuration at {config_path}: {reason}"
    )


def _parse_target(raw_target: Any, config_path: str, context: str) -> RouteTarget:
    if not isinstance(raw_target, dict):
        raise _config_error(config_path, f"{context} target must be an object")

    model = raw_target.get("model")
    if not isinstance(model, str) or not model:
        raise _config_error(config_path, f"{context} target requires a model name")

    if "weight" in raw_target:
        weight = raw_target["weight"]
        if isinstance(weight, bool) or not isinstance(weight, int):
            raise _config_error(
                config_path, f"{context} target weight must be a non-negative integer"
            )
        if weight < 0:
            raise _config_error(
                config_path, f"{context} target weight must be non-negative"
            )

    version = raw_target.get("version")
    if version is not None and not isinstance(version, str):
        raise _config_error(config_path, f"{context} target version must be a string")

    try:
        return RouteTarget.model_validate(raw_target)
    except Exception as exc:
        raise _config_error(config_path, f"invalid {context} target: {exc}") from exc


def _parse_route(raw_route: Any, config_path: str, index: int) -> RouteSpec:
    context = f"routes[{index}]"
    if not isinstance(raw_route, dict):
        raise _config_error(config_path, f"{context} must be an object")

    name = raw_route.get("name")
    if not isinstance(name, str) or not name:
        raise _config_error(config_path, f"{context} requires a name")
    if not ROUTE_NAME_PATTERN.fullmatch(name):
        raise _config_error(
            config_path,
            f"route name {name!r} does not match ^[a-zA-Z0-9_-]+$",
        )

    raw_strategy = raw_route.get("strategy", RouteStrategy.LATEST.value)
    try:
        strategy = RouteStrategy(raw_strategy)
    except ValueError:
        allowed = ", ".join(strategy.value for strategy in RouteStrategy)
        raise _config_error(
            config_path,
            f"route {name} has unknown strategy {raw_strategy!r}; "
            f"expected one of {allowed}",
        )

    raw_targets = raw_route.get("targets")
    if not isinstance(raw_targets, list):
        raise _config_error(config_path, f"route {name} requires a targets array")
    if len(raw_targets) == 0:
        raise _config_error(config_path, f"route {name} has empty targets")

    targets = [
        _parse_target(raw_target, config_path, f"route {name}")
        for raw_target in raw_targets
    ]

    models = {target.model for target in targets}
    if len(models) != 1:
        raise _config_error(
            config_path,
            f"route {name} targets must all reference the same model",
        )

    if strategy == RouteStrategy.LATEST:
        if len(targets) != 1:
            raise _config_error(
                config_path,
                f"route {name} using latest strategy requires exactly one target",
            )
        if "version" in raw_targets[0]:
            raise _config_error(
                config_path,
                f"route {name} using latest strategy must not set target version",
            )

    if strategy == RouteStrategy.PINNED:
        if len(targets) != 1:
            raise _config_error(
                config_path,
                f"route {name} using pinned strategy requires exactly one target",
            )
        if not targets[0].version:
            raise _config_error(
                config_path,
                f"route {name} using pinned strategy requires target version",
            )

    if strategy == RouteStrategy.WEIGHTED:
        for target_index, target in enumerate(targets):
            if not target.version:
                raise _config_error(
                    config_path,
                    f"route {name} weighted target[{target_index}] requires version",
                )
        if sum(target.weight for target in targets) == 0:
            raise _config_error(
                config_path,
                f"route {name} weighted target weights sum to zero",
            )

    try:
        return RouteSpec(
            name=name,
            strategy=strategy,
            targets=targets,
        )
    except Exception as exc:
        raise _config_error(config_path, f"invalid route {name}: {exc}") from exc


def load_routing_config(
    path: Optional[Union[PathLike, Any]] = None,
    model_repository_root: Optional[Union[PathLike, Any]] = None,
    *,
    config_path: Optional[PathLike] = None,
    routing_config_path: Optional[PathLike] = None,
    settings: Optional[Any] = None,
) -> RoutingConfig:
    """Load and validate the JSON routing configuration."""

    resolved_path = _resolve_config_path(
        path=path,
        model_repository_root=model_repository_root,
        settings=settings,
        config_path=config_path,
        routing_config_path=routing_config_path,
    )

    try:
        with open(resolved_path, "r", encoding="utf-8") as config_file:
            raw_config = json.load(config_file)
    except FileNotFoundError as exc:
        raise _config_error(
            resolved_path, f"configuration file not found: {exc.filename}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise _config_error(resolved_path, f"invalid JSON: {exc.msg}") from exc
    except OSError as exc:
        raise _config_error(
            resolved_path, f"could not read configuration: {exc}"
        ) from exc

    if not isinstance(raw_config, dict):
        raise _config_error(resolved_path, "configuration root must be a JSON object")

    raw_routes = raw_config.get("routes")
    if not isinstance(raw_routes, list):
        raise _config_error(resolved_path, "configuration requires a routes array")

    routes = [
        _parse_route(raw_route, resolved_path, index)
        for index, raw_route in enumerate(raw_routes)
    ]

    names = [route.name for route in routes]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        duplicates = ", ".join(duplicate_names)
        raise _config_error(resolved_path, f"duplicate route name(s): {duplicates}")

    return RoutingConfig(routes, config_path=resolved_path)
