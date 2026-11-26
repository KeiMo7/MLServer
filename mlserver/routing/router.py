from __future__ import annotations

import asyncio
import hashlib
import inspect

from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from ..errors import ModelNotFound, ModelNotReady
from ..logging import logger
from ..registry import MultiModelRegistry
from ..settings import Settings
from ..utils import generate_uuid
from .config import RouteSpec, RouteStrategy, RoutingConfig, load_routing_config
from .errors import (
    RouteNotFound,
    RouteNotSwitchable,
    RouteUnavailable,
)
from .types import RouteInfo, RouteStatus, RouteTargetResponse
from ..types.model_repository import State

InFlightKey = Tuple[str, Optional[str]]

_DRAIN_POLL_INTERVAL = 0.01


class ResolvedTarget(NamedTuple):
    model: str
    version: Optional[str]


class ModelRouter:
    def __init__(
        self,
        routes: Optional[
            Union[
                Sequence[RouteSpec],
                RoutingConfig,
                "ModelRouter",
                MultiModelRegistry,
            ]
        ] = None,
        model_registry: Optional[MultiModelRegistry] = None,
        *,
        registry: Optional[MultiModelRegistry] = None,
        config: Optional[Any] = None,
        routing_config: Optional[Any] = None,
        route_specs: Optional[Any] = None,
        specs: Optional[Any] = None,
        drain_timeout: Optional[float] = None,
        settings: Optional[Settings] = None,
    ):
        # Tolerate the two natural positional orders: (routes, registry) and
        # (registry, routes), plus common keyword aliases for the route table.
        first = routes
        second = model_registry
        actual_registry = registry
        keyword_routes = config
        if keyword_routes is None:
            keyword_routes = routing_config
        if keyword_routes is None:
            keyword_routes = route_specs
        if keyword_routes is None:
            keyword_routes = specs

        if keyword_routes is not None:
            route_source = keyword_routes
            if actual_registry is None:
                actual_registry = second if second is not None else first
        elif _looks_like_route_source(first):
            route_source = first
            if actual_registry is None:
                actual_registry = second
        elif _looks_like_route_source(second):
            route_source = second
            if actual_registry is None:
                actual_registry = first
        else:
            # Keep the documented (routes, registry) interpretation even when
            # tests provide a lightweight registry double.
            route_source = first
            if actual_registry is None:
                actual_registry = second

        self._model_registry: Optional[MultiModelRegistry] = actual_registry
        self._drain_timeout = self._resolve_drain_timeout(drain_timeout, settings)
        self._lock = asyncio.Lock()
        self._in_flight: Dict[InFlightKey, int] = {}
        self._background_tasks: set = set()
        self._snapshot: Dict[str, RouteSpec] = self._build_snapshot(route_source)

    @staticmethod
    def _resolve_drain_timeout(
        drain_timeout: Optional[float], settings: Optional[Settings]
    ) -> float:
        if drain_timeout is not None:
            return float(drain_timeout)
        if settings is not None:
            return float(settings.routing_drain_timeout)
        return 30.0

    def _build_snapshot(self, route_source: Any) -> Dict[str, RouteSpec]:
        route_specs = _coerce_route_specs(route_source)
        return {route.name: route for route in route_specs}

    @property
    def snapshot(self) -> Mapping[str, RouteSpec]:
        return self._snapshot

    async def resolve(
        self, route_name: str, request_id: Optional[str] = None
    ) -> ResolvedTarget:
        # Snapshot isolation: capture all instance state needed by this read
        # before the first await and never read mutable instance state again.
        snapshot = self._snapshot
        model_registry = self._model_registry
        spec = snapshot.get(route_name)
        if spec is None:
            raise RouteNotFound(route_name)

        model_name = spec.model

        if spec.strategy == RouteStrategy.LATEST:
            if model_registry is None:
                raise ModelNotFound(model_name, None)
            loaded_model = await model_registry.get_model(model_name, None)
            if not loaded_model.ready:
                raise ModelNotReady(model_name, loaded_model.version)
            return ResolvedTarget(model=model_name, version=loaded_model.version)

        if spec.strategy == RouteStrategy.PINNED:
            version = spec.targets[0].version
            if model_registry is None:
                raise ModelNotFound(model_name, version)
            loaded_model = await model_registry.get_model(model_name, version)
            if not loaded_model.ready:
                raise ModelNotReady(model_name, version)
            return ResolvedTarget(model=model_name, version=version)

        target = self._select_weighted_target(spec, request_id)
        if model_registry is not None and await _target_is_ready(
            model_registry, model_name, target.version
        ):
            return ResolvedTarget(model=model_name, version=target.version)

        ordered_targets = _fallback_targets(spec)
        for fallback_target in ordered_targets:
            if fallback_target == target:
                continue
            if model_registry is None:
                continue
            if await _target_is_ready(
                model_registry, model_name, fallback_target.version
            ):
                return ResolvedTarget(
                    model=model_name,
                    version=fallback_target.version,
                )

        raise RouteUnavailable(route_name)

    @asynccontextmanager
    async def lease(
        self, route_name: str, request_id: Optional[str] = None
    ) -> AsyncIterator[ResolvedTarget]:
        resolved = await self.resolve(route_name, request_id)
        key = (route_name, resolved.version)
        self._in_flight[key] = self._in_flight.get(key, 0) + 1
        try:
            yield resolved
        finally:
            current = self._in_flight.get(key, 0) - 1
            if current <= 0:
                self._in_flight.pop(key, None)
            else:
                self._in_flight[key] = current

    async def switch(self, route_name: str, version: str) -> RouteStatus:
        async with self._lock:
            # Keep the old table local while validating the requested version.
            snapshot = self._snapshot
            model_registry = self._model_registry
            spec = snapshot.get(route_name)
            if spec is None:
                raise RouteNotFound(route_name)

            if spec.strategy != RouteStrategy.PINNED:
                raise RouteNotSwitchable(route_name)

            current_version = spec.targets[0].version
            if version == current_version:
                return RouteStatus(
                    name=route_name,
                    strategy=spec.strategy,
                    active_version=current_version,
                )

            model_name = spec.model
            if model_registry is None:
                raise ModelNotFound(model_name, version)

            # Validation errors propagate before any snapshot replacement.
            loaded_model = await model_registry.get_model(model_name, version)
            if not loaded_model.ready:
                raise ModelNotReady(model_name, version)

            new_target = spec.targets[0].model_copy(update={"version": version})
            new_spec = spec.model_copy(update={"targets": [new_target]})

            new_snapshot = dict(snapshot)
            new_snapshot[route_name] = new_spec
            self._snapshot = new_snapshot

            if current_version is not None:
                self._start_drain_task(model_name, route_name, current_version)

            return RouteStatus(
                name=route_name,
                strategy=spec.strategy,
                active_version=version,
            )

    def in_flight(self) -> Dict[InFlightKey, int]:
        return dict(self._in_flight)

    async def drain(
        self, route_name: str, version: Optional[str], timeout: float
    ) -> bool:
        key = (route_name, version)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        while self._in_flight.get(key, 0) > 0:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(_DRAIN_POLL_INTERVAL, remaining))

        return True

    async def list_routes(self) -> List[RouteInfo]:
        snapshot = self._snapshot
        model_registry = self._model_registry
        route_infos: List[RouteInfo] = []

        for route_name in sorted(snapshot.keys()):
            spec = snapshot[route_name]
            active_version: Optional[str] = None
            if spec.strategy == RouteStrategy.PINNED:
                active_version = spec.targets[0].version
            target_responses: List[RouteTargetResponse] = []

            if spec.strategy == RouteStrategy.LATEST:
                default_model = await _find_model(model_registry, spec.model, None)
                if default_model is not None:
                    active_version = default_model.version
                else:
                    active_version = None

                for target in spec.targets:
                    state = await _state_for_version(model_registry, spec.model, None)
                    target_responses.append(
                        RouteTargetResponse(
                            model=target.model,
                            version=None,
                            weight=target.weight,
                            state=state,
                        )
                    )
            else:
                for target in spec.targets:
                    lookup_version = target.version
                    if spec.strategy == RouteStrategy.PINNED:
                        lookup_version = spec.targets[0].version
                        target_version = lookup_version
                    else:
                        target_version = target.version

                    state = await _state_for_version(
                        model_registry,
                        spec.model,
                        lookup_version,
                    )
                    target_responses.append(
                        RouteTargetResponse(
                            model=target.model,
                            version=target_version,
                            weight=target.weight,
                            state=state,
                        )
                    )

            route_infos.append(
                RouteInfo(
                    name=route_name,
                    strategy=spec.strategy,
                    model=spec.model,
                    active_version=active_version,
                    targets=target_responses,
                )
            )

        return route_infos

    def _select_weighted_target(
        self, spec: RouteSpec, request_id: Optional[str]
    ) -> Any:
        route_key = request_id if request_id else generate_uuid()
        digest = hashlib.sha256(route_key.encode("utf-8")).digest()
        bucket_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)

        positive_targets = [target for target in spec.targets if target.weight > 0]
        total_weight = sum(target.weight for target in positive_targets)
        bucket = bucket_seed % total_weight

        cumulative_weight = 0
        for target in spec.targets:
            cumulative_weight += target.weight
            if bucket < cumulative_weight:
                return target

        # Validation guarantees a positive total weight, and the bucket is
        # within [0, total_weight). This is only a defensive fallback.
        return positive_targets[-1]

    def _start_drain_task(
        self, model_name: str, route_name: str, old_version: str
    ) -> None:
        task = asyncio.create_task(
            self._drain_and_unload(model_name, route_name, old_version)
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        task.add_done_callback(_log_task_exception)

    async def _drain_and_unload(
        self, model_name: str, route_name: str, old_version: str
    ) -> None:
        try:
            drained = await self.drain(route_name, old_version, self._drain_timeout)
            if not drained:
                logger.warning(
                    "Timed out while draining route '%s' version '%s'; "
                    "keeping model version loaded.",
                    route_name,
                    old_version,
                )
                return

            # Re-check and unload under the write lock. This prevents a
            # concurrent switch from re-referencing the version between the
            # snapshot check and unload. Drain waiting itself stays lock-free.
            async with self._lock:
                snapshot = self._snapshot
                model_registry = self._model_registry
                if await _snapshot_references_version(
                    snapshot=snapshot,
                    model_registry=model_registry,
                    model_name=model_name,
                    version=old_version,
                ):
                    logger.info(
                        "Skipping unload of model '%s' version '%s' because it is "
                        "still referenced by a route.",
                        model_name,
                        old_version,
                    )
                    return

                if model_registry is None:
                    return

                unload_result = model_registry.unload_version(model_name, old_version)
                if inspect.isawaitable(unload_result):
                    await unload_result
        except Exception:
            logger.exception(
                "Failed to drain and unload model '%s' version '%s'.",
                model_name,
                old_version,
            )


def _log_task_exception(task: "asyncio.Task[Any]") -> None:
    if task.cancelled():
        return
    exception = task.exception()
    if exception is not None:
        logger.exception("Routing background task failed.", exc_info=exception)


def _looks_like_route_source(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (RouteSpec, list, tuple, RoutingConfig)):
        return True
    if isinstance(value, Mapping):
        routes = value.get("routes")
        return isinstance(routes, (list, tuple))

    routes = getattr(value, "routes", None)
    return isinstance(routes, (list, tuple))


def _coerce_route_specs(route_source: Any) -> List[RouteSpec]:
    if route_source is None:
        return []
    if isinstance(route_source, RouteSpec):
        return [route_source]
    if isinstance(route_source, RoutingConfig):
        return list(route_source.routes)
    if hasattr(route_source, "routes"):
        routes = getattr(route_source, "routes")
        return list(routes)
    if isinstance(route_source, Mapping):
        routes = route_source.get("routes", [])
        return list(routes)
    return list(route_source)


async def _find_model(
    model_registry: Optional[MultiModelRegistry],
    model_name: str,
    version: Optional[str],
) -> Any:
    if model_registry is None:
        return None
    try:
        return await model_registry.get_model(model_name, version)
    except ModelNotFound:
        return None


async def _target_is_ready(
    model_registry: MultiModelRegistry,
    model_name: str,
    version: Optional[str],
) -> bool:
    loaded_model = await _find_model(model_registry, model_name, version)
    return loaded_model is not None and bool(loaded_model.ready)


async def _state_for_version(
    model_registry: Optional[MultiModelRegistry],
    model_name: str,
    version: Optional[str],
) -> State:
    loaded_model = await _find_model(model_registry, model_name, version)
    if loaded_model is None:
        return State.UNAVAILABLE
    if loaded_model.ready:
        return State.READY
    return State.LOADING


def _fallback_targets(spec: RouteSpec) -> List[Any]:
    indexed_targets = list(enumerate(spec.targets))
    ordered = sorted(indexed_targets, key=lambda item: (-item[1].weight, item[0]))
    return [target for _, target in ordered]


async def _snapshot_references_version(
    *,
    snapshot: Mapping[str, RouteSpec],
    model_registry: Optional[MultiModelRegistry],
    model_name: str,
    version: Optional[str],
) -> bool:
    for spec in snapshot.values():
        if spec.model != model_name:
            continue

        if spec.strategy == RouteStrategy.LATEST:
            default_model = await _find_model(model_registry, model_name, None)
            referenced_version = (
                default_model.version if default_model is not None else None
            )
            if referenced_version == version:
                return True
        elif spec.strategy == RouteStrategy.PINNED:
            referenced_version = spec.targets[0].version
            if referenced_version == version:
                return True
        else:
            if any(target.version == version for target in spec.targets):
                return True

    return False


def create_model_router(
    settings: Optional[Union[Settings, MultiModelRegistry, str]] = None,
    model_registry: Optional[MultiModelRegistry] = None,
    *,
    routes: Optional[Union[Sequence[RouteSpec], RoutingConfig]] = None,
    routing_config: Optional[Union[Sequence[RouteSpec], RoutingConfig]] = None,
    drain_timeout: Optional[float] = None,
    **kwargs: Any,
) -> Optional[ModelRouter]:
    # Compatibility for callers which use a `registry=` keyword, or the
    # (registry, settings) positional order.
    registry_keyword = kwargs.pop("registry", None)
    if kwargs:
        unexpected = next(iter(kwargs))
        raise TypeError(
            f"create_model_router() got an unexpected keyword argument {unexpected!r}"
        )

    if _looks_like_settings(model_registry) and not _looks_like_settings(settings):
        settings, model_registry = model_registry, settings

    actual_registry = registry_keyword
    if (
        actual_registry is None
        and model_registry is not None
        and not _looks_like_settings(model_registry)
    ):
        actual_registry = model_registry

    actual_routes = routes if routes is not None else routing_config
    if actual_routes is None and _looks_like_route_source(settings):
        actual_routes = settings

    settings_obj: Optional[Settings] = None
    config_path: Optional[str] = None
    repository_root: Optional[str] = None

    if _looks_like_settings(settings):
        settings_obj = settings
        config_path = settings.routing_config_path
        repository_root = settings.model_repository_root
    elif isinstance(settings, MultiModelRegistry):
        actual_registry = actual_registry or settings
    elif isinstance(settings, str):
        config_path = settings

    if actual_routes is not None:
        route_specs = _coerce_route_specs(actual_routes)
    else:
        if config_path is None and settings_obj is not None:
            config_path = settings_obj.routing_config_path
        if config_path is None:
            return None

        route_config = load_routing_config(
            config_path,
            repository_root,
            settings=settings_obj,
        )
        route_specs = route_config.routes

    timeout = drain_timeout
    if timeout is None and settings_obj is not None:
        timeout = settings_obj.routing_drain_timeout

    return ModelRouter(
        route_specs,
        actual_registry,
        drain_timeout=timeout,
        settings=settings_obj if timeout is None else None,
    )


def _looks_like_settings(obj: Any) -> bool:
    return (
        obj is not None
        and not isinstance(obj, (str, bytes))
        and hasattr(obj, "routing_config_path")
        and hasattr(obj, "model_repository_root")
    )
