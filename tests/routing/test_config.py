import json

from pathlib import Path

import pytest

from mlserver import MLServer, Settings
from mlserver.routing import (
    RouteSpec,
    RouteStrategy,
    RoutingConfigError,
    create_model_router,
    load_routing_config,
)


def write_config(tmp_path: Path, payload: object, name: str = "routes.json") -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def valid_route() -> dict:
    return {
        "name": "sum-route",
        "strategy": "pinned",
        "targets": [{"model": "sum-model", "version": "1"}],
    }


def test_load_routing_config_defaults(tmp_path: Path):
    path = write_config(
        tmp_path,
        {
            "routes": [
                {
                    "name": "sum-route",
                    "targets": [{"model": "sum-model"}],
                }
            ]
        },
    )

    config = load_routing_config(path)

    assert len(config) == 1
    route = config[0]
    assert isinstance(route, RouteSpec)
    assert route.strategy == RouteStrategy.LATEST
    assert route.targets[0].weight == 1
    assert route.model == "sum-model"


def test_load_routing_config_accepts_settings_and_relative_path(tmp_path: Path):
    config_path = write_config(tmp_path, {"routes": [valid_route()]})
    settings = Settings(
        model_repository_root=str(tmp_path),
        routing_config_path="routes.json",
    )

    config = load_routing_config(settings)

    assert config.config_path == str(Path(config_path).resolve())
    assert config[0].name == "sum-route"


@pytest.mark.parametrize(
    "route_mutator, reason",
    [
        (lambda route: route.update(name="bad name"), "route name"),
        (lambda route: route.update(strategy="unknown"), "unknown strategy"),
        (lambda route: route.update(targets=[]), "empty targets"),
        (
            lambda route: route.update(
                targets=[
                    {"model": "sum-model", "version": "1"},
                    {"model": "other-model", "version": "2"},
                ]
            ),
            "same model",
        ),
        (
            lambda route: route.update(
                strategy="latest",
                targets=[{"model": "sum-model", "version": "1"}],
            ),
            "latest",
        ),
        (
            lambda route: route.update(
                strategy="pinned",
                targets=[{"model": "sum-model"}],
            ),
            "pinned",
        ),
        (
            lambda route: route.update(
                strategy="weighted",
                targets=[{"model": "sum-model", "weight": 1}],
            ),
            "weighted",
        ),
        (
            lambda route: route.update(
                strategy="weighted",
                targets=[
                    {"model": "sum-model", "version": "1", "weight": 0},
                    {"model": "sum-model", "version": "2", "weight": 0},
                ],
            ),
            "sum to zero",
        ),
    ],
)
def test_invalid_route_config(tmp_path: Path, route_mutator, reason: str):
    route = valid_route()
    route_mutator(route)
    path = write_config(tmp_path, {"routes": [route]})

    with pytest.raises(RoutingConfigError) as exc_info:
        load_routing_config(path)

    message = str(exc_info.value)
    assert path in message
    assert reason in message


def test_duplicate_route_names_are_invalid(tmp_path: Path):
    path = write_config(
        tmp_path,
        {"routes": [valid_route(), valid_route()]},
    )

    with pytest.raises(RoutingConfigError) as exc_info:
        load_routing_config(path)

    assert path in str(exc_info.value)
    assert "duplicate" in str(exc_info.value)


def test_missing_config_file_is_invalid(tmp_path: Path):
    path = str(tmp_path / "missing.json")

    with pytest.raises(RoutingConfigError) as exc_info:
        load_routing_config(path)

    assert path in str(exc_info.value)


def test_invalid_json_is_invalid(tmp_path: Path):
    path = tmp_path / "routes.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(RoutingConfigError) as exc_info:
        load_routing_config(str(path))

    assert str(path) in str(exc_info.value)
    assert "JSON" in str(exc_info.value)


async def test_bad_config_factory_and_server_raise_with_path(
    tmp_path: Path, model_registry
):
    path = write_config(
        tmp_path,
        {"routes": [{**valid_route(), "strategy": "bad"}]},
    )
    settings = Settings(routing_config_path=path)

    with pytest.raises(RoutingConfigError) as factory_exc:
        create_model_router(settings, model_registry)
    assert path in str(factory_exc.value)

    with pytest.raises(RoutingConfigError) as server_exc:
        MLServer(settings)
    assert path in str(server_exc.value)


def test_disabled_routing_returns_none():
    settings = Settings(routing_config_path=None)

    assert create_model_router(settings) is None


def test_package_all_contains_public_api():
    import mlserver.routing as routing

    for name in [
        "ModelRouter",
        "RouteSpec",
        "load_routing_config",
        "RoutingConfigError",
    ]:
        assert name in routing.__all__
        assert hasattr(routing, name)
