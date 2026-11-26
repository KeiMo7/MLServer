from typing import List

import httpx

from mlserver import Settings
from mlserver.handlers import DataPlane, ModelRepositoryHandlers
from mlserver.registry import MultiModelRegistry
from mlserver.routing import (
    ModelRouter,
    RouteSpec,
    RouteStrategy,
    RouteTarget,
)
from mlserver.rest.app import create_app

MODEL_NAME = "sum-model"


async def load_versions(
    model_registry: MultiModelRegistry,
    sum_model_settings,
    versions: List[str],
) -> None:
    for version in versions:
        sum_model_settings.parameters.version = version
        await model_registry.load(sum_model_settings)


def make_router(model_registry: MultiModelRegistry) -> ModelRouter:
    routes = [
        RouteSpec(
            name="a-latest",
            strategy=RouteStrategy.LATEST,
            targets=[RouteTarget(model=MODEL_NAME)],
        ),
        RouteSpec(
            name="b-pinned",
            strategy=RouteStrategy.PINNED,
            targets=[RouteTarget(model=MODEL_NAME, version="1")],
        ),
        RouteSpec(
            name="c-weighted",
            strategy=RouteStrategy.WEIGHTED,
            targets=[
                RouteTarget(model=MODEL_NAME, version="1", weight=1),
                RouteTarget(model=MODEL_NAME, version="2", weight=2),
            ],
        ),
    ]
    return ModelRouter(routes, model_registry)


async def test_disabled_routing_endpoints(
    settings: Settings,
    data_plane: DataPlane,
    model_repository_handlers: ModelRepositoryHandlers,
    inference_request,
):
    app = create_app(settings, data_plane, model_repository_handlers)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        list_response = await client.get("/v2/routing/routes")
        assert list_response.status_code == 200
        assert list_response.json() == {"routes": []}

        infer_response = await client.post(
            "/v2/routing/routes/missing/infer",
            json=inference_request.model_dump(mode="json"),
        )
        assert infer_response.status_code == 404
        assert infer_response.json()["error"].startswith("Route missing not found")

        switch_response = await client.post(
            "/v2/routing/routes/missing/switch",
            json={"version": "1"},
        )
        assert switch_response.status_code == 404

        malformed_switch = await client.post(
            "/v2/routing/routes/missing/switch",
            json={},
        )
        assert malformed_switch.status_code == 404

        live_response = await client.get("/v2/health/live")
        assert live_response.status_code == 200


async def test_list_routes_sorted_states_and_omitted_none_keys(
    settings: Settings,
    data_plane: DataPlane,
    model_repository_handlers: ModelRepositoryHandlers,
    model_registry: MultiModelRegistry,
    sum_model_settings,
):
    await load_versions(model_registry, sum_model_settings, ["1", "2"])
    missing_state_route = RouteSpec(
        name="d-missing",
        strategy=RouteStrategy.PINNED,
        targets=[RouteTarget(model=MODEL_NAME, version="9")],
    )
    not_ready_route = RouteSpec(
        name="e-loading",
        strategy=RouteStrategy.PINNED,
        targets=[RouteTarget(model=MODEL_NAME, version="1")],
    )
    router = make_router(model_registry)
    router._snapshot["d-missing"] = router._build_snapshot([missing_state_route])[
        "d-missing"
    ]
    router._snapshot["e-loading"] = router._build_snapshot([not_ready_route])[
        "e-loading"
    ]

    model_v1 = await model_registry.get_model(MODEL_NAME, "1")
    model_v1.ready = False

    app = create_app(
        settings,
        data_plane,
        model_repository_handlers,
        model_router=router,
    )
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v2/routing/routes")

    assert response.status_code == 200
    body = response.json()
    assert [route["name"] for route in body["routes"]] == [
        "a-latest",
        "b-pinned",
        "c-weighted",
        "d-missing",
        "e-loading",
    ]

    latest = body["routes"][0]
    assert latest["strategy"] == "latest"
    assert latest["model"] == MODEL_NAME
    assert latest["active_version"] == "2"
    assert "version" not in latest["targets"][0]
    assert latest["targets"][0]["state"] == "READY"

    pinned = body["routes"][1]
    assert pinned["active_version"] == "1"
    assert pinned["targets"][0] == {
        "model": MODEL_NAME,
        "version": "1",
        "weight": 1,
        "state": "LOADING",
    }

    weighted = body["routes"][2]
    assert "active_version" not in weighted
    assert [target["version"] for target in weighted["targets"]] == ["1", "2"]

    missing = body["routes"][3]
    assert missing["targets"][0]["state"] == "UNAVAILABLE"


async def test_route_infer_uses_lease_target_and_sets_response_fields(
    settings: Settings,
    data_plane: DataPlane,
    model_repository_handlers: ModelRepositoryHandlers,
    model_registry: MultiModelRegistry,
    sum_model_settings,
    inference_request,
):
    await load_versions(model_registry, sum_model_settings, ["1", "2"])
    router = make_router(model_registry)
    app = create_app(
        settings,
        data_plane,
        model_repository_handlers,
        model_router=router,
    )
    transport = httpx.ASGITransport(app=app)
    payload = inference_request.model_dump(mode="json")

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        latest_response = await client.post(
            "/v2/routing/routes/a-latest/infer",
            json=payload,
        )
        pinned_response = await client.post(
            "/v2/routing/routes/b-pinned/infer",
            json={**payload, "id": "pinned-request"},
        )
        weighted_response = await client.post(
            "/v2/routing/routes/c-weighted/infer",
            json={**payload, "id": "weighted-request"},
        )

    assert latest_response.status_code == 200
    latest_body = latest_response.json()
    assert latest_body["model_name"] == MODEL_NAME
    assert latest_body["model_version"] == "2"
    assert latest_body["id"]

    assert pinned_response.status_code == 200
    pinned_body = pinned_response.json()
    assert pinned_body["model_version"] == "1"
    assert pinned_body["id"] == "pinned-request"

    assert weighted_response.status_code == 200
    weighted_body = weighted_response.json()
    assert weighted_body["model_name"] == MODEL_NAME
    assert weighted_body["model_version"] in {"1", "2"}
    assert weighted_body["id"] == "weighted-request"


async def test_route_infer_error_statuses(
    settings: Settings,
    data_plane: DataPlane,
    model_repository_handlers: ModelRepositoryHandlers,
    model_registry: MultiModelRegistry,
    sum_model_settings,
    inference_request,
):
    await load_versions(model_registry, sum_model_settings, ["1", "2"])
    missing_version_route = RouteSpec(
        name="missing-version",
        strategy=RouteStrategy.PINNED,
        targets=[RouteTarget(model=MODEL_NAME, version="9")],
    )
    not_ready_route = RouteSpec(
        name="not-ready",
        strategy=RouteStrategy.PINNED,
        targets=[RouteTarget(model=MODEL_NAME, version="1")],
    )
    unavailable_weighted = RouteSpec(
        name="bad-weighted",
        strategy=RouteStrategy.WEIGHTED,
        targets=[
            RouteTarget(model=MODEL_NAME, version="1", weight=1),
            RouteTarget(model=MODEL_NAME, version="2", weight=1),
        ],
    )
    router = ModelRouter(
        [missing_version_route, not_ready_route, unavailable_weighted],
        model_registry,
    )
    model_v1 = await model_registry.get_model(MODEL_NAME, "1")
    model_v2 = await model_registry.get_model(MODEL_NAME, "2")
    model_v1.ready = False
    model_v2.ready = False

    app = create_app(
        settings,
        data_plane,
        model_repository_handlers,
        model_router=router,
    )
    transport = httpx.ASGITransport(app=app)
    payload = inference_request.model_dump(mode="json")

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        missing_route = await client.post(
            "/v2/routing/routes/nope/infer",
            json=payload,
        )
        missing_version = await client.post(
            "/v2/routing/routes/missing-version/infer",
            json=payload,
        )
        not_ready = await client.post(
            "/v2/routing/routes/not-ready/infer",
            json=payload,
        )
        weighted = await client.post(
            "/v2/routing/routes/bad-weighted/infer",
            json=payload,
        )

    assert missing_route.status_code == 404
    assert missing_route.json()["error"] == "Route nope not found"

    assert missing_version.status_code == 404
    assert missing_version.json()["error"] == (
        "Model sum-model with version 9 not found"
    )

    assert not_ready.status_code == 400
    assert not_ready.json()["error"].startswith(
        "Model sum-model with version 1 is not ready"
    )

    assert weighted.status_code == 503
    assert weighted.json()["error"] == ("Route bad-weighted has no ready target")


async def test_switch_endpoint_success_validation_and_422(
    settings: Settings,
    data_plane: DataPlane,
    model_repository_handlers: ModelRepositoryHandlers,
    model_registry: MultiModelRegistry,
    sum_model_settings,
):
    await load_versions(model_registry, sum_model_settings, ["1", "2"])
    router = ModelRouter(
        [
            RouteSpec(
                name="pinned",
                strategy=RouteStrategy.PINNED,
                targets=[RouteTarget(model=MODEL_NAME, version="1")],
            ),
            RouteSpec(
                name="latest",
                strategy=RouteStrategy.LATEST,
                targets=[RouteTarget(model=MODEL_NAME)],
            ),
        ],
        model_registry,
    )
    app = create_app(
        settings,
        data_plane,
        model_repository_handlers,
        model_router=router,
    )
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        missing_body = await client.post(
            "/v2/routing/routes/pinned/switch",
            json={},
        )
        wrong_type = await client.post(
            "/v2/routing/routes/pinned/switch",
            json={"version": 2},
        )
        not_switchable = await client.post(
            "/v2/routing/routes/latest/switch",
            json={"version": "2"},
        )
        invalid_version = await client.post(
            "/v2/routing/routes/pinned/switch",
            json={"version": "9"},
        )
        assert invalid_version.status_code == 404
        assert router._snapshot["pinned"].targets[0].version == "1"

        switched = await client.post(
            "/v2/routing/routes/pinned/switch",
            json={"version": "2"},
        )
        inference = await client.post(
            "/v2/routing/routes/pinned/infer",
            json={
                "inputs": [
                    {
                        "name": "input-0",
                        "shape": [1, 3],
                        "datatype": "INT32",
                        "data": [1, 2, 3],
                    }
                ]
            },
        )

    assert missing_body.status_code == 422
    assert wrong_type.status_code == 422

    assert not_switchable.status_code == 409
    assert not_switchable.json()["error"] == (
        "Route latest does not support version switching"
    )

    assert switched.status_code == 200
    assert switched.json() == {
        "name": "pinned",
        "strategy": "pinned",
        "active_version": "2",
    }
    assert inference.status_code == 200
    assert inference.json()["model_version"] == "2"


async def test_route_recovers_after_repository_unload_and_load(
    settings: Settings,
    data_plane: DataPlane,
    model_repository_handlers: ModelRepositoryHandlers,
    model_registry: MultiModelRegistry,
):
    router = ModelRouter(
        [
            RouteSpec(
                name="latest",
                strategy=RouteStrategy.LATEST,
                targets=[RouteTarget(model=MODEL_NAME)],
            )
        ],
        model_registry,
    )
    app = create_app(
        settings,
        data_plane,
        model_repository_handlers,
        model_router=router,
    )
    transport = httpx.ASGITransport(app=app)
    payload = {
        "inputs": [
            {
                "name": "input-0",
                "shape": [1, 3],
                "datatype": "INT32",
                "data": [1, 2, 3],
            }
        ]
    }

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        before = await client.get("/v2/routing/routes")
        assert before.json()["routes"][0]["targets"][0]["state"] == "READY"

        unload = await client.post(f"/v2/repository/models/{MODEL_NAME}/unload")
        assert unload.status_code == 200

        after_unload = await client.get("/v2/routing/routes")
        assert after_unload.json()["routes"][0]["targets"][0]["state"] == (
            "UNAVAILABLE"
        )

        infer_after_unload = await client.post(
            "/v2/routing/routes/latest/infer",
            json=payload,
        )
        assert infer_after_unload.status_code == 404

        load = await client.post(f"/v2/repository/models/{MODEL_NAME}/load")
        assert load.status_code == 200

        after_load = await client.get("/v2/routing/routes")
        assert after_load.json()["routes"][0]["targets"][0]["state"] == "READY"

        infer_after_load = await client.post(
            "/v2/routing/routes/latest/infer",
            json=payload,
        )
        assert infer_after_load.status_code == 200
