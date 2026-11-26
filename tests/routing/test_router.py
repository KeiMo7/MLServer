import asyncio
import hashlib

from typing import List, Optional
from unittest.mock import AsyncMock

import pytest

from mlserver.errors import ModelNotFound, ModelNotReady
from mlserver.registry import MultiModelRegistry
from mlserver.routing import (
    ModelRouter,
    RouteSpec,
    RouteStrategy,
    RouteTarget,
)
from mlserver.routing.errors import (
    RouteNotSwitchable,
    RouteUnavailable,
)


MODEL_NAME = "sum-model"


async def load_versions(
    model_registry: MultiModelRegistry,
    sum_model_settings,
    versions: List[str],
) -> None:
    for version in versions:
        sum_model_settings.parameters.version = version
        await model_registry.load(sum_model_settings)


def pinned_route(version: str = "1", name: str = "pinned") -> RouteSpec:
    return RouteSpec(
        name=name,
        strategy=RouteStrategy.PINNED,
        targets=[RouteTarget(model=MODEL_NAME, version=version)],
    )


def latest_route(name: str = "latest") -> RouteSpec:
    return RouteSpec(
        name=name,
        strategy=RouteStrategy.LATEST,
        targets=[RouteTarget(model=MODEL_NAME)],
    )


def weighted_route(
    versions: List[str],
    weights: List[int],
    name: str = "weighted",
) -> RouteSpec:
    return RouteSpec(
        name=name,
        strategy=RouteStrategy.WEIGHTED,
        targets=[
            RouteTarget(model=MODEL_NAME, version=version, weight=weight)
            for version, weight in zip(versions, weights)
        ],
    )


def expected_weighted_version(route: RouteSpec, request_id: str) -> str:
    digest = hashlib.sha256(request_id.encode("utf-8")).digest()
    bucket_seed = int.from_bytes(digest[:8], byteorder="big")
    positive_weight = sum(
        target.weight for target in route.targets if target.weight > 0
    )
    bucket = bucket_seed % positive_weight

    cumulative = 0
    for target in route.targets:
        cumulative += target.weight
        if bucket < cumulative:
            return target.version  # type: ignore[return-value]

    raise AssertionError("No weighted target selected")


async def test_latest_and_pinned_resolution(
    model_registry: MultiModelRegistry,
    sum_model_settings,
):
    await load_versions(model_registry, sum_model_settings, ["1", "2"])
    router = ModelRouter([latest_route(), pinned_route("1")], model_registry)

    latest = await router.resolve("latest", "request-1")
    pinned = await router.resolve("pinned", "request-2")

    assert latest.model == MODEL_NAME
    assert latest.version == "2"
    assert pinned == (MODEL_NAME, "1")


async def test_weighted_selection_matches_sha256_algorithm(
    model_registry: MultiModelRegistry,
    sum_model_settings,
):
    await load_versions(model_registry, sum_model_settings, ["1", "2", "3"])
    route = weighted_route(["1", "2", "3"], [2, 3, 5])
    router = ModelRouter([route], model_registry)

    for index in range(40):
        request_id = f"weighted-request-{index}"
        resolved = await router.resolve("weighted", request_id)
        assert resolved.version == expected_weighted_version(route, request_id)


async def test_zero_weight_target_is_never_selected(
    model_registry: MultiModelRegistry,
    sum_model_settings,
):
    await load_versions(model_registry, sum_model_settings, ["1", "2", "3"])
    route = weighted_route(["1", "2", "3"], [1, 1, 0])
    router = ModelRouter([route], model_registry)

    for index in range(100):
        resolved = await router.resolve("weighted", f"zero-{index}")
        assert resolved.version in {"1", "2"}


async def test_weighted_fallback_uses_weight_then_declaration_order(
    model_registry: MultiModelRegistry,
    sum_model_settings,
):
    await load_versions(model_registry, sum_model_settings, ["1", "2", "3"])
    route = weighted_route(["1", "2", "3"], [1, 10, 5])
    router = ModelRouter([route], model_registry)

    preferred_id: Optional[str] = None
    for index in range(200):
        request_id = f"fallback-{index}"
        if expected_weighted_version(route, request_id) == "1":
            preferred_id = request_id
            break

    assert preferred_id is not None
    model_v1 = await model_registry.get_model(MODEL_NAME, "1")
    model_v1.ready = False

    resolved = await router.resolve("weighted", preferred_id)
    assert resolved.version == "2"


async def test_weighted_all_targets_unavailable_returns_503(
    model_registry: MultiModelRegistry,
    sum_model_settings,
):
    await load_versions(model_registry, sum_model_settings, ["1", "2"])
    route = weighted_route(["1", "2"], [1, 1])
    router = ModelRouter([route], model_registry)

    for version in ["1", "2"]:
        model = await model_registry.get_model(MODEL_NAME, version)
        model.ready = False

    with pytest.raises(RouteUnavailable) as exc_info:
        await router.resolve("weighted", "unavailable")

    assert exc_info.value.status_code == 503
    assert str(exc_info.value) == "Route weighted has no ready target"


async def test_latest_and_pinned_error_contracts(
    model_registry: MultiModelRegistry,
    sum_model_settings,
):
    await load_versions(model_registry, sum_model_settings, ["1", "2"])
    router = ModelRouter([latest_route(), pinned_route("1")], model_registry)

    default_model = await model_registry.get_model(MODEL_NAME)
    default_model.ready = False
    with pytest.raises(ModelNotReady) as not_ready:
        await router.resolve("latest")
    assert not_ready.value.status_code == 400
    assert "sum-model" in str(not_ready.value)

    missing_router = ModelRouter(
        [
            RouteSpec(
                name="missing-latest",
                strategy=RouteStrategy.LATEST,
                targets=[RouteTarget(model="missing-model")],
            )
        ],
        model_registry,
    )
    with pytest.raises(ModelNotFound) as not_found:
        await missing_router.resolve("missing-latest")
    assert not_found.value.status_code == 404
    assert str(not_found.value) == "Model missing-model not found"

    missing_version_router = ModelRouter(
        [pinned_route("9")],
        model_registry,
    )
    with pytest.raises(ModelNotFound) as pinned_missing:
        await missing_version_router.resolve("pinned")
    assert pinned_missing.value.status_code == 404
    assert str(pinned_missing.value) == ("Model sum-model with version 9 not found")


async def test_switch_rejects_non_pinned_routes(model_registry):
    router = ModelRouter(
        [latest_route(), weighted_route(["1", "2"], [1, 1])],
        model_registry,
    )

    with pytest.raises(RouteNotSwitchable) as latest_error:
        await router.switch("latest", "2")
    assert latest_error.value.status_code == 409
    assert str(latest_error.value) == (
        "Route latest does not support version switching"
    )

    with pytest.raises(RouteNotSwitchable):
        await router.switch("weighted", "2")


async def test_switch_idempotent_keeps_snapshot_and_avoids_registry(
    model_registry: MultiModelRegistry,
    sum_model_settings,
    monkeypatch,
):
    await load_versions(model_registry, sum_model_settings, ["1", "2"])
    router = ModelRouter([pinned_route("1")], model_registry)
    snapshot = router._snapshot

    load_mock = AsyncMock()
    unload_mock = AsyncMock()
    monkeypatch.setattr(model_registry, "load", load_mock)
    monkeypatch.setattr(model_registry, "unload_version", unload_mock)

    first = await router.switch("pinned", "1")
    second = await router.switch("pinned", "1")

    assert first.active_version == "1"
    assert second.active_version == "1"
    assert router._snapshot is snapshot
    load_mock.assert_not_called()
    unload_mock.assert_not_called()


async def test_fifty_concurrent_leases_keep_old_version_and_decrement(
    model_registry: MultiModelRegistry,
    sum_model_settings,
    monkeypatch,
):
    await load_versions(model_registry, sum_model_settings, ["1", "2"])
    unload_mock = AsyncMock()
    monkeypatch.setattr(model_registry, "unload_version", unload_mock)
    router = ModelRouter([pinned_route("1")], model_registry)

    entered = 0
    release = asyncio.Event()
    held_versions: List[Optional[str]] = []

    async def hold(index: int) -> None:
        nonlocal entered
        async with router.lease("pinned", f"lease-{index}") as target:
            entered += 1
            while entered < 50:
                await asyncio.sleep(0)
            held_versions.append(target.version)
            await release.wait()

    tasks = [asyncio.create_task(hold(index)) for index in range(50)]
    for _ in range(1000):
        if sum(router.in_flight().values()) == 50:
            break
        await asyncio.sleep(0)

    assert router.in_flight() == {("pinned", "1"): 50}

    status = await router.switch("pinned", "2")
    assert status.active_version == "2"
    new_target = await router.resolve("pinned", "new-request")
    assert new_target.version == "2"

    release.set()
    await asyncio.gather(*tasks)

    assert held_versions == ["1"] * 50
    assert router.in_flight() == {}

    for _ in range(100):
        if unload_mock.await_count:
            break
        await asyncio.sleep(0.01)
    unload_mock.assert_awaited_once_with(MODEL_NAME, "1")


async def test_switch_drains_and_unloads_old_version(
    model_registry: MultiModelRegistry,
    sum_model_settings,
    monkeypatch,
):
    await load_versions(model_registry, sum_model_settings, ["1", "2"])
    router = ModelRouter(
        [pinned_route("1")],
        model_registry,
        drain_timeout=1,
    )
    unload_mock = AsyncMock()
    monkeypatch.setattr(model_registry, "unload_version", unload_mock)

    await router.switch("pinned", "2")

    for _ in range(100):
        if unload_mock.await_count:
            break
        await asyncio.sleep(0.01)

    unload_mock.assert_awaited_once_with(MODEL_NAME, "1")


async def test_drain_timeout_keeps_old_version(
    model_registry: MultiModelRegistry,
    sum_model_settings,
    monkeypatch,
):
    await load_versions(model_registry, sum_model_settings, ["1", "2"])
    router = ModelRouter(
        [pinned_route("1")],
        model_registry,
        drain_timeout=0.05,
    )
    unload_mock = AsyncMock()
    monkeypatch.setattr(model_registry, "unload_version", unload_mock)

    async def hold_old_version() -> None:
        async with router.lease("pinned", "long-request"):
            await asyncio.sleep(0.2)

    holder = asyncio.create_task(hold_old_version())
    for _ in range(100):
        if router.in_flight():
            break
        await asyncio.sleep(0)

    await router.switch("pinned", "2")
    await asyncio.sleep(0.1)

    unload_mock.assert_not_called()
    assert await router.resolve("pinned", "explicit-old") == (MODEL_NAME, "2")
    await holder

    # The timed-out background task does not retry after the lease ends.
    await asyncio.sleep(0.05)
    unload_mock.assert_not_called()


async def test_drain_returns_false_and_true(
    model_registry: MultiModelRegistry,
    sum_model_settings,
):
    await load_versions(model_registry, sum_model_settings, ["1"])
    router = ModelRouter([pinned_route("1")], model_registry)

    assert await router.drain("pinned", "1", 0.01) is True

    async def hold() -> None:
        async with router.lease("pinned", "drain-id"):
            await asyncio.sleep(0.08)

    task = asyncio.create_task(hold())
    while not router.in_flight():
        await asyncio.sleep(0)

    assert await router.drain("pinned", "1", 0.01) is False
    assert await router.drain("pinned", "1", 0.2) is True
    await task


async def test_other_pinned_route_reference_skips_unload(
    model_registry: MultiModelRegistry,
    sum_model_settings,
    monkeypatch,
):
    await load_versions(model_registry, sum_model_settings, ["1", "2"])
    routes = [pinned_route("1", "switching"), pinned_route("1", "watcher")]
    router = ModelRouter(routes, model_registry)
    unload_mock = AsyncMock()
    monkeypatch.setattr(model_registry, "unload_version", unload_mock)

    await router.switch("switching", "2")
    await asyncio.sleep(0.05)

    unload_mock.assert_not_called()
    assert await router.resolve("watcher", "id") == (MODEL_NAME, "1")
