from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, StrictStr

from ..types.model_repository import State
from .config import RouteStrategy


class RoutingApiModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class RouteSwitchRequest(RoutingApiModel):
    version: StrictStr


class RouteTargetResponse(RoutingApiModel):
    model: str
    version: Optional[str] = None
    weight: int
    state: State


class RouteStatus(RoutingApiModel):
    name: str
    strategy: RouteStrategy
    active_version: Optional[str] = None


class RouteInfo(RouteStatus):
    model: str
    targets: List[RouteTargetResponse]


class RouteListResponse(RoutingApiModel):
    routes: List[RouteInfo]


SwitchRouteRequest = RouteSwitchRequest
RouteResponse = RouteInfo
RoutesResponse = RouteListResponse
RouteStatusResponse = RouteStatus
