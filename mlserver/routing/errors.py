from fastapi import status

from ..errors import MLServerError


class RoutingConfigError(Exception):
    """Raised when the routing configuration is missing or invalid."""

    def __init__(self, msg: str):
        super().__init__(msg)


class RouteNotFound(MLServerError):
    def __init__(self, route_name: str):
        super().__init__(
            f"Route {route_name} not found",
            status.HTTP_404_NOT_FOUND,
        )


class RouteNotSwitchable(MLServerError):
    def __init__(self, route_name: str):
        super().__init__(
            f"Route {route_name} does not support version switching",
            status.HTTP_409_CONFLICT,
        )


class RouteUnavailable(MLServerError):
    def __init__(self, route_name: str):
        super().__init__(
            f"Route {route_name} has no ready target",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )
