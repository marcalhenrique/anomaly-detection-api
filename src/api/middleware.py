import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.services.metrics_collector import HTTP_REQUEST_DURATION

_SKIP_PATHS = {"/metrics", "/metrics/", "/docs", "/openapi.json", "/healthcheck"}


class HTTPLatencyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in _SKIP_PATHS:
            return await call_next(request)

        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        # Normalise path: strip series/version IDs so cardinality stays low
        # e.g. /predict/sensor-42 → /predict/{series_id}
        path = request.url.path
        route = request.scope.get("route")
        if route is not None:
            path = route.path

        HTTP_REQUEST_DURATION.labels(
            method=request.method,
            path=path,
            status_code=response.status_code,
        ).observe(duration_ms)

        return response
