from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from src.core.database import engine
from src.api.dependencies import get_mlflow_service, get_metrics_collector
from src.api.schemas import HealthCheckResponse, HealthCheckMetrics

router = APIRouter(tags=["Health"])


@router.get("/healthcheck", response_model=HealthCheckResponse)
async def healthcheck() -> HealthCheckResponse | JSONResponse:
    healthy = True

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:
        healthy = False

    try:
        mlflow_svc = get_mlflow_service()
        await __import__("asyncio").to_thread(
            mlflow_svc._client.search_experiments, max_results=1
        )
    except Exception:
        healthy = False

    collector = get_metrics_collector()
    response_data = HealthCheckResponse(
        series_trained=collector.get_series_trained(),
        inference_latency_ms=HealthCheckMetrics(**collector.get_inference_stats()),
        training_latency_ms=HealthCheckMetrics(**collector.get_training_stats()),
    )

    if not healthy:
        return JSONResponse(
            content=response_data.model_dump(),
            status_code=503,
        )

    return response_data
