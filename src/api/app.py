import os
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from prometheus_client import make_asgi_app

from src.core.config import get_settings
from src.core.structlog_config import get_logger
from src.api.routes.fit import router as training_router
from src.api.routes.predict import router as prediction_router
from src.api.routes.health import router as health_router
from src.api.middleware import HTTPLatencyMiddleware
from src.core.database import AsyncSessionFactory, engine
from src.repositories.model_metadata import ModelMetadataRepository
from src.api.dependencies import _mlflow_service, _metadata_cache, get_metrics_collector

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    os.environ.setdefault("AWS_ACCESS_KEY_ID", settings.minio_access_key)
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", settings.minio_secret_key)
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", settings.mlflow_s3_endpoint_url)
    logger.info("starting_up")

    async with AsyncSessionFactory() as session:
        repo = ModelMetadataRepository(session)
        records = await repo.get_all_latest()
        logger.info("cache_warming_start", total_models=len(records))

    from src.services.metrics_collector import SERIES_TRAINED

    SERIES_TRAINED.set(len(records))

    for record in records:
        _metadata_cache.set(record)

    for record in records:
        try:
            await asyncio.to_thread(
                _mlflow_service.load_model,
                record.mlflow_run_id,
            )
        except Exception as e:
            logger.warning(
                "startup_cache_warm_failed",
                series_id=record.series_id,
                run_id=record.mlflow_run_id,
                error=str(e),
            )
    logger.info("cache_warming_complete", total_models=len(records))

    metrics_collector = get_metrics_collector()

    async def _cache_metrics_loop() -> None:
        while True:
            try:
                await asyncio.to_thread(
                    metrics_collector.update_cache_metrics,
                    _metadata_cache,
                    _mlflow_service,
                )
            except Exception as e:
                logger.warning("cache_metrics_refresh_failed", error=str(e))
            await asyncio.sleep(10)

    cache_metrics_task = asyncio.create_task(_cache_metrics_loop())
    yield
    cache_metrics_task.cancel()

    logger.info("shutting_down")
    await engine.dispose()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Time Series Anomaly Detection API",
        version="0.0.0",
        lifespan=lifespan,
    )

    app.include_router(health_router)
    app.include_router(training_router)
    app.include_router(prediction_router)

    app.add_middleware(HTTPLatencyMiddleware)

    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    logger.info("app_created", port=settings.api_port)
    return app
