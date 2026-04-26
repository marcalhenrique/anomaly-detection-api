import os
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from src.core.config import get_settings
from src.core.structlog_config import get_logger
from src.api.routes.fit import router as training_router
from src.api.routes.predict import router as prediction_router
from src.core.database import AsyncSessionFactory
from src.repositories.model_metadata import ModelMetadataRepository
from src.api.dependencies import _mlflow_service

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

    for record in records:
        try:
            await asyncio.to_thread(
                _mlflow_service.load_model,
                record.mlflow_run_id,
            )
            logger.info(
                "startup_cache_warmed",
                series_id=record.series_id,
                run_id=record.mlflow_run_id,
            )
        except Exception as e:
            logger.warning(
                "startup_cache_warm_failed",
                series_id=record.series_id,
                run_id=record.mlflow_run_id,
                error=str(e),
            )

    yield

    logger.info("shutting_down")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Time Series Anomaly Detection API",
        version="0.0.0",
        lifespan=lifespan,
    )

    app.include_router(training_router)
    app.include_router(prediction_router)
    # app.include_router(healthcheck_router)

    logger.info("app_created", port=settings.api_port)
    return app
