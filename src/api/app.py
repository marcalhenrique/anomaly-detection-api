import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from src.core.config import get_settings
from src.core.structlog_config import get_logger
from src.api.routes.fit import router as training_router

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    os.environ.setdefault("AWS_ACCESS_KEY_ID", settings.minio_access_key)
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", settings.minio_secret_key)
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", settings.mlflow_s3_endpoint_url)
    logger.info("starting_up", cache_backend=settings.cache_backend)

    if settings.cache_backend == "redis":
        # TODO: start Redis Pub/Sub subscriber loop for L1 cache invalidation
        pass

    yield

    logger.info("shutting_down")
    if settings.cache_backend == "redis":
        # TODO: cancel subscriber loop task
        pass


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Time Series Anomaly Detection API",
        version="0.0.0",
        lifespan=lifespan,
    )

    # TODO: include routers
    app.include_router(training_router)
    # app.include_router(prediction_router)
    # app.include_router(healthcheck_router)

    logger.info("app_created", port=settings.api_port)
    return app
