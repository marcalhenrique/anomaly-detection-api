import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from src.core.config import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info("Starting up", extra={"cache_backend": settings.cache_backend})

    if settings.cache_backend == "redis":
        # TODO: start Redis Pub/Sub subscriber loop for L1 cache invalidation
        pass

    yield

    logger.info("Shutting down")
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
    # app.include_router(training_router)
    # app.include_router(prediction_router)
    # app.include_router(healthcheck_router)

    logger.info("App created", extra={"port": settings.api_port})
    return app
