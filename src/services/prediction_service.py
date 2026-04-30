import asyncio
import time

from src.api.schemas import PredictResponse
from src.repositories.model_metadata import ModelMetadataRepository
from src.services.mlflow_service import MLflowService
from src.core.structlog_config import get_logger
from src.core.time_series import DataPoint
from src.services.metrics_collector import timed
from src.services.metadata_cache import MetadataCache
from src.core.database import AsyncSessionFactory

logger = get_logger(__name__)

class ModelNotFoundError(Exception):
    pass

class PredictionService:
    def __init__(
        self,
        mlflow_svc: MLflowService,
        metadata_cache: MetadataCache,
    ) -> None:
        self._mlflow_svc = mlflow_svc
        self._metadata_cache = metadata_cache

    async def predict(
        self,
        series_id: str,
        timestamp: str,
        value: float,
        version: str | None = None,
    ) -> tuple[PredictResponse, dict[str, float]]:
        timings: dict[str, float] = {}
        start = time.perf_counter()

        with timed("metadata_ms", timings):
            if version:
                metadata = self._metadata_cache.get_by_version(series_id, version)
            else:
                metadata = self._metadata_cache.get_latest(series_id)

            if metadata is None:
                async with AsyncSessionFactory() as session:
                    repo = ModelMetadataRepository(session)
                    if version:
                        metadata = await repo.get_by_version(series_id, version)
                    else:
                        metadata = await repo.get_latest_by_series_id(series_id)

                if metadata is None:
                    raise ModelNotFoundError(
                        f"No model found for series_id={series_id!r}"
                    )

                self._metadata_cache.set(metadata)

        with timed("model_load_ms", timings):
            model = self._mlflow_svc.get_cached_model(metadata.mlflow_run_id)
            if model is None:
                model = await asyncio.to_thread(
                    self._mlflow_svc.load_model, metadata.mlflow_run_id
                )

        with timed("inference_ms", timings):
            is_anomaly = model.predict(DataPoint(timestamp=int(timestamp), value=value))

        timings["predict_operation_ms"] = round((time.perf_counter() - start) * 1000, 3)

        logger.info(
            "predict_finished",
            series_id=series_id,
            version=metadata.version,
            timestamp=timestamp,
            value=value,
            is_anomaly=is_anomaly,
            **timings,
        )

        return PredictResponse(
            anomaly=is_anomaly, model_version=metadata.version
        ), timings

