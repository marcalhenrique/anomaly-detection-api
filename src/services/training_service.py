import asyncio
import hashlib
import json
import time
from structlog import get_logger

from src.services.training_lock import TrainingLockProtocol
from src.services.mlflow_service import MLflowService
from src.repositories.model_metadata import ModelMetadataRepository
from src.api.schemas import TrainRequest, TrainResponse
from src.services.anomaly_detection import AnomalyDetectionModel
from src.core.time_series import TimeSeries, DataPoint
from src.services.metrics_collector import MetricsCollector
from src.services.metadata_cache import MetadataCache

logger = get_logger(__name__)

def _compute_data_hash(body: "TrainRequest") -> str:
    payload = json.dumps(
        {"timestamps": body.timestamps, "values": body.values}, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()

class TrainingService:
    def __init__(
        self,
        lock: TrainingLockProtocol,
        mlflow_svc: MLflowService,
        repo: ModelMetadataRepository,
        metrics_collector: MetricsCollector,
        metadata_cache: MetadataCache,
    ) -> None:
        self._lock = lock
        self._mlflow_svc = mlflow_svc
        self._repo = repo
        self._metrics_collector = metrics_collector
        self._metadata_cache = metadata_cache

    async def fit(self, series_id: str, body: TrainRequest) -> TrainResponse:
        incoming_hash = _compute_data_hash(body)

        async with self._lock.acquire(series_id):
            latest = await self._repo.get_latest_by_series_id(series_id)
            if latest is not None and latest.data_hash == incoming_hash:
                logger.info(
                    "training_skipped_duplicate",
                    series_id=series_id,
                    version=latest.version,
                )
                return TrainResponse(
                    series_id=series_id,
                    version=latest.version,
                    points_used=latest.points_used,
                )

            start = time.perf_counter()
            logger.info(
                "training_started", series_id=series_id, points=len(body.timestamps)
            )
            time_series = TimeSeries(
                data=[
                    DataPoint(timestamp=t, value=v)
                    for t, v in zip(body.timestamps, body.values)
                ]
            )

            model = AnomalyDetectionModel().fit(time_series)

            try:
                run_id, model_version = await asyncio.to_thread(
                    self._mlflow_svc.save_model,
                    series_id,
                    model,
                    len(body.timestamps),
                    body.timestamps,
                    body.values,
                )
            except Exception as e:
                logger.error("mlflow_save_failed", series_id=series_id, error=str(e))
                raise

            try:
                metadata = await self._repo.save(
                    series_id=series_id,
                    run_id=run_id,
                    model_version=model_version,
                    points_used=len(body.timestamps),
                    data_hash=incoming_hash,
                )
                self._metadata_cache.set(metadata)
            except Exception as e:
                logger.error(
                    "metadata_save_failed",
                    series_id=series_id,
                    version=model_version,
                    run_id=run_id,
                    error=str(e),
                )
                raise
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._metrics_collector.record_training(elapsed_ms)
            if latest is None:
                self._metrics_collector.inc_series_trained()
            logger.info(
                "training_completed",
                series_id=series_id,
                version=model_version,
                run_id=run_id,
                elapsed_ms=elapsed_ms,
            )

            try:
                await asyncio.to_thread(self._mlflow_svc.load_model, run_id)
                logger.info("cache_warmed", series_id=series_id, run_id=run_id)
            except Exception as e:
                logger.warning(
                    "cache_warm_failed",
                    series_id=series_id,
                    run_id=run_id,
                    error=str(e),
                )

        return TrainResponse(
            series_id=series_id,
            version=model_version,
            points_used=len(body.timestamps),
        )

