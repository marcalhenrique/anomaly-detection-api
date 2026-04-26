import asyncio

from structlog import get_logger

from src.services.training_lock import TrainingLockProtocol
from src.services.mlflow_service import MLflowService
from src.repositories.model_metadata import ModelMetadataRepository
from src.api.schemas import TrainRequest, TrainResponse
from src.services.anomaly_detection import AnomalyDetectionModel
from src.core.time_series import TimeSeries, DataPoint

logger = get_logger(__name__)


class TrainingService:
    def __init__(
        self,
        lock: TrainingLockProtocol,
        mlflow_svc: MLflowService,
        repo: ModelMetadataRepository,
    ) -> None:
        self._lock = lock
        self._mlflow_svc = mlflow_svc
        self._repo = repo

    async def fit(self, series_id: str, body: TrainRequest) -> TrainResponse:
        async with self._lock.acquire(series_id):
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
                )
            except Exception as e:
                logger.error("mlflow_save_failed", series_id=series_id, error=str(e))
                raise

            try:
                await self._repo.save(
                    series_id=series_id,
                    run_id=run_id,
                    model_version=model_version,
                    points_used=len(body.timestamps),
                )
            except Exception as e:
                logger.error(
                    "metadata_save_failed",
                    series_id=series_id,
                    version=model_version,
                    run_id=run_id,
                    error=str(e),
                )
                raise

            logger.info(
                "training_completed",
                series_id=series_id,
                version=model_version,
                run_id=run_id,
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
