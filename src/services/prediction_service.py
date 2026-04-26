import asyncio

from src.api.schemas import PredictResponse
from src.repositories.model_metadata import ModelMetadataRepository
from src.services.mlflow_service import MLflowService
from src.core.structlog_config import get_logger
from src.core.time_series import DataPoint

logger = get_logger(__name__)


class ModelNotFoundError(Exception):
    pass


class PredictionService:
    def __init__(
        self,
        mlflow_svc: MLflowService,
        repo: ModelMetadataRepository,
    ) -> None:
        self._mlflow_svc = mlflow_svc
        self._repo = repo

    async def predict(
        self,
        series_id: str,
        timestamp: int,
        value: float,
        version: str | None = None,
    ) -> PredictResponse:
        if version:
            metadata = await self._repo.get_by_version(series_id, version)
        else:
            metadata = await self._repo.get_latest_by_series_id(series_id)

        if metadata is None:
            raise ModelNotFoundError(f"No model found for series_id={series_id!r}")

        logger.info(
            "predict_started",
            series_id=series_id,
            version=metadata.version,
            timestamp=timestamp,
            value=value,
        )

        model = await asyncio.to_thread(
            self._mlflow_svc.load_model,
            metadata.mlflow_run_id,
        )
        data_point = DataPoint(timestamp=timestamp, value=value)
        is_anomaly = await asyncio.to_thread(model.predict, data_point)
        return PredictResponse(anomaly=is_anomaly, model_version=metadata.version)
