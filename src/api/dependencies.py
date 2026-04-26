from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import get_settings
from src.core.database import get_db
from src.repositories.model_metadata import ModelMetadataRepository
from src.services.mlflow_service import MLflowService
from src.services.training_lock import LocalTrainingLock
from src.services.training_service import TrainingService
from src.services.prediction_service import PredictionService

settings = get_settings()
_lock = LocalTrainingLock()
_mlflow_service = MLflowService(
    tracking_uri=settings.mlflow_tracking_uri,
    artifact_bucket=settings.minio_bucket_name,
)


def get_mlflow_service() -> MLflowService:
    return _mlflow_service


def get_training_service(
    db: AsyncSession = Depends(get_db),
    mlflow_svc: MLflowService = Depends(get_mlflow_service),
) -> TrainingService:
    return TrainingService(
        lock=_lock,
        mlflow_svc=mlflow_svc,
        repo=ModelMetadataRepository(session=db),
    )


def get_prediction_service(
    db: AsyncSession = Depends(get_db),
    mlflow_svc: MLflowService = Depends(get_mlflow_service),
) -> PredictionService:
    return PredictionService(
        mlflow_svc=mlflow_svc,
        repo=ModelMetadataRepository(session=db),
    )
