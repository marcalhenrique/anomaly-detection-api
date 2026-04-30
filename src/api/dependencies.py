from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import get_settings
from src.core.database import get_db
from src.repositories.model_metadata import ModelMetadataRepository
from src.services.mlflow_service import MLflowService
from src.services.training_lock import RedisTrainingLock
from src.services.training_service import TrainingService
from src.services.prediction_service import PredictionService
from src.services.metrics_collector import MetricsCollector
from src.services.metadata_cache import MetadataCache

settings = get_settings()

_metrics_collector = MetricsCollector(
    latency_window=settings.healthcheck_latency_window
)

_lock = RedisTrainingLock()

_metadata_cache = MetadataCache()

_mlflow_service = MLflowService(
    tracking_uri=settings.mlflow_tracking_uri,
    artifact_bucket=settings.minio_bucket_name,
)

def get_metadata_cache() -> MetadataCache:
    return _metadata_cache

def get_mlflow_service() -> MLflowService:
    return _mlflow_service

def get_metrics_collector() -> MetricsCollector:
    return _metrics_collector

def get_training_service(
    db: AsyncSession = Depends(get_db),
    mlflow_svc: MLflowService = Depends(get_mlflow_service),
    metadata_cache: MetadataCache = Depends(get_metadata_cache),
    metrics_collector: MetricsCollector = Depends(get_metrics_collector),
) -> TrainingService:
    return TrainingService(
        lock=_lock,
        mlflow_svc=mlflow_svc,
        repo=ModelMetadataRepository(session=db),
        metrics_collector=metrics_collector,
        metadata_cache=metadata_cache,
    )

def get_prediction_service(
    mlflow_svc: MLflowService = Depends(get_mlflow_service),
    metadata_cache: MetadataCache = Depends(get_metadata_cache),
) -> PredictionService:
    return PredictionService(
        mlflow_svc=mlflow_svc,
        metadata_cache=metadata_cache,
    )

