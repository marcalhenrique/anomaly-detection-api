"""Unit tests for API dependency factories."""

from unittest.mock import MagicMock

from src.api.dependencies import (
    _metadata_cache,
    _metrics_collector,
    _mlflow_service,
    get_metadata_cache,
    get_metrics_collector,
    get_mlflow_service,
    get_prediction_service,
    get_training_service,
)
from src.services.prediction_service import PredictionService
from src.services.training_service import TrainingService


# ---------------------------------------------------------------------------
# Singleton getters
# ---------------------------------------------------------------------------


def test_get_metadata_cache_returns_module_singleton():
    assert get_metadata_cache() is _metadata_cache


def test_get_mlflow_service_returns_module_singleton():
    assert get_mlflow_service() is _mlflow_service


def test_get_metrics_collector_returns_module_singleton():
    assert get_metrics_collector() is _metrics_collector


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def test_get_training_service_returns_training_service_instance():
    """get_training_service() must construct and return a TrainingService."""
    svc = get_training_service(
        db=MagicMock(),
        mlflow_svc=MagicMock(),
        metadata_cache=MagicMock(),
        metrics_collector=MagicMock(),
    )
    assert isinstance(svc, TrainingService)


def test_get_prediction_service_returns_prediction_service_instance():
    """get_prediction_service() must construct and return a PredictionService."""
    svc = get_prediction_service(
        mlflow_svc=MagicMock(),
        metadata_cache=MagicMock(),
    )
    assert isinstance(svc, PredictionService)
