"""Unit tests for PredictionService.predict()

All external collaborators (MLflow, repository) are mocked so these
tests remain fast, deterministic, and free of I/O.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.prediction_service import PredictionService, ModelNotFoundError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SERIES_ID = "sensor-vibration-01"
RUN_ID = "run-abc123"
MODEL_VERSION = "3"
TIMESTAMP = 1745000600
VALUE_NORMAL = 10.0
VALUE_ANOMALY = 99.9


def _make_metadata(version: str = MODEL_VERSION, run_id: str = RUN_ID):
    metadata = MagicMock()
    metadata.version = version
    metadata.mlflow_run_id = run_id
    return metadata


def _make_repo(metadata=None):
    repo = MagicMock()
    repo.get_latest_by_series_id = AsyncMock(return_value=metadata)
    repo.get_by_version = AsyncMock(return_value=metadata)
    return repo


def _make_mlflow(is_anomaly: bool = False):
    model = MagicMock()
    model.predict.return_value = is_anomaly
    mlflow_svc = MagicMock()
    mlflow_svc.load_model.return_value = model
    return mlflow_svc


def _make_service(mlflow_svc=None, repo=None) -> PredictionService:
    if mlflow_svc is None:
        mlflow_svc = _make_mlflow()
    if repo is None:
        repo = _make_repo(metadata=_make_metadata())
    return PredictionService(mlflow_svc=mlflow_svc, repo=repo)


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_predict_returns_correct_response_for_normal_value():
    """predict() must return anomaly=False and correct model_version for normal values."""
    svc = _make_service(mlflow_svc=_make_mlflow(is_anomaly=False))
    response = await svc.predict(
        series_id=SERIES_ID, timestamp=TIMESTAMP, value=VALUE_NORMAL
    )

    assert response.anomaly is False
    assert response.model_version == MODEL_VERSION


@pytest.mark.asyncio
async def test_predict_returns_correct_response_for_anomalous_value():
    """predict() must return anomaly=True for values that exceed the threshold."""
    svc = _make_service(mlflow_svc=_make_mlflow(is_anomaly=True))
    response = await svc.predict(
        series_id=SERIES_ID, timestamp=TIMESTAMP, value=VALUE_ANOMALY
    )

    assert response.anomaly is True
    assert response.model_version == MODEL_VERSION


@pytest.mark.asyncio
async def test_predict_uses_latest_model_when_version_not_provided():
    """predict() must call get_latest_by_series_id when no version is given."""
    repo = _make_repo(metadata=_make_metadata())
    svc = _make_service(repo=repo)

    await svc.predict(series_id=SERIES_ID, timestamp=TIMESTAMP, value=VALUE_NORMAL)

    repo.get_latest_by_series_id.assert_awaited_once_with(SERIES_ID)
    repo.get_by_version.assert_not_awaited()


@pytest.mark.asyncio
async def test_predict_uses_specific_version_when_provided():
    """predict() must call get_by_version when a version is explicitly requested."""
    repo = _make_repo(metadata=_make_metadata())
    svc = _make_service(repo=repo)

    await svc.predict(
        series_id=SERIES_ID, timestamp=TIMESTAMP, value=VALUE_NORMAL, version="2"
    )

    repo.get_by_version.assert_awaited_once_with(SERIES_ID, "2")
    repo.get_latest_by_series_id.assert_not_awaited()


@pytest.mark.asyncio
async def test_predict_loads_model_using_run_id_from_metadata():
    """predict() must load the model using the run_id stored in metadata."""
    mlflow_svc = _make_mlflow()
    svc = _make_service(mlflow_svc=mlflow_svc)

    await svc.predict(series_id=SERIES_ID, timestamp=TIMESTAMP, value=VALUE_NORMAL)

    mlflow_svc.load_model.assert_called_once_with(RUN_ID)


@pytest.mark.asyncio
async def test_predict_calls_model_predict_with_correct_data_point():
    """predict() must call model.predict() with the correct DataPoint."""
    model = MagicMock()
    model.predict.return_value = False
    mlflow_svc = MagicMock()
    mlflow_svc.load_model.return_value = model
    svc = _make_service(mlflow_svc=mlflow_svc)

    await svc.predict(series_id=SERIES_ID, timestamp=TIMESTAMP, value=VALUE_NORMAL)

    model.predict.assert_called_once()
    data_point = model.predict.call_args.args[0]
    assert data_point.timestamp == TIMESTAMP
    assert data_point.value == VALUE_NORMAL


# ---------------------------------------------------------------------------
# Error-path tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_predict_raises_model_not_found_when_series_does_not_exist():
    """predict() must raise ModelNotFoundError when the series has no trained model."""
    repo = _make_repo(metadata=None)  # simulates no model in DB
    svc = _make_service(repo=repo)

    with pytest.raises(ModelNotFoundError, match=SERIES_ID):
        await svc.predict(series_id=SERIES_ID, timestamp=TIMESTAMP, value=VALUE_NORMAL)


@pytest.mark.asyncio
async def test_predict_raises_model_not_found_when_version_does_not_exist():
    """predict() must raise ModelNotFoundError when the requested version is not found."""
    repo = _make_repo(metadata=None)  # simulates version not found
    svc = _make_service(repo=repo)

    with pytest.raises(ModelNotFoundError):
        await svc.predict(
            series_id=SERIES_ID, timestamp=TIMESTAMP, value=VALUE_NORMAL, version="999"
        )
