"""Unit tests for PredictionService.predict()

All external collaborators (MLflow, metadata cache) are mocked so these
tests remain fast, deterministic, and free of I/O.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.prediction_service import PredictionService, ModelNotFoundError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SERIES_ID = "sensor-01"
RUN_ID = "run-abc123"
MODEL_VERSION = "3"
TIMESTAMP = "1745000600"
VALUE_NORMAL = 10.0
VALUE_ANOMALY = 99.9


def _make_metadata(version: str = MODEL_VERSION, run_id: str = RUN_ID):
    metadata = MagicMock()
    metadata.version = version
    metadata.mlflow_run_id = run_id
    return metadata


def _make_cache(latest=None, by_version=None):
    cache = MagicMock()
    cache.get_latest.return_value = latest
    cache.get_by_version.return_value = by_version if by_version is not None else latest
    return cache


def _make_mlflow(is_anomaly: bool = False):
    model = MagicMock()
    model.predict.return_value = is_anomaly
    mlflow_svc = MagicMock()
    mlflow_svc.get_cached_model.return_value = None  # force load_model path
    mlflow_svc.load_model.return_value = model
    return mlflow_svc


def _make_service(mlflow_svc=None, cache=None) -> PredictionService:
    if mlflow_svc is None:
        mlflow_svc = _make_mlflow()
    if cache is None:
        cache = _make_cache(latest=_make_metadata())
    return PredictionService(mlflow_svc=mlflow_svc, metadata_cache=cache)


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_predict_returns_correct_response_for_normal_value():
    """predict() must return anomaly=False and correct model_version for normal values."""
    svc = _make_service(mlflow_svc=_make_mlflow(is_anomaly=False))
    response, _ = await svc.predict(
        series_id=SERIES_ID, timestamp=TIMESTAMP, value=VALUE_NORMAL
    )

    assert response.anomaly is False
    assert response.model_version == MODEL_VERSION


@pytest.mark.asyncio
async def test_predict_returns_correct_response_for_anomalous_value():
    """predict() must return anomaly=True for values that exceed the threshold."""
    svc = _make_service(mlflow_svc=_make_mlflow(is_anomaly=True))
    response, _ = await svc.predict(
        series_id=SERIES_ID, timestamp=TIMESTAMP, value=VALUE_ANOMALY
    )

    assert response.anomaly is True
    assert response.model_version == MODEL_VERSION


@pytest.mark.asyncio
async def test_predict_uses_latest_model_when_version_not_provided():
    """predict() must call cache.get_latest when no version is given."""
    cache = _make_cache(latest=_make_metadata())
    svc = _make_service(cache=cache)

    await svc.predict(series_id=SERIES_ID, timestamp=TIMESTAMP, value=VALUE_NORMAL)

    cache.get_latest.assert_called_once_with(SERIES_ID)
    cache.get_by_version.assert_not_called()


@pytest.mark.asyncio
async def test_predict_uses_specific_version_when_provided():
    """predict() must call cache.get_by_version when a version is explicitly requested."""
    cache = _make_cache(by_version=_make_metadata())
    svc = _make_service(cache=cache)

    await svc.predict(
        series_id=SERIES_ID, timestamp=TIMESTAMP, value=VALUE_NORMAL, version="2"
    )

    cache.get_by_version.assert_called_once_with(SERIES_ID, "2")
    cache.get_latest.assert_not_called()


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
    mlflow_svc.get_cached_model.return_value = None  # force load_model path
    mlflow_svc.load_model.return_value = model
    svc = _make_service(mlflow_svc=mlflow_svc)

    await svc.predict(series_id=SERIES_ID, timestamp=TIMESTAMP, value=VALUE_NORMAL)

    model.predict.assert_called_once()
    data_point = model.predict.call_args.args[0]
    assert data_point.timestamp == int(TIMESTAMP)
    assert data_point.value == VALUE_NORMAL


# ---------------------------------------------------------------------------
# Error-path tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_predict_raises_model_not_found_when_series_does_not_exist():
    """predict() must raise ModelNotFoundError when cache and DB have no model."""
    cache = _make_cache(latest=None)
    svc = _make_service(cache=cache)

    mock_repo = MagicMock()
    mock_repo.get_latest_by_series_id = AsyncMock(return_value=None)
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch(
            "src.services.prediction_service.ModelMetadataRepository",
            return_value=mock_repo,
        ),
        patch(
            "src.services.prediction_service.AsyncSessionFactory",
            return_value=mock_session,
        ),
    ):
        with pytest.raises(ModelNotFoundError, match=SERIES_ID):
            await svc.predict(
                series_id=SERIES_ID, timestamp=TIMESTAMP, value=VALUE_NORMAL
            )


@pytest.mark.asyncio
async def test_predict_raises_model_not_found_when_version_does_not_exist():
    """predict() must raise ModelNotFoundError when the requested version is not found."""
    cache = _make_cache(by_version=None, latest=None)
    svc = _make_service(cache=cache)

    mock_repo = MagicMock()
    mock_repo.get_by_version = AsyncMock(return_value=None)
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch(
            "src.services.prediction_service.ModelMetadataRepository",
            return_value=mock_repo,
        ),
        patch(
            "src.services.prediction_service.AsyncSessionFactory",
            return_value=mock_session,
        ),
    ):
        with pytest.raises(ModelNotFoundError):
            await svc.predict(
                series_id=SERIES_ID,
                timestamp=TIMESTAMP,
                value=VALUE_NORMAL,
                version="999",
            )


# ---------------------------------------------------------------------------
# DB fallback — metadata is stored in cache after DB hit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_predict_populates_cache_after_db_fallback():
    """When cache misses but DB has the model, predict() must call cache.set() with the metadata."""
    meta = _make_metadata()
    cache = _make_cache(latest=None)
    mlflow_svc = _make_mlflow()
    svc = _make_service(mlflow_svc=mlflow_svc, cache=cache)

    mock_repo = MagicMock()
    mock_repo.get_latest_by_series_id = AsyncMock(return_value=meta)
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch(
            "src.services.prediction_service.ModelMetadataRepository",
            return_value=mock_repo,
        ),
        patch(
            "src.services.prediction_service.AsyncSessionFactory",
            return_value=mock_session,
        ),
    ):
        await svc.predict(series_id=SERIES_ID, timestamp=TIMESTAMP, value=VALUE_NORMAL)

    cache.set.assert_called_once_with(meta)


@pytest.mark.asyncio
async def test_predict_returns_correct_response_after_db_fallback():
    """predict() must return a correct response when metadata is fetched from DB."""
    meta = _make_metadata(version="7")
    cache = _make_cache(latest=None)
    mlflow_svc = _make_mlflow(is_anomaly=False)
    svc = _make_service(mlflow_svc=mlflow_svc, cache=cache)

    mock_repo = MagicMock()
    mock_repo.get_latest_by_series_id = AsyncMock(return_value=meta)
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch(
            "src.services.prediction_service.ModelMetadataRepository",
            return_value=mock_repo,
        ),
        patch(
            "src.services.prediction_service.AsyncSessionFactory",
            return_value=mock_session,
        ),
    ):
        response, _ = await svc.predict(
            series_id=SERIES_ID, timestamp=TIMESTAMP, value=VALUE_NORMAL
        )

    assert response.model_version == "7"
    assert response.anomaly is False
