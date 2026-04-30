"""Unit tests for TrainingService.fit()

All external collaborators (lock, MLflow, repository) are mocked so these
tests remain fast, deterministic, and free of I/O.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.api.schemas import TrainRequest
from src.services.training_service import TrainingService

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SERIES_ID = "sensor-vibration-01"
RUN_ID = "run-abc123"
MODEL_VERSION = "3"
N_POINTS = 10

_TIMESTAMPS = list(range(N_POINTS))
_VALUES = [float(i) for i in range(N_POINTS)]


def _make_request() -> TrainRequest:
    return TrainRequest(timestamps=_TIMESTAMPS, values=_VALUES)


def _make_service(
    lock=None,
    mlflow_svc=None,
    repo=None,
    metrics_collector=None,
    metadata_cache=None,
    latest=None,
) -> TrainingService:
    if lock is None:
        lock = _make_lock()
    if mlflow_svc is None:
        mlflow_svc = _make_mlflow()
    if repo is None:
        repo = _make_repo(latest=latest)
    if metrics_collector is None:
        metrics_collector = MagicMock()
    if metadata_cache is None:
        metadata_cache = MagicMock()
    return TrainingService(
        lock=lock,
        mlflow_svc=mlflow_svc,
        repo=repo,
        metrics_collector=metrics_collector,
        metadata_cache=metadata_cache,
    )


def _make_lock():
    """Return a mock that behaves as an async context manager."""
    lock = MagicMock()

    @asynccontextmanager
    async def acquire(series_id: str):
        yield

    lock.acquire = acquire
    return lock


def _make_mlflow(run_id: str = RUN_ID, version: str = MODEL_VERSION):
    svc = MagicMock()
    svc.save_model.return_value = (run_id, version)
    return svc


def _make_repo(latest=None):
    repo = MagicMock()
    repo.save = AsyncMock(return_value=None)
    repo.get_latest_by_series_id = AsyncMock(return_value=latest)
    return repo


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fit_returns_correct_train_response():
    """fit() must return a TrainResponse with correct series_id, version, points_used."""
    svc = _make_service()
    response = await svc.fit(series_id=SERIES_ID, body=_make_request())

    assert response.series_id == SERIES_ID
    assert response.version == MODEL_VERSION
    assert response.points_used == N_POINTS


@pytest.mark.asyncio
async def test_fit_calls_mlflow_save_model_with_correct_args():
    """fit() must pass series_id, a fitted model, point count, and training data to MLflow."""
    mlflow_svc = _make_mlflow()
    svc = _make_service(mlflow_svc=mlflow_svc)
    body = _make_request()

    await svc.fit(series_id=SERIES_ID, body=body)

    mlflow_svc.save_model.assert_called_once()
    call_args = mlflow_svc.save_model.call_args
    assert call_args.args[0] == SERIES_ID  # series_id
    assert call_args.args[2] == N_POINTS  # points_used
    assert call_args.args[3] == _TIMESTAMPS  # timestamps
    assert call_args.args[4] == _VALUES  # values


@pytest.mark.asyncio
async def test_fit_calls_repo_save_with_correct_args():
    """fit() must persist metadata with the correct identifiers and point count."""
    from src.services.training_service import _compute_data_hash

    repo = _make_repo()
    svc = _make_service(repo=repo)
    body = _make_request()

    await svc.fit(series_id=SERIES_ID, body=body)

    repo.save.assert_awaited_once_with(
        series_id=SERIES_ID,
        run_id=RUN_ID,
        model_version=MODEL_VERSION,
        points_used=N_POINTS,
        data_hash=_compute_data_hash(body),
    )


@pytest.mark.asyncio
async def test_fit_acquires_lock_for_series_id():
    """fit() must acquire the training lock using the correct series_id."""
    acquired_ids: list[str] = []

    class TrackingLock:
        @asynccontextmanager
        async def acquire(self, series_id: str):
            acquired_ids.append(series_id)
            yield

    svc = _make_service(lock=TrackingLock())
    await svc.fit(series_id=SERIES_ID, body=_make_request())

    assert acquired_ids == [SERIES_ID]


# ---------------------------------------------------------------------------
# Failure propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fit_raises_when_mlflow_save_fails():
    """If MLflow raises, fit() must propagate the exception."""
    mlflow_svc = _make_mlflow()
    mlflow_svc.save_model.side_effect = RuntimeError("mlflow unavailable")

    svc = _make_service(mlflow_svc=mlflow_svc)

    with pytest.raises(RuntimeError, match="mlflow unavailable"):
        await svc.fit(series_id=SERIES_ID, body=_make_request())


@pytest.mark.asyncio
async def test_fit_raises_when_repo_save_fails():
    """If the repository raises, fit() must propagate the exception."""
    repo = _make_repo()
    repo.save = AsyncMock(side_effect=RuntimeError("db unavailable"))

    svc = _make_service(repo=repo)

    with pytest.raises(RuntimeError, match="db unavailable"):
        await svc.fit(series_id=SERIES_ID, body=_make_request())


@pytest.mark.asyncio
async def test_fit_does_not_call_repo_when_mlflow_fails():
    """If MLflow fails, the repository must not be called."""
    mlflow_svc = _make_mlflow()
    mlflow_svc.save_model.side_effect = RuntimeError("mlflow down")
    repo = _make_repo()

    svc = _make_service(mlflow_svc=mlflow_svc, repo=repo)

    with pytest.raises(RuntimeError):
        await svc.fit(series_id=SERIES_ID, body=_make_request())

    repo.save.assert_not_awaited()


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def _make_existing_metadata(body: "TrainRequest", version: str = MODEL_VERSION):
    """Return a mock ModelMetadata whose data_hash matches body."""
    from src.services.training_service import _compute_data_hash

    meta = MagicMock()
    meta.data_hash = _compute_data_hash(body)
    meta.version = version
    meta.points_used = len(body.timestamps)
    return meta


@pytest.mark.asyncio
async def test_fit_returns_existing_version_when_data_is_identical():
    """If the same data is submitted again, fit() must return the existing version."""
    body = _make_request()
    svc = _make_service(latest=_make_existing_metadata(body, version="5"))

    response = await svc.fit(series_id=SERIES_ID, body=body)

    assert response.series_id == SERIES_ID
    assert response.version == "5"
    assert response.points_used == N_POINTS


@pytest.mark.asyncio
async def test_fit_skips_mlflow_and_repo_when_data_is_identical():
    """Duplicate data must not trigger MLflow save or repo.save."""
    body = _make_request()
    mlflow_svc = _make_mlflow()
    repo = _make_repo(latest=_make_existing_metadata(body))
    svc = _make_service(mlflow_svc=mlflow_svc, repo=repo)

    await svc.fit(series_id=SERIES_ID, body=body)

    mlflow_svc.save_model.assert_not_called()
    repo.save.assert_not_awaited()


@pytest.mark.asyncio
async def test_fit_trains_when_values_differ():
    """Even with the same timestamps, different values must trigger a new training."""
    body = _make_request()
    different_body = TrainRequest(
        timestamps=_TIMESTAMPS,
        values=[v + 1.0 for v in _VALUES],
    )
    repo = _make_repo(latest=_make_existing_metadata(body))
    mlflow_svc = _make_mlflow()
    svc = _make_service(mlflow_svc=mlflow_svc, repo=repo)

    await svc.fit(series_id=SERIES_ID, body=different_body)

    mlflow_svc.save_model.assert_called_once()


@pytest.mark.asyncio
async def test_fit_trains_when_no_existing_model():
    """First training for a series must always proceed."""
    mlflow_svc = _make_mlflow()
    svc = _make_service(mlflow_svc=mlflow_svc, latest=None)

    await svc.fit(series_id=SERIES_ID, body=_make_request())

    mlflow_svc.save_model.assert_called_once()


# ---------------------------------------------------------------------------
# Post-training cache warm failure is swallowed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fit_returns_successfully_when_post_training_cache_warm_fails():
    """If load_model raises after training, fit() must log a warning and still return normally."""
    mlflow_svc = _make_mlflow()
    mlflow_svc.load_model.side_effect = RuntimeError("storage unavailable")

    svc = _make_service(mlflow_svc=mlflow_svc)
    response = await svc.fit(series_id=SERIES_ID, body=_make_request())

    assert response.series_id == SERIES_ID
    assert response.version == MODEL_VERSION
    assert response.points_used == N_POINTS


@pytest.mark.asyncio
async def test_fit_does_not_propagate_load_model_exception_after_save():
    """The cache-warming load_model exception must NOT bubble up out of fit()."""
    mlflow_svc = _make_mlflow()
    mlflow_svc.load_model.side_effect = OSError("disk full")

    svc = _make_service(mlflow_svc=mlflow_svc)

    # Must not raise
    await svc.fit(series_id=SERIES_ID, body=_make_request())
