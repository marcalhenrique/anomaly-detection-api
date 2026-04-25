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
) -> TrainingService:
    if lock is None:
        lock = _make_lock()
    if mlflow_svc is None:
        mlflow_svc = _make_mlflow()
    if repo is None:
        repo = _make_repo()
    return TrainingService(lock=lock, mlflow_svc=mlflow_svc, repo=repo)


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


def _make_repo():
    repo = MagicMock()
    repo.save = AsyncMock(return_value=None)
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
    """fit() must pass series_id, a fitted model, and point count to MLflow."""
    mlflow_svc = _make_mlflow()
    svc = _make_service(mlflow_svc=mlflow_svc)

    await svc.fit(series_id=SERIES_ID, body=_make_request())

    mlflow_svc.save_model.assert_called_once()
    call_args = mlflow_svc.save_model.call_args
    assert call_args.args[0] == SERIES_ID  # series_id
    assert call_args.args[2] == N_POINTS  # points_used


@pytest.mark.asyncio
async def test_fit_calls_repo_save_with_correct_args():
    """fit() must persist metadata with the correct identifiers and point count."""
    repo = _make_repo()
    svc = _make_service(repo=repo)

    await svc.fit(series_id=SERIES_ID, body=_make_request())

    repo.save.assert_awaited_once_with(
        series_id=SERIES_ID,
        run_id=RUN_ID,
        model_version=MODEL_VERSION,
        points_used=N_POINTS,
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
