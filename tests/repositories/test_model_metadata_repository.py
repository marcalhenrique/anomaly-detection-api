"""Unit tests for ModelMetadataRepository.

The SQLAlchemy AsyncSession is mocked so no real database is required.
Covers save(), get_latest_by_series_id(), get_by_version(),
get_all_latest(), and count_distinct_series().
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.model_metadata import ModelMetadata
from src.repositories.model_metadata import ModelMetadataRepository

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SERIES_ID = "sensor-vibration-01"
RUN_ID = "run-abc123"
MODEL_VERSION = "3"
POINTS_USED = 42


def _make_session() -> AsyncMock:
    session = MagicMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    return session


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_adds_record_to_session():
    """save() must add exactly one record to the session."""
    session = _make_session()
    repo = ModelMetadataRepository(session=session)

    await repo.save(
        series_id=SERIES_ID,
        run_id=RUN_ID,
        model_version=MODEL_VERSION,
        points_used=POINTS_USED,
    )

    session.add.assert_called_once()


@pytest.mark.asyncio
async def test_save_calls_commit():
    """save() must commit the session after adding the record."""
    session = _make_session()
    repo = ModelMetadataRepository(session=session)

    await repo.save(
        series_id=SERIES_ID,
        run_id=RUN_ID,
        model_version=MODEL_VERSION,
        points_used=POINTS_USED,
    )

    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_save_returns_model_metadata_with_correct_fields():
    """save() must return a ModelMetadata instance with the provided field values."""
    session = _make_session()
    repo = ModelMetadataRepository(session=session)

    record = await repo.save(
        series_id=SERIES_ID,
        run_id=RUN_ID,
        model_version=MODEL_VERSION,
        points_used=POINTS_USED,
    )

    assert record.series_id == SERIES_ID
    assert record.mlflow_run_id == RUN_ID
    assert record.version == MODEL_VERSION
    assert record.points_used == POINTS_USED


@pytest.mark.asyncio
async def test_save_preserves_add_then_commit_order():
    """add() must be called before commit() - wrong order would lose data."""
    call_order: list[str] = []
    session = MagicMock()
    session.add = MagicMock(side_effect=lambda _: call_order.append("add"))
    session.commit = AsyncMock(side_effect=lambda: call_order.append("commit"))

    repo = ModelMetadataRepository(session=session)
    await repo.save(
        series_id=SERIES_ID,
        run_id=RUN_ID,
        model_version=MODEL_VERSION,
        points_used=POINTS_USED,
    )

    assert call_order == ["add", "commit"]


# ---------------------------------------------------------------------------
# Helpers for query tests
# ---------------------------------------------------------------------------


def _make_record(
    series_id: str = SERIES_ID,
    version: str = MODEL_VERSION,
    run_id: str = RUN_ID,
    points_used: int = POINTS_USED,
) -> ModelMetadata:
    record = ModelMetadata(
        series_id=series_id,
        version=version,
        mlflow_run_id=run_id,
        points_used=points_used,
    )
    return record


def _make_execute_result(scalar=None, scalars=None, scalar_one=None) -> MagicMock:
    """Return a mock that simulates an AsyncSession.execute() result."""
    result = MagicMock()
    result.scalar_one_or_none.return_value = scalar
    result.scalar_one.return_value = scalar_one
    if scalars is not None:
        result.scalars.return_value.all.return_value = scalars
    return result


def _make_query_session(execute_result: MagicMock) -> AsyncMock:
    session = MagicMock()
    session.execute = AsyncMock(return_value=execute_result)
    return session


# ---------------------------------------------------------------------------
# get_latest_by_series_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_latest_by_series_id_returns_record_when_found():
    """get_latest_by_series_id() must return the record when one exists."""
    record = _make_record()
    session = _make_query_session(_make_execute_result(scalar=record))
    repo = ModelMetadataRepository(session=session)

    result = await repo.get_latest_by_series_id(SERIES_ID)

    assert result is record
    session.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_latest_by_series_id_returns_none_when_not_found():
    """get_latest_by_series_id() must return None when no record exists."""
    session = _make_query_session(_make_execute_result(scalar=None))
    repo = ModelMetadataRepository(session=session)

    result = await repo.get_latest_by_series_id("unknown-series")

    assert result is None


# ---------------------------------------------------------------------------
# get_by_version
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_by_version_returns_record_when_found():
    """get_by_version() must return the matching record when it exists."""
    record = _make_record(version="5")
    session = _make_query_session(_make_execute_result(scalar=record))
    repo = ModelMetadataRepository(session=session)

    result = await repo.get_by_version(SERIES_ID, "5")

    assert result is record
    session.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_by_version_returns_none_when_not_found():
    """get_by_version() must return None when the requested version does not exist."""
    session = _make_query_session(_make_execute_result(scalar=None))
    repo = ModelMetadataRepository(session=session)

    result = await repo.get_by_version(SERIES_ID, "999")

    assert result is None


# ---------------------------------------------------------------------------
# get_all_latest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_all_latest_returns_list_of_records():
    """get_all_latest() must return a list of the latest record per series."""
    records = [_make_record(series_id="s1"), _make_record(series_id="s2")]
    session = _make_query_session(_make_execute_result(scalars=records))
    repo = ModelMetadataRepository(session=session)

    result = await repo.get_all_latest()

    assert result == records


@pytest.mark.asyncio
async def test_get_all_latest_returns_empty_list_when_no_records():
    """get_all_latest() must return an empty list when no records exist."""
    session = _make_query_session(_make_execute_result(scalars=[]))
    repo = ModelMetadataRepository(session=session)

    result = await repo.get_all_latest()

    assert result == []


# ---------------------------------------------------------------------------
# count_distinct_series
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_count_distinct_series_returns_correct_count():
    """count_distinct_series() must return the count from the database."""
    session = _make_query_session(_make_execute_result(scalar_one=7))
    repo = ModelMetadataRepository(session=session)

    count = await repo.count_distinct_series()

    assert count == 7


@pytest.mark.asyncio
async def test_count_distinct_series_returns_zero_when_empty():
    """count_distinct_series() must return 0 when the table is empty."""
    session = _make_query_session(_make_execute_result(scalar_one=0))
    repo = ModelMetadataRepository(session=session)

    count = await repo.count_distinct_series()

    assert count == 0
