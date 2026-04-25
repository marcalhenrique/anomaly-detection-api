"""Unit tests for ModelMetadataRepository.save()

The SQLAlchemy AsyncSession is mocked so no real database is required.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

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
