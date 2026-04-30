"""Unit tests for src.core.database.get_db."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_get_db_yields_session_and_commits():
    """get_db must yield a session and commit on successful exit."""
    from src.core.database import get_db

    mock_session = AsyncMock()
    mock_factory = MagicMock()
    mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch("src.core.database.AsyncSessionFactory", mock_factory):
        gen = get_db()
        session = await gen.__anext__()

        assert session is mock_session

        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()

    mock_session.commit.assert_awaited_once()
    mock_session.rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_db_rolls_back_on_exception():
    """get_db must rollback and re-raise when an exception is thrown inside the context."""
    from src.core.database import get_db

    mock_session = AsyncMock()
    mock_factory = MagicMock()
    mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch("src.core.database.AsyncSessionFactory", mock_factory):
        gen = get_db()
        await gen.__anext__()

        with pytest.raises(RuntimeError, match="boom"):
            await gen.athrow(RuntimeError("boom"))

    mock_session.rollback.assert_awaited_once()
    mock_session.commit.assert_not_awaited()
