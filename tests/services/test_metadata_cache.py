"""Unit tests for MetadataCache.

fakeredis is used via monkeypatch so no live Redis is required.
"""

from datetime import datetime, timezone
from unittest.mock import patch

import fakeredis
import pytest

from src.models.model_metadata import ModelMetadata
from src.services.metadata_cache import MetadataCache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SERIES_ID = "sensor-vibration-01"
VERSION = "3"
RUN_ID = "run-abc123"
POINTS_USED = 100
DATA_HASH = "deadbeef" * 8


def _make_metadata(
    series_id: str = SERIES_ID,
    version: str = VERSION,
    run_id: str = RUN_ID,
    points_used: int = POINTS_USED,
    data_hash: str | None = DATA_HASH,
) -> ModelMetadata:
    meta = ModelMetadata(
        series_id=series_id,
        version=version,
        mlflow_run_id=run_id,
        points_used=points_used,
        data_hash=data_hash,
    )
    meta.id = 1
    meta.trained_at = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    return meta


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_redis():
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def cache(fake_redis):
    """MetadataCache backed by an in-memory fakeredis instance."""
    with patch("src.services.metadata_cache.get_redis_client", return_value=fake_redis):
        return MetadataCache()


# ---------------------------------------------------------------------------
# set → get_latest round-trip
# ---------------------------------------------------------------------------


def test_get_latest_returns_none_when_cache_is_empty(cache):
    """get_latest() must return None when nothing has been stored."""
    assert cache.get_latest(SERIES_ID) is None


def test_get_latest_returns_metadata_after_set(cache):
    """get_latest() must return the stored metadata after a set() call."""
    meta = _make_metadata()
    cache.set(meta)
    result = cache.get_latest(SERIES_ID)
    assert result is not None
    assert result.series_id == SERIES_ID


def test_get_latest_preserves_all_fields(cache):
    """get_latest() must reconstruct all fields faithfully."""
    meta = _make_metadata()
    cache.set(meta)
    result = cache.get_latest(SERIES_ID)
    assert result.version == VERSION
    assert result.mlflow_run_id == RUN_ID
    assert result.points_used == POINTS_USED
    assert result.data_hash == DATA_HASH


def test_get_latest_preserves_trained_at(cache):
    """get_latest() must preserve the trained_at timestamp."""
    meta = _make_metadata()
    cache.set(meta)
    result = cache.get_latest(SERIES_ID)
    assert result.trained_at == datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# set → get_by_version round-trip
# ---------------------------------------------------------------------------


def test_get_by_version_returns_none_when_cache_is_empty(cache):
    """get_by_version() must return None when nothing has been stored."""
    assert cache.get_by_version(SERIES_ID, VERSION) is None


def test_get_by_version_returns_metadata_after_set(cache):
    """get_by_version() must return the stored metadata after a set() call."""
    meta = _make_metadata()
    cache.set(meta)
    result = cache.get_by_version(SERIES_ID, VERSION)
    assert result is not None
    assert result.version == VERSION


def test_get_by_version_preserves_all_fields(cache):
    """get_by_version() must reconstruct all fields faithfully."""
    meta = _make_metadata()
    cache.set(meta)
    result = cache.get_by_version(SERIES_ID, VERSION)
    assert result.series_id == SERIES_ID
    assert result.mlflow_run_id == RUN_ID
    assert result.points_used == POINTS_USED
    assert result.data_hash == DATA_HASH


def test_get_by_version_returns_none_for_wrong_version(cache):
    """get_by_version() must return None when a different version is requested."""
    cache.set(_make_metadata(version="1"))
    assert cache.get_by_version(SERIES_ID, "99") is None


# ---------------------------------------------------------------------------
# Independence across series IDs
# ---------------------------------------------------------------------------


def test_get_latest_returns_none_for_different_series_id(cache):
    """Caching one series must not affect another series' lookup."""
    cache.set(_make_metadata(series_id="sensor-A"))
    assert cache.get_latest("sensor-B") is None


def test_set_multiple_series_are_independent(cache):
    """Each series_id must maintain its own cache entry."""
    cache.set(_make_metadata(series_id="sensor-A", version="1", run_id="run-A"))
    cache.set(_make_metadata(series_id="sensor-B", version="2", run_id="run-B"))
    assert cache.get_latest("sensor-A").mlflow_run_id == "run-A"
    assert cache.get_latest("sensor-B").mlflow_run_id == "run-B"


# ---------------------------------------------------------------------------
# Optional data_hash (nullable field)
# ---------------------------------------------------------------------------


def test_set_and_get_latest_with_no_data_hash(cache):
    """set() and get_latest() must work correctly when data_hash is None."""
    meta = _make_metadata(data_hash=None)
    cache.set(meta)
    result = cache.get_latest(SERIES_ID)
    assert result.data_hash is None


# ---------------------------------------------------------------------------
# Redis fallback paths (L1 cold, Redis warm)
# ---------------------------------------------------------------------------


def test_get_latest_falls_back_to_redis_when_l1_is_cold(cache):
    """get_latest() must fetch from Redis and repopulate L1 when L1 has expired."""
    meta = _make_metadata()
    cache.set(meta)
    cache._local.clear()

    result = cache.get_latest(SERIES_ID)

    assert result is not None
    assert result.series_id == SERIES_ID
    assert result.version == VERSION


def test_get_latest_repopulates_l1_after_redis_hit(cache):
    """After a Redis fallback, the next get_latest() must be served from L1."""
    meta = _make_metadata()
    cache.set(meta)
    cache._local.clear()

    cache.get_latest(SERIES_ID)  # Redis hit — repopulates L1
    # Clear Redis to prove the second call is served from L1 only
    cache._redis.flushall()

    result = cache.get_latest(SERIES_ID)
    assert result is not None
    assert result.series_id == SERIES_ID


def test_get_by_version_falls_back_to_redis_when_l1_is_cold(cache):
    """get_by_version() must fetch from Redis and repopulate L1 when L1 has expired."""
    meta = _make_metadata()
    cache.set(meta)
    cache._local.clear()

    result = cache.get_by_version(SERIES_ID, VERSION)

    assert result is not None
    assert result.version == VERSION


def test_get_by_version_repopulates_l1_after_redis_hit(cache):
    """After a Redis fallback, the next get_by_version() must be served from L1."""
    meta = _make_metadata()
    cache.set(meta)
    cache._local.clear()

    cache.get_by_version(SERIES_ID, VERSION)  # Redis hit — repopulates L1
    cache._redis.flushall()

    result = cache.get_by_version(SERIES_ID, VERSION)
    assert result is not None
    assert result.version == VERSION


# ---------------------------------------------------------------------------
# _dict_to_metadata (covered via Redis fallback round-trip)
# ---------------------------------------------------------------------------


def test_dict_to_metadata_preserves_all_fields(cache):
    """Fields serialised to Redis must be fully reconstructed via _dict_to_metadata."""
    meta = _make_metadata()
    cache.set(meta)
    cache._local.clear()

    result = cache.get_latest(SERIES_ID)

    assert result.id == meta.id
    assert result.series_id == meta.series_id
    assert result.version == meta.version
    assert result.mlflow_run_id == meta.mlflow_run_id
    assert result.points_used == meta.points_used
    assert result.data_hash == meta.data_hash
    assert result.trained_at == meta.trained_at


def test_dict_to_metadata_handles_null_trained_at(cache):
    """_dict_to_metadata must reconstruct metadata with trained_at=None."""
    meta = _make_metadata()
    meta.trained_at = None
    cache.set(meta)
    cache._local.clear()

    result = cache.get_latest(SERIES_ID)
    assert result.trained_at is None
