"""Unit tests for MetricsCollector.

Covers record_predict, record_training, inc_series_trained, and the
derived stats (get_inference_stats, get_training_stats).
"""

import pytest

from src.services.metrics_collector import MetricsCollector

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FULL_TIMINGS = {
    "metadata_ms": 1.0,
    "model_load_ms": 5.0,
    "inference_ms": 0.5,
    "predict_operation_ms": 6.5,
    "total_ms": 10.0,
}


@pytest.fixture
def collector():
    """Fresh MetricsCollector for each test."""
    return MetricsCollector(latency_window=1000)


# ---------------------------------------------------------------------------
# record_predict
# ---------------------------------------------------------------------------


def test_record_predict_appends_total_ms_to_latencies(collector):
    """record_predict() must store total_ms in the rolling latency buffer."""
    collector.record_predict({**_FULL_TIMINGS, "total_ms": 42.0})
    assert list(collector._predict_latencies) == [42.0]


def test_record_predict_multiple_calls_accumulate(collector):
    """Multiple calls to record_predict() must accumulate in the buffer."""
    for ms in [10.0, 20.0, 30.0]:
        collector.record_predict({**_FULL_TIMINGS, "total_ms": ms})
    assert list(collector._predict_latencies) == [10.0, 20.0, 30.0]


def test_record_predict_updates_inference_stats(collector):
    """After record_predict(), get_inference_stats() must reflect recorded latencies."""
    for ms in [10.0, 20.0, 30.0]:
        collector.record_predict({**_FULL_TIMINGS, "total_ms": ms})
    stats = collector.get_inference_stats()
    assert stats["avg"] == pytest.approx(20.0)
    assert stats["p95"] == pytest.approx(29.0)  # linear interpolation between 20 and 30


# ---------------------------------------------------------------------------
# record_training
# ---------------------------------------------------------------------------


def test_record_training_appends_ms_to_latencies(collector):
    """record_training() must store ms in the training latency buffer."""
    collector.record_training(150.0)
    assert list(collector._train_latencies) == [150.0]


def test_record_training_multiple_calls_accumulate(collector):
    """Multiple calls to record_training() must accumulate in the buffer."""
    for ms in [100.0, 200.0, 300.0]:
        collector.record_training(ms)
    assert list(collector._train_latencies) == [100.0, 200.0, 300.0]


def test_record_training_updates_training_stats(collector):
    """After record_training(), get_training_stats() must reflect recorded latencies."""
    collector.record_training(100.0)
    collector.record_training(200.0)
    stats = collector.get_training_stats()
    assert stats["avg"] == pytest.approx(150.0)
    assert stats["p95"] == pytest.approx(195.0)


# ---------------------------------------------------------------------------
# inc_series_trained
# ---------------------------------------------------------------------------


def test_inc_series_trained_increments_by_one(collector):
    """inc_series_trained() must increase the series trained count by 1."""
    collector.set_series_trained(0)
    collector.inc_series_trained()
    assert collector.get_series_trained() == 1


def test_inc_series_trained_is_cumulative(collector):
    """Multiple inc_series_trained() calls must accumulate."""
    collector.set_series_trained(5)
    collector.inc_series_trained()
    collector.inc_series_trained()
    assert collector.get_series_trained() == 7


# ---------------------------------------------------------------------------
# Latency window cap
# ---------------------------------------------------------------------------


def test_latency_window_caps_buffer_size():
    """Latency buffer must not exceed the configured window size."""
    small_window = MetricsCollector(latency_window=3)
    for i in range(5):
        small_window.record_predict({**_FULL_TIMINGS, "total_ms": float(i)})
    assert len(small_window._predict_latencies) == 3
    assert list(small_window._predict_latencies) == [2.0, 3.0, 4.0]


# ---------------------------------------------------------------------------
# update_cache_metrics
# ---------------------------------------------------------------------------


def test_update_cache_metrics_completes_without_error(collector):
    """update_cache_metrics() must succeed when Redis and cache objects are provided."""
    from unittest.mock import MagicMock, patch

    fake_redis = MagicMock()
    fake_redis.scan_iter.return_value = iter([])
    fake_redis.info.return_value = {"used_memory": 0}

    mock_meta_cache = MagicMock()
    mock_meta_cache._local = {}
    mock_mlflow_svc = MagicMock()
    mock_mlflow_svc._local = {}

    with patch(
        "src.services.metrics_collector.get_redis_client", return_value=fake_redis
    ):
        collector.update_cache_metrics(mock_meta_cache, mock_mlflow_svc)


def test_update_cache_metrics_counts_redis_keys_by_prefix(collector):
    """update_cache_metrics() must count model:* and metadata:* keys separately."""
    from unittest.mock import MagicMock, patch
    from src.services.metrics_collector import REDIS_MODEL_KEYS, REDIS_METADATA_KEYS

    def _scan_iter(match=None):
        if match == "model:*":
            return iter(["model:run-1", "model:run-2"])
        if match == "metadata:*":
            return iter(["metadata:latest:s1", "metadata:version:s1:1"])
        return iter([])

    fake_redis = MagicMock()
    fake_redis.scan_iter.side_effect = _scan_iter
    fake_redis.info.return_value = {"used_memory": 1024}

    mock_meta_cache = MagicMock()
    mock_meta_cache._local = {}
    mock_mlflow_svc = MagicMock()
    mock_mlflow_svc._local = {}

    with patch(
        "src.services.metrics_collector.get_redis_client", return_value=fake_redis
    ):
        collector.update_cache_metrics(mock_meta_cache, mock_mlflow_svc)

    assert REDIS_MODEL_KEYS._value.get() == pytest.approx(2)
    assert REDIS_METADATA_KEYS._value.get() == pytest.approx(2)


def test_update_cache_metrics_sums_l1_cache_lengths(collector):
    """update_cache_metrics() must set L1_CACHE_ITEMS to len(meta._local) + len(mlflow._local)."""
    from unittest.mock import MagicMock, patch
    from src.services.metrics_collector import L1_CACHE_ITEMS

    fake_redis = MagicMock()
    fake_redis.scan_iter.return_value = iter([])
    fake_redis.info.return_value = {"used_memory": 0}

    mock_meta_cache = MagicMock()
    mock_meta_cache._local = {"a": 1, "b": 2, "c": 3}
    mock_mlflow_svc = MagicMock()
    mock_mlflow_svc._local = {"x": 1, "y": 2}

    with patch(
        "src.services.metrics_collector.get_redis_client", return_value=fake_redis
    ):
        collector.update_cache_metrics(mock_meta_cache, mock_mlflow_svc)

    assert L1_CACHE_ITEMS._value.get() == pytest.approx(5)
