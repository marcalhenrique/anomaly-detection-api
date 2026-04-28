import numpy as np
import pytest

from src.core.time_series import DataPoint, TimeSeries
from src.services.anomaly_detection import AnomalyDetectionModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_time_series(values: list[float]) -> TimeSeries:
    return TimeSeries(
        data=[DataPoint(timestamp=i, value=v) for i, v in enumerate(values)]
    )


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------


def test_fit_computes_correct_mean():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    model = AnomalyDetectionModel().fit(_make_time_series(values))
    assert model.mean == pytest.approx(np.mean(values))


def test_fit_computes_correct_std():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    model = AnomalyDetectionModel().fit(_make_time_series(values))
    assert model.std == pytest.approx(np.std(values))


def test_fit_returns_self():
    """fit() must return the model instance for fluent chaining."""
    ts = _make_time_series([1.0] * 10)
    model = AnomalyDetectionModel()
    result = model.fit(ts)
    assert result is model


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------


def test_predict_returns_false_for_normal_value():
    """A value near the mean must NOT be flagged as anomaly."""
    values = [10.0] * 10
    model = AnomalyDetectionModel().fit(_make_time_series(values))
    normal_point = DataPoint(timestamp=99, value=10.0)
    assert not model.predict(normal_point)


def test_predict_returns_true_for_outlier():
    """A value clearly above mean + 3σ must be flagged."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    model = AnomalyDetectionModel().fit(_make_time_series(values))
    outlier = DataPoint(timestamp=99, value=model.mean + 4 * model.std)
    assert model.predict(outlier)
