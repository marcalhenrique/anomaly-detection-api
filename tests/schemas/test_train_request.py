import pytest
from pydantic import ValidationError

from src.api.schemas import TrainRequest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(n: int = 10, **overrides) -> dict:
    """Build a minimal valid payload with *n* points."""
    timestamps = list(range(n))
    values = [float(i) for i in range(n)]
    payload = {"timestamps": timestamps, "values": values}
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_valid_request_passes_validation():
    """A request with 10 valid, sorted, unique timestamps must be accepted."""
    req = TrainRequest(**_make_request(10))
    assert len(req.timestamps) == 10
    assert len(req.values) == 10


def test_valid_request_with_more_than_minimum_points():
    """Requests with more than 10 points are also valid."""
    req = TrainRequest(**_make_request(50))
    assert len(req.timestamps) == 50


# ---------------------------------------------------------------------------
# min_length constraint (Pydantic field-level)
# ---------------------------------------------------------------------------


def test_raises_when_fewer_than_10_timestamps():
    """Fewer than 10 timestamps must fail field validation."""
    with pytest.raises(ValidationError, match="timestamps"):
        TrainRequest(**_make_request(9))


def test_raises_when_fewer_than_10_values():
    """Fewer than 10 values must fail field validation."""
    payload = _make_request(10)
    payload["values"] = payload["values"][:9]  # shrink values only
    with pytest.raises(ValidationError, match="values"):
        TrainRequest(**payload)


# ---------------------------------------------------------------------------
# model_validator checks
# ---------------------------------------------------------------------------


def test_raises_when_timestamps_and_values_length_mismatch():
    """timestamps and values with different lengths must be rejected."""
    payload = _make_request(10)
    payload["values"] = payload["values"] + [99.0]  # 11 values, 10 timestamps
    with pytest.raises(ValidationError, match="same length"):
        TrainRequest(**payload)


def test_raises_when_timestamps_not_unique():
    """Duplicate timestamps must be rejected."""
    payload = _make_request(10)
    payload["timestamps"][5] = payload["timestamps"][4]  # introduce duplicate
    with pytest.raises(ValidationError, match="unique"):
        TrainRequest(**payload)


def test_raises_when_timestamps_not_sorted():
    """Out-of-order timestamps must be rejected."""
    payload = _make_request(10)
    payload["timestamps"][0], payload["timestamps"][1] = (
        payload["timestamps"][1],
        payload["timestamps"][0],
    )
    with pytest.raises(ValidationError, match="ascending"):
        TrainRequest(**payload)


def test_raises_when_timestamp_is_negative():
    """Negative timestamps must be rejected."""
    payload = _make_request(10)
    payload["timestamps"][0] = -1
    # After setting [0] to -1, timestamps are no longer sorted either;
    # the validator checks negatives explicitly.
    with pytest.raises(ValidationError, match="non-negative"):
        TrainRequest(**payload)
