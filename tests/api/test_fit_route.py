"""Integration tests for the POST /fit/{series_id} HTTP route.

FastAPI's TestClient is used together with dependency overrides so that no
real TrainingService (or any I/O) is involved.
"""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_training_service
from src.api.schemas import TrainResponse

app = create_app()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_POINTS = 10
_VALID_BODY = {
    "timestamps": list(range(N_POINTS)),
    "values": [float(i) for i in range(N_POINTS)],
}
_EXPECTED_RESPONSE = {
    "series_id": "sensor-1",
    "version": "1",
    "points_used": N_POINTS,
}


def _override_service(response: TrainResponse | None = None, raises: Exception | None = None):
    """Return a FastAPI dependency override for TrainingService."""
    mock_svc = AsyncMock()
    if raises is not None:
        mock_svc.fit.side_effect = raises
    else:
        mock_svc.fit.return_value = response or TrainResponse(**_EXPECTED_RESPONSE)

    async def _dep():
        return mock_svc

    return _dep


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """TestClient with a happy-path TrainingService override."""
    app.dependency_overrides[get_training_service] = _override_service()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def client_with_error():
    """TestClient whose TrainingService always raises."""
    app.dependency_overrides[get_training_service] = _override_service(
        raises=RuntimeError("unexpected failure")
    )
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_fit_returns_200_with_valid_body(client):
    response = client.post("/fit/sensor-1", json=_VALID_BODY)
    assert response.status_code == 200


def test_fit_response_body_matches_schema(client):
    response = client.post("/fit/sensor-1", json=_VALID_BODY)
    data = response.json()
    assert data["series_id"] == "sensor-1"
    assert data["version"] == "1"
    assert data["points_used"] == N_POINTS


# ---------------------------------------------------------------------------
# Input validation (handled by Pydantic before the service is even called)
# ---------------------------------------------------------------------------


def test_fit_returns_422_when_body_missing_fields(client):
    response = client.post("/fit/sensor-1", json={})
    assert response.status_code == 422


def test_fit_returns_422_when_fewer_than_10_timestamps(client):
    body = {"timestamps": list(range(9)), "values": [0.0] * 9}
    response = client.post("/fit/sensor-1", json=body)
    assert response.status_code == 422


def test_fit_returns_422_when_timestamps_and_values_length_mismatch(client):
    body = {
        "timestamps": list(range(10)),
        "values": [0.0] * 11,
    }
    response = client.post("/fit/sensor-1", json=body)
    assert response.status_code == 422


def test_fit_returns_422_when_timestamps_not_sorted(client):
    timestamps = list(range(10))
    timestamps[0], timestamps[1] = timestamps[1], timestamps[0]
    body = {"timestamps": timestamps, "values": [0.0] * 10}
    response = client.post("/fit/sensor-1", json=body)
    assert response.status_code == 422


def test_fit_returns_422_when_timestamps_not_unique(client):
    timestamps = list(range(10))
    timestamps[5] = timestamps[4]
    body = {"timestamps": timestamps, "values": [0.0] * 10}
    response = client.post("/fit/sensor-1", json=body)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Service-level errors → 500
# ---------------------------------------------------------------------------


def test_fit_returns_500_when_service_raises(client_with_error):
    response = client_with_error.post("/fit/sensor-1", json=_VALID_BODY)
    assert response.status_code == 500


def test_fit_500_response_contains_error_detail(client_with_error):
    response = client_with_error.post("/fit/sensor-1", json=_VALID_BODY)
    assert "unexpected failure" in response.json()["detail"]
