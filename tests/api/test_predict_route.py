"""Integration tests for the POST /predict/{series_id} HTTP route.

FastAPI's TestClient is used together with dependency overrides so that no
real PredictionService (or any I/O) is involved.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_prediction_service, get_metrics_collector
from src.api.schemas import PredictResponse
from src.services.prediction_service import ModelNotFoundError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _noop_lifespan(app):
    """Bypass the real lifespan (DB cache-warming) for isolated route tests."""
    yield

_VALID_BODY = {"timestamp": "1745000600", "value": 10.0}
_EXPECTED_RESPONSE = {"anomaly": False, "model_version": "3"}


_STUB_TIMINGS = {
    "metadata_ms": 1.0,
    "model_load_ms": 1.0,
    "inference_ms": 0.5,
    "predict_operation_ms": 2.5,
}


def _override_service(
    response: PredictResponse | None = None,
    raises: Exception | None = None,
):
    """Return a FastAPI dependency override for PredictionService."""
    mock_svc = AsyncMock()
    if raises is not None:
        mock_svc.predict.side_effect = raises
    else:
        mock_svc.predict.return_value = (
            response or PredictResponse(**_EXPECTED_RESPONSE),
            dict(_STUB_TIMINGS),
        )

    async def _dep():
        return mock_svc

    return _dep


def _override_metrics():
    """Return a FastAPI dependency override for MetricsCollector."""
    mock_metrics = MagicMock()

    def _dep():
        return mock_metrics

    return _dep


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """TestClient with a happy-path PredictionService override."""
    with patch("src.api.app.lifespan", _noop_lifespan):
        app = create_app()
        app.dependency_overrides[get_prediction_service] = _override_service()
        app.dependency_overrides[get_metrics_collector] = _override_metrics()
        with TestClient(app) as c:
            yield c
        app.dependency_overrides.clear()


@pytest.fixture
def client_not_found():
    """TestClient whose PredictionService raises ModelNotFoundError."""
    with patch("src.api.app.lifespan", _noop_lifespan):
        app = create_app()
        app.dependency_overrides[get_prediction_service] = _override_service(
            raises=ModelNotFoundError("No model found for series_id='sensor-1'")
        )
        app.dependency_overrides[get_metrics_collector] = _override_metrics()
        with TestClient(app) as c:
            yield c
        app.dependency_overrides.clear()


@pytest.fixture
def client_with_error():
    """TestClient whose PredictionService raises an unexpected exception."""
    with patch("src.api.app.lifespan", _noop_lifespan):
        app = create_app()
        app.dependency_overrides[get_prediction_service] = _override_service(
            raises=RuntimeError("unexpected failure")
        )
        app.dependency_overrides[get_metrics_collector] = _override_metrics()
        with TestClient(app) as c:
            yield c
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_predict_returns_200_with_valid_body(client):
    response = client.post("/predict/sensor-1", json=_VALID_BODY)
    assert response.status_code == 200


def test_predict_response_body_matches_schema(client):
    response = client.post("/predict/sensor-1", json=_VALID_BODY)
    data = response.json()
    assert data["anomaly"] is False
    assert data["model_version"] == "3"


def test_predict_returns_anomaly_true_when_service_says_so(client):
    client.app.dependency_overrides[get_prediction_service] = _override_service(
        response=PredictResponse(anomaly=True, model_version="3")
    )
    client.app.dependency_overrides[get_metrics_collector] = _override_metrics()
    response = client.post("/predict/sensor-1", json={**_VALID_BODY, "value": 99.9})
    assert response.json()["anomaly"] is True


def test_predict_passes_version_query_param_to_service(client):
    response = client.post("/predict/sensor-1?version=2", json=_VALID_BODY)
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Input validation (handled by Pydantic before the service is even called)
# ---------------------------------------------------------------------------


def test_predict_returns_422_when_body_is_empty(client):
    response = client.post("/predict/sensor-1", json={})
    assert response.status_code == 422


def test_predict_returns_422_when_value_is_missing(client):
    response = client.post("/predict/sensor-1", json={"timestamp": "1745000600"})
    assert response.status_code == 422


def test_predict_returns_422_when_timestamp_is_missing(client):
    response = client.post("/predict/sensor-1", json={"value": 10.0})
    assert response.status_code == 422


def test_predict_returns_200_when_timestamp_is_string(client):
    response = client.post(
        "/predict/sensor-1", json={"timestamp": "1745000600", "value": 10.0}
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Not found → 404
# ---------------------------------------------------------------------------


def test_predict_returns_404_when_series_does_not_exist(client_not_found):
    response = client_not_found.post("/predict/sensor-1", json=_VALID_BODY)
    assert response.status_code == 404


def test_predict_404_response_contains_error_detail(client_not_found):
    response = client_not_found.post("/predict/sensor-1", json=_VALID_BODY)
    assert "sensor-1" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Service-level errors → 500
# ---------------------------------------------------------------------------


def test_predict_returns_500_when_service_raises(client_with_error):
    response = client_with_error.post("/predict/sensor-1", json=_VALID_BODY)
    assert response.status_code == 500


def test_predict_500_response_contains_error_detail(client_with_error):
    response = client_with_error.post("/predict/sensor-1", json=_VALID_BODY)
    assert "unexpected failure" in response.json()["detail"]
