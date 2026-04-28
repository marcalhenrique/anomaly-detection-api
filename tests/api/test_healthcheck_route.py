"""Integration tests for the GET /healthcheck HTTP route.

The route checks external dependencies (database, MLflow) and returns
system-level metrics as defined in the OpenAPI spec.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.services.metrics_collector import MetricsCollector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _noop_lifespan(app):
    """Bypass the real lifespan (DB cache-warming) for isolated route tests."""
    yield


def _make_metrics_collector(
    predict_latencies: list[float] | None = None,
    train_latencies: list[float] | None = None,
    series_trained: int = 0,
) -> MetricsCollector:
    """Build a real MetricsCollector seeded with optional latency data."""
    collector = MetricsCollector(latency_window=1000)
    for ms in predict_latencies or []:
        collector._predict_latencies.append(ms)
    for ms in train_latencies or []:
        collector._train_latencies.append(ms)
    collector.set_series_trained(series_trained)
    return collector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client_healthy_empty_metrics():
    """TestClient with healthy dependencies and empty metrics."""
    with (
        patch("src.api.routes.health.engine") as mock_engine,
        patch("src.api.routes.health.get_mlflow_service") as mock_mlflow,
        patch("src.api.routes.health.get_metrics_collector") as mock_metrics_factory,
        patch("src.api.app.lifespan", _noop_lifespan),
    ):
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)
        mock_engine.connect = MagicMock(return_value=mock_conn)

        mock_mlflow_client = MagicMock()
        mock_mlflow_client._client.search_experiments = MagicMock(return_value=[])
        mock_mlflow.return_value = mock_mlflow_client

        mock_metrics_factory.return_value = _make_metrics_collector()

        app = create_app()
        with TestClient(app) as c:
            yield c


@pytest.fixture
def client_healthy_with_metrics():
    """TestClient with healthy dependencies and populated metrics."""
    with (
        patch("src.api.routes.health.engine") as mock_engine,
        patch("src.api.routes.health.get_mlflow_service") as mock_mlflow,
        patch("src.api.routes.health.get_metrics_collector") as mock_metrics_factory,
        patch("src.api.app.lifespan", _noop_lifespan),
    ):
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)
        mock_engine.connect = MagicMock(return_value=mock_conn)

        mock_mlflow_client = MagicMock()
        mock_mlflow_client._client.search_experiments = MagicMock(return_value=[])
        mock_mlflow.return_value = mock_mlflow_client

        collector = _make_metrics_collector(
            predict_latencies=[10.0, 20.0, 30.0],
            train_latencies=[100.0, 200.0],
            series_trained=7,
        )
        mock_metrics_factory.return_value = collector

        app = create_app()
        with TestClient(app) as c:
            yield c


@pytest.fixture
def client_unhealthy_db():
    """TestClient whose database dependency is down."""
    with (
        patch("src.api.routes.health.engine") as mock_engine,
        patch("src.api.routes.health.get_mlflow_service") as mock_mlflow,
        patch("src.api.routes.health.get_metrics_collector") as mock_metrics_factory,
        patch("src.api.app.lifespan", _noop_lifespan),
    ):
        mock_engine.connect = MagicMock(side_effect=Exception("connection refused"))

        mock_mlflow_client = MagicMock()
        mock_mlflow_client._client.search_experiments = MagicMock(return_value=[])
        mock_mlflow.return_value = mock_mlflow_client

        mock_metrics_factory.return_value = _make_metrics_collector(series_trained=3)

        app = create_app()
        with TestClient(app) as c:
            yield c


@pytest.fixture
def client_unhealthy_mlflow():
    """TestClient whose MLflow dependency is down."""
    with (
        patch("src.api.routes.health.engine") as mock_engine,
        patch("src.api.routes.health.get_mlflow_service") as mock_mlflow,
        patch("src.api.routes.health.get_metrics_collector") as mock_metrics_factory,
        patch("src.api.app.lifespan", _noop_lifespan),
    ):
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)
        mock_engine.connect = MagicMock(return_value=mock_conn)

        mock_mlflow.side_effect = Exception("mlflow unreachable")

        mock_metrics_factory.return_value = _make_metrics_collector(series_trained=3)

        app = create_app()
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_healthcheck_returns_200_when_healthy(client_healthy_empty_metrics):
    response = client_healthy_empty_metrics.get("/healthcheck")
    assert response.status_code == 200
    body = response.json()
    assert body["series_trained"] == 0
    assert body["inference_latency_ms"] == {"avg": 0.0, "p95": 0.0}
    assert body["training_latency_ms"] == {"avg": 0.0, "p95": 0.0}


def test_healthcheck_returns_populated_metrics(client_healthy_with_metrics):
    response = client_healthy_with_metrics.get("/healthcheck")
    assert response.status_code == 200
    body = response.json()
    assert body["series_trained"] == 7
    # avg of [10, 20, 30] = 20.0
    assert body["inference_latency_ms"]["avg"] == 20.0
    # p95 of [10, 20, 30] = 29.0 (linear interpolation between 20 and 30)
    assert body["inference_latency_ms"]["p95"] == 29.0
    # avg of [100, 200] = 150.0
    assert body["training_latency_ms"]["avg"] == 150.0
    # p95 of [100, 200] = 195.0
    assert body["training_latency_ms"]["p95"] == 195.0


def test_healthcheck_returns_503_when_db_down(client_unhealthy_db):
    response = client_unhealthy_db.get("/healthcheck")
    assert response.status_code == 503
    body = response.json()
    assert "series_trained" in body
    assert "inference_latency_ms" in body
    assert "training_latency_ms" in body


def test_healthcheck_returns_503_when_mlflow_down(client_unhealthy_mlflow):
    response = client_unhealthy_mlflow.get("/healthcheck")
    assert response.status_code == 503
    body = response.json()
    assert "series_trained" in body
    assert "inference_latency_ms" in body
    assert "training_latency_ms" in body
