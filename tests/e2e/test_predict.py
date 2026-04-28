"""E2E tests for POST /predict/{series_id}."""

import httpx
import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_predict_returns_200(client: httpx.AsyncClient, trained_series: dict):
    series_id = trained_series["series_id"]
    r = await client.post(
        f"/predict/{series_id}",
        json={"timestamp": "1745100000", "value": trained_series["mean"]},
    )
    assert r.status_code == 200


async def test_predict_response_has_anomaly_field(
    client: httpx.AsyncClient, trained_series: dict
):
    r = await client.post(
        f"/predict/{trained_series['series_id']}",
        json={"timestamp": "1745100060", "value": trained_series["mean"]},
    )
    assert "anomaly" in r.json()


async def test_predict_response_has_model_version(
    client: httpx.AsyncClient, trained_series: dict
):
    r = await client.post(
        f"/predict/{trained_series['series_id']}",
        json={"timestamp": "1745100120", "value": trained_series["mean"]},
    )
    assert "model_version" in r.json()


async def test_predict_model_version_matches_trained(
    client: httpx.AsyncClient, trained_series: dict
):
    r = await client.post(
        f"/predict/{trained_series['series_id']}",
        json={"timestamp": "1745100180", "value": trained_series["mean"]},
    )
    assert r.json()["model_version"] == trained_series["version"]


# ---------------------------------------------------------------------------
# Anomaly detection correctness
# ---------------------------------------------------------------------------


async def test_predict_normal_point_is_not_anomaly(
    client: httpx.AsyncClient, trained_series: dict
):
    """A value at the mean must not be flagged as anomalous."""
    r = await client.post(
        f"/predict/{trained_series['series_id']}",
        json={"timestamp": "1745200000", "value": trained_series["mean"]},
    )
    assert r.json()["anomaly"] is False


async def test_predict_point_within_3sigma_is_not_anomaly(
    client: httpx.AsyncClient, trained_series: dict
):
    """A value at mean + 2σ must not be flagged (threshold is > 3σ)."""
    value = trained_series["mean"] + 2 * trained_series["std"]
    r = await client.post(
        f"/predict/{trained_series['series_id']}",
        json={"timestamp": "1745200060", "value": value},
    )
    assert r.json()["anomaly"] is False


async def test_predict_point_above_3sigma_is_anomaly(
    client: httpx.AsyncClient, trained_series: dict
):
    """A value at mean + 5σ must be flagged as anomalous."""
    value = trained_series["mean"] + 5 * trained_series["std"]
    r = await client.post(
        f"/predict/{trained_series['series_id']}",
        json={"timestamp": "1745200120", "value": value},
    )
    assert r.json()["anomaly"] is True


# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------


async def test_predict_with_explicit_version(
    client: httpx.AsyncClient, trained_series: dict
):
    r = await client.post(
        f"/predict/{trained_series['series_id']}",
        params={"version": trained_series["version"]},
        json={"timestamp": "1745300000", "value": trained_series["mean"]},
    )
    assert r.status_code == 200
    assert r.json()["model_version"] == trained_series["version"]


async def test_predict_with_unknown_version_returns_404(
    client: httpx.AsyncClient, trained_series: dict
):
    r = await client.post(
        f"/predict/{trained_series['series_id']}",
        params={"version": "999999"},
        json={"timestamp": "1745300060", "value": trained_series["mean"]},
    )
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


async def test_predict_unknown_series_returns_404(client: httpx.AsyncClient):
    r = await client.post(
        "/predict/e2e-series-does-not-exist",
        json={"timestamp": "1745000000", "value": 42.0},
    )
    assert r.status_code == 404


async def test_predict_returns_422_without_body(client: httpx.AsyncClient):
    r = await client.post("/predict/any-series", json={})
    assert r.status_code == 422


async def test_predict_returns_422_without_value(client: httpx.AsyncClient):
    r = await client.post("/predict/any-series", json={"timestamp": "1745000000"})
    assert r.status_code == 422


async def test_predict_returns_422_without_timestamp(client: httpx.AsyncClient):
    r = await client.post("/predict/any-series", json={"value": 42.0})
    assert r.status_code == 422
