"""
E2E test fixtures.

The tests run against a live API stack (Postgres, MinIO, MLflow, Redis, API).
Set API_BASE_URL to override the default (useful for local runs against
docker compose).

    API_BASE_URL=http://localhost:8000 pytest tests/e2e/ -v
"""

import os

import httpx
import numpy as np
import pytest_asyncio

API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")

# Fixed series used as the shared trained fixture across all predict tests.
_E2E_SERIES_ID = "e2e-sensor-01"
_MEAN = 100.0
_STD = 5.0
_N_POINTS = 200


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def client():
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as c:
        yield c


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def trained_series(client: httpx.AsyncClient) -> dict:
    """Train a series with a known Gaussian distribution.

    Returns a dict with series_id, version, mean and std so that tests
    can construct points that are guaranteed normal or anomalous.
    """
    rng = np.random.default_rng(seed=42)
    base_ts = 1_745_000_000
    timestamps = [base_ts + i * 60 for i in range(_N_POINTS)]
    values = [float(rng.normal(_MEAN, _STD)) for _ in range(_N_POINTS)]

    r = await client.post(
        f"/fit/{_E2E_SERIES_ID}",
        json={"timestamps": timestamps, "values": values},
    )
    assert r.status_code == 200, f"E2E setup — training failed: {r.text}"

    data = r.json()
    return {
        "series_id": _E2E_SERIES_ID,
        "version": data["version"],
        "mean": _MEAN,
        "std": _STD,
    }
