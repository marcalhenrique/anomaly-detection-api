"""E2E tests for POST /fit/{series_id}."""

import httpx
import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")


_BASE_TS = 1_745_000_000
_VALID_TIMESTAMPS = [_BASE_TS + i * 60 for i in range(10)]
_VALID_VALUES = [10.1, 10.3, 9.8, 10.0, 10.2, 9.9, 10.1, 10.4, 9.7, 10.0]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_fit_returns_200(client: httpx.AsyncClient):
    r = await client.post(
        "/fit/e2e-fit-basic",
        json={"timestamps": _VALID_TIMESTAMPS, "values": _VALID_VALUES},
    )
    assert r.status_code == 200


async def test_fit_response_contains_series_id(client: httpx.AsyncClient):
    r = await client.post(
        "/fit/e2e-fit-schema",
        json={"timestamps": _VALID_TIMESTAMPS, "values": _VALID_VALUES},
    )
    assert r.json()["series_id"] == "e2e-fit-schema"


async def test_fit_response_contains_version(client: httpx.AsyncClient):
    r = await client.post(
        "/fit/e2e-fit-version-check",
        json={"timestamps": _VALID_TIMESTAMPS, "values": _VALID_VALUES},
    )
    assert "version" in r.json()


async def test_fit_response_contains_points_used(client: httpx.AsyncClient):
    r = await client.post(
        "/fit/e2e-fit-points",
        json={"timestamps": _VALID_TIMESTAMPS, "values": _VALID_VALUES},
    )
    assert r.json()["points_used"] == len(_VALID_TIMESTAMPS)


# ---------------------------------------------------------------------------
# Versioning — retrain with different data increments version
# ---------------------------------------------------------------------------


async def test_fit_retrain_increments_version(client: httpx.AsyncClient):
    series_id = "e2e-fit-retrain"
    body1 = {"timestamps": _VALID_TIMESTAMPS, "values": _VALID_VALUES}
    body2 = {
        "timestamps": _VALID_TIMESTAMPS,
        "values": [v + 1.0 for v in _VALID_VALUES],
    }

    r1 = await client.post(f"/fit/{series_id}", json=body1)
    r2 = await client.post(f"/fit/{series_id}", json=body2)

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert int(r2.json()["version"]) > int(r1.json()["version"])


# ---------------------------------------------------------------------------
# Idempotency — same data returns existing model without retraining
# ---------------------------------------------------------------------------


async def test_fit_same_data_returns_same_version(client: httpx.AsyncClient):
    series_id = "e2e-fit-idempotent"
    body = {"timestamps": _VALID_TIMESTAMPS, "values": _VALID_VALUES}

    r1 = await client.post(f"/fit/{series_id}", json=body)
    r2 = await client.post(f"/fit/{series_id}", json=body)

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["version"] == r2.json()["version"]


# ---------------------------------------------------------------------------
# Input validation → 422
# ---------------------------------------------------------------------------


async def test_fit_rejects_too_few_points(client: httpx.AsyncClient):
    r = await client.post(
        "/fit/e2e-fit-few",
        json={"timestamps": _VALID_TIMESTAMPS[:3], "values": _VALID_VALUES[:3]},
    )
    assert r.status_code == 422


async def test_fit_rejects_mismatched_lengths(client: httpx.AsyncClient):
    r = await client.post(
        "/fit/e2e-fit-mismatch",
        json={"timestamps": _VALID_TIMESTAMPS, "values": _VALID_VALUES[:-2]},
    )
    assert r.status_code == 422


async def test_fit_rejects_duplicate_timestamps(client: httpx.AsyncClient):
    duped_ts = _VALID_TIMESTAMPS[:9] + [_VALID_TIMESTAMPS[0]]
    r = await client.post(
        "/fit/e2e-fit-dupes",
        json={"timestamps": duped_ts, "values": _VALID_VALUES},
    )
    assert r.status_code == 422


async def test_fit_rejects_unsorted_timestamps(client: httpx.AsyncClient):
    unsorted_ts = list(reversed(_VALID_TIMESTAMPS))
    r = await client.post(
        "/fit/e2e-fit-unsorted",
        json={"timestamps": unsorted_ts, "values": _VALID_VALUES},
    )
    assert r.status_code == 422


async def test_fit_rejects_negative_timestamps(client: httpx.AsyncClient):
    negative_ts = [-_BASE_TS + i * 60 for i in range(10)]
    r = await client.post(
        "/fit/e2e-fit-negative-ts",
        json={"timestamps": negative_ts, "values": _VALID_VALUES},
    )
    assert r.status_code == 422


async def test_fit_rejects_empty_body(client: httpx.AsyncClient):
    r = await client.post("/fit/e2e-fit-empty", json={})
    assert r.status_code == 422
