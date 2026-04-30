# Anomaly Detection API

A REST API for real-time anomaly detection on univariate time series data. Train per-series models, version them automatically, persist weights in MLflow/MinIO, and run predictions at scale.

## Table of Contents

- [How to Use](#how-to-use)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Nuances](#nuances)
- [Benchmark Results](#benchmark-results)

---

## How to Use

### Requirements

Docker and Docker Compose. The following host ports must be free before starting:

| Port | Service       |
| ---- | ------------- |
| 8000 | API           |
| 5432 | PostgreSQL    |
| 5001 | MLflow UI     |
| 9000 | MinIO (S3)    |
| 9001 | MinIO console |
| 6379 | Redis         |
| 9090 | Prometheus    |
| 3000 | Grafana       |

### Stack

| Service        | Description                          |
| -------------- | ------------------------------------ |
| **api**        | FastAPI application                  |
| **postgres**   | Model metadata persistence           |
| **mlflow**     | Model registry and artifact tracking |
| **minio**      | S3-compatible artifact store         |
| **redis**      | Distributed cache & training locks   |
| **prometheus** | Metrics collection                   |
| **grafana**    | Metrics dashboard                    |

### Environment

Copy `.env.example` to `.env` before starting:

```bash
cp .env.example .env
```

The `compose.yaml` reads all variables directly via `${VAR}` interpolation — port mappings, credentials and stack behaviour are all controlled from `.env`. Key variables:

| Variable                     | Default  | Description                                   |
| ---------------------------- | -------- | --------------------------------------------- |
| `API_PORT`                   | `8000`   | API server port                               |
| `ZSCORE_THRESHOLD`           | `3.0`    | Standard deviations used as anomaly threshold |
| `LOG_LEVEL`                  | `DEBUG`  | Log verbosity (`DEBUG`, `INFO`, `WARNING`)    |
| `LOCAL_CACHE_MAXSIZE`        | `100`    | Max models held in the in-process LRU cache   |
| `LOCAL_CACHE_TTL_SECONDS`    | `60`     | TTL for in-process cache entries (seconds)    |
| `REDIS_MODEL_TTL_SECONDS`    | `86400`  | Sliding TTL for model objects in Redis (24 h) |
| `REDIS_METADATA_TTL_SECONDS` | `3600`   | Fixed TTL for metadata keys in Redis (1 h)    |
| `POSTGRES_*`                 | see file | PostgreSQL connection settings                |
| `MINIO_*`                    | see file | MinIO / S3 credentials and bucket             |
| `MLFLOW_*`                   | see file | MLflow tracking server settings               |

### Start

```bash
make run
```

Services will be available at:

- **API docs:** http://localhost:8000/docs
- **MLflow UI:** http://localhost:5001
- **MinIO console:** http://localhost:9001 — user: `minioadmin` / password: `minioadmin`
- **Grafana:** http://localhost:3000 — user: `admin` / password: `admin`

```bash
make stop    # stop containers, preserve volumes
make clean   # stop containers and delete all volumes (full reset)
```

### Tests

```bash
make test-unit   # unit + integration tests, no running stack required
make test-e2e    # end-to-end tests against the running stack
make test        # both
```

### Populate, Inference and Benchmark

These commands run inside the Docker network and write Markdown reports to `reports/`.

```bash
make populate    # train 100 models at concurrency 20
make inference   # run 1 000 predictions across 50 series
make benchmark   # full 4-scenario benchmark
```

> **Note:** `make inference` requires trained models. Run `make populate` first — without it every prediction returns `404`.

To change the parameters (number of series, concurrency, etc.) edit the corresponding service command in `compose.yaml` or override via `.env`.

---

## API Reference

### `POST /fit/{series_id}`

Train or retrain a model for a given series.

**Request body:**

```json
{
  "timestamps": [1745000000, 1745000060, "..."],
  "values": [10.1, 10.3, "..."]
}
```

| Field        | Type          | Constraints                                       |
| ------------ | ------------- | ------------------------------------------------- |
| `timestamps` | `list[int]`   | Unix timestamps, unique, ascending, min 10 points |
| `values`     | `list[float]` | Same length as `timestamps`, must not be constant |

**Response `200`:**

```json
{ "series_id": "sensor-vibration-01", "version": "3", "points_used": 200 }
```

**Other responses:** `422` validation error · `500` internal error.

---

### `POST /predict/{series_id}`

Predict whether a data point is anomalous. A value is flagged when it exceeds `mean ± ZSCORE_THRESHOLD × std` of the training distribution.

**Query parameter:** `version` (optional) — pin to a specific model version; defaults to latest.

**Request body:**

```json
{ "timestamp": "1745000600", "value": 10.2 }
```

**Response `200`:**

```json
{ "anomaly": false, "model_version": "1" }
```

**Other responses:** `404` model not found · `422` validation error · `500` internal error.

---

### `GET /healthcheck`

Returns liveness status and rolling latency metrics. Returns `503` when a critical dependency is unreachable.

```json
{
  "series_trained": 5,
  "inference_latency_ms": { "avg": 12.4, "p95": 28.1 },
  "training_latency_ms": { "avg": 340.2, "p95": 512.0 }
}
```

---

## Nuances

**Idempotent training** — `/fit` hashes the incoming payload. If identical data was already trained for a `series_id`, the existing version is returned without touching MLflow or MinIO.

**Distributed training lock** — each `series_id` is protected by a Redis distributed lock (`SET NX EX 60`). Concurrent `/fit` calls for the same series are serialised, not rejected. This guarantees a correct monotonically-increasing version counter regardless of concurrency.

**Two-level prediction cache** — models are cached first in an in-process `TTLCache` (L1, `LOCAL_CACHE_TTL_SECONDS`) and then in Redis (L2, `REDIS_MODEL_TTL_SECONDS` with sliding expiry — TTL resets on every hit). A miss on both levels falls back to MLflow/MinIO.

**Version pinning** — `POST /predict/{series_id}?version=<n>` targets any historical model version, useful for A/B comparison or reproducing past results.

**Anomaly threshold** — detection is purely statistical (Z-score). Changing `ZSCORE_THRESHOLD` in `.env` and restarting the API adjusts sensitivity without any retraining.

**Client vs server latency** — benchmark numbers are measured client-side over the Docker bridge and include TCP/kernel overhead. Server-side latency tracked by Prometheus is significantly lower (inference p99 < 1 ms). See the note in the benchmark section.

For the full architectural breakdown and flow diagrams see [`docs/architecture.md`](docs/architecture.md).

---

## API Reference

Four scenarios measured with the full stack on a local machine. Latencies are client-side (include Docker bridge overhead); server-side inference latency tracked by Prometheus is < 1 ms p99.

| Scenario                  | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Errors |
| ------------------------- | ------------------ | -------- | -------- | -------- | ------ |
| A · Concurrent Training   | 9.1                | 1044.98  | 1514.62  | 1522.12  | 0/20   |
| B · Inference Cache Hit   | 417.9              | 78.49    | 340.94   | 443.65   | 0/500  |
| C · Inference Cache Miss  | 401.2              | 80.90    | 334.35   | 590.00   | 0/500  |
| D · Concurrent Retraining | 1.7                | 3267.83  | 5710.65  | 5927.74  | 0/10   |

Scenario D serialises 10 concurrent retrains for the same series through the distributed lock — all succeed and the version counter reaches 10 with no errors.

Full per-request statistics in [`reports/benchmark.md`](reports/benchmark.md). To re-run: `make benchmark`.

---

## Architecture

For the full architectural breakdown, layered design, flow diagrams (fit, predict, cache/TTL) and design decisions see [`docs/architecture.md`](docs/architecture.md).
