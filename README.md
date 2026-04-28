# Anomaly Detection API

A production-ready REST API for real-time anomaly detection on univariate time series data. Train per-series models, version them automatically, persist weights in MLflow/MinIO, and run predictions at scale.

## Table of Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Sample Requests](#sample-requests)
- [Configuration](#configuration)
- [Testing](#testing)
- [Scripts](#scripts)
- [Benchmark Results](#benchmark-results)

---

## Architecture

| Service        | Description                          | Port |
| -------------- | ------------------------------------ | ---- |
| **api**        | FastAPI application                  | 8000 |
| **postgres**   | Model metadata persistence           | 5432 |
| **mlflow**     | Model registry and artifact tracking | 5000 |
| **minio**      | S3-compatible artifact store         | 9000 |
| **prometheus** | Metrics collection                   | 9090 |
| **grafana**    | Metrics dashboard                    | 3000 |

**Request flow:**

```
POST /fit/{series_id}
  -> validate input
  -> acquire per-series lock
  -> check data hash (skip if identical data was already trained)
  -> train AnomalyDetectionModel
  -> persist model + training data to MLflow/MinIO
  -> save metadata to PostgreSQL
  -> warm LRU cache

POST /predict/{series_id}
  -> look up latest (or versioned) run_id in PostgreSQL
  -> load model from LRU cache (or pull from MLflow/MinIO on miss)
  -> run inference
  -> record latency metrics
```

---

## Quick Start

**Requirements:** Docker and Docker Compose.

```bash
# 1. clone and configure
git clone https://github.com/marcalhenrique/anomaly-detection-api.git
cd anomaly-detection-api
cp .env.example .env

# 2. start all services
make run

# 3. check the API is healthy
curl http://localhost:8000/healthcheck
```

Services will be available at:

- API docs: http://localhost:8000/docs
- MLflow UI: http://localhost:5001
- MinIO console: http://localhost:9001 (user: `minioadmin`, password: `minioadmin`)
- Grafana: http://localhost:3000 (user: `admin`, password: `admin`)

---

## API Reference

### `POST /fit/{series_id}`

Train or retrain a model for a given series. Retraining with identical data returns the existing version without re-training (idempotent).

**Path parameter:** `series_id` — unique identifier for the time series (e.g. `sensor-vibration-01`).

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
{
  "series_id": "sensor-vibration-01",
  "version": "3",
  "points_used": 200
}
```

---

### `POST /predict/{series_id}`

Predict whether a data point is anomalous. A value is flagged as an anomaly when it exceeds `mean + 3 * std` of the training distribution.

**Path parameter:** `series_id`

**Query parameter:** `version` (optional) — target a specific model version; defaults to latest.

**Request body:**

```json
{
  "timestamp": "1745000600",
  "value": 10.2
}
```

**Response `200`:**

```json
{
  "anomaly": false,
  "model_version": "3"
}
```

**Response `404`:** returned when `series_id` has no trained model (or the requested version does not exist).

---

### `GET /healthcheck`

Returns the liveness status of the API and its critical dependencies.

**Response `200` — all systems healthy:**

```json
{
  "series_trained": 5,
  "inference_latency_ms": {
    "avg": 12.4,
    "p95": 28.1
  },
  "training_latency_ms": {
    "avg": 340.2,
    "p95": 512.0
  }
}
```

**Response `503`** — one or more dependencies are unreachable (same schema, HTTP status only changes).

---

## Sample Requests

The examples below use a 10-point series with values around `10.0`. Copy and run them against a running instance.

### Train a model

```bash
curl -s -X POST http://localhost:8000/fit/sensor-vibration-01 \
  -H "Content-Type: application/json" \
  -d '{
    "timestamps": [1745000000, 1745000060, 1745000120, 1745000180, 1745000240,
                   1745000300, 1745000360, 1745000420, 1745000480, 1745000540],
    "values":     [10.1, 10.3, 9.8, 10.0, 10.2, 9.9, 10.1, 10.4, 9.7, 10.0]
  }'
```

```json
{ "series_id": "sensor-vibration-01", "version": "1", "points_used": 10 }
```

### Predict — normal value

```bash
curl -s -X POST http://localhost:8000/predict/sensor-vibration-01 \
  -H "Content-Type: application/json" \
  -d '{"timestamp": "1745000600", "value": 10.2}'
```

```json
{ "anomaly": false, "model_version": "1" }
```

### Predict — anomalous value

```bash
curl -s -X POST http://localhost:8000/predict/sensor-vibration-01 \
  -H "Content-Type: application/json" \
  -d '{"timestamp": "1745000660", "value": 50.0}'
```

```json
{ "anomaly": true, "model_version": "1" }
```

### Predict — specific version

```bash
curl -s -X POST "http://localhost:8000/predict/sensor-vibration-01?version=1" \
  -H "Content-Type: application/json" \
  -d '{"timestamp": "1745000720", "value": 10.0}'
```

### Retrain with new data (increments version)

```bash
curl -s -X POST http://localhost:8000/fit/sensor-vibration-01 \
  -H "Content-Type: application/json" \
  -d '{
    "timestamps": [1745100000, 1745100060, 1745100120, 1745100180, 1745100240,
                   1745100300, 1745100360, 1745100420, 1745100480, 1745100540],
    "values":     [20.5, 20.1, 19.8, 20.3, 20.0, 19.9, 20.2, 20.6, 19.7, 20.1]
  }'
```

```json
{ "series_id": "sensor-vibration-01", "version": "2", "points_used": 10 }
```

### Health check

```bash
curl -s http://localhost:8000/healthcheck
```

---

## Configuration

Copy `.env.example` to `.env` and adjust as needed:

```bash
cp .env.example .env
```

Key variables:

| Variable           | Default  | Description                                |
| ------------------ | -------- | ------------------------------------------ |
| `API_PORT`         | `8000`   | API server port                            |
| `LRU_CACHE_SIZE`   | `10000`  | Number of models kept in memory            |
| `ZSCORE_THRESHOLD` | `3.0`    | Standard deviations for anomaly threshold  |
| `LOG_LEVEL`        | `DEBUG`  | Log verbosity (`DEBUG`, `INFO`, `WARNING`) |
| `POSTGRES_*`       | see file | PostgreSQL connection settings             |
| `MINIO_*`          | see file | MinIO / S3 credentials and bucket          |
| `MLFLOW_*`         | see file | MLflow tracking server settings            |

---

## Testing

```bash
# unit tests only (no Docker required)
make test-unit

# end-to-end tests (spins up full Docker stack)
make test-e2e

# both
make test
```

The test suite covers:

- `POST /fit` — training, validation, idempotency, version increment
- `POST /predict` — normal and anomalous values, version pinning, 404 on missing model
- `GET /healthcheck` — liveness checks
- Service-level unit tests for training, prediction, MLflow integration, and caching

---

## Scripts

All scripts run inside the Docker stack and output a Markdown report to `reports/`.

| Command          | Description                                          | Output                 |
| ---------------- | ---------------------------------------------------- | ---------------------- |
| `make populate`  | Train 100 models (concurrency 20)                    | `reports/populate.md`  |
| `make inference` | 1 000 predictions across 50 series (concurrency 100) | `reports/inference.md` |
| `make benchmark` | Full benchmark — see below                           | `reports/benchmark.md` |

> **Note:** `make inference` requires trained models. Run `make populate` first to train the series, otherwise all prediction requests will return `404 Not Found`.

---

## Benchmark Results

Four scenarios measured with the full Docker stack on a local machine.

> **Note on latency measurements:** benchmark latencies are measured **client-side** using
> `httpx` inside the same Docker network. They include HTTP round-trip overhead (TCP stack,
> kernel scheduling, Docker bridge) in addition to actual API processing time.
> Server-side latency (measured by Prometheus inside the API process) is significantly lower:
>
> | Metric        | Client-side (benchmark) | Server-side (Prometheus) |
> | ------------- | ----------------------- | ------------------------ |
> | Inference p50 | ~56 ms                  | < 1 ms                   |
> | Inference p99 | ~350 ms                 | < 1 ms                   |
> | Training p50  | ~99 ms                  | ~817 ms\*                |
> | Training p99  | ~185 ms                 | ~4 810 ms\*              |
>
> \*Training is CPU-bound (NumPy + MLflow artifact upload); the server-side value is higher
> because it includes the full MLflow/MinIO write, while the client sees requests completing
> in parallel thanks to concurrency.

### Scenario A — Concurrent Training

> 20 models trained simultaneously at concurrency 10.

| Metric     | Value      |
| ---------- | ---------- |
| Throughput | 93.3 req/s |
| p50        | 98.83 ms   |
| p95        | 180.92 ms  |
| p99        | 184.75 ms  |
| Errors     | 0 / 20     |

### Scenario B — Inference · Cache Hit

> 500 predictions against models held in LRU memory cache (concurrency 50).

| Metric     | Value       |
| ---------- | ----------- |
| Throughput | 584.1 req/s |
| p50        | 56.31 ms    |
| p95        | 241.76 ms   |
| p99        | 350.45 ms   |
| Errors     | 0 / 500     |

### Scenario C — Inference · Cache Miss

> 500 predictions against evicted models — each request loads the model from MLflow/MinIO (concurrency 50).

| Metric     | Value       |
| ---------- | ----------- |
| Throughput | 621.6 req/s |
| p50        | 54.61 ms    |
| p95        | 202.89 ms   |
| p99        | 340.43 ms   |
| Errors     | 0 / 500     |

### Scenario D — Concurrent Retraining (same series)

> 10 concurrent `/fit` requests for the same `series_id` with distinct data. Tests the per-series lock and version counter under concurrent load.

| Metric     | Value     |
| ---------- | --------- |
| Throughput | 1.3 req/s |
| p50        | 4 108 ms  |
| p95        | 7 218 ms  |
| p99        | 7 481 ms  |
| Errors     | 0 / 10    |

> Retraining is serialised by a per-series async lock — all 10 requests succeed with no errors and the version counter increments correctly.

### Comparison

| Scenario                  | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Error rate |
| ------------------------- | ------------------ | -------- | -------- | -------- | ---------- |
| A · Concurrent Training   | 93.3               | 98.83    | 180.92   | 184.75   | 0.0%       |
| B · Inference Cache Hit   | 584.1              | 56.31    | 241.76   | 350.45   | 0.0%       |
| C · Inference Cache Miss  | 621.6              | 54.61    | 202.89   | 340.43   | 0.0%       |
| D · Concurrent Retraining | 1.3                | 4108.26  | 7218.37  | 7481.44  | 0.0%       |

Full results with per-request statistics are saved to [`reports/benchmark.md`](reports/benchmark.md).

To re-run:

```bash
make benchmark
```
