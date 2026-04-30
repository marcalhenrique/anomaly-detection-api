# API Benchmark Report

## Configuration

| Parameter | Value |
|-----------|-------|
| **Generated at** | 2026-04-30T03:38:00.210373+00:00 |
| **Base URL** | http://api:8000 |
| **Redis cache size** | 50 |
| **Models evicted (cold-start)** | 10 |
| **SLA min throughput** | 200.0 req/s |
| **SLA max p99 latency** | 500.0 ms |
| **SLA max error rate** | 1% |

## Overview

This benchmark exercises the anomaly-detection API across four scenarios:

| # | Scenario | What it measures |
|---|----------|------------------|
| A | **Concurrent Training** | Throughput of parallel `/fit` requests for *distinct* series. |
| B | **Inference · Cache Hit** | Prediction latency when model + metadata are **present in Redis**. |
| C | **Inference · Cache Miss** | Prediction latency when model + metadata were **purged from Redis** (cold-start: forces MLflow/MinIO download + Postgres lookup). |
| D | **Concurrent Retraining** | Correctness of the per-series lock and version counter when multiple `/fit` requests target the *same* series simultaneously. |

## SLA Summary

**Overall:** ✅ PASSED

### Scenario B — Inference · Cache Hit

- ✅ Throughput 1080.4 req/s ≥ 200.0 req/s
- ✅ p99 latency 149.1 ms ≤ 500.0 ms
- ✅ Error rate 0.00% ≤ 1.00%

### Scenario C — Inference · Cache Miss

- ✅ Throughput 825.4 req/s ≥ 200.0 req/s
- ✅ p99 latency 301.1 ms ≤ 500.0 ms
- ✅ Error rate 0.00% ≤ 1.00%

---

## Scenario A — Concurrent Training

> Trains **50 models** simultaneously with concurrency **20**.

| Metric | Value |
|--------|-------|
| Requests OK      | 50 / 50 |
| Errors           | 0 |
| Throughput       | 12.1 req/s |
| Mean (ms)        | 1433.56 |
| Median p50 (ms)  | 1275.07 |
| p95 (ms)         | 2732.22 |
| p99 (ms)         | 2760.05 |
| Max (ms)         | 2785.51 |

---

## Scenario B — Inference · Cache Hit

> **1000 predictions** against models that are **in Redis cache**,
> concurrency **100**.  
> Redis model + metadata keys are present — no MLflow/MinIO download required.

| Metric | Value |
|--------|-------|
| Requests OK      | 1000 / 1000 |
| Errors           | 0 |
| Throughput       | 1080.4 req/s |
| Mean (ms)        | 83.13 |
| Median p50 (ms)  | 85.37 |
| p95 (ms)         | 99.15 |
| p99 (ms)         | 149.05 |
| Max (ms)         | 168.06 |

---

## Scenario C — Inference · Cache Miss (Cold Start)

> **1000 predictions** against models **evicted from Redis cache**,
> concurrency **100**.  
> Redis keys (model + metadata) were deleted before the benchmark.  
> Each request forces MLflow/MinIO artifact download + Postgres metadata lookup.

| Metric | Value |
|--------|-------|
| Requests OK      | 1000 / 1000 |
| Errors           | 0 |
| Throughput       | 825.4 req/s |
| Mean (ms)        | 112.15 |
| Median p50 (ms)  | 87.28 |
| p95 (ms)         | 288.61 |
| p99 (ms)         | 301.06 |
| Max (ms)         | 304.00 |

---

## Scenario D — Concurrent Retraining (same series)

> Fires **10 concurrent `/fit` requests** all targeting **`bench-d-retrain`**,
> each with distinct training data, concurrency **10**.  
> Tests the per-series lock and version increment under concurrent load.  
> **Final version reached: 10**

| Metric | Value |
|--------|-------|
| Requests OK      | 10 / 10 |
| Errors           | 0 |
| Throughput       | 1.7 req/s |
| Mean (ms)        | 3192.60 |
| Median p50 (ms)  | 3194.29 |
| p95 (ms)         | 5540.00 |
| p99 (ms)         | 5756.88 |
| Max (ms)         | 5811.10 |

---

## Comparison

| Scenario | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Error rate |
|----------|--------------------|----------|----------|----------|------------|
| A · Concurrent Training | 12.1 | 1275.07 | 2732.22 | 2760.05 | 0.0% |
| B · Inference Cache Hit | 1080.4 | 85.37 | 99.15 | 149.05 | 0.0% |
| C · Inference Cache Miss | 825.4 | 87.28 | 288.61 | 301.06 | 0.0% |
| D · Concurrent Retraining | 1.7 | 3194.29 | 5540.00 | 5756.88 | 0.0% |

---

## Notes

- **Cache architecture:** The API uses Redis as a *single-source-of-truth* distributed cache.  
  Model artifacts (`model:{run_id}`) and metadata (`metadata:latest:{series_id}`, `metadata:version:{series_id}:{version}`)  
  are stored with TTL and shared across all API workers, eliminating the cache-coherency issues of the previous in-process LRU.
- **Scenario C evictions:** Keys are manually deleted from Redis before the benchmark to simulate a genuine cold-start  
  (MLflow/MinIO artifact download + Postgres metadata query).
- **Scenario D expectations:** The per-series lock serialises concurrent `/fit` calls, so the final version should equal  
  the number of successful requests (or total if all succeed).

