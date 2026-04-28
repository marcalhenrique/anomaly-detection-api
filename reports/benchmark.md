# API Benchmark Report

**Generated at:** 2026-04-28T18:08:45.559083+00:00  
**Base URL:** http://api:8000  
**LRU cache size:** 50  
**Models evicted for cache-miss scenario:** 10  

---

## Scenario A — Concurrent Training

> Trains **20 models** simultaneously with concurrency **10**.

| Metric | Value |
|--------|-------|
| Requests OK      | 20 / 20 |
| Errors           | 0 |
| Throughput       | 9.5 req/s |
| Mean (ms)        | 979.59 |
| Median p50 (ms)  | 959.91 |
| p95 (ms)         | 1201.74 |
| p99 (ms)         | 1215.84 |
| Max (ms)         | 1219.36 |

---

## Scenario B — Inference · Cache Hit

> **500 predictions** against models that are **in LRU cache**,
> concurrency **50**.  
> No MLflow/MinIO load is required — prediction runs entirely from memory.

| Metric | Value |
|--------|-------|
| Requests OK      | 500 / 500 |
| Errors           | 0 |
| Throughput       | 544.3 req/s |
| Mean (ms)        | 86.61 |
| Median p50 (ms)  | 59.20 |
| p95 (ms)         | 242.23 |
| p99 (ms)         | 316.55 |
| Max (ms)         | 532.15 |

---

## Scenario C — Inference · Cache Miss

> **500 predictions** against models **evicted from LRU cache**,
> concurrency **50**.  
> Each request must load the model from MLflow/MinIO before predicting.

| Metric | Value |
|--------|-------|
| Requests OK      | 500 / 500 |
| Errors           | 0 |
| Throughput       | 488.2 req/s |
| Mean (ms)        | 89.99 |
| Median p50 (ms)  | 64.00 |
| p95 (ms)         | 261.01 |
| p99 (ms)         | 369.18 |
| Max (ms)         | 618.22 |

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
| Throughput       | 1.4 req/s |
| Mean (ms)        | 4042.03 |
| Median p50 (ms)  | 4048.87 |
| p95 (ms)         | 6964.87 |
| p99 (ms)         | 7220.17 |
| Max (ms)         | 7284.00 |

---

## Comparison

| Scenario | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Error rate |
|----------|--------------------|----------|----------|----------|------------|
| A · Concurrent Training | 9.5 | 959.91 | 1201.74 | 1215.84 | 0.0% |
| B · Inference Cache Hit | 544.3 | 59.20 | 242.23 | 316.55 | 0.0% |
| C · Inference Cache Miss | 488.2 | 64.00 | 261.01 | 369.18 | 0.0% |
| D · Concurrent Retraining | 1.4 | 4048.87 | 6964.87 | 7220.17 | 0.0% |

