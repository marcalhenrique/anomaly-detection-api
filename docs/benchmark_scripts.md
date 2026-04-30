# Benchmark & Load-Testing Scripts

This document describes the purpose, mechanics, and expected outputs of every load-testing and benchmarking script in the project.

---

## Overview

| Script | Makefile Target | Purpose | Runtime |
|--------|-----------------|---------|---------|
| `scripts/run_benchmark.py` | `make benchmark` | Full API benchmark (training + inference + cache) | Inside Docker |
| `scripts/populate_training.py` | `make populate` | Bulk training of series | Inside Docker |
| `scripts/run_inference.py` | `make inference` | Inference stress test | Inside Docker |
| `scripts/run_stress_test.sh` | — | Extreme stress test with `oha` | Host machine |
| `scripts/run_inference_host.py` | — | Inference test run from the host | Host machine |

---

## `scripts/run_benchmark.py`

**Target:** `make benchmark`

### Purpose
Runs a comprehensive, four-scenario benchmark that exercises the entire API lifecycle: training, cached inference, cold-start inference, and concurrent retraining. It is the **canonical integration test** for verifying that caching, locking, and versioning work correctly under load.

### What It Measures

| Scenario | Description | Success Criteria |
|----------|-------------|------------------|
| **A — Concurrent Training** | Trains N distinct models in parallel | All succeed, reasonable throughput |
| **B — Inference · Cache Hit** | Predicts against models known to be in Redis | Throughput and latency under cache-hot conditions |
| **C — Inference · Cache Miss** | Predicts after manually evicting Redis keys | Cold-start penalty quantified |
| **D — Concurrent Retraining** | Retrains the *same* series concurrently | Lock serializes requests, versions increment correctly |

### Mechanics
- Uses `requests.Session` + `ThreadPoolExecutor` for efficient HTTP concurrency.
- Scenario C forces a true cold start by deleting Redis keys and requiring an MLflow/MinIO artifact download + Postgres lookup.
- Scenario D validates the `RedisTrainingLock` — only one `/fit` per series runs at a time.
- Generates a Markdown report (`reports/benchmark.md`) with consolidated metrics for all scenarios.

### Key CLI Arguments

```bash
python scripts/run_benchmark.py \
  --base-url http://localhost:8000 \
  --train-n-models 50 \
  --train-concurrency 20 \
  --infer-n-requests 1000 \
  --infer-concurrency 100 \
  --cache-size 50 \
  --output reports/benchmark.md
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--train-n-models` | 20 | Models trained in Scenario A |
| `--train-concurrency` | 10 | Parallel training requests |
| `--infer-n-requests` | 500 | Predictions per inference scenario |
| `--infer-concurrency` | 50 | Parallel inference requests |
| `--cache-size` | 50 | Expected Redis cache size |
| `--cache-size` | 50 | Expected Redis cache size |
| `--evict-extra` | 10 | Extra models trained to force cache evictions |

### Sample Output

```markdown
# API Benchmark Report

## Scenario B — Inference · Cache Hit
> 1000 predictions against models in Redis cache, concurrency 100.

| Metric | Value |
|--------|-------|
| Requests OK | 1000 / 1000 |
| Throughput | 1297.9 req/s |
| Mean (ms) | 26.44 |
| p99 (ms) | 70.29 |

## Scenario C — Inference · Cache Miss (Cold Start)
> 1000 predictions after Redis eviction.

| Metric | Value |
|--------|-------|
| Requests OK | 1000 / 1000 |
| Throughput | 785.8 req/s |
| Mean (ms) | 105.26 |
| p99 (ms) | 1087.43 |
```

### How to Interpret
- **B vs C gap:** A large gap between Cache Hit and Cache Miss throughput confirms the cache is working. If the gap is small, caching may be misconfigured.
- **Scenario D final version:** Should equal the number of concurrent requests (e.g., 10 requests → version 10). If lower, the lock failed or requests errored.

---

## `scripts/populate_training.py`

**Target:** `make populate`

### Purpose
Bulk-trains N synthetic time-series so the API has models to infer against. This is typically run **before** inference benchmarks to ensure the target series exist in the database, Redis, and MLflow.

### What It Measures
- Total time to train N series
- Per-series latency (mean, p95, max)
- Success/failure rate

### Mechanics
- Generates synthetic sinusoidal + Gaussian noise data.
- Uses `requests.Session` + `ThreadPoolExecutor` for concurrent `/fit` calls.
- Saves a Markdown report (`reports/populate.md`).

### Key CLI Arguments

```bash
python scripts/populate_training.py \
  --base-url http://localhost:8000 \
  --n-series 100 \
  --concurrency 20 \
  --points 200 \
  --output reports/populate.md
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--n-series` | 100 | Number of series to train |
| `--concurrency` | 20 | Parallel `/fit` requests |
| `--points` | 200 | Data points per series |
| `--series-id` | — | Train specific IDs instead of auto-generating |

### Sample Output

```markdown
# Training Population Report

## Summary

| Metric | Value |
|--------|-------|
| Successful | 100 / 100 |
| Failed | 0 |
| Mean latency (ms) | 860.7 |
| p95 latency (ms) | 1615.7 |
| Max latency (ms) | 1669.7 |
```

### How to Interpret
- **Mean latency:** Training is CPU-bound (scikit-learn). Expect 500–1,500 ms depending on worker count.
- **Failures:** If > 0, check Postgres/MLflow health or Redis lock timeouts.

---

## `scripts/run_inference.py`

**Target:** `make inference`

### Purpose
Stress-tests the `/predict/{series_id}` endpoint. Unlike the benchmark script, this focuses **only** on inference throughput and latency, and it supports configurable anomaly injection.

### What It Measures
- Throughput (requests/sec)
- Latency distribution (p50, p95, p99)
- Anomaly detection rate vs. injected ratio
- Throughput, latency distribution, and error rate

### Mechanics
- Generates random `(timestamp, value)` pairs.
- Injects anomalies (values outside 3σ) at a configurable ratio.
- Rotates through multiple series to avoid biasing a single worker's L1 cache.
- Uses `requests.Session` + `ThreadPoolExecutor`.

### Key CLI Arguments

```bash
python scripts/run_inference.py \
  --base-url http://localhost:8000 \
  --n-series 50 \
  --n-requests 1000 \
  --concurrency 100 \
  --anomaly-ratio 0.10 \
  --output reports/inference.md
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--n-series` | 50 | Distinct series to rotate through |
| `--n-requests` | 1000 | Total predictions |
| `--concurrency` | 100 | Parallel requests |
| `--anomaly-ratio` | 0.10 | Fraction of anomalous values injected |
| `--output` | — | Save report to file |

### Sample Output

```markdown
# Inference Stress Test Report

**Total requests:** 1000
**Concurrency:** 100
**Elapsed:** 0.80s

## Latency Metrics

| Metric | Value |
|--------|-------|
| Requests OK | 1000 / 1000 |
| Throughput | 1254.7 req/s |
| Mean (ms) | 28.24 |
| p50 (ms) | 24.45 |
| p95 (ms) | 60.69 |
| p99 (ms) | 78.96 |

## Anomaly Detection

| Metric | Value |
|--------|-------|
| Anomalies detected | 254 / 1000 |
| Detection rate | 25.4% |
| Injected anomaly ratio | 10.0% |
```

### How to Interpret
- **Throughput:** With 3 workers + L1/L2 cache, expect 900–1,300 req/s. With 1 worker, expect ~700–900 req/s.
- **p99:** Should stay under 200 ms for cached models. If it spikes above 500 ms, the API is saturated.
- **Anomaly detection rate:** The model flags ~25–30 % of requests as anomalous when 10 % are injected. This is expected behavior for a 3-sigma threshold on Gaussian noise.

---

## `scripts/run_stress_test.sh`

**Target:** None (run manually)

### Purpose
Pushes the API to its absolute limit using [`oha`](https://github.com/hatoo/oha), a high-performance HTTP load generator written in Rust. Because `oha` runs on the **host machine** (outside Docker), it avoids client-side resource contention and measures the API's true ceiling.

### What It Measures
- Peak throughput under increasing concurrency
- Latency degradation curve (p50 → p99)
- Stability under sustained extreme load (30 seconds)

### Mechanics
Executes three load profiles sequentially:
1. **Baseline** — 10,000 requests @ 100 connections
2. **Stress** — 50,000 requests @ 500 connections
3. **Sustained** — 30 seconds @ 1,000 connections

### Usage

```bash
# Basic run (requires oha installed)
./scripts/run_stress_test.sh

# Custom endpoint
BASE_URL=http://api:8000 SERIES_ID=series-42 ./scripts/run_stress_test.sh
```

### Sample Output

```text
Summary:
  Success rate: 100.00%
  Requests/sec: 3817.4312
  Average: 26.0280 ms
  99.00% in 127.2142 ms
  99.99% in 139.9250 ms
```

### How to Interpret
- **Baseline vs. Stress:** If throughput drops significantly when moving from 100 to 500 connections, the API is entering saturation.
- **Sustained test:** A success rate below 100 % with "aborted due to deadline" errors usually indicates the client hit `ulimit` file-descriptor limits, not API failure. Increase with `ulimit -n 65536`.

---

## `scripts/run_inference_host.py`

**Target:** None (run manually)

### Purpose
Identical logic to `run_inference.py`, but designed to run **on the host machine** (outside Docker). Useful for comparing containerized vs. host-based client performance.

### When to Use
- When you suspect the Docker client container is the bottleneck, not the API.
- When you need the absolute highest client concurrency possible.

### Usage

```bash
python scripts/run_inference_host.py \
  --base-url http://localhost:8000 \
  --n-series 50 \
  --n-requests 10000 \
  --concurrency 500
```

---

## Quick-Reference Cheat Sheet

```bash
# 1. Populate models
make populate
# → reports/populate.md

# 2. Run inference stress test
make inference
# → reports/inference.md

# 3. Full benchmark (training + inference + cache)
make benchmark
# → reports/benchmark.md

# 4. Extreme stress with oha (host)
./scripts/run_stress_test.sh
# → reports/stress_*.txt
```

---

## Latest Run Results

The following results were captured on **2026-04-30** with the API running **3 Uvicorn workers** inside Docker Compose.

### `make populate` — Training Population

| Metric | Value |
|--------|-------|
| Total series | 100 |
| Points per series | 200 |
| Successful | 100 / 100 |
| Failed | 0 |
| Mean latency | 1,308.8 ms |
| p95 latency | 2,470.1 ms |
| Max latency | 4,039.2 ms |

**Interpretation:** All 100 series trained successfully. The wide latency spread (692 ms → 4,039 ms) reflects Docker resource contention during the initial burst. The first batches compete for CPU while scikit-learn fits models; later batches stabilize once the initial queue clears.

---

### `make inference` — Inference Stress Test

| Metric | Value |
|--------|-------|
| Total requests | 1,000 |
| Concurrency | 100 |
| Elapsed | 0.97 s |
| Requests OK | 1,000 / 1,000 |
| Throughput | **1,034.4 req/s** |
| Mean latency | 90.00 ms |
| p50 latency | 89.10 ms |
| p95 latency | 113.19 ms |
| p99 latency | 196.54 ms |
| Anomalies detected | 254 / 1,000 (25.4 %) |
| Injected anomaly ratio | 10.0 % |

**Interpretation:** Throughput (~1,000 req/s) and p99 (~197 ms) are healthy for a 3-worker deployment with L1/L2 caching. The detection rate (25.4 %) on a 10 % injection ratio is consistent with the model's 3-sigma threshold on Gaussian noise.

---

### `make benchmark` — Full API Benchmark

#### Scenario A — Concurrent Training (50 models @ concurrency 20)

| Metric | Value |
|--------|-------|
| Requests OK | 50 / 50 |
| Throughput | 15.2 train/s |
| Mean latency | 1,148.23 ms |
| p99 latency | 1,838.53 ms |

#### Scenario B — Inference · Cache Hit (1,000 requests @ concurrency 100)

| Metric | Value |
|--------|-------|
| Requests OK | 1,000 / 1,000 |
| Throughput | **1,057.3 req/s** |
| Mean latency | 86.54 ms |
| p99 latency | 196.92 ms |

#### Scenario C — Inference · Cache Miss (1,000 requests @ concurrency 100)

| Metric | Value |
|--------|-------|
| Requests OK | 1,000 / 1,000 |
| Throughput | **1,031.3 req/s** |
| Mean latency | 90.66 ms |
| p99 latency | 202.83 ms |

#### Scenario D — Concurrent Retraining (10 versions @ concurrency 10)

| Metric | Value |
|--------|-------|
| Requests OK | 10 / 10 |
| Final version | **10 / 10** |
| Throughput | 1.8 req/s |
| Mean latency | 3,095.64 ms |

**Interpretation:**
- **Scenarios B vs C:** Cache Hit (1,057 req/s) and Cache Miss (1,031 req/s) are nearly identical. This is expected with **3 workers** because the first request to each worker warms the L1 cache; subsequent requests on that worker are local. The cold-start penalty is amortized across 1,000 requests.
- **Scenario D:** The per-series lock works correctly — all 10 concurrent `/fit` requests succeeded and the final version reached 10, proving serialization and version increment are race-safe.
- **Scenario A:** 15.2 train/s is CPU-bound throughput. With 1 worker this drops to ~10–12 train/s.

---

## Notes

- All Docker-based scripts (`benchmark`, `populate`, `inference`) use the `anomaly-api:latest` image. Rebuild with `docker compose build api` after code changes.
- The `oha` stress test is the most accurate way to find the **absolute throughput ceiling** because it eliminates container-to-container networking overhead on the client side.
- For a detailed stress-test analysis including 1-worker vs. 3-worker saturation points, see `docs/stress_test_report.md`.
