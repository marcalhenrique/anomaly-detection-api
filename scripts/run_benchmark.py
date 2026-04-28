"""
Comprehensive API benchmark.

Four scenarios are run sequentially and the results are combined into a
single Markdown report:

  A – Concurrent Training     fires N parallel /fit requests for N distinct series
  B – Inference · Cache Hit   infers on models known to be in LRU cache
  C – Inference · Cache Miss  infers on models that were evicted from LRU
                              (each request forces an MLflow/MinIO load)
  D – Concurrent Retraining   fires N parallel /fit requests for the SAME series_id
                              with different data each time; tests the per-series
                              lock and version increment under concurrent load

Usage:
    python scripts/run_benchmark.py --base-url http://localhost:8000 \\
        --cache-size 50 --output reports/benchmark.md

SLA checks are applied to Scenarios B and C (inference).  Training (A, D) is
measured but not gated.

Defaults:
    --cache-size           50
    --evict-extra          10   (models trained beyond cache to force evictions)
    --train-n-models       20
    --train-concurrency    10
    --infer-n-requests     500  (per scenario)
    --infer-concurrency    50
    --retrain-n-versions   10   (number of retrain requests in Scenario D)
    --retrain-concurrency  10
    --sla-min-throughput   200  req/s
    --sla-max-p99-ms       500  ms
    --sla-max-error-rate   0.01 (1 %)
"""

import argparse
import asyncio
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_training_data(rng: np.random.Generator) -> dict:
    base = 1_745_000_000
    timestamps = [base + i * 60 for i in range(200)]
    values = [float(rng.normal(100.0, 5.0)) for _ in range(200)]
    return {"timestamps": timestamps, "values": values}


def _make_training_data_for_version(version_index: int) -> dict:
    """Generate deterministically distinct training data per version index.

    Using a seeded RNG per index guarantees each retrain request carries
    unique data — the idempotency check won't short-circuit.
    """
    rng = np.random.default_rng(seed=1_000_000 + version_index)
    base = 1_745_000_000
    timestamps = [base + i * 60 for i in range(200)]
    values = [float(rng.normal(100.0 + version_index * 0.1, 5.0)) for _ in range(200)]
    return {"timestamps": timestamps, "values": values}


async def _fit(
    client: httpx.AsyncClient,
    base_url: str,
    series_id: str,
    semaphore: asyncio.Semaphore,
    rng: np.random.Generator,
) -> dict:
    body = _make_training_data(rng)
    async with semaphore:
        start = time.perf_counter()
        try:
            r = await client.post(
                f"{base_url}/fit/{series_id}",
                json=body,
                timeout=120.0,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            r.raise_for_status()
            return {"latency_ms": elapsed_ms, "status": "ok", "error": None}
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {"latency_ms": elapsed_ms, "status": "error", "error": str(exc)}


async def _fit_with_body(
    client: httpx.AsyncClient,
    base_url: str,
    series_id: str,
    body: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Send a /fit request with a pre-built body (used for Scenario D)."""
    async with semaphore:
        start = time.perf_counter()
        try:
            r = await client.post(
                f"{base_url}/fit/{series_id}",
                json=body,
                timeout=120.0,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            r.raise_for_status()
            data = r.json()
            return {
                "latency_ms": elapsed_ms,
                "status": "ok",
                "error": None,
                "version": data.get("version"),
            }
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {
                "latency_ms": elapsed_ms,
                "status": "error",
                "error": str(exc),
                "version": None,
            }


async def _predict(
    client: httpx.AsyncClient,
    base_url: str,
    series_id: str,
    semaphore: asyncio.Semaphore,
    rng: np.random.Generator,
) -> dict:
    ts = 1_750_000_000 + int(rng.integers(0, 1_000_000))
    value = float(rng.normal(100.0, 5.0))
    async with semaphore:
        start = time.perf_counter()
        try:
            r = await client.post(
                f"{base_url}/predict/{series_id}",
                json={"timestamp": str(ts), "value": value},
                timeout=30.0,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            r.raise_for_status()
            return {"latency_ms": elapsed_ms, "status": "ok", "error": None}
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {"latency_ms": elapsed_ms, "status": "error", "error": str(exc)}


# ──────────────────────────────────────────────────────────────────────────────
# Scenario runners
# ──────────────────────────────────────────────────────────────────────────────


async def run_scenario_a(
    client: httpx.AsyncClient,
    base_url: str,
    n_models: int,
    concurrency: int,
    rng: np.random.Generator,
) -> tuple[list[dict], float]:
    """Train N models concurrently; return (results, elapsed_s)."""
    semaphore = asyncio.Semaphore(concurrency)
    series_ids = [f"bench-a-{i:04d}" for i in range(n_models)]
    t0 = time.perf_counter()
    results = await asyncio.gather(
        *[_fit(client, base_url, sid, semaphore, rng) for sid in series_ids]
    )
    return list(results), time.perf_counter() - t0


async def run_scenario_d(
    client: httpx.AsyncClient,
    base_url: str,
    series_id: str,
    n_versions: int,
    concurrency: int,
) -> tuple[list[dict], float, int]:
    """Fire n_versions concurrent /fit requests for the SAME series_id,
    each with distinct data.

    Returns (results, elapsed_s, final_version).
    The per-series lock in TrainingService serialises the requests, so all
    n_versions should succeed and the version counter should reach n_versions
    (or n_versions + any pre-existing versions for this series).
    """
    semaphore = asyncio.Semaphore(concurrency)
    bodies = [_make_training_data_for_version(i) for i in range(n_versions)]
    t0 = time.perf_counter()
    results = await asyncio.gather(
        *[
            _fit_with_body(client, base_url, series_id, body, semaphore)
            for body in bodies
        ]
    )
    elapsed = time.perf_counter() - t0
    ok_results = [r for r in results if r["status"] == "ok"]
    versions = [int(r["version"]) for r in ok_results if r["version"] is not None]
    final_version = max(versions) if versions else 0
    return list(results), elapsed, final_version


async def _setup_cache(
    client: httpx.AsyncClient,
    base_url: str,
    cache_size: int,
    evict_extra: int,
    rng: np.random.Generator,
) -> tuple[list[str], list[str]]:
    """Train cache_size + evict_extra models sequentially to get a
    deterministic LRU state.

    Returns (hit_series, miss_series):
      - hit_series  → last cache_size trained, still in LRU
      - miss_series → first evict_extra trained, evicted from LRU
    """
    total = cache_size + evict_extra
    series_ids = [f"bench-cache-{i:04d}" for i in range(total)]
    print(f"    training {total} models sequentially for cache setup …", flush=True)
    for sid in series_ids:
        await _fit(client, base_url, sid, asyncio.Semaphore(1), rng)
    return series_ids[evict_extra:], series_ids[:evict_extra]


async def run_scenario_b(
    client: httpx.AsyncClient,
    base_url: str,
    hit_series: list[str],
    n_requests: int,
    concurrency: int,
    rng: np.random.Generator,
) -> tuple[list[dict], float]:
    """Infer against models that are in LRU cache."""
    semaphore = asyncio.Semaphore(concurrency)
    t0 = time.perf_counter()
    results = await asyncio.gather(
        *[
            _predict(client, base_url, hit_series[i % len(hit_series)], semaphore, rng)
            for i in range(n_requests)
        ]
    )
    return list(results), time.perf_counter() - t0


async def run_scenario_c(
    client: httpx.AsyncClient,
    base_url: str,
    miss_series: list[str],
    n_requests: int,
    concurrency: int,
    rng: np.random.Generator,
) -> tuple[list[dict], float]:
    """Infer against models that were evicted from LRU cache."""
    semaphore = asyncio.Semaphore(concurrency)
    t0 = time.perf_counter()
    results = await asyncio.gather(
        *[
            _predict(
                client, base_url, miss_series[i % len(miss_series)], semaphore, rng
            )
            for i in range(n_requests)
        ]
    )
    return list(results), time.perf_counter() - t0


# ──────────────────────────────────────────────────────────────────────────────
# Markdown report
# ──────────────────────────────────────────────────────────────────────────────


def _stats_rows(results: list[dict], elapsed_s: float) -> list[str]:
    ok = [r for r in results if r["status"] == "ok"]
    errors = [r for r in results if r["status"] == "error"]
    lats = np.array([r["latency_ms"] for r in ok]) if ok else np.array([0.0])
    return [
        f"| Requests OK      | {len(ok)} / {len(results)} |",
        f"| Errors           | {len(errors)} |",
        f"| Throughput       | {len(ok) / elapsed_s:.1f} req/s |",
        f"| Mean (ms)        | {float(np.mean(lats)):.2f} |",
        f"| Median p50 (ms)  | {float(np.median(lats)):.2f} |",
        f"| p95 (ms)         | {float(np.percentile(lats, 95)):.2f} |",
        f"| p99 (ms)         | {float(np.percentile(lats, 99)):.2f} |",
        f"| Max (ms)         | {float(np.max(lats)):.2f} |",
    ]


def _comparison_row(label: str, results: list[dict], elapsed_s: float) -> str:
    ok = [r for r in results if r["status"] == "ok"]
    if not ok:
        return f"| {label} | N/A | N/A | N/A | N/A | N/A |"
    lats = np.array([r["latency_ms"] for r in ok])
    err_pct = (1 - len(ok) / len(results)) * 100 if results else 100.0
    return (
        f"| {label} | {len(ok) / elapsed_s:.1f} | "
        f"{float(np.median(lats)):.2f} | "
        f"{float(np.percentile(lats, 95)):.2f} | "
        f"{float(np.percentile(lats, 99)):.2f} | "
        f"{err_pct:.1f}% |"
    )


def build_markdown(
    *,
    base_url: str,
    started_at: str,
    cache_size: int,
    evict_extra: int,
    a_results: list[dict],
    a_elapsed: float,
    a_n_models: int,
    a_concurrency: int,
    b_results: list[dict],
    b_elapsed: float,
    b_n_requests: int,
    b_concurrency: int,
    c_results: list[dict],
    c_elapsed: float,
    c_n_requests: int,
    c_concurrency: int,
    d_results: list[dict],
    d_elapsed: float,
    d_n_versions: int,
    d_concurrency: int,
    d_series_id: str,
    d_final_version: int,
) -> str:
    lines = [
        "# API Benchmark Report",
        "",
        f"**Generated at:** {started_at}  ",
        f"**Base URL:** {base_url}  ",
        f"**LRU cache size:** {cache_size}  ",
        f"**Models evicted for cache-miss scenario:** {evict_extra}  ",
        "",
        "---",
        "",
        "## Scenario A — Concurrent Training",
        "",
        f"> Trains **{a_n_models} models** simultaneously with concurrency **{a_concurrency}**.",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        *_stats_rows(a_results, a_elapsed),
        "",
        "---",
        "",
        "## Scenario B — Inference · Cache Hit",
        "",
        f"> **{b_n_requests} predictions** against models that are **in LRU cache**,",
        f"> concurrency **{b_concurrency}**.  ",
        "> No MLflow/MinIO load is required — prediction runs entirely from memory.",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        *_stats_rows(b_results, b_elapsed),
        "",
        "---",
        "",
        "## Scenario C — Inference · Cache Miss",
        "",
        f"> **{c_n_requests} predictions** against models **evicted from LRU cache**,",
        f"> concurrency **{c_concurrency}**.  ",
        "> Each request must load the model from MLflow/MinIO before predicting.",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        *_stats_rows(c_results, c_elapsed),
        "",
        "---",
        "",
        "## Scenario D — Concurrent Retraining (same series)",
        "",
        f"> Fires **{d_n_versions} concurrent `/fit` requests** all targeting **`{d_series_id}`**,",
        f"> each with distinct training data, concurrency **{d_concurrency}**.  ",
        "> Tests the per-series lock and version increment under concurrent load.  ",
        f"> **Final version reached: {d_final_version}**",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        *_stats_rows(d_results, d_elapsed),
        "",
        "---",
        "",
        "## Comparison",
        "",
        "| Scenario | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Error rate |",
        "|----------|--------------------|----------|----------|----------|------------|",
        _comparison_row("A · Concurrent Training", a_results, a_elapsed),
        _comparison_row("B · Inference Cache Hit", b_results, b_elapsed),
        _comparison_row("C · Inference Cache Miss", c_results, c_elapsed),
        _comparison_row("D · Concurrent Retraining", d_results, d_elapsed),
        "",
    ]
    return "\n".join(lines) + "\n"


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


async def main(args: argparse.Namespace) -> None:
    started_at = datetime.now(timezone.utc).isoformat()
    rng = np.random.default_rng(seed=args.seed)

    print("=" * 60)
    print("  API Benchmark")
    print("=" * 60)
    print(f"  Base URL:          {args.base_url}")
    print(f"  LRU cache size:    {args.cache_size}")
    print(f"  Evict extra:       {args.evict_extra}")
    print(
        f"  Train models (A):  {args.train_n_models} @ concurrency {args.train_concurrency}"
    )
    print(
        f"  Infer requests:    {args.infer_n_requests} per scenario @ concurrency {args.infer_concurrency}"
    )
    print(
        f"  Retrain (D):       {args.retrain_n_versions} versions @ concurrency {args.retrain_concurrency}"
    )
    print("=" * 60)
    print()

    async with httpx.AsyncClient() as client:
        # ── Scenario A: Concurrent Training ────────────────────────────────
        print("[A] Concurrent Training …", flush=True)
        a_results, a_elapsed = await run_scenario_a(
            client,
            args.base_url,
            n_models=args.train_n_models,
            concurrency=args.train_concurrency,
            rng=rng,
        )
        a_ok = [r for r in a_results if r["status"] == "ok"]
        a_lats = np.array([r["latency_ms"] for r in a_ok]) if a_ok else np.array([0.0])
        print(
            f"    {len(a_ok)}/{len(a_results)} ok  |  "
            f"{len(a_ok) / a_elapsed:.1f} train/s  |  "
            f"p99 {float(np.percentile(a_lats, 99)):.0f} ms"
        )
        print()

        # ── Setup cache for B + C ───────────────────────────────────────────
        print("[setup] Preparing cache state for scenarios B and C …", flush=True)
        hit_series, miss_series = await _setup_cache(
            client,
            args.base_url,
            cache_size=args.cache_size,
            evict_extra=args.evict_extra,
            rng=rng,
        )
        print(
            f"    in-cache: {len(hit_series)} series | evicted: {len(miss_series)} series"
        )
        print()

        # ── Scenario B: Inference Cache Hit ────────────────────────────────
        print("[B] Inference Cache Hit …", flush=True)
        b_results, b_elapsed = await run_scenario_b(
            client,
            args.base_url,
            hit_series=hit_series,
            n_requests=args.infer_n_requests,
            concurrency=args.infer_concurrency,
            rng=rng,
        )
        b_ok = [r for r in b_results if r["status"] == "ok"]
        b_lats = np.array([r["latency_ms"] for r in b_ok]) if b_ok else np.array([0.0])
        print(
            f"    {len(b_ok)}/{len(b_results)} ok  |  "
            f"{len(b_ok) / b_elapsed:.1f} req/s  |  "
            f"p99 {float(np.percentile(b_lats, 99)):.0f} ms"
        )
        print()

        # ── Scenario C: Inference Cache Miss ───────────────────────────────
        print("[C] Inference Cache Miss …", flush=True)
        c_results, c_elapsed = await run_scenario_c(
            client,
            args.base_url,
            miss_series=miss_series,
            n_requests=args.infer_n_requests,
            concurrency=args.infer_concurrency,
            rng=rng,
        )
        c_ok = [r for r in c_results if r["status"] == "ok"]
        c_lats = np.array([r["latency_ms"] for r in c_ok]) if c_ok else np.array([0.0])
        print(
            f"    {len(c_ok)}/{len(c_results)} ok  |  "
            f"{len(c_ok) / c_elapsed:.1f} req/s  |  "
            f"p99 {float(np.percentile(c_lats, 99)):.0f} ms"
        )
        print()

        # ── Scenario D: Concurrent Retraining ──────────────────────────────
        d_series_id = "bench-d-retrain"
        print(
            f"[D] Concurrent Retraining ({args.retrain_n_versions}x '{d_series_id}') …",
            flush=True,
        )
        d_results, d_elapsed, d_final_version = await run_scenario_d(
            client,
            args.base_url,
            series_id=d_series_id,
            n_versions=args.retrain_n_versions,
            concurrency=args.retrain_concurrency,
        )
        d_ok = [r for r in d_results if r["status"] == "ok"]
        d_lats = np.array([r["latency_ms"] for r in d_ok]) if d_ok else np.array([0.0])
        print(
            f"    {len(d_ok)}/{len(d_results)} ok  |  "
            f"{len(d_ok) / d_elapsed:.1f} train/s  |  "
            f"p99 {float(np.percentile(d_lats, 99)):.0f} ms  |  "
            f"final version: {d_final_version}"
        )
        print()

    # ── Report ───────────────────────────────────────────────────────────────
    if args.output:
        md = build_markdown(
            base_url=args.base_url,
            started_at=started_at,
            cache_size=args.cache_size,
            evict_extra=args.evict_extra,
            a_results=a_results,
            a_elapsed=a_elapsed,
            a_n_models=args.train_n_models,
            a_concurrency=args.train_concurrency,
            b_results=b_results,
            b_elapsed=b_elapsed,
            b_n_requests=args.infer_n_requests,
            b_concurrency=args.infer_concurrency,
            c_results=c_results,
            c_elapsed=c_elapsed,
            c_n_requests=args.infer_n_requests,
            c_concurrency=args.infer_concurrency,
            d_results=d_results,
            d_elapsed=d_elapsed,
            d_n_versions=args.retrain_n_versions,
            d_concurrency=args.retrain_concurrency,
            d_series_id=d_series_id,
            d_final_version=d_final_version,
        )
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md)
        print(f"Report saved: {out}")
        print()

    # ── SLA checks (inference scenarios only) ────────────────────────────────
    violations: list[str] = []
    for label, results, elapsed in [
        ("B Cache Hit", b_results, b_elapsed),
        ("C Cache Miss", c_results, c_elapsed),
    ]:
        ok = [r for r in results if r["status"] == "ok"]
        lats = (
            np.array([r["latency_ms"] for r in ok]) if ok else np.array([float("inf")])
        )
        thr = len(ok) / elapsed
        p99 = float(np.percentile(lats, 99))
        err_rate = 1 - len(ok) / len(results) if results else 1.0

        if thr < args.sla_min_throughput:
            violations.append(
                f"[{label}] throughput {thr:.1f} req/s < {args.sla_min_throughput} req/s"
            )
        if p99 > args.sla_max_p99_ms:
            violations.append(f"[{label}] p99 {p99:.1f} ms > {args.sla_max_p99_ms} ms")
        if err_rate > args.sla_max_error_rate:
            violations.append(
                f"[{label}] error rate {err_rate:.1%} > {args.sla_max_error_rate:.1%}"
            )

    if violations:
        print("SLA violations:")
        for v in violations:
            print(f"   - {v}")
        sys.exit(1)
    else:
        print("All SLA checks passed")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Comprehensive API benchmark (training + cache-hit + cache-miss inference)"
    )
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument(
        "--cache-size",
        type=int,
        default=50,
        help="LRU model cache size — must match LRU_CACHE_SIZE in .env (default: 50)",
    )
    p.add_argument(
        "--evict-extra",
        type=int,
        default=10,
        help="Extra models trained beyond cache size to force LRU evictions (default: 10)",
    )
    p.add_argument(
        "--train-n-models",
        type=int,
        default=20,
        help="Number of models to train in Scenario A (default: 20)",
    )
    p.add_argument(
        "--train-concurrency",
        type=int,
        default=10,
        help="Concurrency for Scenario A training (default: 10)",
    )
    p.add_argument(
        "--infer-n-requests",
        type=int,
        default=500,
        help="Inference requests for scenarios B and C each (default: 500)",
    )
    p.add_argument(
        "--infer-concurrency",
        type=int,
        default=50,
        help="Concurrency for inference scenarios (default: 50)",
    )
    p.add_argument(
        "--retrain-n-versions",
        type=int,
        default=10,
        help="Number of concurrent retrain requests in Scenario D (default: 10)",
    )
    p.add_argument(
        "--retrain-concurrency",
        type=int,
        default=10,
        help="Concurrency for Scenario D retraining (default: 10)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output", metavar="FILE", default=None, help="Markdown report path"
    )
    p.add_argument("--sla-min-throughput", type=float, default=200.0, metavar="N")
    p.add_argument("--sla-max-p99-ms", type=float, default=500.0, metavar="N")
    p.add_argument("--sla-max-error-rate", type=float, default=0.01, metavar="R")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
