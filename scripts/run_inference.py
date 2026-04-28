"""
Run inference stress test against the anomaly detection API.

Usage:
    python scripts/run_inference.py [OPTIONS]

Examples:
    # 500 requests across 5 auto-generated series, concurrency 20
    python scripts/run_inference.py --n-series 5

    # Custom series, 1000 requests, concurrency 50, 15% anomalies
    python scripts/run_inference.py --series-id sensor-01 --series-id sensor-02 \\
        --n-requests 1000 --concurrency 50 --anomaly-ratio 0.15

    # Save report
    python scripts/run_inference.py --n-series 5 --output reports/inference.md

    # Via Docker
    docker exec anomaly-api python scripts/run_inference.py --base-url http://localhost:8000 --n-series 5

SLA defaults (override via CLI flags):
    --sla-min-throughput  400   req/s
    --sla-max-p99-ms      300   ms
    --sla-max-error-rate  0.01  (1 %)
"""

import argparse
import asyncio
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

# Approximate distribution per series — populated after training (or set fixed defaults)
# series_id -> (mean, std) used to generate anomalous points outside 3σ
_SERIES_STATS: dict[str, tuple[float, float]] = {}


def _make_point(
    series_id: str,
    rng: np.random.Generator,
    anomaly: bool,
    base_ts: int,
) -> tuple[int, float]:
    """Generate a (timestamp, value) pair, optionally anomalous."""
    mean, std = _SERIES_STATS.get(series_id, (100.0, 5.0))
    ts = base_ts + int(rng.integers(0, 10_000_000))
    if anomaly:
        # Push well outside 3-sigma to guarantee detection
        direction = rng.choice([-1, 1])
        value = mean + direction * rng.uniform(4.0, 6.0) * std
    else:
        value = float(rng.normal(mean, std))
    return int(ts), float(value)


async def _predict(
    client: httpx.AsyncClient,
    base_url: str,
    series_id: str,
    timestamp: int,
    value: float,
    version: str | None,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        start = time.perf_counter()
        try:
            params = {"version": version} if version else {}
            response = await client.post(
                f"{base_url}/predict/{series_id}",
                json={"timestamp": str(timestamp), "value": value},
                params=params,
                timeout=30.0,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            response.raise_for_status()
            data = response.json()
            return {
                "series_id": series_id,
                "latency_ms": elapsed_ms,
                "anomaly": data.get("anomaly", False),
                "model_version": data.get("model_version", "?"),
                "status": "ok",
                "error": None,
            }
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {
                "series_id": series_id,
                "latency_ms": elapsed_ms,
                "anomaly": False,
                "model_version": "-",
                "status": "error",
                "error": str(exc),
            }


def build_markdown(
    results: list[dict],
    series_ids: list[str],
    n_requests: int,
    concurrency: int,
    anomaly_ratio: float,
    base_url: str,
    started_at: str,
    total_elapsed_s: float,
) -> str:
    ok = [r for r in results if r["status"] == "ok"]
    errors = [r for r in results if r["status"] == "error"]
    latencies = np.array([r["latency_ms"] for r in ok]) if ok else np.array([0.0])
    anomalies_detected = sum(1 for r in ok if r["anomaly"])

    lines = [
        "# Inference Stress Test Report",
        "",
        f"**Generated at:** {started_at}  ",
        f"**Base URL:** {base_url}  ",
        f"**Series:** {', '.join(series_ids)}  ",
        f"**Total requests:** {n_requests}  ",
        f"**Concurrency:** {concurrency}  ",
        f"**Anomaly ratio (injected):** {anomaly_ratio:.0%}  ",
        f"**Elapsed:** {total_elapsed_s:.2f}s  ",
        "",
        "## Latency Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Requests OK | {len(ok)} / {n_requests} |",
        f"| Errors | {len(errors)} |",
        f"| Throughput | {len(ok) / total_elapsed_s:.1f} req/s |",
        f"| Mean (ms) | {float(np.mean(latencies)):.2f} |",
        f"| Median / p50 (ms) | {float(np.median(latencies)):.2f} |",
        f"| p95 (ms) | {float(np.percentile(latencies, 95)):.2f} |",
        f"| p99 (ms) | {float(np.percentile(latencies, 99)):.2f} |",
        f"| Max (ms) | {float(np.max(latencies)):.2f} |",
        "",
        "## Anomaly Detection",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Anomalies detected | {anomalies_detected} / {len(ok)} |",
        f"| Detection rate | {anomalies_detected / len(ok):.1%} |"
        if ok
        else "| Detection rate | N/A |",
        f"| Injected anomaly ratio | {anomaly_ratio:.1%} |",
    ]

    if errors:
        lines += ["", "## Errors", ""]
        from collections import Counter

        counts = Counter(r["error"] for r in errors)
        for msg, count in counts.most_common():
            lines.append(f"- ({count}x) `{msg}`")

    return "\n".join(lines) + "\n"


async def main(args: argparse.Namespace) -> None:
    started_at = datetime.now(timezone.utc).isoformat()
    rng = np.random.default_rng(seed=args.seed)

    if args.series_id:
        series_ids = args.series_id
    else:
        series_ids = [f"sensor-{i:02d}" for i in range(1, args.n_series + 1)]

    # Set default stats for each series (override with observed values if desired)
    for sid in series_ids:
        if sid not in _SERIES_STATS:
            _SERIES_STATS[sid] = (100.0, 5.0)

    print(
        f"Inference stress test: {args.n_requests} requests across {len(series_ids)} series"
    )
    print(f"Concurrency: {args.concurrency} | Anomaly ratio: {args.anomaly_ratio:.0%}")
    print(f"Base URL: {args.base_url}")
    print()

    # Build request list (round-robin series assignment)
    n_anomalies = int(args.n_requests * args.anomaly_ratio)
    anomaly_flags = [True] * n_anomalies + [False] * (args.n_requests - n_anomalies)
    rng.shuffle(anomaly_flags)

    base_ts = 1_750_000_000
    request_params = []
    for i, is_anomaly in enumerate(anomaly_flags):
        series_id = series_ids[i % len(series_ids)]
        ts, value = _make_point(series_id, rng, is_anomaly, base_ts)
        request_params.append((series_id, ts, value, is_anomaly))

    semaphore = asyncio.Semaphore(args.concurrency)
    t0 = time.perf_counter()

    async with httpx.AsyncClient() as client:
        tasks = [
            _predict(client, args.base_url, sid, ts, val, args.version, semaphore)
            for sid, ts, val, _ in request_params
        ]
        results = await asyncio.gather(*tasks)

    total_elapsed = time.perf_counter() - t0

    ok = [r for r in results if r["status"] == "ok"]
    errors = [r for r in results if r["status"] == "error"]
    latencies = np.array([r["latency_ms"] for r in ok]) if ok else np.array([0.0])
    anomalies_detected = sum(1 for r in ok if r["anomaly"])

    print(f"Results: {len(ok)} ok / {len(errors)} errors in {total_elapsed:.2f}s")
    print(f"Throughput:        {len(ok) / total_elapsed:.1f} req/s")
    print(f"Latency mean:      {float(np.mean(latencies)):.2f} ms")
    print(f"Latency p95:       {float(np.percentile(latencies, 95)):.2f} ms")
    print(f"Latency p99:       {float(np.percentile(latencies, 99)):.2f} ms")
    print(f"Latency max:       {float(np.max(latencies)):.2f} ms")
    print(
        f"Anomalies detected: {anomalies_detected}/{len(ok)} ({anomalies_detected / len(ok):.1%} rate)"
        if ok
        else ""
    )

    if args.output:
        md = build_markdown(
            results,
            series_ids,
            args.n_requests,
            args.concurrency,
            args.anomaly_ratio,
            args.base_url,
            started_at,
            total_elapsed,
        )
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md)
        print(f"\nReport saved to: {out_path}")

    # ------------------------------------------------------------------
    # SLA validation — exits with code 1 if any threshold is breached.
    # ------------------------------------------------------------------
    throughput = len(ok) / total_elapsed
    p99 = float(np.percentile(latencies, 99)) if ok else float("inf")
    error_rate = len(errors) / len(results) if results else 0.0

    violations: list[str] = []
    if throughput < args.sla_min_throughput:
        violations.append(
            f"throughput {throughput:.1f} req/s < {args.sla_min_throughput} req/s"
        )
    if p99 > args.sla_max_p99_ms:
        violations.append(f"p99 {p99:.1f} ms > {args.sla_max_p99_ms} ms")
    if error_rate > args.sla_max_error_rate:
        violations.append(
            f"error rate {error_rate:.1%} > {args.sla_max_error_rate:.1%}"
        )

    print()
    if violations:
        print("SLA violations:")
        for v in violations:
            print(f"   - {v}")
        sys.exit(1)
    else:
        print("All SLA checks passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference stress test against the anomaly detection API."
    )
    parser.add_argument(
        "--base-url", default="http://localhost:8000", help="API base URL"
    )
    parser.add_argument(
        "--series-id",
        action="append",
        metavar="ID",
        help="Series ID to query (can be repeated). Overrides --n-series.",
    )
    parser.add_argument(
        "--n-series",
        type=int,
        default=5,
        metavar="N",
        help="Number of series to use (default: 5)",
    )
    parser.add_argument(
        "--n-requests",
        type=int,
        default=500,
        metavar="N",
        help="Total inference requests (default: 500)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        metavar="N",
        help="Max concurrent requests (default: 20)",
    )
    parser.add_argument(
        "--anomaly-ratio",
        type=float,
        default=0.1,
        metavar="R",
        help="Fraction of injected anomalies (default: 0.1)",
    )
    parser.add_argument(
        "--version",
        default=None,
        metavar="V",
        help="Model version to query (optional, uses latest if omitted)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Save Markdown report to this path (optional)",
    )
    # SLA thresholds
    parser.add_argument(
        "--sla-min-throughput",
        type=float,
        default=400.0,
        metavar="N",
        help="Minimum acceptable throughput in req/s (default: 400)",
    )
    parser.add_argument(
        "--sla-max-p99-ms",
        type=float,
        default=300.0,
        metavar="N",
        help="Maximum acceptable p99 latency in ms (default: 300)",
    )
    parser.add_argument(
        "--sla-max-error-rate",
        type=float,
        default=0.01,
        metavar="R",
        help="Maximum acceptable error rate 0-1 (default: 0.01)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
