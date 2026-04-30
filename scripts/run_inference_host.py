"""
Inference stress test using ThreadPoolExecutor (optimized for host execution).

Avoids asyncio/httpx overhead inside Docker containers.
Usage:
    python scripts/run_inference_host.py [OPTIONS]
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

_SERIES_STATS: dict[str, tuple[float, float]] = {}


def _make_point(
    series_id: str,
    rng: np.random.Generator,
    anomaly: bool,
    base_ts: int,
) -> tuple[int, float]:
    mean, std = _SERIES_STATS.get(series_id, (100.0, 5.0))
    ts = base_ts + int(rng.integers(0, 10_000_000))
    if anomaly:
        direction = rng.choice([-1, 1])
        value = mean + direction * rng.uniform(4.0, 6.0) * std
    else:
        value = float(rng.normal(mean, std))
    return int(ts), float(value)


def _predict(
    session: requests.Session,
    base_url: str,
    series_id: str,
    timestamp: int,
    value: float,
    version: str | None,
) -> dict:
    start = time.perf_counter()
    try:
        params = {"version": version} if version else {}
        response = session.post(
            f"{base_url}/predict/{series_id}",
            json={"timestamp": str(timestamp), "value": value},
            params=params,
            timeout=30.0,
        )
        response.raise_for_status()
        elapsed_ms = (time.perf_counter() - start) * 1000
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
        "# Inference Stress Test Report (Host)",
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference stress test from host using threads"
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--n-series", type=int, default=50)
    parser.add_argument("--n-requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--anomaly-ratio", type=float, default=0.10)
    parser.add_argument("--version", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="reports/inference_host.md")
    args = parser.parse_args()

    started_at = datetime.now(timezone.utc).isoformat()
    rng = np.random.default_rng(seed=args.seed)

    series_ids = [f"sensor-{i:02d}" for i in range(1, args.n_series + 1)]
    for sid in series_ids:
        if sid not in _SERIES_STATS:
            _SERIES_STATS[sid] = (100.0, 5.0)

    n_anomalies = int(args.n_requests * args.anomaly_ratio)
    anomaly_flags = [True] * n_anomalies + [False] * (args.n_requests - n_anomalies)
    rng.shuffle(anomaly_flags)

    base_ts = 1_750_000_000
    request_params = []
    for i, is_anomaly in enumerate(anomaly_flags):
        series_id = series_ids[i % len(series_ids)]
        ts, value = _make_point(series_id, rng, is_anomaly, base_ts)
        request_params.append((series_id, ts, value))

    print(
        f"Inference stress test: {args.n_requests} requests across {len(series_ids)} series"
    )
    print(f"Concurrency: {args.concurrency} | Anomaly ratio: {args.anomaly_ratio:.0%}")
    print(f"Base URL: {args.base_url}")
    print()

    results: list[dict] = []
    t0 = time.perf_counter()

    session = requests.Session()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                _predict, session, args.base_url, sid, ts, val, args.version
            ): (sid, ts, val)
            for sid, ts, val in request_params
        }
        for future in as_completed(futures):
            results.append(future.result())

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

    throughput = len(ok) / total_elapsed
    p99 = float(np.percentile(latencies, 99)) if ok else float("inf")
    error_rate = len(errors) / len(results) if results else 0.0

    violations: list[str] = []
    if throughput < 400.0:
        violations.append(f"throughput {throughput:.1f} req/s < 400 req/s")
    if p99 > 300.0:
        violations.append(f"p99 {p99:.1f} ms > 300 ms")
    if error_rate > 0.01:
        violations.append(f"error rate {error_rate:.1%} > 1%")

    print()
    if violations:
        print("SLA violations:")
        for v in violations:
            print(f"   - {v}")
        raise SystemExit(1)
    else:
        print("All SLA checks passed")


if __name__ == "__main__":
    main()
