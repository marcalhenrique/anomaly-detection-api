"""
Populate training data for anomaly detection series.

Uses requests + ThreadPoolExecutor for efficient concurrency inside Docker.

Usage:
    python scripts/populate_training.py [OPTIONS]

Examples:
    # Train 5 auto-generated series with 200 points each
    python scripts/populate_training.py --n-series 5

    # Train specific series with 300 points
    python scripts/populate_training.py --series-id sensor-01 --series-id sensor-02 --points 300

    # Save report to a custom path
    python scripts/populate_training.py --n-series 3 --output reports/training.md

    # Via Docker
    docker exec anomaly-api python scripts/populate_training.py --base-url http://localhost:8000 --n-series 5
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests


def generate_series_data(
    n_points: int,
    rng: np.random.Generator,
) -> tuple[list[int], list[float]]:
    """Generate synthetic sinusoidal + Gaussian noise time series."""
    amplitude = rng.uniform(5.0, 20.0)
    frequency = rng.uniform(0.01, 0.05)
    noise_std = rng.uniform(0.5, 2.0)
    base_value = rng.uniform(50.0, 150.0)

    base_ts = 1_745_000_000
    interval = 60  # 1 minute between points

    timestamps = [base_ts + i * interval for i in range(n_points)]
    values = [
        base_value
        + amplitude * float(np.sin(2 * np.pi * frequency * i))
        + float(rng.normal(0, noise_std))
        for i in range(n_points)
    ]
    return timestamps, values


def train_series(
    session: requests.Session,
    base_url: str,
    series_id: str,
    timestamps: list[int],
    values: list[float],
) -> dict:
    start = time.perf_counter()
    try:
        response = session.post(
            f"{base_url}/fit/{series_id}",
            json={"timestamps": timestamps, "values": values},
            timeout=60.0,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        data = response.json()
        return {
            "series_id": series_id,
            "version": data.get("version", "?"),
            "points_used": data.get("points_used", len(values)),
            "latency_ms": elapsed_ms,
            "status": "ok",
            "error": None,
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "series_id": series_id,
            "version": "-",
            "points_used": 0,
            "latency_ms": elapsed_ms,
            "status": "error",
            "error": str(exc),
        }


def build_markdown(
    results: list[dict],
    n_points: int,
    base_url: str,
    started_at: str,
    total_elapsed_s: float,
) -> str:
    ok = [r for r in results if r["status"] == "ok"]
    errors = [r for r in results if r["status"] == "error"]
    latencies = [r["latency_ms"] for r in ok] if ok else [0.0]

    lines = [
        "# Training Population Report",
        "",
        f"**Generated at:** {started_at}  ",
        f"**Base URL:** {base_url}  ",
        f"**Points per series:** {n_points}  ",
        f"**Total series:** {len(results)}  ",
        f"**Elapsed:** {total_elapsed_s:.2f}s  ",
        "",
        "## Results",
        "",
        "| Series ID | Version | Points Used | Latency (ms) | Status |",
        "|-----------|---------|-------------|--------------|--------|",
    ]
    for r in results:
        status_cell = "ok" if r["status"] == "ok" else f"error: {r['error']}"
        lines.append(
            f"| {r['series_id']} | {r['version']} | {r['points_used']} "
            f"| {r['latency_ms']:.1f} | {status_cell} |"
        )

    lines += [
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Successful | {len(ok)} / {len(results)} |",
        f"| Failed | {len(errors)} |",
        f"| Mean latency (ms) | {float(np.mean(latencies)):.1f} |",
        f"| p95 latency (ms) | {float(np.percentile(latencies, 95)):.1f} |",
        f"| Max latency (ms) | {float(np.max(latencies)):.1f} |",
    ]

    if errors:
        lines += ["", "## Errors", ""]
        for r in errors:
            lines.append(f"- **{r['series_id']}**: {r['error']}")

    return "\n".join(lines) + "\n"


def main(args: argparse.Namespace) -> None:
    started_at = datetime.now(timezone.utc).isoformat()
    rng = np.random.default_rng(seed=args.seed)

    if args.series_id:
        series_ids = args.series_id
    else:
        series_ids = [f"sensor-{i:02d}" for i in range(1, args.n_series + 1)]

    print(f"Training {len(series_ids)} series with {args.points} points each...")
    print(f"Base URL: {args.base_url}")
    print(f"Concurrency: {args.concurrency}")
    print()

    series_data = {}
    for series_id in series_ids:
        timestamps, values = generate_series_data(args.points, rng)
        series_data[series_id] = (timestamps, values)

    results: list[dict] = []
    t0 = time.perf_counter()

    session = requests.Session()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                train_series,
                session,
                args.base_url,
                series_id,
                series_data[series_id][0],
                series_data[series_id][1],
            ): series_id
            for series_id in series_ids
        }
        for future in as_completed(futures):
            results.append(future.result())

    total_elapsed = time.perf_counter() - t0

    # Print table to terminal
    col_w = max(len(r["series_id"]) for r in results) + 2
    print(
        f"{'Series ID':<{col_w}} {'Version':>8} {'Points':>8} {'Latency(ms)':>12} {'Status'}"
    )
    print("-" * (col_w + 40))
    for r in results:
        status = "ok" if r["status"] == "ok" else f"ERROR: {r['error']}"
        print(
            f"{r['series_id']:<{col_w}} {r['version']:>8} {r['points_used']:>8} "
            f"{r['latency_ms']:>11.1f} {status}"
        )

    ok_count = sum(1 for r in results if r["status"] == "ok")
    print()
    print(f"Done: {ok_count}/{len(results)} succeeded in {total_elapsed:.2f}s")

    if args.output:
        md = build_markdown(
            results, args.points, args.base_url, started_at, total_elapsed
        )
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md)
        print(f"Report saved to: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate training data for anomaly detection."
    )
    parser.add_argument(
        "--base-url", default="http://localhost:8000", help="API base URL"
    )
    parser.add_argument(
        "--series-id",
        action="append",
        metavar="ID",
        help="Series ID to train (can be repeated). Overrides --n-series.",
    )
    parser.add_argument(
        "--n-series",
        type=int,
        default=5,
        metavar="N",
        help="Number of series to generate (default: 5)",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=200,
        metavar="N",
        help="Data points per series (default: 200)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        metavar="N",
        help="Max concurrent requests (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Save Markdown report to this path (optional)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
