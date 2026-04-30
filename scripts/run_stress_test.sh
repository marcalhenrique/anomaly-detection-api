#!/usr/bin/env bash
set -euo pipefail

# Stress test automation script for the Anomaly Detection API.
# Requires:
#   - API running on localhost:8000
#   - `oha` installed and available in PATH
#   - A trained series (default: sensor-01)
#
# Usage:
#   ./scripts/run_stress_test.sh
#   BASE_URL=http://api:8000 SERIES_ID=series-42 ./scripts/run_stress_test.sh

BASE_URL="${BASE_URL:-http://localhost:8000}"
SERIES_ID="${SERIES_ID:-sensor-01}"
ENDPOINT="${BASE_URL}/predict/${SERIES_ID}"
PAYLOAD='{"timestamp":"1750000000","value":100.0}'

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORTS_DIR="reports"
mkdir -p "${REPORTS_DIR}"

# Ensure oha is available
if ! command -v oha &>/dev/null; then
    echo "Error: 'oha' not found in PATH. Install it with:"
    echo "  cargo install oha"
    exit 1
fi

# Ensure API is reachable
echo "Checking API health at ${BASE_URL}/metrics/ ..."
if ! curl -sf -o /dev/null "${BASE_URL}/metrics/"; then
    echo "Error: API is not responding at ${BASE_URL}"
    exit 1
fi

echo "Target endpoint: ${ENDPOINT}"
echo "Reports will be saved to: ${REPORTS_DIR}/"
echo ""

# Increase file-descriptor limit to avoid client-side "Too many open files"
# This may fail in some restricted shells; we warn but continue.
if ! ulimit -n 65536 2>/dev/null; then
    echo "Warning: could not raise ulimit -n. High-concurrency tests may hit client limits."
fi

run_test() {
    local name=$1
    local requests=$2
    local concurrency=$3
    local duration_flag=$4  # empty for -n, or "-z 30s" for sustained
    local out_file="${REPORTS_DIR}/stress_${name}_${TIMESTAMP}.txt"

    echo "=============================================="
    echo "Test: ${name}"
    if [ -n "${duration_flag}" ]; then
        echo "Duration: 30 seconds | Concurrency: ${concurrency}"
    else
        echo "Requests: ${requests} | Concurrency: ${concurrency}"
    fi
    echo "Saving output to: ${out_file}"
    echo "=============================================="

    if [ -n "${duration_flag}" ]; then
        oha ${duration_flag} -c "${concurrency}" --no-tui \
            -m POST -T "application/json" -d "${PAYLOAD}" \
            "${ENDPOINT}" | tee "${out_file}"
    else
        oha -n "${requests}" -c "${concurrency}" --no-tui \
            -m POST -T "application/json" -d "${PAYLOAD}" \
            "${ENDPOINT}" | tee "${out_file}"
    fi

    echo ""
    echo "Done: ${name}"
    echo ""
}

# ── Test Matrix ──────────────────────────────────────────────────────────

run_test "baseline" 10000 100 ""
run_test "stress" 50000 500 ""
run_test "sustained" "" 1000 "-z 30s"

echo "All tests completed. Reports saved to ${REPORTS_DIR}/"
