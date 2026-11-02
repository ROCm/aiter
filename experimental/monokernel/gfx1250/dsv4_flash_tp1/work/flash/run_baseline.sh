#!/bin/bash
set -euo pipefail
trap 'echo $? > /work/flash/baseline.status' EXIT
for i in $(seq 1 180); do
 if curl -sf --max-time 2 http://localhost:8000/health >/dev/null; then break; fi
 sleep 10
done
for c in 1 16 128; do bash /work/flash/run_bench.sh baseline "$c" 1; done
for c in 1 8 16 32 64 128; do
 bash /work/flash/run_bench.sh baseline_run1 "$c" 0
 bash /work/flash/run_bench.sh baseline_run2 "$c" 0
done
