#!/bin/bash
set -euo pipefail
trap 'echo $? > /work/flash/bucketed.status' EXIT
for i in $(seq 1 180); do
 if curl -sf --max-time 2 http://localhost:8000/health >/dev/null; then break; fi
 sleep 10
done
for c in 32 128; do bash /work/flash/run_bench.sh bucketed "$c" 1; done
for c in 32 64 128; do
 bash /work/flash/run_bench.sh bucketed_run1 "$c" 0
 bash /work/flash/run_bench.sh bucketed_run2 "$c" 0
done
