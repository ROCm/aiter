#!/bin/bash
set -euo pipefail
trap 'echo $? > /work/baseline-recheck.status' EXIT
for i in $(seq 1 180); do
 if curl -sf --max-time 2 http://localhost:8000/health >/dev/null; then break; fi
 sleep 10
done
# Match hybrid profile warmups before the unprofiled sweep.
bash /work/run_bench.sh baseline_recheck 16 1
bash /work/run_bench.sh baseline_recheck 64 1
for c in 1 8 16 32 64 128; do bash /work/run_bench.sh baseline_recheck "$c" 0; done
python /work/save_completions.py baseline_recheck
