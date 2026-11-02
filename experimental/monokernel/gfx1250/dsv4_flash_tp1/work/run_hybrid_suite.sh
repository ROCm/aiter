#!/bin/bash
set -euo pipefail
trap 'echo $? > /work/hybrid-suite.status' EXIT
for i in $(seq 1 180); do
 if curl -sf --max-time 2 http://localhost:8000/health >/dev/null; then break; fi
 sleep 10
done
bash /work/run_bench.sh hybrid 16 1
bash /work/run_bench.sh hybrid 64 1
for c in 1 8 16 32 64 128; do bash /work/run_bench.sh hybrid "$c" 0; done
python /work/save_completions.py hybrid
