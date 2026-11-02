#!/bin/bash
set -euo pipefail
trap 'echo $? > /work/quant-profile.status' EXIT
for i in $(seq 1 180); do
 if curl -sf --max-time 2 http://localhost:8000/health >/dev/null; then break; fi
 sleep 10
done
bash /work/run_bench.sh hybrid_quant 16 1
bash /work/run_bench.sh hybrid_quant 64 1
python /work/save_completions.py hybrid_quant
