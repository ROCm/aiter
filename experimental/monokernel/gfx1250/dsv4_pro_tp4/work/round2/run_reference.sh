#!/bin/bash
set -euo pipefail
trap 'echo $? > /work/round2/reference.status' EXIT
for i in $(seq 1 180); do
 if curl -sf --max-time 2 http://localhost:8000/health >/dev/null; then break; fi
 sleep 10
done
bash /work/run_bench.sh round2_v1 128 1
for c in 64 128; do bash /work/run_bench.sh round2_v1 "$c" 0; done
