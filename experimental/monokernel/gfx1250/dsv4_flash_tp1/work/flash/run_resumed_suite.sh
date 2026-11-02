#!/bin/bash
set -euo pipefail
mode=${1:?arm name}
shift
for i in $(seq 1 120); do
 if curl -sf --max-time 2 http://localhost:8000/health >/dev/null; then break; fi
 sleep 5
done
for c in "${@:-1 8 16 32 64 128}"; do
 for bucket in $c; do
  bash /work/flash/run_bench.sh "${mode}_run1" "$bucket" 0
  bash /work/flash/run_bench.sh "${mode}_run2" "$bucket" 0
 done
done
