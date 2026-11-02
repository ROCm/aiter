#!/bin/bash
set -euo pipefail
mode=${1:?mode}
trap 'echo $? > /work/round2/'"$mode"'.status' EXIT
for i in $(seq 1 180); do
 if curl -sf --max-time 2 http://localhost:8000/health >/dev/null; then break; fi
 sleep 10
done
for c in 64 128; do bash /work/run_bench.sh "$mode" "$c" 1; done
for c in 64 128; do
 bash /work/run_bench.sh "${mode}_warm" "$c" 0
 bash /work/run_bench.sh "$mode" "$c" 0
done
python /work/round2/save_batch_completions.py "$mode"
if [ "$mode" = round2_control ]; then
 for trial in 2 3; do bash /work/run_bench.sh "round2_control_repeat${trial}" 128 0; done
fi
