#!/bin/bash
set -euo pipefail
trap 'echo $? > /work/round2/candidate-repeats.status' EXIT
for i in $(seq 1 180); do
 if curl -sf --max-time 2 http://localhost:8000/health >/dev/null; then break; fi
 sleep 10
done
bash /work/run_bench.sh round2_candidate_repeat_warm 128 1
bash /work/run_bench.sh round2_candidate_repeat_warm 128 0
for trial in 2 3; do bash /work/run_bench.sh "round2_candidate_repeat${trial}" 128 0; done
