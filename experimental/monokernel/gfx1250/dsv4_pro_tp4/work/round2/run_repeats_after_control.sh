#!/bin/bash
set -euo pipefail
root=/home/vigokhal/dsv4-monokernel
while [ ! -f "$root/round2/round2_control.status" ]; do sleep 10; done
[ "$(cat "$root/round2/round2_control.status")" = 0 ]
sleep 3
docker stop -t 3 vigokhal_dsv4_tp4
docker start vigokhal_dsv4_tp4
docker exec -d -w /work -e ATOM_DSV4_HYBRID=1 -e ATOM_DSV4_HYBRID_GEMM=1 -e ATOM_DSV4_HYBRID_ATTN=1 -e ATOM_DSV4_HYBRID_CHECK=1 -e ATOM_FORCE_ATTN_TRITON=1 -e FAKE_EPLB=0 -e TRACE=1 -e TRACE_DIR=/work/traces/round2_candidate_repeat -e TRITON_CACHE_DIR=/work/triton-cache vigokhal_dsv4_tp4 bash -c 'bash /work/serve.sh > /work/round2/candidate-repeat-launch.log 2>&1'
docker exec -w /work vigokhal_dsv4_tp4 python /work/round2/analyze_serving.py round2_control > "$root/round2/control-analysis.log" 2>&1
docker exec -w /work vigokhal_dsv4_tp4 python /work/round2/summarize_serving.py > "$root/round2/initial-comparison.log" 2>&1
docker exec -w /work vigokhal_dsv4_tp4 bash -c 'bash /work/round2/run_candidate_repeats.sh > /work/round2/candidate-repeat-suite.log 2>&1'
