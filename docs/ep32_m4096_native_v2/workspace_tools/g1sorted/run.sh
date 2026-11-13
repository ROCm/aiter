#!/usr/bin/env bash
set -euo pipefail
CX=/root/workspace/workflow_34/moonshot_ep32_moe_v2/cx
CC=/root/workspace/workflow_34/moonshot_ep32_moe_v2/cc
PY=/tmp/aiter_ep32_branch_review_20261112/bin/python
BENCH=$CX/../amd_moe_bench_share/bench_moe.py
GPU_HIP=${1:-5}
ENTRY=${2:-check}
O=$CX/g1sorted/${ENTRY}_$(date -u +%Y%m%d_%H%M%S)_hip$GPU_HIP
source "$CC/opt_gemm1/env.sh"
export WT=$CX/aiter PYTHONPATH=$CX/aiter AITER_META_DIR=$CX/aiter AITER_JIT_DIR=$CX/aiter/aiter/jit
export HIP_VISIBLE_DEVICES=$GPU_HIP FLYDSL_RUNTIME_CACHE_DIR=$CC/wt_cache/flydsl050
export FLYDSL_EXTRA_SOURCE_DIRS=$WT/aiter/ops/flydsl EP32_G2_NATIVE=1
export AITER_CONFIG_FMOE=$WT/aiter/configs/model_configs/kimik3_a8w4_ep32_fmoe_mxmoe_g1g2ep32.csv
cd "$CX"
mkdir -p "$O"
bash gpu_idle_for_bench.sh "$GPU_HIP" > "$O/gpu_idle.log" 2>&1 || {
    cat "$O/gpu_idle.log"
    exit 1
}
cat "$O/gpu_idle.log"
"$PY" -B g1sorted/prepare.py install > "$O/install.log"
trap '"$PY" -B "$CX/g1sorted/prepare.py" restore' EXIT
sha256sum "$BENCH" "$AITER_CONFIG_FMOE" "$WT/aiter/fused_moe.py" \
    "$WT/aiter/ops/flydsl/kernels/mxfp4_gemm1_ep32.py" \
    "$WT/aiter/ops/flydsl/mxfp4_gemm1_kernels.py" \
    "$WT/aiter/ops/flydsl/ep32_sorted_a.py" \
    "$WT/aiter/ops/flydsl/kernels/ep32_sorted_a.py" \
    "$WT/aiter/ops/flydsl/kernels/mxmoe_gemm2_ep32_bm80var.py" > "$O/source_sha256.txt"
if [ "$ENTRY" = check ]; then
    "$PY" -B g1sorted/check.py "$O" > "$O/check.log" 2>&1
    tail -n 3 "$O/check.log"
elif [ "$ENTRY" = default ]; then
    rocprofv3 --kernel-trace --marker-trace -f csv -o trace -d "$O/trace" -- \
        "$PY" -B "$BENCH" run --output-dir "$O/bench" --batch-sizes 4096 \
        --ep 32 --rank 0 > "$O/run.log" 2>&1
    "$PY" -B "$BENCH" parse --output-dir "$O/bench" --trace-dir "$O/trace" > "$O/parse.log" 2>&1
    cat "$O/bench/metrics.csv"
else
    exit 2
fi
printf '%s\n' "$O" > "$CX/g1sorted/latest_${ENTRY}.txt"
printf 'Results: %s\n' "$O"
