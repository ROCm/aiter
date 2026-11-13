#!/usr/bin/env bash
set -euo pipefail

CX=/root/workspace/workflow_34/moonshot_ep32_moe_v2/cx
CC=/root/workspace/workflow_34/moonshot_ep32_moe_v2/cc
BENCH=/root/workspace/workflow_34/moonshot_ep32_moe_v2/amd_moe_bench_share/bench_moe.py
PY=/tmp/aiter_ep32_branch_review_20261112/bin/python
GPU_HIP=${1:-5}
O=$CX/results/ep32_m4096_$(date -u +%Y%m%d_%H%M%S)_hip$GPU_HIP

source "$CC/opt_gemm1/env.sh"
export WT=$CX/aiter
export PYTHONPATH=$WT AITER_META_DIR=$WT AITER_JIT_DIR=$WT/aiter/jit
export HIP_VISIBLE_DEVICES=$GPU_HIP
export EP32_G2_ROUTE_MERGE=0
export EP32_G2_BPACK=0
export EP32_G2_BNT=0
export FLYDSL_RUNTIME_CACHE_DIR=$CC/wt_cache/flydsl050
export FLYDSL_EXTRA_SOURCE_DIRS=$WT/aiter/ops/flydsl
export AITER_CONFIG_FMOE=$WT/aiter/configs/model_configs/kimik3_a8w4_ep32_fmoe_mxmoe_g1g2ep32.csv

cd "$CX"
mkdir -p "$O"
git -C "$WT" rev-parse HEAD > "$O/head.txt"
sha256sum "$BENCH" "$AITER_CONFIG_FMOE" \
    "$WT/aiter/ops/flydsl/kernels/mxfp4_gemm1_ep32.py" \
    "$WT/aiter/ops/flydsl/kernels/mxmoe_gemm2_ep32.py" \
    "$WT/aiter/ops/flydsl/kernels/mxmoe_gemm2_ep32_bm80var.py" \
    "$WT/aiter/ops/flydsl/kernels/mxmoe_gemm2_ep32_bm80base.py" \
    "$WT/aiter/ops/flydsl/kernels/mxmoe_reduce_ep32.py" \
    "$WT/aiter/ops/flydsl/ep32_native80.py" \
    "$WT/aiter/ops/flydsl/mxmoe_native_reduce.py" \
    "$WT/aiter/fused_moe.py" \
    > "$O/source_sha256.txt"
printf '%s\n' "$O" > "$CX/results/latest_baseline.txt"

bash "$CX/gpu_idle_for_bench.sh" "$GPU_HIP" 2>&1 | tee "$O/gpu_idle.log"
"$PY" -B -c 'import flydsl; print(flydsl.__version__, flydsl.__file__)' \
    > "$O/flydsl_version.txt"
rocprofv3 --kernel-trace --marker-trace -f csv \
    -o trace -d "$O/trace" -- \
    "$PY" -B "$BENCH" run --output-dir "$O/bench" \
    --batch-sizes 4096 --ep 32 --rank 0 \
    > "$O/run.log" 2>&1
"$PY" -B "$BENCH" parse --output-dir "$O/bench" --trace-dir "$O/trace" \
    > "$O/parse.log" 2>&1
cat "$O/bench/metrics.csv"
printf 'Results: %s\n' "$O"
