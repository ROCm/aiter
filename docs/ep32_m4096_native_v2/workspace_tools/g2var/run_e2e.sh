#!/usr/bin/env bash
set -euo pipefail
CX=/root/workspace/workflow_34/moonshot_ep32_moe_v2/cx
CC=/root/workspace/workflow_34/moonshot_ep32_moe_v2/cc
PY=/tmp/aiter_ep32_branch_review_20261112/bin/python
GPU_HIP=${1:-5}
source "$CC/opt_gemm1/env.sh"
export WT=$CX/aiter PYTHONPATH=$CX/aiter AITER_META_DIR=$CX/aiter AITER_JIT_DIR=$CX/aiter/aiter/jit
export HIP_VISIBLE_DEVICES=$GPU_HIP FLYDSL_RUNTIME_CACHE_DIR=$CC/wt_cache/flydsl050
export FLYDSL_EXTRA_SOURCE_DIRS=$WT/aiter/ops/flydsl EP32_G2_NATIVE=1
export AITER_CONFIG_FMOE=$WT/aiter/configs/model_configs/kimik3_a8w4_ep32_fmoe_mxmoe_g1g2ep32.csv
cd "$CX"
bash gpu_idle_for_bench.sh "$GPU_HIP"
"$PY" -B g2var/e2e.py -t g1u1_a8w4_mxfp4 -m 4096 -hd 3584 -id 3072 \
    -e 896 -k 16 -ep 32 -a situv2 --beta 4 --linear-beta 25 > g2var/e2e.log 2>&1
tail -n 24 g2var/e2e.log
