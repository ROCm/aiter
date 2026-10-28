#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

export AITER_USE_GROUPED_GEMM=1
export AITER_GROUPED_DEBUG=0
export ENABLE_CK=0
export FLYDSL_DUMP_IR=1
export AITER_LOG_MORE=1
export AITER_MOE_EXPERT_BALANCE=true

# Workload-specific configuration: GEMM1 WPT2, GEMM2 WPT1, rolled drain off.
export AITER_CONFIG_GROUPED_FMOE="$script_dir/aiter/configs/benchmarks/a4w4_conc16384.csv"
# Validated for this balanced workload, not a general cluster-sync default.
export AITER_FLYDSL_TDM_EARLY_TIMEOUT=0
export AITER_FLYDSL_DISABLE_CLUSTER_SYNC=1

# Keep A-preshuffled inputs, with GEMM1 producing a BF16 intermediate and
# a separate quant/preshuffle kernel preparing GEMM2's input.
export AITER_GROUPED_A4W4_A_PRESHUFFLE=1
export AITER_FLYDSL_DISABLE_GEMM1_REQUANT=1
# Workload-specific IGLP defaults; allow overrides for comparison runs.
export AITER_FLYDSL_MMA_FIRST_GROUP="${AITER_FLYDSL_MMA_FIRST_GROUP:-4}"
export AITER_FLYDSL_MMA_GROUP="${AITER_FLYDSL_MMA_GROUP:-4}"
export AITER_FLYDSL_DS_FIRST_N="${AITER_FLYDSL_DS_FIRST_N:-8}"
export AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=1
# export AITER_FLYDSL_PLANAR_LDS=1
# export AITER_FLYDSL_INTERLEAVED_LDS_LOAD=1
export AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1
export AITER_FLYDSL_DISABLE_GEMM1_ACT=1

exec python3 -u op_tests/flydsl_tests/test_flydsl_grouped_gemm.py \
  --scenario kernel \
  --data-format a4w4 \
  --experts 96 \
  --tokens 16384 \
  --topk 6 \
  --model-dim 7168 \
  --inter-dim 3072 \
  --act silu \
  --no-bias \
  --iters 64 \
  --no-check-aot-cache \
  "$@"
