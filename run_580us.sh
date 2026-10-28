#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

export AITER_USE_GROUPED_GEMM=1
export AITER_GROUPED_DEBUG=0
export ENABLE_CK=0
export FLYDSL_DUMP_IR=0
export AITER_LOG_MORE=1
export AITER_MOE_EXPERT_BALANCE=true

# The ~580 us GEMM1 path disables the fused GEMM1 requant stage.
export AITER_FLYDSL_DISABLE_GEMM1_REQUANT=1
export AITER_FLYDSL_MMA_FIRST_GROUP=4
export AITER_FLYDSL_MMA_GROUP=5
export AITER_FLYDSL_DS_FIRST_N=1
export AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=1
export AITER_FLYDSL_PLANAR_LDS=1
export AITER_FLYDSL_INTERLEAVED_LDS_LOAD=1
export AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1

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
  --data-init zero \
  --scale-init zero
