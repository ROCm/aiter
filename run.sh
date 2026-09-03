#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

HIP_VISIBLE_DEVICES=0 \
AITER_USE_GROUPED_GEMM=1 \
AITER_GROUPED_DEBUG=0 \
ENABLE_CK=0 \
FLYDSL_DUMP_IR=0 \
AITER_LOG_MORE=1 \
AITER_MOE_EXPERT_BALANCE=true \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
AITER_TDM_TILE_M=16 \
AITER_TDM_TILE_N2=512 \
AITER_TDM_TILE_K2=256 \
AITER_TDM_NUM_BUFFERS2=4 \
AITER_TDM_B_TH=1 \
AITER_FLYDSL_NUM_WAVES_PER_TENSOR_TDM=4 \
AITER_FLYDSL_MAX_TASKS_PER_WORKER=7 \
python3 -u "${SCRIPT_DIR}/op_tests/test_flydsl_grouped_gemm_gfx1250.py" \
  --scenario kernel \
  --data-format a4w4 \
  --experts 96 \
  --tokens 128 \
  --topk 6 \
  --model-dim 7168 \
  --inter-dim 2048 \
  --act silu \
  --no-bias \
  --warmup 5 \
  --iters 101 \
  --no-check-aot-cache
