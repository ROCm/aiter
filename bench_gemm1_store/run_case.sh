#!/bin/bash
# One measurement process. Identical to run_a4w4_v1.sh except --model-dim 7168
# (no --real-k) and the AITER_TDM_NO_STORE switch.
#
#   NO_STORE=0  baseline
#   NO_STORE=1  epilogue global stores fall fully OOB -- output is WRONG on
#               purpose (logits_diff = 1.0); measurement only.
#
# Prints the gemm1 trace row. device_time_avg is the 2nd-to-last number on it.
# The trace prints BEFORE the sanity assert, so the timing survives NO_STORE=1.
set -u
NO_STORE=${NO_STORE:-0}
cd "$(dirname "$0")/.."
AITER_USE_GROUPED_GEMM=1 \
AITER_GROUPED_DEBUG=0 \
ENABLE_CK=0 \
AITER_LOG_MORE=1 \
AITER_MOE_EXPERT_BALANCE=true \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
AITER_TDM_NO_STORE="$NO_STORE" \
python3 -u op_tests/flydsl_tests/test_flydsl_grouped_gemm.py \
  --scenario kernel --data-format a4w4 \
  --experts 96 --tokens 512 --topk 6 \
  --model-dim 7168 --inter-dim 3072 --act silu \
  --data-init constant --scale-init constant \
  --iters 16 --no-bias --no-check-aot-cache 2>&1 \
 | grep -E "^[0-9]+ +a8w4_tdm_fp4_t64x256x256_w1x4_b3_K7168" \
 | sed "s/^/store$([ "$NO_STORE" = 1 ] && echo OFF || echo ON) /"
