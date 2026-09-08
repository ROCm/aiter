#!/bin/bash
# Reliable gemm1/gemm2 timing: many iters so the harness IQR filter cannot
# collapse the sample set, and only trust rows whose surviving count is high.
AITER_USE_GROUPED_GEMM=1 AITER_GROUPED_DEBUG=0 ENABLE_CK=0 FLYDSL_DUMP_IR=${DUMP:-0} \
AITER_LOG_MORE=1 AITER_MOE_EXPERT_BALANCE=true \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
python3 -u op_tests/test_flydsl_grouped_gemm_gfx1250.py \
  --scenario kernel --data-format a4w4 --experts 96 --tokens 16384 --topk 6 \
  --model-dim 7168 --inter-dim 3072 --act silu --no-bias --iters ${ITERS:-128} \
  --no-check-aot-cache ${EXTRA} --const-init 0 2>&1 \
| grep -E "^0 +a4w4_quad" | tr -s ' ' | awk '{printf "%-72s cnt=%s us=%s\n", $2, $3, $6}'
