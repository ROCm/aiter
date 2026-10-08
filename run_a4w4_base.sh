  AITER_USE_GROUPED_GEMM=1 \
  AITER_GROUPED_DEBUG=0 \
  ENABLE_CK=0 \
  FLYDSL_DUMP_IR=1 \
  AITER_FLYDSL_DISABLE_GEMM1_REQUANT=${AITER_FLYDSL_DISABLE_GEMM1_REQUANT:-0} \
  AITER_LOG_MORE=1 \
  AITER_MOE_EXPERT_BALANCE=true \
  python3 -u op_tests/flydsl_tests/test_flydsl_grouped_gemm.py \
    --scenario bench \
    --data-format a4w4 \
    --experts 64 \
    --tokens 1536 \
    --topk 8 \
    --model-dim 7168 \
    --inter-dim 2048 \
    --act silu \
    --no-bias \
    --iters 128 \
    --no-check-aot-cache \
    --data-init ${AITER_DATA_INIT:-zero} \
    --scale-init ${AITER_SCALE_INIT:-zero}
