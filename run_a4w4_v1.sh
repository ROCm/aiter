  AITER_USE_GROUPED_GEMM=1 \
  AITER_GROUPED_DEBUG=0 \
  ENABLE_CK=0 \
  FLYDSL_DUMP_IR=1 \
  AITER_LOG_MORE=1 \
  AITER_MOE_EXPERT_BALANCE=true \
  AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
  python3 -u op_tests/flydsl_tests/test_flydsl_grouped_gemm.py \
    --scenario kernel \
    --data-format a4w4 \
    --experts 96 \
    --tokens 512 \
    --topk 6 \
    --model-dim 8192 \
    --real-k 7168 \
    --inter-dim 3072 \
    --act silu \
    --data-init constant \
    --scale-init constant \
    --iters 16 \
    --no-bias \
    --no-check-aot-cache
