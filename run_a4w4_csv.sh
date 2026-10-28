  AITER_USE_GROUPED_GEMM=1 \
  AITER_GROUPED_DEBUG=0 \
  ENABLE_CK=0 \
  FLYDSL_DUMP_IR=0 \
  AITER_LOG_MORE=1 \
  AITER_MOE_EXPERT_BALANCE=true \
  python3 -u op_tests/flydsl_tests/test_flydsl_grouped_gemm.py \
    --scenario csv \
    --data-format a4w4 \
    --experts 96 \
    --tokens 512 \
    --topk 6 \
    --model-dim 7168 \
    --inter-dim 3072 \
    --act silu \
    --no-bias \
    --iters 64 \
    --no-check-aot-cache
