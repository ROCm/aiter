  # AITER_USE_GROUPED_GEMM=1 \
  # AITER_GROUPED_DEBUG=0 \
  # ENABLE_CK=0 \
  # FLYDSL_DUMP_IR=1 \
  # AITER_LOG_MORE=1 \
  # AITER_MOE_EXPERT_BALANCE=true \
  # AITER_TDM_TILE_M=32 \
  # AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
  # python3 -u op_tests/test_flydsl_grouped_gemm_gfx1250.py \
  #   --scenario bench \
  #   --data-format a4w4 \
  #   --experts 96 \
  #   --tokens 512 \
  #   --topk 6 \
  #   --model-dim 7168 \
  #   --inter-dim 3072 \
  #   --act silu \
  #   --no-bias \
  #   --no-check-aot-cache

  AITER_USE_GROUPED_GEMM=1 \
  AITER_GROUPED_DEBUG=0 \
  ENABLE_CK=0 \
  FLYDSL_DUMP_IR=1 \
  AITER_LOG_MORE=1 \
  AITER_MOE_EXPERT_BALANCE=true \
  AITER_TDM_TILE_M=32 \
  AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
  python3 -u op_tests/test_flydsl_grouped_gemm_gfx1250.py \
    --scenario bench \
    --data-format a4w4 \
    --experts 256 \
    --tokens 512 \
    --topk 6 \
    --model-dim 16384 \
    --inter-dim 4096 \
    --act silu \
    --no-bias \
    --no-check-aot-cache
