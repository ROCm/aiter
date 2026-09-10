  AITER_USE_GROUPED_GEMM=1 \
  AITER_GROUPED_DEBUG=0 \
  ENABLE_CK=0 \
  FLYDSL_DUMP_IR=1 \
  AITER_LOG_MORE=1 \
  AITER_MOE_EXPERT_BALANCE=true \
  AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
  AITER_FLYDSL_MMA_FIRST_GROUP=${AITER_FLYDSL_MMA_FIRST_GROUP:-10} \
  AITER_FLYDSL_MMA_GROUP=${AITER_FLYDSL_MMA_GROUP:-10} \
  AITER_FLYDSL_DS_FIRST_N=${AITER_FLYDSL_DS_FIRST_N:-1} \
  AITER_FLYDSL_WMMA_COLUMN_MAJOR=${AITER_FLYDSL_WMMA_COLUMN_MAJOR:-1} \
  AITER_FLYDSL_SCALE_LO256=${AITER_FLYDSL_SCALE_LO256:-2} \
  python3 -u op_tests/test_flydsl_grouped_gemm_gfx1250.py \
    --scenario kernel \
    --data-format a4w4 \
    --experts 96 \
    --tokens 16384 \
    --topk 6 \
    --model-dim 7168 \
    --inter-dim 3072 \
    --act silu \
    --no-bias \
    --iters 32 \
    --no-check-aot-cache \
    --const-init 0
