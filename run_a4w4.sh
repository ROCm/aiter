# Input initialization defaults to zero for stable performance measurements.
# Run a nonzero randomized correctness check with:
#   AITER_DATA_INIT=uniform AITER_SCALE_INIT=auto bash run_a4w4.sh
  AITER_USE_GROUPED_GEMM=1 \
  AITER_GROUPED_DEBUG=0 \
  ENABLE_CK=0 \
  FLYDSL_DUMP_IR=1 \
  AITER_FLYDSL_DISABLE_GEMM1_REQUANT=${AITER_FLYDSL_DISABLE_GEMM1_REQUANT:-0} \
  AITER_LOG_MORE=1 \
  AITER_MOE_EXPERT_BALANCE=true \
  AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
  AITER_FLYDSL_MMA_FIRST_GROUP=${AITER_FLYDSL_MMA_FIRST_GROUP:-4} \
  AITER_FLYDSL_MMA_GROUP=${AITER_FLYDSL_MMA_GROUP:-5} \
  AITER_FLYDSL_DS_FIRST_N=${AITER_FLYDSL_DS_FIRST_N:-1} \
  AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=${AITER_FLYDSL_EXPLICIT_VGPR_PARTITION:-1} \
  AITER_FLYDSL_PLANAR_LDS=${AITER_FLYDSL_PLANAR_LDS:-1} \
  AITER_FLYDSL_INTERLEAVED_LDS_LOAD=${AITER_FLYDSL_INTERLEAVED_LDS_LOAD:-1} \
  python3 -u op_tests/flydsl_tests/test_flydsl_grouped_gemm.py \
    --scenario kernel \
    --data-format a4w4 \
    --experts 64 \
    --tokens 1536 \
    --topk 8 \
    --model-dim 7168 \
    --inter-dim 2048 \
    --act silu \
    --no-bias \
    --iters 32 \
    --no-check-aot-cache \
    --data-init ${AITER_DATA_INIT:-zero} \
    --scale-init ${AITER_SCALE_INIT:-zero}
