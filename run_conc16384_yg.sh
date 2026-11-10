export AITER_GROUPED_A4W4_A_PRESHUFFLE=1
export AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=1
export AITER_CONFIG_GROUPED_FMOE=$PWD/aiter/configs/tuned_grouped_fmoe.csv
export AITER_FLYDSL_DISABLE_GEMM1_REQUANT=1
export AITER_FLYDSL_DISABLE_GEMM1_ACT=1

PYTHONPATH=$PWD \
AITER_USE_GROUPED_GEMM=1 \
AITER_GROUPED_DEBUG=0 \
ENABLE_CK=0 \
FLYDSL_DUMP_IR=1 \
AITER_LOG_MORE=0 \
AITER_FORCE_GFX1250=1 \
GPU_ARCHS=gfx1250 \
AITER_MOE_EXPERT_BALANCE=true \
CU_NUM=256 \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
python3 -u op_tests/flydsl_tests/test_flydsl_grouped_gemm.py \
  --scenario kernel \
  --data-format a4w4 \
  --experts 96 \
  --tokens 16384 \
  --topk 6 \
  --model-dim 7168 \
  --inter-dim 3072 \
  --act silu \
  --no-bias \
  --no-check-aot-cache \
  --warmup 5 \
  --iters 101 \
  --seed 0 
