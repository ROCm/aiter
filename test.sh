AITER_LOG_MORE=1 AITER_MOE_EXPERT_BALANCE=true AITER_BF16_FP8_BOUND=-1 HIP_VISIBLE_DEVICES=2 python3 op_tests/test_moe_2stage.py \
  -q 7 -d bf16 -dim 7168,256 -e 384 -k 8 \
  -t 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 -hip 0,0 > a8w4.log 2>&1
exit 

AITER_LOG_MORE=1 AITER_MOE_EXPERT_BALANCE=true AITER_BF16_FP8_BOUND=999999 HIP_VISIBLE_DEVICES=7 python3 op_tests/test_moe_2stage.py \
  -q 6 -d bf16 -dim 7168,256 -e 384 -k 8 \
  -t 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 -hip 0,0 > a16w4.log 2>&1

AITER_LOG_MORE=1 AITER_MOE_EXPERT_BALANCE=true AITER_BF16_FP8_BOUND=-1 HIP_VISIBLE_DEVICES=7 python3 op_tests/test_moe_2stage.py \
  -q 8 -d bf16 -dim 7168,256 -e 384 -k 8 \
  -t 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 -hip 0,0 > a16wi4.log 2>&1
