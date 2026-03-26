
export FLIR_PATH=/workspace/FlyDSL
export AITER_USE_FLYDSL=1
# Env vars used by FLIR/kernels/moe_gemm_2stage.py (a8w4smooth path)
export FLIR_A8W4SMOOTH_QPARAM_FORMAT=packed4
export FLIR_A8W4SMOOTH_INTERLEAVE_K64=1
export FLIR_A8W4SMOOTH_OVERFLOW_GUARD=0
export FLIR_MOE_STAGE1_CSHUFFLE=0
export FLIR_MOE_STAGE2_CSHUFFLE=1
export AITER_LOG_MORE=1
export FLIR_DUMP_IR=1
export FLIR_DUMP_DIR=dumps

# Env vars used by aiter/ops/moe_flydsl.py (tile override)
# export AITER_FLYDSL_MOE_TILE_M=64
export AITER_FLYDSL_MOE_TILE_N1=64
export AITER_FLYDSL_MOE_TILE_K1=128
export AITER_FLYDSL_MOE_TILE_N2=128
export AITER_FLYDSL_MOE_TILE_K2=128

rm -fr ~/.cache/flydsl
export HIP_VISIBLE_DEVICES=5

# DeepSeek-V3: E=128, topk=6, mdim=5120, idim=1536, shared_E=2, ep=8
# python op_tests/test_moe_lqq.py -t 32 64 128 256 512 1024 -e 128 -k 6 -md 5120 -id 1536 -ep 8 -se 2 -x 32

# python op_tests/test_moe_lqq.py -t 256 512 1024 1664 -e 128 -k 6 -md 5120 -id 1536 -ep 8 -se 2 -x 32 64 80 --evenly
# python op_tests/test_moe_lqq.py -t 1664 -e 128 -k 6 -md 5120 -id 1536 -ep 8 -se 2 -x 32 64 80 --evenly
AITER_FLYDSL_GEMM2_MODE=REDUCE AITER_FLYDSL_GEMM2_VALID_MASK=1 python op_tests/test_moe_lqq.py -t 1664 -e 128 -k 6 -md 5120 -id 1536 -ep 8 -se 2 -x 32 64 80 --evenly
# AITER_FLYDSL_GEMM2_MODE=REDUCE AITER_FLYDSL_GEMM2_VALID_MASK=1 python op_tests/test_moe_lqq.py -t 256 512 1024 1664 -e 128 -k 6 -md 5120 -id 1536 -ep 8 -se 2 -x 32 64 80 --evenly
# AITER_FLYDSL_GEMM2_MODE=REDUCE AITER_FLYDSL_GEMM2_VALID_MASK=1 python op_tests/test_moe_lqq.py -t 1664 -e 128 -k 6 -md 5120 -id 1536 -ep 8 -se 2 -x 32 64 80 96 --evenly
# python op_tests/test_moe_lqq.py -t 440 -e 128 -k 6 -md 5120 -id 1536 -ep 8 -se 2 -x 64


# python op_tests/test_moe_lqq.py -t 256 -e 384 -k 8 -md 3584 -id 1024 -ep 8 -se 0 -x 32 64 80 --evenly
# AITER_FLYDSL_GEMM2_MODE=REDUCE  python op_tests/test_moe_lqq.py -t 256 512 1024 1664 2048 -e 384 -k 8 -md 3584 -id 1024 -ep 8 -se 0 -x 32 64 80 --evenly
# python op_tests/test_moe_lqq.py -t 256 512 1024 1664 2048 -e 384 -k 8 -md 3584 -id 1024 -ep 8 -se 0 -x 32 64 80 --evenly
# AITER_FLYDSL_GEMM2_MODE=REDUCE AITER_FLYDSL_GEMM2_VALID_MASK=1 python op_tests/test_moe_lqq.py -t 256 512 1024 1664 2048 -e 384 -k 8 -md 3584 -id 1024 -ep 8 -se 0 -x 32 64 80 --evenly
# AITER_FLYDSL_GEMM2_MODE=ATOMIC  python op_tests/test_moe_lqq.py -t 256 -e 384 -k 8 -md 3584 -id 1024 -ep 8 -se 0 -x 32 --evenly
# AITER_FLYDSL_GEMM2_MODE=REDUCE python op_tests/test_moe_lqq.py -t 1024 -e 384 -k 8 -md 3584 -id 1024 -ep 8 -se 0 -x 32 --evenly
# AITER_FLYDSL_GEMM2_MODE=REDUCE python op_tests/test_moe_lqq.py -t 32 -e 384 -k 8 -md 3584 -id 1024 -ep 1 -se 0 -x 32 --evenly
# AITER_FLYDSL_GEMM2_MODE=REDUCE AITER_FLYDSL_GEMM2_VALID_MASK=0 python op_tests/test_moe_lqq.py -t 1024 -e 384 -k 8 -md 3584 -id 1024 -ep 8 -se 0 -x 32 --evenly
