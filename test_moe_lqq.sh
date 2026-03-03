
export FLIR_PATH=/workspace/FLIR
export AITER_USE_FLYDSL=1
# Env vars used by FLIR/kernels/moe_gemm_2stage.py (a8w4smooth path)
export FLIR_A8W4SMOOTH_QPARAM_FORMAT=packed4
export FLIR_A8W4SMOOTH_INTERLEAVE_K64=1
export FLIR_A8W4SMOOTH_OVERFLOW_GUARD=0
export FLIR_MOE_STAGE1_CSHUFFLE=0
export FLIR_MOE_STAGE2_CSHUFFLE=1
export HIP_VISIBLE_DEVICES=6
export AITER_LOG_MORE=1
export FLIR_DUMP_IR=1
export FLIR_DUMP_DIR=dumps

rm -fr ~/.cache/flydsl

# DeepSeek-V3: E=128, topk=6, mdim=5120, idim=1536, shared_E=2, ep=8
python /workspace/aiter_dev/op_tests/test_moe_lqq.py -t 32 64 128 -e 128 -k 6 -md 5120 -id 1536 -ep 8 -se 2
