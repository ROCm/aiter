source $GFX1250_MODEL_PATH/ffmlite_env.sh
export AITER_LOG_MORE=1
export USE_FFM=1
# - /workspace/ffm/triton/python:  triton isn't pip-installed in the venv;
#   without it, aiter/ops/quant.py's `import triton` fails -> aiter silently
#   disables HIP ops -> `from aiter import get_hip_quant` ImportError.
# - /workspace/ffm/FlyDSL:         test_moe_2stage._gfx1250_fp8_round_trip_bf16
#   imports tests.kernels.utils.fp4_utils from the FlyDSL repo.
export PYTHONPATH=/workspace/ffm/triton/python:/workspace/ffm/FlyDSL:${PYTHONPATH}
# Note: topk (-k) must be <= number of experts (-e), otherwise topk_softmax errors.
# ENABLE_CK=0 AITER_USE_OPUS_MOE_SORTING=0 python op_tests/test_moe_2stage.py -t 1 -dim 256,256 -e 1 -k 1 -q 4 --no-flydsl-csv -hip 0,0

# Explicitly pass the historical hardcoded gfx1250 FlyDSL tile config
# (-tn / -tk / -bm = stage1; --stage2-tile-n / --stage2-tile-k = stage2)
# so subsequent sweeps can tweak any of them without falling back to
# the silent default in aiter/fused_moe.py::get_2stage_cfgs.
# AITER_GFX1250_NUM_BUFFERS sets the FlyDSL K-pipeline depth shared by
# stage1 and stage2 (1..4; default 1 = no pipelining). Both stages must
# be able to fit it (num_buffers <= K_padded // tile_k); the kernel
# raises otherwise.
AITER_GFX1250_NUM_BUFFERS=2 AITER_LOG_MORE=1 ENABLE_CK=0 AITER_USE_OPUS_MOE_SORTING=1 python op_tests/test_moe_2stage.py \
  -t 64 -dim 7168,2048 -e 256 -k 8 -q 4 --no-flydsl-csv -hip 0,0 \
  -tn 128 -tk 512 -bm 16 --stage2-tile-n 128 --stage2-tile-k 128

# ENABLE_CK=0 AITER_USE_OPUS_MOE_SORTING=0 python op_tests/test_moe_2stage.py -t 1 -dim 256,1024 -e 8 -k 2 -q 4 --no-flydsl-csv -hip 0,0
