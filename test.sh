source $GFX1250_MODEL_PATH/ffmlite_env.sh
export AITER_LOG_MORE=1
# - /workspace/ffm/triton/python:  triton isn't pip-installed in the venv;
#   without it, aiter/ops/quant.py's `import triton` fails -> aiter silently
#   disables HIP ops -> `from aiter import get_hip_quant` ImportError.
# - /workspace/ffm/FlyDSL:         test_moe_2stage._gfx1250_fp8_round_trip_bf16
#   imports tests.kernels.utils.fp4_utils from the FlyDSL repo.
export PYTHONPATH=/workspace/ffm/triton/python:/workspace/ffm/FlyDSL:${PYTHONPATH}
# Note: topk (-k) must be <= number of experts (-e), otherwise topk_softmax errors.
# ENABLE_CK=0 AITER_USE_OPUS_MOE_SORTING=0 python op_tests/test_moe_2stage.py -t 1 -dim 256,256 -e 1 -k 1 -q 4 --no-flydsl-csv -hip 0,0

# ENABLE_CK=0 AITER_USE_OPUS_MOE_SORTING=0 python op_tests/test_moe_2stage.py -t 1 -dim 256,256 -e 2 -k 2 -q 4 --no-flydsl-csv -hip 0,0

ENABLE_CK=0 AITER_USE_OPUS_MOE_SORTING=0 python op_tests/test_moe_2stage.py -t 1 -dim 256,1024 -e 8 -k 2 -q 4 --no-flydsl-csv -hip 0,0
