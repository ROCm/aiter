#!/bin/bash

source $GFX1250_MODEL_PATH/ffmlite_env.sh
export USE_FFM=1
export AITER_LOG_MORE=1
DUMP_DIR=dumps
mkdir -p "${DUMP_DIR}"
export FLYDSL_DEBUG_DUMP_ASM=1
export FLYDSL_DUMP_IR=1
export FLYDSL_DUMP_DIR="${DUMP_DIR}"
export FLYDSL_REBUILD_KERNELS=1
export HSA_MODEL_NUM_THREADS=8
# - /workspace/ffm/triton/python:  triton isn't pip-installed in the venv;
#   without it, aiter/ops/quant.py's `import triton` fails -> aiter silently
#   disables HIP ops -> `from aiter import get_hip_quant` ImportError.
# - /workspace/ffm/FlyDSL:         test_moe_2stage._gfx1250_fp8_round_trip_bf16
#   imports tests.kernels.utils.fp4_utils from the FlyDSL repo.
export PYTHONPATH=/workspace/ffm/triton/python:/workspace/ffm/FlyDSL:${PYTHONPATH}
# tile_k=128 default doesn't satisfy the perfect-tile contract
# (tile_m * scale_k_per_tile % block_threads == 0) at this small
# config, so bump to 256 (16 * 8 == 128 OK, no extra block_threads).
export AITER_GFX1250_STAGE1_TILE_K=256
export AITER_GFX1250_STAGE2_TILE_K=256
# Note: topk (-k) must be <= number of experts (-e), otherwise topk_softmax errors.
# ENABLE_CK=0 AITER_USE_OPUS_MOE_SORTING=1 python -m pdb op_tests/test_moe_2stage.py -t 65536 -dim 256,256 -e 1 -k 1 -q 4 --no-flydsl-csv -hip 0,0
ENABLE_CK=0 AITER_USE_OPUS_MOE_SORTING=1 python op_tests/test_moe_2stage.py -t 4 -dim 256,256 -e 2 -k 2 -q 4 --no-flydsl-csv -hip 0,0
