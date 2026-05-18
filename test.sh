source $GFX1250_MODEL_PATH/ffmlite_env.sh
export AITER_LOG_MORE=1
export USE_FFM=1
# - /workspace/ffm/triton/python:  triton isn't pip-installed in the venv;
#   without it, aiter/ops/quant.py's `import triton` fails -> aiter silently
#   disables HIP ops -> `from aiter import get_hip_quant` ImportError.
# - /workspace/ffm/FlyDSL:         test_moe_2stage._gfx1250_fp8_round_trip_bf16
#   imports tests.kernels.utils.fp4_utils from the FlyDSL repo.
export PYTHONPATH=/workspace/ffm/triton/python:/workspace/ffm/FlyDSL:${PYTHONPATH}
export AITER_USE_OPUS_MOE_SORTING=1;export ENABLE_CK=0;
# Note: topk (-k) must be <= number of experts (-e), otherwise topk_softmax errors.
# ENABLE_CK=0 AITER_USE_OPUS_MOE_SORTING=0 python op_tests/test_moe_2stage.py -t 1 -dim 256,256 -e 1 -k 1 -q 4 --no-flydsl-csv -hip 0,0
python op_tests/test_moe_2stage.py -t 64 -dim 256,128 -e 4 -k 2 -q 0 --no-flydsl-csv
python op_tests/test_moe_2stage.py -t 64 -dim 256,128 -e 4 -k 2 -q 4 --no-flydsl-csv
python op_tests/test_moe_2stage.py -t 64 -dim 256,128 -e 4 -k 2 -q 7 --no-flydsl-csv
python op_tests/test_moe_2stage.py -t 512 -dim 3072,3072 -e 128 -k 4 -q 7 --no-flydsl-csv
python op_tests/test_moe_2stage.py -t 16384 -dim 7168,256 -e 257 -k 9 -q 4 --no-flydsl-csv --num_buffers 2

# Explicitly pass the historical hardcoded gfx1250 FlyDSL tile config
# (-tn / -tk / -bm = stage1; --stage2-tile-n / --stage2-tile-k = stage2)
# so subsequent sweeps can tweak any of them without falling back to
# the silent default in aiter/fused_moe.py::get_2stage_cfgs.
# AITER_GFX1250_NUM_BUFFERS sets the FlyDSL K-pipeline depth shared by
# stage1 and stage2 (1..4; default 1 = no pipelining). Both stages must
# be able to fit it (num_buffers <= K_padded // tile_k); the kernel
# raises otherwise.
AITER_GFX1250_NUM_BUFFERS=2 AITER_LOG_MORE=1 ENABLE_CK=0 AITER_USE_OPUS_MOE_SORTING=1 python op_tests/test_moe_2stage.py \
  -t 1 -dim 7168,2048 -e 256 -k 8 -q 4 --no-flydsl-csv -hip 0,0 \
  -tn 128 -tk 512 -bm 16 --stage2-tile-n 256 --stage2-tile-k 256

AITER_GFX1250_NUM_BUFFERS=2 AITER_LOG_MORE=1 ENABLE_CK=0 AITER_USE_OPUS_MOE_SORTING=1 python op_tests/test_moe_2stage.py \
  -t 64 -dim 7168,2048 -e 256 -k 8 -q 4 --no-flydsl-csv -hip 0,0 \
  -tn 128 -tk 512 -bm 16 --stage2-tile-n 256 --stage2-tile-k 256

# GPT-OSS 120B TP1 (a8w4 / MXFP8 x MXFP4, SwiGLU)
#   input : [M, 2880] bf16
#   w1    : [128, 2*2880, 2880]   (gate||up packed, MXFP8 act x MXFP4 wt)
#   w2    : [128, 2880,   2880]
#   topk=4 -- ``-q 7`` (per_1x32 fp8/fp4x2) is force-routed to
#   ActivationType.Swiglu by ``_effective_act_type``; passing
#   ``-a swiglu`` makes the intent explicit.
# Accuracy note: with ``-hip 0,0`` (no K-padding on w1/w2) the FlyDSL
# a8w4 verdict will fail at K=2880 (per-1x32 quant noise ~sqrt(K) blows
# past mismatch_ratio=0.05 / logits_diff=0.5). Perf numbers are still
# valid; drop the ``-hip 0,0`` flag to let the script auto-pick
# ``-hip 768,768`` via ``_gfx1250_a8w4_default_kpad`` if a PASS verdict
# is required.

AITER_MOE_SKIP_REF=0 AITER_GFX1250_NUM_BUFFERS=2 AITER_LOG_MORE=1 ENABLE_CK=0 AITER_USE_OPUS_MOE_SORTING=1 python op_tests/test_moe_2stage.py \
  -t 1 -dim 2880,2880 -e 128 -k 4 -q 7 -a swiglu --no-flydsl-csv -hip 0,0 \
  -tn 128 -tk 256 -bm 16 --stage2-tile-n 128 --stage2-tile-k 256

AITER_MOE_SKIP_REF=0 AITER_GFX1250_NUM_BUFFERS=2 AITER_LOG_MORE=1 ENABLE_CK=0 AITER_USE_OPUS_MOE_SORTING=1 python op_tests/test_moe_2stage.py \
  -t 64 -dim 2880,2880 -e 128 -k 4 -q 7 -a swiglu --no-flydsl-csv -hip 0,0 \
  -tn 128 -tk 512 -bm 16 --stage2-tile-n 256 --stage2-tile-k 256

# GPT-OSS BS=1 / BS=64 with physical K-tile alignment padding
# (AITER_K_TILE_ALIGN_PAD=1): rounds model_dim / inter_dim up to the
# next multiple of stage1 / stage2 ``tile_k`` so there is no partial
# K-tile.  At ``-dim 2880,2880 -tk 256 --stage2-tile-k 256``,
# 2880 -> 3072 (+192) per stage, eliminating the trailing 25%-utilised
# K-tile.  Padded K-columns are zero-valued so kernel output on the
# original ``[:, :2880]`` slice is bit-identical to the unpadded run.
AITER_K_TILE_ALIGN_PAD=1 AITER_MOE_SKIP_REF=0 AITER_GFX1250_NUM_BUFFERS=2 AITER_LOG_MORE=1 ENABLE_CK=0 AITER_USE_OPUS_MOE_SORTING=1 python op_tests/test_moe_2stage.py \
  -t 1 -dim 2880,2880 -e 128 -k 4 -q 7 -a swiglu --no-flydsl-csv -hip 0,0 \
  -tn 256 -tk 512 -bm 16 --stage2-tile-n 256 --stage2-tile-k 256

AITER_K_TILE_ALIGN_PAD=1 AITER_MOE_SKIP_REF=0 AITER_GFX1250_NUM_BUFFERS=2 AITER_LOG_MORE=1 ENABLE_CK=0 AITER_USE_OPUS_MOE_SORTING=1 python op_tests/test_moe_2stage.py \
  -t 64 -dim 2880,2880 -e 128 -k 4 -q 7 -a swiglu --no-flydsl-csv -hip 0,0 \
  -tn 256 -tk 512 -bm 16 --stage2-tile-n 256 --stage2-tile-k 256

AITER_GFX1250_NUM_BUFFERS=2 AITER_LOG_MORE=1 ENABLE_CK=0 AITER_USE_OPUS_MOE_SORTING=1 python op_tests/test_moe_2stage.py \
 -t 64 -dim 3072,3072 -e 128 -k 4 -q 7 --no-flydsl-csv -hip 0,0 \
  -tn 128 -tk 512 -bm 16 --stage2-tile-n 256 --stage2-tile-k 256

AITER_GFX1250_PROBE=1 AITER_GFX1250_NUM_BUFFERS=2 AITER_LOG_MORE=1 ENABLE_CK=0 AITER_USE_OPUS_MOE_SORTING=1 python op_tests/test_moe_2stage.py \
  -t 64 -dim 2880,2880 -e 64 -k 4 -q 7 -a swiglu --no-flydsl-csv -hip 0,0 \
  -tn 128 -tk 512 -bm 16 --stage2-tile-n 128 --stage2-tile-k 128

# ENABLE_CK=0 AITER_USE_OPUS_MOE_SORTING=0 python op_tests/test_moe_2stage.py -t 1 -dim 256,1024 -e 8 -k 2 -q 4 --no-flydsl-csv -hip 0,0
