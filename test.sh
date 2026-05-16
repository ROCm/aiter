#!/bin/bash
# DeepSeek V3 TP1 stage1+stage2 perftest (gfx1250, real HW — no FFM)

export AITER_LOG_MORE=1
export AITER_GFX1250_TDM_GATHER=1
# export AITER_GFX1250_CONST_ONE=1
# Dump compiled kernel assembly + IR to $FLYDSL_DUMP_DIR (default ~/.flydsl/debug).
export FLYDSL_DEBUG_DUMP_ASM=1
export FLYDSL_DUMP_IR=1
export FLYDSL_DUMP_DIR=/app/aiter/.flydsl_dump

# Explicit gfx1250 FlyDSL tile config:
#   stage1: -tn / -tk / -bm
#   stage2: --stage2-tile-n / --stage2-tile-k
# Pass them explicitly so sweeps can tweak any without falling back to
# the silent default in aiter/fused_moe.py::get_2stage_cfgs.
ENABLE_CK=0 AITER_USE_OPUS_MOE_SORTING=1 python op_tests/test_moe_2stage.py \
  -t 64 -dim 7168,2048 -e 256 -k 8 -q 4 --no-flydsl-csv -hip 0,0 \
  -tn 128 -tk 512 -bm 16 --stage2-tile-n 128 --stage2-tile-k 128 \
  --num-buffers ${NUM_BUFFERS:-1}
