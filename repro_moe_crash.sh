#!/usr/bin/env bash
# Minimal reproducer for the gfx1250 a8w4 mxfp4 MoE crash.
#
# Symptom:
#   HSA_STATUS_ERROR_EXCEPTION: An HSAIL operation resulted in a hardware
#   exception. code: 0x1016   (illegal memory access)
#
# Root cause (observed, not GPU hardware):
#   - Each token size passes on its own; basic torch kernels (copy/add/matmul)
#     are fine on gfx1250.
#   - Running >=3 distinct token sizes in ONE process corrupts GPU memory; the
#     abort surfaces on whatever torch kernel touches the poisoned region next
#     (varies run-to-run: direct_copy_kernel / index_elementwise_kernel).
#   - Still crashes with HIP_LAUNCH_BLOCKING=1 -> NOT an async race; it is an
#     accumulative out-of-bounds bug in the aiter mxfp4 grouped-gemm path.
#
# Note: these runs are SERIALIZED (single process, tokens run one after another),
# NOT parallel and NOT multi-stream. Parallelism is not involved in the crash.
#
# Usage:
#   docker exec felix bash -lc 'cd /home/felix/aiter && ./repro_moe_crash.sh'
#   MODE=blocking ./repro_moe_crash.sh   # fully serialize kernels too
#   MODE=single   ./repro_moe_crash.sh   # control: one token only, should PASS
set -u

AITER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$AITER_DIR" || { echo "no $AITER_DIR"; exit 1; }
PY=${PY:-python3}
MODE=${MODE:-repro}

# ck_tile has no gfx1250 target yet -> build via the ck_tile_shim.
export ENABLE_CK=${ENABLE_CK:-0}
export AITER_FORCE_A8W4=${AITER_FORCE_A8W4:-1}
export AITER_USE_GROUPED_GEMM=${AITER_USE_GROUPED_GEMM:-1}
export AITER_BF16_FP8_MOE_BOUND=${AITER_BF16_FP8_MOE_BOUND:-0}

SHAPE=(-hd 7168 -id 3072 -e 384 -k 6 -ep 8)

case "$MODE" in
  blocking)
    echo "[repro] SERIALIZED + HIP_LAUNCH_BLOCKING=1 (expect CRASH)"
    export HIP_LAUNCH_BLOCKING=1
    export AMD_SERIALIZE_KERNEL=3
    TOKENS="1 16 128 512"
    ;;
  single)
    echo "[repro] control: single token in its own process (expect PASS)"
    TOKENS="1"
    ;;
  *)
    echo "[repro] SERIALIZED multi-token in one process (expect CRASH)"
    TOKENS="1 16 128 512"
    ;;
esac

set -x
"$PY" -u op_tests/test_moe_ep.py -t g1u1_a8w4_mxfp4 -m $TOKENS "${SHAPE[@]}"
