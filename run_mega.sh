#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# The installed MORI predates TokOffExt; allow newer checkouts to opt in.
MORI_V2_KERNEL_BACKEND=hip \
MORI_EP_TOKOFF_EXT="${MORI_EP_TOKOFF_EXT:-0}" \
MEGA_DISPATCH=mori \
FLYDSL_DUMP_IR=1 \
MORI_GPU_ARCHS=gfx1250 \
FLYDSL_GPU_ARCH=gfx1250 \
AITER_FORCE_A8W4=0 \
AITER_MOE_EXPERT_BALANCE=True \
exec torchrun --standalone --nproc_per_node=4 \
  "${script_dir}/op_tests/multigpu_tests/bench_mega_moe.py" \
  -e 256 \
  -k 8 \
  -hd 7168 \
  -id 2048 \
  --layers 61 \
  -tpr 1536 \
  -q a4w4_mxfp4 \
  --combine fused \
  --acc_verify 0 \
  --profile_table 1 \
  --dispatch_wire fp4 \
  "$@"
