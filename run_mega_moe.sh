#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export AITER_MEGA_S1F=0 AITER_FLYDSL_A2_PRESHUFFLE=1 DEBUG_HIP_GRAPH_CLASSIC_PATH=1
torchrun --standalone --nproc_per_node=4 -m op_tests.multigpu_tests.bench_mega_moe \
  -q a4w4_mxfp4 -tpr 16384 -hd 7168 -id 2048 -e 256 -k 8 \
  --layers 1 --stage1_fused true --stage2_fused true --dispatch_wire fp4 \
  --warmup 10 --iters 20 --prof_replays 10 \
  "$@"
