#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

python3 -u -m op_tests.flydsl_tests.test_flydsl_grouped_gemm \
  --scenario kernel --data-format a4w4 --experts 96 \
  --tokens 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 \
  --topk 6 --model-dim 5120 --inter-dim 2304 --act silu --no-bias \
  --balanced-routing --warmup 5 --iters 32 --seed 0 --no-check-aot-cache \
  --data-init constant uniform --scale-init constant \
  "$@"
