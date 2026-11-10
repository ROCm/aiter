#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# DSR1 problem size, with the same balanced routing for both input modes.
exec python3 -u -m op_tests.flydsl_tests.test_flydsl_grouped_gemm \
  --scenario kernel \
  --data-format a4w4 \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048 \
  --act silu \
  --no-bias \
  --balanced-routing \
  --warmup 5 \
  --iters 512 \
  --seed 0 \
  --no-check-aot-cache \
  --data-init constant uniform \
  --scale-init constant auto
