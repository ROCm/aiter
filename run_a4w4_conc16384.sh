#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# DSV4 problem size; tuning comes from model_configs/dsv4_tuned_grouped_fmoe.csv.
export AITER_FLYDSL_DISABLE_GEMM1_ACT=1
export AITER_FLYDSL_DISABLE_GEMM1_REQUANT=1

exec python3 -u -m op_tests.flydsl_tests.test_flydsl_grouped_gemm \
  --scenario kernel \
  --data-format a4w4 \
  --experts 96 \
  --tokens 16384 \
  --topk 6 \
  --model-dim 7168 \
  --inter-dim 3072 \
  --act silu \
  --no-bias \
  --balanced-routing \
  --warmup 5 \
  --iters 512 \
  --seed 0 \
  --no-check-aot-cache \
  --data-init constant uniform \
  --scale-init constant auto
