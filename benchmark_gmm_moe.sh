#!/bin/bash

# Benchmark Triton GMM kernel on a set of representative MoE shapes and
# print Triton autotuning choices (best configs) for each shape.
#
# Usage (recommended):
#   cd /root/aiter
#   ./benchmark_gmm_moe.sh | tee benchmark_gmm_moe.log
#
# The log will contain TRITON_PRINT_AUTOTUNING output, including the
# "best config" line per (M, K, N, G) shape, which you can then use to
# populate MI300X-GMM.json.
#
# Note: We use --num-group-sizes 1 because autotuning configs are the same
# regardless of group_sizes, and using 64 would run autotuning 64 times.

set -euo pipefail

# Always print Triton autotuning info for chosen configs.
export TRITON_PRINT_AUTOTUNING=1

# Enable AITER autotuning mode to use @triton.autotune decorators.
export AITER_AUTOTUNE=1

# Shapes: (M, K, N, G)
# ---------------------------------------------------------------------------
# Existing / deepseekv2-style shapes.
SHAPES=(
  # Generic DeepSeek-like MoE projection you already benchmarked.
  "24576 2048 2816 64"

  # DeepSeek v2 16B (from test_gmm REAL_SHAPES).
  "49152 1408 2048 64"
  "3145728 2048 1408 8"
  "393216 2048 1408 64"

  # Mixtral 8x22B (from test_gmm REAL_SHAPES).
  "32768 6144 16384 8"
  "32768 16384 6144 8"

  # Mixtral 8x7B-style (approx): 4k hidden, ~3.5x FFN, 8 experts.
  "32768 4096 14336 8"
  "32768 14336 4096 8"

  # DBRX-style (approx): 5k hidden, 4x FFN, 16 experts.
  "32768 5120 20480 16"
  "32768 20480 5120 16"

  # Qwen/Gemma-class MoE (approx): 5k hidden, ~3x FFN, 32 experts.
  "32768 5120 15360 32"
  "32768 15360 5120 32"

  # Large MoE (LLaMA3/DeepSeek-V3-class, approx): 8k hidden, ~3.5x FFN.
  "32768 8192 28672 64"
  "32768 28672 8192 64"
)

for shape in "${SHAPES[@]}"; do
  echo "================================================================"
  echo "Benchmarking Triton GMM for shape (M, K, N, G) = ($shape)"
  echo "================================================================"

  # Redirect stderr to stdout to capture all Triton autotuning output
  python op_tests/op_benchmarks/triton/bench_gmm.py \
    $shape \
    --gmm-type ptgmm \
    --input-type bf16 \
    --output-type bf16 \
    --metric throughput \
    --trans-lhs

  echo
done