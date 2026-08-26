#!/bin/bash
set -euo pipefail

SHAPES=(
    "16 1024 7168"
    "16 128 2880"
    "16 5120 2880"
    "16 2880 4096"
    "16 256 7168"
    "16 128 5120"
    "16 128 4096"
    "64 1024 7168"
    "512 1024 7168"
    "2048 128 2880"
    "2048 5120 2880"
    "2048 2880 4096"
    "2048 256 7168"
    "2048 128 5120"
    "2048 128 4096"
)

TOTAL=${#SHAPES[@]}
IDX=0

for shape in "${SHAPES[@]}"; do
    IDX=$((IDX + 1))
    echo "===== [$IDX/$TOTAL] Running shape: $shape ====="
    python3 op_tests/op_benchmarks/triton/bench_gemm_a16w16.py \
        --backend gluon --layout TN --shape $shape --metric time --persistent
    echo ""
done

echo "===== All $TOTAL benchmarks complete ====="
