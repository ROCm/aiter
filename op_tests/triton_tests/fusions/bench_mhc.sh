#!/bin/bash
# Benchmark script for mHC (manifold-constrained Hyper Connection) kernel
#
# Runs benchmarks across various configurations:
# - Different batch sizes (M)
# - Different stream counts (n)
# - Different hidden dimensions (C)
# - With and without Sinkhorn-Knopp projection
# - Different Sinkhorn iteration counts
# - Different data types (fp16 vs bf16)
#
# Note: This benchmarks PERFORMANCE. For CORRECTNESS testing of edge cases
# (zero input, large values, different epsilon/alpha values), see test_mhc.py

# Helper function to echo and run a command
run_cmd() {
    echo "$@"
    "$@"
}

echo "========================================"
echo "mHC Benchmark Suite"
echo "========================================"
echo ""

# ========================================
# 1. Batch Size Sweep (fixed n=4, C=1024)
# ========================================
echo "=========================================="
echo "1. Batch Size Sweep"
echo "=========================================="

BATCH_SIZES=(32 64 128 256 512 1024 2048 4096)

for M in ${BATCH_SIZES[@]}; do
    echo "--- Batch size M=$M ---"
    run_cmd python op_tests/op_benchmarks/triton/bench_mhc.py \
        -M $M \
        -n 4 \
        -C 1024 \
        --dtype bf16
    echo ""
done

# ========================================
# 2. Stream Count Sweep (fixed M=128, C=1024)
# ========================================
echo "=========================================="
echo "2. Stream Count Sweep"
echo "=========================================="

STREAM_COUNTS=(2 4 8 16)

for n in ${STREAM_COUNTS[@]}; do
    echo "--- Stream count n=$n ---"
    run_cmd python op_tests/op_benchmarks/triton/bench_mhc.py \
        -M 128 \
        -n $n \
        -C 1024 \
        --dtype bf16
    echo ""
done

# ========================================
# 3. Hidden Dimension Sweep (fixed M=128, n=4)
# ========================================
echo "=========================================="
echo "3. Hidden Dimension Sweep"
echo "=========================================="

HIDDEN_DIMS=(512 1024 2048 4096)

for C in ${HIDDEN_DIMS[@]}; do
    echo "--- Hidden dimension C=$C ---"
    run_cmd python op_tests/op_benchmarks/triton/bench_mhc.py \
        -M 128 \
        -n 4 \
        -C $C \
        --dtype bf16
    echo ""
done

# ========================================
# 4. Fused-only vs Full Pipeline Comparison
# ========================================
echo "=========================================="
echo "4. Fused-only vs Full Pipeline"
echo "=========================================="

echo "--- Fused mHC only (no Sinkhorn-Knopp) ---"
run_cmd python op_tests/op_benchmarks/triton/bench_mhc.py \
    -M 1024 \
    -n 4 \
    -C 1024 \
    --dtype bf16 \
    --no_sinkhorn

echo ""
echo "--- Full mHC (with Sinkhorn-Knopp, 20 iters) ---"
run_cmd python op_tests/op_benchmarks/triton/bench_mhc.py \
    -M 1024 \
    -n 4 \
    -C 1024 \
    --dtype bf16

echo ""

# ========================================
# 5. Sinkhorn Iteration Count Sweep
# ========================================
echo "=========================================="
echo "5. Sinkhorn Iteration Count Sweep"
echo "=========================================="

SINKHORN_ITERS=(5 10 20 50)

for iters in ${SINKHORN_ITERS[@]}; do
    echo "--- Sinkhorn iterations=$iters ---"
    run_cmd python op_tests/op_benchmarks/triton/bench_mhc.py \
        -M 1024 \
        -n 4 \
        -C 1024 \
        --dtype bf16 \
        -sinkhorn_iters $iters
    echo ""
done

# ========================================
# 6. Full benchmark suite (all configs)
# ========================================
echo "=========================================="
echo "6. Full Benchmark Suite"
echo "=========================================="

echo "--- Running full benchmark suite (all default configs) ---"
run_cmd python op_tests/op_benchmarks/triton/bench_mhc.py \
    --dtype bf16 \
    -o

echo ""
echo "========================================"
echo "mHC Benchmark Suite Complete"
echo "========================================"
# ========================================
# 7. Data Type Comparison
# ========================================
echo "=========================================="
echo "7. Data Type Comparison (fp16 vs bf16)"
echo "=========================================="

DTYPES=(fp16 bf16)

for dtype in ${DTYPES[@]}; do
    echo "--- Data type: $dtype ---"
    run_cmd python op_tests/op_benchmarks/triton/bench_mhc.py \
        -M 1024 \
        -n 4 \
        -C 1024 \
        --dtype $dtype
    echo ""
done
