#!/bin/bash
# Benchmark script for mHC (manifold-constrained Hyper Connection) kernel

# Helper function to echo and run a command
run_cmd() {
    echo "$@"
    "$@"
}

modes=("fused" "sinkhorn" "full")

for mode in "${modes[@]}"; do
    echo "----------------------------------------"
    echo "--- Running benchmark in $mode mode ---"
    echo "----------------------------------------"
    run_cmd python op_tests/op_benchmarks/triton/bench_mhc.py --mode $mode
done
