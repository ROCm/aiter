#!/bin/bash

BATCH_SIZES=(1 1 1 2)
NUM_HEADS=(5 24 3 2)
SEQ_LENS=(75600 16452 118808 29760)

# Helper function to echo and run a command
run_cmd() {
    echo "$@"
    "$@"
}

# Run benchmarks for all configurations
for i in ${!BATCH_SIZES[@]}; do
    echo "=========================================="
    echo "Running benchmark $((i+1))/4: batch_size=${BATCH_SIZES[i]}, num_heads=${NUM_HEADS[i]}, seq_len=${SEQ_LENS[i]}"
    echo "=========================================="
    
    echo ""
    echo "--- SageAttnV1 (i.e -qk_int8) ---"
    run_cmd python op_tests/op_benchmarks/triton/bench_mha.py \
        -b ${BATCH_SIZES[i]} \
        -hq ${NUM_HEADS[i]} \
        -sq ${SEQ_LENS[i]} \
        -d 128 \
        -qk_int8 \
        -real_quant \
        -metric all
    
    echo ""
    echo "--- FAv3 FP8 (i.e -fp8) ---"
    run_cmd python op_tests/op_benchmarks/triton/bench_mha.py \
        -b ${BATCH_SIZES[i]} \
        -hq ${NUM_HEADS[i]} \
        -sq ${SEQ_LENS[i]} \
        -d 128 \
        -causal False \
        -fp8 \
        -metric all
    
    echo ""
done

echo "Benchmarks complete."
