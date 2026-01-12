#!/bin/bash
BATCH_SIZES=(1 1)
NUM_HEADS=(5 5)
SEQ_LENS_Q=(75352 75352)
SEQ_LENS_K=(75352 512)

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

    echo "--- FAv3 Sage (i.e -fav3_sage, sage features fused on fav3 fp8 pipeline) ---"
    run_cmd python op_tests/op_benchmarks/triton/bench_diffusion_attention.py \
        -b ${BATCH_SIZES[i]} \
        -hq ${NUM_HEADS[i]} \
        -sq ${SEQ_LENS_Q[i]} \
        -sk ${SEQ_LENS_K[i]} \
        -d 128 \
        -fav3_sage \
        -metric all \
        # -print_vgpr \
    
    echo ""
done

echo "Benchmarks complete."
