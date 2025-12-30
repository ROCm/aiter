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

    echo "--- FAv2 (no quantization) ---"
    run_cmd python op_tests/op_benchmarks/triton/bench_diffusion_attention.py \
        -b ${BATCH_SIZES[i]} \
        -hq ${NUM_HEADS[i]} \
        -sq ${SEQ_LENS[i]} \
        -d 128 \
        -metric all \
        # -print_vgpr \

    echo ""
    
    echo "--- SageAttnV1 (i.e -sagev1) ---"
    run_cmd python op_tests/op_benchmarks/triton/bench_diffusion_attention.py \
        -b ${BATCH_SIZES[i]} \
        -hq ${NUM_HEADS[i]} \
        -sq ${SEQ_LENS[i]} \
        -d 128 \
        -sagev1 \
        -metric all \
        # -print_vgpr \
    
    echo ""
    echo "--- FAv3 FP8 (i.e -fav3_fp8) ---"
    run_cmd python op_tests/op_benchmarks/triton/bench_diffusion_attention.py \
        -b ${BATCH_SIZES[i]} \
        -hq ${NUM_HEADS[i]} \
        -sq ${SEQ_LENS[i]} \
        -d 128 \
        -fav3_fp8 \
        -metric all
        
    echo ""

    echo "--- FAv3 Sage (i.e -fav3_sage, sage features fused on fav3 fp8 pipeline) ---"
    run_cmd python op_tests/op_benchmarks/triton/bench_diffusion_attention.py \
        -b ${BATCH_SIZES[i]} \
        -hq ${NUM_HEADS[i]} \
        -sq ${SEQ_LENS[i]} \
        -d 128 \
        -fav3_sage \
        -metric all \
        # -print_vgpr \
    
    echo ""
done

echo "Benchmarks complete."
