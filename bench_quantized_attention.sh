#!/bin/bash

# Default: run benchmark only
RUN_BENCHMARK=false
RUN_COMPARE=false

# Parse CLI arguments
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -b, --benchmark    Run benchmarks"
    echo "  -c, --compare      Run output comparison"
    echo "  -h, --help         Show this help message"
    echo ""
    echo "If no options are provided, only benchmarks will run."
    echo "Use -b -c together to run both."
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--benchmark)
            RUN_BENCHMARK=true
            shift
            ;;
        -c|--compare)
            RUN_COMPARE=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# If no flags were set, default to running benchmark only
if [[ "$RUN_BENCHMARK" == "false" && "$RUN_COMPARE" == "false" ]]; then
    RUN_BENCHMARK=true
fi

BATCH_SIZES=(1 1 1 2)
NUM_HEADS=(5 24 3 2)
SEQ_LENS=(75600 16452 118808 29760)
OUTPUT_DIR=./debug

# Helper function to echo and run a command
run_cmd() {
    echo "$@"
    "$@"
}

# Run benchmarks for all configurations
if [[ "$RUN_BENCHMARK" == "true" ]]; then
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
        
        echo "--- SageAttnV1 (i.e -qk_int8) ---"
        run_cmd python op_tests/op_benchmarks/triton/bench_diffusion_attention.py \
            -b ${BATCH_SIZES[i]} \
            -hq ${NUM_HEADS[i]} \
            -sq ${SEQ_LENS[i]} \
            -d 128 \
            -qk_int8 \
            -metric all \
            # -print_vgpr \
        
        echo ""
        echo "--- FAv3 FP8 (i.e -fp8) ---"
        run_cmd python op_tests/op_benchmarks/triton/bench_diffusion_attention.py \
            -b ${BATCH_SIZES[i]} \
            -hq ${NUM_HEADS[i]} \
            -sq ${SEQ_LENS[i]} \
            -d 128 \
            -fp8 \
            -metric all \
            # -print_vgpr \
            
        echo ""
    done

    echo "Benchmarks complete."
fi