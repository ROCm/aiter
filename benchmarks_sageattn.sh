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

# Only save outputs if comparison is enabled
SAVE_OUTPUT_ARGS=""
if [[ "$RUN_COMPARE" == "true" ]]; then
    SAVE_OUTPUT_ARGS="--save_output --output_dir ${OUTPUT_DIR}"
fi

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
        
        echo ""
        echo "--- SageAttnV1 (i.e -qk_int8) ---"
        run_cmd python op_tests/op_benchmarks/triton/bench_mha.py \
            -b ${BATCH_SIZES[i]} \
            -hq ${NUM_HEADS[i]} \
            -sq ${SEQ_LENS[i]} \
            -d 128 \
            -qk_int8 \
            -real_quant \
            -metric all \
            ${SAVE_OUTPUT_ARGS}
        
        echo ""
        echo "--- FAv3 FP8 (i.e -fp8) ---"
        run_cmd python op_tests/op_benchmarks/triton/bench_mha.py \
            -b ${BATCH_SIZES[i]} \
            -hq ${NUM_HEADS[i]} \
            -sq ${SEQ_LENS[i]} \
            -d 128 \
            -causal False \
            -fp8 \
            -metric all \
            ${SAVE_OUTPUT_ARGS}
        
        echo ""
    done

    echo "Benchmarks complete."
fi

# Run output comparison
if [[ "$RUN_COMPARE" == "true" ]]; then
    echo ""
    echo "=========================================="
    echo "Comparing outputs between FP8 and QK-INT8 implementations..."
    echo "=========================================="

    # Compare each fp8 output file against its qk-int8 counterpart when both exist.
    for fp8_file in ${OUTPUT_DIR}/*_fp8.pt; do
        [[ -e "${fp8_file}" ]] || continue
        base_path=${fp8_file%_fp8.pt}
        int8_file="${base_path}_qk-int8.pt"
        if [[ -f "${int8_file}" ]]; then
            run_cmd python compare_outputs.py "${fp8_file}" "${int8_file}"
        else
            echo "Skipping ${fp8_file}: missing ${int8_file}"
        fi
    done
fi
