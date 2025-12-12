#!/bin/bash

# Directory containing captured inputs from sageattn_cogvideo.py --save_inputs
CAPTURED_DIR="${CAPTURED_DIR:-./captured_inputs}"

# Optional: limit number of inputs or sample rate
MAX_INPUTS="${MAX_INPUTS:-}"  # e.g., 10 to limit to first 10 inputs
SAMPLE_RATE="${SAMPLE_RATE:-}"  # e.g., 5 to sample every 5th input

# Helper function to echo and run a command
run_cmd() {
    echo "$@"
    "$@"
}

# Build optional args
OPTIONAL_ARGS=""
if [ -n "$MAX_INPUTS" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --max_inputs $MAX_INPUTS"
fi
if [ -n "$SAMPLE_RATE" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --sample_rate $SAMPLE_RATE"
fi

echo "=========================================="
echo "Running benchmarks with captured inputs"
echo "Captured dir: $CAPTURED_DIR"
echo "Optional args: $OPTIONAL_ARGS"
echo "=========================================="

echo ""
echo "--- FAv2 (no quantization) ---"
run_cmd python op_tests/op_benchmarks/triton/bench_diffusion_attention.py \
    --load_captured \
    --captured_dir "$CAPTURED_DIR" \
    $OPTIONAL_ARGS \
    -metric all

echo ""
echo "--- SageAttnV1 (i.e -qk_int8) ---"
run_cmd python op_tests/op_benchmarks/triton/bench_diffusion_attention.py \
    --load_captured \
    --captured_dir "$CAPTURED_DIR" \
    $OPTIONAL_ARGS \
    -qk_int8 \
    -metric all

echo ""
echo "--- FAv3 FP8 (i.e -fp8) ---"
run_cmd python op_tests/op_benchmarks/triton/bench_diffusion_attention.py \
    --load_captured \
    --captured_dir "$CAPTURED_DIR" \
    $OPTIONAL_ARGS \
    -fp8 \
    -metric all

echo ""
echo "--- SageAttnV1 (i.e -sagev1, fused on fa3 fp8) ---"
run_cmd python op_tests/op_benchmarks/triton/bench_diffusion_attention.py \
    --load_captured \
    --captured_dir "$CAPTURED_DIR" \
    $OPTIONAL_ARGS \
    -sagev1_fa3 \
    -metric all

echo ""
echo "Benchmarks complete."

