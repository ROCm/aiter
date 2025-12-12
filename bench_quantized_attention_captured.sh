#!/bin/bash

# Directory containing captured inputs from sageattn_cogvideo.py --save_inputs
CAPTURED_DIR="${CAPTURED_DIR:-./captured_inputs}"

# Helper function to echo and run a command
run_cmd() {
    echo "$@"
    "$@"
}

echo "=========================================="
echo "Running benchmarks with captured inputs"
echo "Captured dir: $CAPTURED_DIR"
echo "=========================================="

echo ""
echo "--- FAv2 (no quantization) ---"
run_cmd python op_tests/op_benchmarks/triton/bench_diffusion_attention.py \
    --load_captured \
    --captured_dir "$CAPTURED_DIR" \
    -metric all

echo ""
echo "--- SageAttnV1 (i.e -qk_int8) ---"
run_cmd python op_tests/op_benchmarks/triton/bench_diffusion_attention.py \
    --load_captured \
    --captured_dir "$CAPTURED_DIR" \
    -qk_int8 \
    -metric all

echo ""
echo "--- FAv3 FP8 (i.e -fp8) ---"
run_cmd python op_tests/op_benchmarks/triton/bench_diffusion_attention.py \
    --load_captured \
    --captured_dir "$CAPTURED_DIR" \
    -fp8 \
    -metric all

echo ""
echo "--- SageAttnV1 (i.e -sagev1, fused on fa3 fp8) ---"
run_cmd python op_tests/op_benchmarks/triton/bench_diffusion_attention.py \
    --load_captured \
    --captured_dir "$CAPTURED_DIR" \
    -sagev1_fa3 \
    -metric all

echo ""
echo "Benchmarks complete."

