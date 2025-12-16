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
echo "--- SageAttnV1 (i.e -sagev1) ---"
run_cmd python op_tests/op_benchmarks/triton/bench_diffusion_attention.py \
    --load_captured \
    --captured_dir "$CAPTURED_DIR" \
    -sagev1 \
    -metric all

echo ""
echo "--- FAv3 FP8 (i.e -fav3_fp8) ---"
run_cmd python op_tests/op_benchmarks/triton/bench_diffusion_attention.py \
    --load_captured \
    --captured_dir "$CAPTURED_DIR" \
    -fav3_fp8 \
    -metric all

echo ""
echo "--- FAv3 Sage (i.e -fav3_sage, sage features fused on fav3 fp8 pipeline) ---"
run_cmd python op_tests/op_benchmarks/triton/bench_diffusion_attention.py \
    --load_captured \
    --captured_dir "$CAPTURED_DIR" \
    -fav3_sage \
    -metric all

echo ""
echo "Benchmarks complete."

