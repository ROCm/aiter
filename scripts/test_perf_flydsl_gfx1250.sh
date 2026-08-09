#!/bin/bash
# Performance benchmark for FlyDSL gfx1250 kernels (hgemm, moe, gemm).
# Reports kernel timing (us) and throughput for key operator shapes.
# Usage:  PYTHONPATH=. bash scripts/test_perf_flydsl_gfx1250.sh

set -euo pipefail
export HSA_ENABLE_SDMA=0

PYTHON="${PYTHON:-/opt/venv/bin/python3}"

failed=()

run() {
    echo "========================================"
    echo ">>> $*"
    echo "========================================"
    if "$@"; then
        echo "--- PASS: $* ---"
    else
        echo "--- FAIL: $* ---"
        failed+=("$*")
    fi
    echo
}

# --- hgemm / mhc (bf16 WMMA path on gfx1250) ---
# Sweeps default hidden sizes with fused RMSNorm and bf16 fn-pack variants
run $PYTHON op_tests/test_mhc.py --fuse_rmsnorm --fn_pack_bf16

# --- moe grouped gemm: end-to-end bench (CUDA graph, DSV4-like shape) ---
run $PYTHON op_tests/test_flydsl_grouped_gemm_gfx1250.py \
    --scenario bench \
    --tokens 64 128 256 \
    --model-dim 7168 --inter-dim 512 \
    --experts 256 --topk 8 \
    --no-check-aot-cache

# --- moe grouped gemm: per-kernel bench (gemm1 + gemm2 in isolation) ---
run $PYTHON op_tests/test_flydsl_grouped_gemm_gfx1250.py \
    --scenario kernel \
    --tokens 64 128 256 \
    --model-dim 7168 --inter-dim 512 \
    --experts 256 --topk 8 \
    --no-check-aot-cache

# --- moe 2-stage (FlyDSL SiTUv2 path, DSV4 shape) ---
run $PYTHON op_tests/test_moe_2stage.py \
    --no-flydsl-csv --no-legacy \
    -dim 7168,512 -t 64 128 256

# --- gemm a4w4 / f4gemm perf ---
# Default 16384³ hits a GPU memory fault on some configs; use 4096³ instead
run $PYTHON op_tests/test_f4gemm.py --mode perf -mnk 4096,4096,4096

echo "========================================"
if [ ${#failed[@]} -eq 0 ]; then
    echo "ALL PASSED"
else
    echo "FAILURES (${#failed[@]}):"
    for f in "${failed[@]}"; do
        echo "  - $f"
    done
    exit 1
fi
