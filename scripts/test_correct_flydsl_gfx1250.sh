#!/bin/bash
# Smoke-test all FlyDSL gfx1250-relevant op_tests after a FlyDSL version bump.
# Each test self-skips on unsupported arches, so this is safe to run anywhere.
# Usage:  PYTHONPATH=. bash run-mha-single.sh

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

# --- qknorm ---
run $PYTHON op_tests/test_flydsl_qk_norm_rope_quant.py

# --- mha ---
run $PYTHON op_tests/test_mha_flydsl_varlen.py

# --- compress_attn (run via __main__; self-skips on unsupported arches) ---
# FP4 path is not yet implemented for gfx1250; the test raises NotImplementedError
# on those shapes, so restrict to bf16-only shapes to avoid false failures.
run $PYTHON op_tests/test_flydsl_compress_attn.py -s csa_main hca_main

# --- gemm (a4w4 / f4gemm, gfx1250) ---
# Use correctness mode (small shapes) to avoid GPU memory faults on large tiles
run $PYTHON op_tests/test_f4gemm.py --mode func

# --- grouped gemm / moe (gfx1250, allow JIT since AOT cache may not exist) ---
run $PYTHON op_tests/test_flydsl_grouped_gemm_gfx1250.py --scenario verify --model-dim 512 --inter-dim 512 --no-check-aot-cache

# --- moe 2-stage (multi-arch, exercises FlyDSL moe kernels) ---
run $PYTHON op_tests/test_moe_2stage.py --no-flydsl-csv --no-legacy

# --- flydsl fmha (gfx1201/RDNA4, self-skips on CDNA/gfx1250) ---
run $PYTHON -m pytest op_tests/flydsl_tests/test_flydsl_fmha.py -x -q --tb=short

# --- flydsl moe a16wfp4 (gfx942/gfx950, self-skips elsewhere) ---
run $PYTHON -m pytest op_tests/flydsl_tests/test_flydsl_moe_a16wfp4.py -x -q --tb=short

# --- flydsl silu_and_mul_fq ---
run $PYTHON -m pytest op_tests/flydsl_tests/test_silu_and_mul_fq.py -x -q --tb=short

# --- hgemm / mhc (multi-arch, touches FlyDSL wmma on gfx1250) ---
run $PYTHON op_tests/test_mhc.py

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
