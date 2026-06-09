# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.
#!/bin/bash
#
# Smoke test for benchmark_mha_fwd_v3.cpp (ASM FWD with sink, gfx1250 only).
# Runs a set of shapes covering aligned/unaligned sq/sk and D64/D128 variants.
# Does NOT abort on the first failure — all cases run and a summary is printed.
#
# Prerequisites: libmha_fwd_asm.so + fwd.exe built via:
#   python3 compile.py --api=fwd_v3 && bash build_mha.sh fwd_v3
#
# Usage:
#   cd op_tests/cpp/mha
#   bash smoke_test_fwd_sink.sh
#
# Environment:
#   AITER_ASM_DIR  -- path to the hsa/ directory inside the repo.  The kernel
#                     loader appends "/<arch>/<co_name>" to locate .co files, so
#                     this must be the hsa/ dir, NOT the repo root.
#                     If unset, the script auto-detects it from its own location.
#                     Example: export AITER_ASM_DIR=/path/to/aiter/hsa

EXE="$(find . -name fwd.exe -type f | head -n 1)"
if [ -z "$EXE" ]; then
    echo "fwd.exe not found. Build first: bash build_mha.sh fwd_v3"
    exit 1
fi

# AITER_ASM_DIR must point to the repo's hsa/ directory so that AiterAsmKernel
# can locate hsa/gfx1250/fmha_fwd_bf16/*.co at runtime.
# Auto-detect from the script location (op_tests/cpp/mha → repo root is ../../../).
if [ -z "$AITER_ASM_DIR" ]; then
    _SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    _REPO_ROOT=$(cd "$_SCRIPT_DIR/../../.." && pwd)
    export AITER_ASM_DIR="$_REPO_ROOT/hsa"
fi

# Quick validation run (low warmup/repeat to keep wall time short).
COMMON_ARGS="-v=1 -causal=1 -warmup=2 -repeat=3"

NPASS=0
NFAIL=0
FAILED_CASES=""

run_case() {
    local label="$1"
    shift
    if "$EXE" "$@" 2>&1; then
        NPASS=$((NPASS + 1))
    else
        NFAIL=$((NFAIL + 1))
        FAILED_CASES="$FAILED_CASES\n  FAIL: $label ($*)"
    fi
}

echo "============================================================"
echo " Smoke test: benchmark_mha_fwd_v3 (fmha_fwd_with_sink_asm)"
echo " EXE: $EXE"
echo " AITER_ASM_DIR: $AITER_ASM_DIR"
echo "============================================================"

run_d64() {
    local sink=$1
    echo ""
    echo "--- head_dim=64 (D64, ENABLE_SINK=1, top-left causal) ---"
    # D64 sink floor (exp(sink_raw) > 0) prevents div-by-zero even for
    # entirely-masked rows, so sq != sk is safe to validate.

    # Square sq==sk
    for s in 512 1024 2048; do
        run_case "sq=sk=$s d=64" -d=64 -b=1 -h=8 -h_k=1 -s=$s -s_k=$s -sink=$sink $COMMON_ARGS
        run_case "sq=sk=$s d=64 b=2 gqa2" -d=64 -b=2 -h=8 -h_k=2 -s=$s -s_k=$s -sink=$sink $COMMON_ARGS
    done

    # Rectangular sq < sk
    for sq in 128 256 512; do
    for sk in 512 2048; do
    for batch in 1 2; do
    for hq_hk in "8 1" "8 2" "4 4"; do
        read hq hk <<< "$hq_hk"
        run_case "sq=$sq sk=$sk b=$batch h=$hq/$hk d=64" \
            -d=64 -b=$batch -h=$hq -h_k=$hk -s=$sq -s_k=$sk -sink=$sink $COMMON_ARGS
    done
    done
    done
    done

    # Unaligned sq
    for sq in 130 300; do
        run_case "sq=$sq sk=2048 d=64" -d=64 -b=1 -h=8 -h_k=1 -s=$sq -s_k=2048 -sink=$sink $COMMON_ARGS
    done

    # Unaligned sk
    for sk in 768 2300; do
        run_case "sq=128 sk=$sk d=64" -d=64 -b=1 -h=8 -h_k=1 -s=128 -s_k=$sk -sink=$sink $COMMON_ARGS
    done

    # Perf: no validation
    "$EXE" -d=64 -b=2 -h=64 -h_k=8 -s=8192 -s_k=8192 -sink=$sink -v=0 -warmup=5 -repeat=10
}

run_d128() {
    local sink=$1
    echo ""
    echo "--- head_dim=128 (D128, ENABLE_SINK=0, bottom-right causal) ---"
    # Bottom-right causal never creates entirely-masked KV-tiles for large sk,
    # so sq != sk is now safe to validate.

    # Square sq==sk
    for s in 512 1024 2048; do
        run_case "sq=sk=$s d=128" -d=128 -b=1 -h=8 -h_k=1 -s=$s -s_k=$s -sink=$sink $COMMON_ARGS
        run_case "sq=sk=$s d=128 b=2 gqa2" -d=128 -b=2 -h=8 -h_k=2 -s=$s -s_k=$s -sink=$sink $COMMON_ARGS
    done

    # Rectangular sq < sk (matching Python correctness test shapes)
    for sq in 128 256; do
    for hq_hk in "8 1" "8 2" "4 4"; do
        read hq hk <<< "$hq_hk"
        run_case "sq=$sq sk=2048 h=$hq/$hk d=128" \
            -d=128 -b=1 -h=$hq -h_k=$hk -s=$sq -s_k=2048 -sink=$sink $COMMON_ARGS
    done
    done

    # Unaligned sq and sk
    run_case "sq=130 sk=2048 d=128" -d=128 -b=1 -h=8 -h_k=1 -s=130 -s_k=2048 -sink=$sink $COMMON_ARGS
    run_case "sq=128 sk=2300 d=128" -d=128 -b=1 -h=8 -h_k=1 -s=128 -s_k=2300 -sink=$sink $COMMON_ARGS

    # Perf: no validation
    "$EXE" -d=128 -b=2 -h=64 -h_k=4 -s=4096 -s_k=4096 -sink=$sink -v=0 -warmup=5 -repeat=10
}

# D64: kernel ENABLE_SINK=1, must pass a non-zero sink value (1.0 AITER post-scale)
run_d64 1.0

# D128: kernel ENABLE_SINK=0, sink contents ignored (pass 0.0 → zero buffer).
# Validation restricted to sq==sk to avoid NaN from entirely-masked KV-tiles.
run_d128 0.0

echo ""
echo "============================================================"
echo " Results: PASS=$NPASS  FAIL=$NFAIL"
if [ $NFAIL -gt 0 ]; then
    echo " Failed cases:"
    printf "%b\n" "$FAILED_CASES"
    echo "============================================================"
    exit 1
fi
echo " ALL PASSED"
echo "============================================================"
