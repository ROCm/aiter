# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.
#!/bin/bash
#
# Smoke test for benchmark_mha_fwd_v3.cpp (ASM FWD with sink, gfx1250 only).
# Builds the benchmark if fwd.exe is not found, then runs a set of shapes
# covering aligned/unaligned sq/sk and D64/D128 variants.
#
# Prerequisites: libmha_fwd.so built via `python3 compile.py --api=fwd_v3`
# Binary produced by:  bash build_mha.sh fwd_v3   (links benchmark_mha_fwd_v3.cpp)
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

set -e

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

echo "============================================================"
echo " Smoke test: benchmark_mha_fwd_v3 (fmha_fwd_with_sink_asm)"
echo " EXE: $EXE"
echo " AITER_ASM_DIR: $AITER_ASM_DIR"
echo "============================================================"

run_all() {
    local d=$1   # head_dim: 64 or 128
    local sink=$2  # sink value (1.0 for D64; 0.0 for D128, but arg is still passed)

    echo ""
    echo "--- head_dim=$d ---"

    # Aligned sq (mult of 128) + aligned sk (mult of 256)
    for sq in 128 256 512; do
    for sk in 256 2048; do
    for batch in 1 2; do
    for hq_hk in "8 1" "8 2" "4 4"; do
        read hq hk <<< "$hq_hk"
        $EXE -d=$d -b=$batch -h=$hq -h_k=$hk -s=$sq -s_k=$sk -sink=$sink $COMMON_ARGS
    done
    done
    done
    done

    # Unaligned sq (not mult of 128)
    for sq in 130 300; do
    for sk in 2048; do
        $EXE -d=$d -b=1 -h=8 -h_k=1 -s=$sq -s_k=$sk -sink=$sink $COMMON_ARGS
    done
    done

    # Unaligned sk (not mult of 256)
    for sq in 128; do
    for sk in 300 2300; do
        $EXE -d=$d -b=1 -h=8 -h_k=1 -s=$sq -s_k=$sk -sink=$sink $COMMON_ARGS
    done
    done

    # sq == sk (standard square causal)
    for s in 128 512 1024; do
        $EXE -d=$d -b=1 -h=8 -h_k=1 -s=$s -s_k=$s -sink=$sink $COMMON_ARGS
        $EXE -d=$d -b=2 -h=8 -h_k=2 -s=$s -s_k=$s -sink=$sink $COMMON_ARGS
    done

    # Perf-sized shapes (matching Python test defaults)
    if [ "$d" -eq 64 ]; then
        # D64 perf: b=2 hq=64 hk=8 sq=sk=8192
        $EXE -d=$d -b=2 -h=64 -h_k=8 -s=8192 -s_k=8192 -sink=$sink -v=0 -warmup=5 -repeat=10
    else
        # D128 perf: b=2 hq=64 hk=4 sq=sk=4096
        $EXE -d=$d -b=2 -h=64 -h_k=4 -s=4096 -s_k=4096 -sink=$sink -v=0 -warmup=5 -repeat=10
    fi
}

# D64: kernel ENABLE_SINK=1, must pass a non-zero sink value (1.0 AITER post-scale)
run_all 64  1.0

# D128: kernel ENABLE_SINK=0, sink tensor contents ignored (pass 0.0 → zero buffer)
run_all 128 0.0

echo ""
echo "============================================================"
echo " Smoke test PASSED"
echo "============================================================"
