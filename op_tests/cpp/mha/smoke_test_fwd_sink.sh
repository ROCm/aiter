# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.
#!/bin/bash
#
# Smoke test for benchmark_mha_fwd_v3.cpp (gfx1250).
# Supports both kernel call paths via positional arg (default: mha_fwd):
#
#   mha_fwd  -- aiter::mha_fwd() (same path as TE); D64 sink=-1e30
#   direct   -- fmha_fwd_with_sink_asm() directly;  D64 sink=1.0
#
# Prerequisites: libmha_fwd_asm.so + fwd_v3.exe built via:
#   python3 compile.py --api=fwd_v3 && bash build_mha.sh fwd_v3
#
# Usage:
#   bash smoke_test_fwd_v3_gfx1250.sh [mha_fwd|direct]

VIA="${1:-mha_fwd}"

case "$VIA" in
    mha_fwd|direct) ;;
    *) echo "Unknown mode '$VIA'. Use: mha_fwd (default) or direct"; exit 1 ;;
esac

# D64: ENABLE_SINK=1.  In direct mode use real sink=1.0 (post-scale units);
# in mha_fwd mode use sink=-1e30 (exp(-1e30)≈0, mirrors what TE passes).
if [ "$VIA" = "direct" ]; then
    D64_SINK=1.0
else
    D64_SINK=-1e30
fi
# D128: ENABLE_SINK=0, sink value is ignored by the kernel — pass 0 either way.
D128_SINK=0

EXE="$(find . -name fwd_v3.exe -type f | head -n 1)"
if [ -z "$EXE" ]; then
    echo "fwd_v3.exe not found."
    echo "Build first: bash build_mha.sh fwd_v3"
    exit 1
fi

if [ -z "$AITER_ASM_DIR" ]; then
    _SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    _REPO_ROOT=$(cd "$_SCRIPT_DIR/../../.." && pwd)
    export AITER_ASM_DIR="$_REPO_ROOT/hsa"
fi

COMMON_ARGS="-v=1 -causal=1 -warmup=2 -repeat=3 -via=$VIA"

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
echo " Smoke test: benchmark_mha_fwd_v3 -via=$VIA"
echo " EXE: $EXE"
echo " AITER_ASM_DIR: $AITER_ASM_DIR"
echo "============================================================"

echo ""
echo "--- head_dim=64 (D64, ENABLE_SINK=1, top-left causal, sink=$D64_SINK) ---"

for s in 512 1024 2048; do
    run_case "sq=sk=$s d=64 b=1 mha" \
        -d=64 -b=1 -h=8 -h_k=1 -s=$s -s_k=$s -sink=$D64_SINK $COMMON_ARGS
    run_case "sq=sk=$s d=64 b=2 gqa2" \
        -d=64 -b=2 -h=8 -h_k=2 -s=$s -s_k=$s -sink=$D64_SINK $COMMON_ARGS
done

for sq in 128 256 512; do
for sk in 512 2048; do
for batch in 1 2; do
for hq_hk in "8 1" "8 2" "4 4"; do
    read hq hk <<< "$hq_hk"
    run_case "sq=$sq sk=$sk b=$batch h=$hq/$hk d=64" \
        -d=64 -b=$batch -h=$hq -h_k=$hk -s=$sq -s_k=$sk -sink=$D64_SINK $COMMON_ARGS
done; done; done; done

for sq in 130 300; do
    run_case "sq=$sq sk=2048 d=64" \
        -d=64 -b=1 -h=8 -h_k=1 -s=$sq -s_k=2048 -sink=$D64_SINK $COMMON_ARGS
done

for sk in 768 2300; do
    run_case "sq=128 sk=$sk d=64" \
        -d=64 -b=1 -h=8 -h_k=1 -s=128 -s_k=$sk -sink=$D64_SINK $COMMON_ARGS
done

# Perf: no validation
"$EXE" -d=64 -b=2 -h=64 -h_k=8 -s=8192 -s_k=8192 -sink=$D64_SINK -v=0 -warmup=5 -repeat=10 -via=$VIA

echo ""
echo "--- head_dim=128 (D128, ENABLE_SINK=0, bottom-right causal, sink ignored) ---"

for s in 512 1024 2048; do
    run_case "sq=sk=$s d=128 b=1 mha" \
        -d=128 -b=1 -h=8 -h_k=1 -s=$s -s_k=$s -sink=$D128_SINK $COMMON_ARGS
    run_case "sq=sk=$s d=128 b=2 gqa2" \
        -d=128 -b=2 -h=8 -h_k=2 -s=$s -s_k=$s -sink=$D128_SINK $COMMON_ARGS
done

for sq in 128 256; do
for hq_hk in "8 1" "8 2" "4 4"; do
    read hq hk <<< "$hq_hk"
    run_case "sq=$sq sk=2048 h=$hq/$hk d=128" \
        -d=128 -b=1 -h=$hq -h_k=$hk -s=$sq -s_k=2048 -sink=$D128_SINK $COMMON_ARGS
done
done

run_case "sq=130 sk=2048 d=128" \
    -d=128 -b=1 -h=8 -h_k=1 -s=130 -s_k=2048 -sink=$D128_SINK $COMMON_ARGS
run_case "sq=128 sk=2300 d=128" \
    -d=128 -b=1 -h=8 -h_k=1 -s=128 -s_k=2300 -sink=$D128_SINK $COMMON_ARGS

# Perf: no validation
"$EXE" -d=128 -b=2 -h=64 -h_k=4 -s=4096 -s_k=4096 -sink=$D128_SINK -v=0 -warmup=5 -repeat=10 -via=$VIA

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
