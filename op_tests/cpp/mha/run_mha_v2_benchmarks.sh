#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Asm v2 (not v3): -fwd_v3=0, -bwd_v3=0. Precisions: fp16 + bf16 only.
# Timing: -warmup=5 -repeat=25 (matches benchmark ArgParser).
# Batch b=2048,4096; head dim d=64,128,256 (-h=64 num heads); seq 16 and 17.
# Build first: `bash build_mha.sh` (needs FAV2 in compile; default compile.py enables FAV2_ON with CK).
#
#   export AITER_ASM_DIR='<aiter-repo>/hsa/'
#   cd op_tests/cpp/mha && ./run_mha_v2_benchmarks.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# fwd.exe / bwd.exe link libmha_fwd.so / libmha_bwd.so with -Lthis dir — loader needs it on LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$SCRIPT_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export AITER_ASM_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)/hsa/"

if [[ ! -f "$SCRIPT_DIR/libmha_fwd.so" ]] || [[ ! -f "$SCRIPT_DIR/libmha_bwd.so" ]]; then
    echo "error: missing libmha_fwd.so or libmha_bwd.so in $SCRIPT_DIR — run: bash build_mha.sh" >&2
    exit 1
fi

if [[ -x "$SCRIPT_DIR/fwd.exe" ]]; then
    FWD_EXE="$SCRIPT_DIR/fwd.exe"
elif [[ -x "$SCRIPT_DIR/benchmark_mha_fwd" ]]; then
    FWD_EXE="$SCRIPT_DIR/benchmark_mha_fwd"
else
    echo "error: need fwd.exe or benchmark_mha_fwd in $SCRIPT_DIR" >&2
    exit 1
fi

if [[ -x "$SCRIPT_DIR/bwd.exe" ]]; then
    BWD_EXE="$SCRIPT_DIR/bwd.exe"
elif [[ -x "$SCRIPT_DIR/benchmark_mha_bwd" ]]; then
    BWD_EXE="$SCRIPT_DIR/benchmark_mha_bwd"
else
    echo "error: need bwd.exe or benchmark_mha_bwd in $SCRIPT_DIR" >&2
    exit 1
fi

echo "AITER_ASM_DIR=$AITER_ASM_DIR"

for prec in fp16 bf16; do
    for b in 2048 4096; do
        for d in 64 128 256; do
            for s in 16 17; do
                echo "======== prec=${prec} b=${b} d=${d} seqlen=${s} (v2) ========"

                "$FWD_EXE" \
                    "-prec=$prec" "-b=$b" -h=64 "-d=$d" "-s=$s" "-s_k=$s" \
                    -iperm=1 -operm=1 -mask=1 -lse=1 \
                    -fwd_v3=0 -mode=0 -warmup=5 -repeat=25 \
                    -kname=1 -v=0

                "$BWD_EXE" \
                    "-prec=$prec" "-b=$b" -h=64 "-d=$d" "-s=$s" "-s_k=$s" \
                    -iperm=1 -operm=1 -mask=1 \
                    -bwd_v3=0 -mode=0 -warmup=5 -repeat=25 \
                    -kname=1 -v=0
            done
        done
    done
done
