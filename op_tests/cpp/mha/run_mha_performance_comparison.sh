#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Single entry: same problem shape as the published MFMA vs JAX vs TE/CK table —
#   bs=2048, nheads=32, hdim=128, bf16, causal=False (mask=0), seqlen_q == seqlen_kv.
# Runs asm v2 (-fwd_v3=0 -bwd_v3=0), seq SEQ_MIN..SEQ_MAX (default 1–17).
#
# Output: mha_performance_comparison.md (only persistent artifact from this script).
#
#   cd op_tests/cpp/mha && ./run_mha_performance_comparison.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export LD_LIBRARY_PATH="$SCRIPT_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export AITER_ASM_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)/hsa/"

PYTHON="${PYTHON:-python3}"
EXT="$SCRIPT_DIR/extract_mha_timing.py"
SEQ_MIN="${SEQ_MIN:-1}"
SEQ_MAX="${SEQ_MAX:-17}"
OUT_MD="${OUT_MD:-$SCRIPT_DIR/mha_performance_comparison.md}"

TIMINGS_CSV=$(mktemp)
trap 'rm -f "$TIMINGS_CSV"' EXIT

if [[ ! -f "$SCRIPT_DIR/libmha_fwd.so" ]] || [[ ! -f "$SCRIPT_DIR/libmha_bwd.so" ]]; then
    echo "error: missing libmha_fwd.so or libmha_bwd.so — run: bash build_mha.sh" >&2
    exit 1
fi

if [[ -x "$SCRIPT_DIR/fwd.exe" ]]; then
    FWD_EXE="$SCRIPT_DIR/fwd.exe"
elif [[ -x "$SCRIPT_DIR/benchmark_mha_fwd" ]]; then
    FWD_EXE="$SCRIPT_DIR/benchmark_mha_fwd"
else
    echo "error: need fwd.exe or benchmark_mha_fwd" >&2
    exit 1
fi

if [[ -x "$SCRIPT_DIR/bwd.exe" ]]; then
    BWD_EXE="$SCRIPT_DIR/bwd.exe"
elif [[ -x "$SCRIPT_DIR/benchmark_mha_bwd" ]]; then
    BWD_EXE="$SCRIPT_DIR/benchmark_mha_bwd"
else
    echo "error: need bwd.exe or benchmark_mha_bwd" >&2
    exit 1
fi

echo "seq,fwd_ms,bwd_ms" > "$TIMINGS_CSV"
echo "AITER_ASM_DIR=$AITER_ASM_DIR"

for ((s = SEQ_MIN; s <= SEQ_MAX; s++)); do
    echo "======== ck_pr_6764 bench: b=2048 h=32 d=128 bf16 s=${s} (v2) ========"

    fwd_out=$(
        "$FWD_EXE" \
            -prec=bf16 -b=2048 -h=32 -d=128 "-s=$s" "-s_k=$s" \
            -iperm=1 -operm=1 -mask=0 -lse=1 \
            -fwd_v3=0 -mode=0 -warmup=5 -repeat=25 \
            -kname=1 -v=0 2>&1
    )
    printf '%s\n' "$fwd_out"
    fwd_ms=$(printf '%s\n' "$fwd_out" | "$PYTHON" "$EXT" fwd)

    bwd_out=$(
        "$BWD_EXE" \
            -prec=bf16 -b=2048 -h=32 -d=128 "-s=$s" "-s_k=$s" \
            -iperm=1 -operm=1 -mask=0 \
            -bwd_v3=0 -mode=0 -warmup=5 -repeat=25 \
            -kname=1 -v=0 2>&1
    )
    printf '%s\n' "$bwd_out"
    bwd_ms=$(printf '%s\n' "$bwd_out" | "$PYTHON" "$EXT" bwd)

    echo "${s},${fwd_ms},${bwd_ms}" >> "$TIMINGS_CSV"
done

"$PYTHON" "$SCRIPT_DIR/write_mha_performance_comparison_md.py" \
    --timings "$TIMINGS_CSV" \
    --out "$OUT_MD"

echo "wrote $OUT_MD"
