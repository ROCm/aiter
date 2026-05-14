#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Cross-attention shape: s_q=1, s_kv=P for each sweep step P in [KV_MIN, KV_MAX] (default 2–16).
# CK I/O layout: BHSD (-iperm=1 -operm=1), not THD — TE-style THD packed layouts are a different path.
# Batch mode (mode=0): same P for every batch element — comparable forward + backward in this harness.
#
# Batch sweep: space-separated CROSS_ATTN_BATCHES (defaults on the assignment line below). CK + JAX per value.
#
# Optional forward-only group varlen (padding): set CROSS_ATTN_GROUP_FWD=1
#   Uses -mode=1 with per-batch logical s_k in [2,P] and s_kpad=P (Python-generated comma lists).
#   Backward still uses uniform batch mode for the same P (see markdown note).
#
# Optional: SKIP_JAX=1 to skip unfused JAX timings (otherwise needs jax+jaxlib on GPU).
#
# Output: mha_performance_comparison_cross_attn.md (one section per configured sweep; separate forward / backward tables)
#
#   cd op_tests/cpp/mha && ./run_mha_performance_comparison_cross_attn.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export LD_LIBRARY_PATH="$SCRIPT_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export AITER_ASM_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)/hsa/"

PYTHON="${PYTHON:-python3}"
EXT="$SCRIPT_DIR/extract_mha_timing.py"
KV_MIN="${KV_MIN:-2}"
KV_MAX="${KV_MAX:-16}"
# Edit list after :- to choose which runs to sweep (order preserved in markdown § Configuration n).
CROSS_ATTN_BATCHES="${CROSS_ATTN_BATCHES:-2048 4096}"
OUT_MD="${OUT_MD:-$SCRIPT_DIR/mha_performance_comparison_cross_attn.md}"
CROSS_ATTN_GROUP_FWD="${CROSS_ATTN_GROUP_FWD:-0}"

# JAX unfused (no CLI): edit these; run_jax_unfused_cross_attn_benchmark.py reads JAX_UNFUSED_* only.
# JAX_UNFUSED_BATCH is set per batch inside the script loop (do not rely on a single B here).
JAX_UNFUSED_NHEADS="${JAX_UNFUSED_NHEADS:-32}"
JAX_UNFUSED_HDIM="${JAX_UNFUSED_HDIM:-128}"
JAX_UNFUSED_KV_MIN="${JAX_UNFUSED_KV_MIN:-$KV_MIN}"
JAX_UNFUSED_KV_MAX="${JAX_UNFUSED_KV_MAX:-$KV_MAX}"
JAX_UNFUSED_WARMUP="${JAX_UNFUSED_WARMUP:-5}"
JAX_UNFUSED_REPEAT="${JAX_UNFUSED_REPEAT:-25}"
JAX_UNFUSED_SM_SCALE="${JAX_UNFUSED_SM_SCALE:-ck}"
JAX_UNFUSED_NR_SEGMENTS="${JAX_UNFUSED_NR_SEGMENTS:-1}"
JAX_UNFUSED_LAYOUT="${JAX_UNFUSED_LAYOUT:-bshd}"

TIMINGS_CSV=$(mktemp)
JAX_CSV=$(mktemp)
trap 'rm -f "$TIMINGS_CSV" "$JAX_CSV"' EXIT

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

echo "batch,s_kv,fwd_ms,bwd_ms,group_fwd" > "$TIMINGS_CSV"
echo "AITER_ASM_DIR=$AITER_ASM_DIR"

for B in $CROSS_ATTN_BATCHES; do
    for ((P = KV_MIN; P <= KV_MAX; P++)); do
        echo "======== cross-attn ck_pr_6764: h=32 d=128 bf16 s_q=1 s_kv=${P} (v2) ========"

        if [[ "$CROSS_ATTN_GROUP_FWD" == "1" ]]; then
            LISTS=$(mktemp)
            "$PYTHON" - "$B" "$P" "$LISTS" <<'PY'
import random
import sys

b, p, path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
rng = random.Random(6764)
sq = [1] * b
sk = [rng.randint(2, p) for _ in range(b)]
skpad = [p] * b
with open(path, "w", encoding="utf-8") as f:
    f.write(",".join(map(str, sq)) + "\n")
    f.write(",".join(map(str, sk)) + "\n")
    f.write(",".join(map(str, skpad)) + "\n")
PY
            S_Q_CSV=$(sed -n '1p' "$LISTS")
            S_K_CSV=$(sed -n '2p' "$LISTS")
            S_KPAD_CSV=$(sed -n '3p' "$LISTS")
            rm -f "$LISTS"
            fwd_log=$(mktemp)
            "$FWD_EXE" \
                -prec=bf16 "-b=$B" -h=32 -d=128 \
                "-s=$S_Q_CSV" "-s_k=$S_K_CSV" "-s_kpad=$S_KPAD_CSV" \
                -iperm=1 -operm=1 -mask=0 -lse=1 \
                -fwd_v3=0 -mode=1 -warmup=5 -repeat=25 \
                -kname=1 -v=0 2>&1 | tee "$fwd_log"
            fwd_ms=$("$PYTHON" "$EXT" fwd <"$fwd_log")
            rm -f "$fwd_log"
            gf=1
        else
            fwd_log=$(mktemp)
            "$FWD_EXE" \
                -prec=bf16 "-b=$B" -h=32 -d=128 -s=1 "-s_k=$P" \
                -iperm=1 -operm=1 -mask=0 -lse=1 \
                -fwd_v3=0 -mode=0 -warmup=5 -repeat=25 \
                -kname=1 -v=0 2>&1 | tee "$fwd_log"
            fwd_ms=$("$PYTHON" "$EXT" fwd <"$fwd_log")
            rm -f "$fwd_log"
            gf=0
        fi

        bwd_log=$(mktemp)
        "$BWD_EXE" \
            -prec=bf16 "-b=$B" -h=32 -d=128 -s=1 "-s_k=$P" \
            -iperm=1 -operm=1 -mask=0 \
            -bwd_v3=0 -mode=0 -warmup=5 -repeat=25 \
            -kname=1 -v=0 2>&1 | tee "$bwd_log"
        bwd_ms=$("$PYTHON" "$EXT" bwd <"$bwd_log")
        rm -f "$bwd_log"

        echo "${B},${P},${fwd_ms},${bwd_ms},${gf}" >> "$TIMINGS_CSV"
    done
done

JAX_ARGS=()
if [[ "${SKIP_JAX:-0}" != "1" ]]; then
    : >"$JAX_CSV"
    jax_any=0
    export JAX_UNFUSED_NHEADS JAX_UNFUSED_HDIM JAX_UNFUSED_KV_MIN JAX_UNFUSED_KV_MAX
    export JAX_UNFUSED_WARMUP JAX_UNFUSED_REPEAT JAX_UNFUSED_SM_SCALE JAX_UNFUSED_NR_SEGMENTS JAX_UNFUSED_LAYOUT
    for B in $CROSS_ATTN_BATCHES; do
        JAX_CHUNK=$(mktemp)
        export JAX_UNFUSED_OUT="$JAX_CHUNK"
        export JAX_UNFUSED_BATCH="$B"
        if "$PYTHON" "$SCRIPT_DIR/run_jax_unfused_cross_attn_benchmark.py"; then
            if [[ "$jax_any" -eq 0 ]]; then
                cat "$JAX_CHUNK" >>"$JAX_CSV"
                jax_any=1
            else
                tail -n +2 "$JAX_CHUNK" >>"$JAX_CSV"
            fi
        fi
        rm -f "$JAX_CHUNK"
    done
    if [[ "$jax_any" -eq 1 ]]; then
        JAX_ARGS=(--jax-timings "$JAX_CSV")
    else
        echo "warning: JAX unfused benchmark skipped (install jax+jaxlib for your GPU, or set SKIP_JAX=1)" >&2
    fi
fi

"$PYTHON" "$SCRIPT_DIR/write_mha_performance_comparison_md.py" \
    --kind cross_attn \
    --timings "$TIMINGS_CSV" \
    "${JAX_ARGS[@]}" \
    --out "$OUT_MD"

echo "wrote $OUT_MD"
