#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.
# Shared helpers: CK binaries in parent mha/; CSV under results/scenario{N}/.

set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MHA_DIR="$(cd "$BENCH_DIR/.." && pwd)"

export LD_LIBRARY_PATH="$MHA_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export AITER_ASM_DIR="$(cd "$MHA_DIR/../../.." && pwd)/hsa/"

PYTHON="${PYTHON:-python3}"
EXT="$MHA_DIR/extract_mha_timing.py"
GEN="$BENCH_DIR/gen_small_attn_lengths.py"
RESULTS_DIR="${RESULTS_DIR:-$BENCH_DIR/results}"

NHEADS="${NHEADS:-32}"
HDIM="${HDIM:-128}"
WARMUP="${WARMUP:-5}"
REPEAT="${REPEAT:-25}"
BATCHES="${BATCHES:-2048 4096}"
PAD_MIN="${PAD_MIN:-2}"
PAD_MAX="${PAD_MAX:-16}"
SEQ_MIN="${SEQ_MIN:-2}"
SEQ_MAX="${SEQ_MAX:-17}"

bench_check_build() {
    if [[ ! -f "$MHA_DIR/libmha_fwd.so" ]] || [[ ! -f "$MHA_DIR/libmha_bwd.so" ]]; then
        echo "error: run: cd $MHA_DIR && bash build_mha.sh" >&2
        exit 1
    fi
    if [[ -x "$MHA_DIR/fwd.exe" ]]; then
        FWD_EXE="$MHA_DIR/fwd.exe"
    elif [[ -x "$MHA_DIR/benchmark_mha_fwd" ]]; then
        FWD_EXE="$MHA_DIR/benchmark_mha_fwd"
    else
        echo "error: need fwd.exe in $MHA_DIR (bash build_mha.sh)" >&2
        exit 1
    fi
    if [[ -x "$MHA_DIR/bwd.exe" ]]; then
        BWD_EXE="$MHA_DIR/bwd.exe"
    elif [[ -x "$MHA_DIR/benchmark_mha_bwd" ]]; then
        BWD_EXE="$MHA_DIR/benchmark_mha_bwd"
    else
        echo "error: need bwd.exe in $MHA_DIR" >&2
        exit 1
    fi
}

bench_scenario_dir() {
    # Argument is a slug, e.g. 1, 2, or 3_4 → results/scenario1, results/scenario3_4, ...
    echo "$RESULTS_DIR/scenario${1}"
}

bench_csv_path() {
    local scenario="$1" kind="$2"
    echo "$(bench_scenario_dir "$scenario")/${kind}.csv"
}

bench_csv_header() {
    local csv="$1"
    mkdir -p "$(dirname "$csv")"
    echo "batch,s_q,s_kv,ck_pr_6764(ms)" >"$csv"
}

bench_ck_ms() {
    local kind="$1"
    shift
    local log ms
    log=$(mktemp)
    if ! "$@" >"$log" 2>&1; then
        echo "error: CK $kind benchmark failed" >&2
        tail -20 "$log" >&2
        rm -f "$log"
        return 1
    fi
    ms=$("$PYTHON" "$EXT" "$kind" <"$log") || {
        echo "error: could not parse CK $kind timing" >&2
        tail -5 "$log" >&2
        rm -f "$log"
        return 1
    }
    rm -f "$log"
    echo "$ms"
}

bench_ck_fwd_ms() {
    bench_ck_ms fwd "$FWD_EXE" "$@"
}

bench_ck_bwd_ms() {
    bench_ck_ms bwd "$BWD_EXE" "$@"
}

bench_append_row() {
    printf '%s\n' "$2" >>"$1"
}

# Scenarios 1–2: fwd = group packed varlen; bwd = batch uniform (benchmark_mha_bwd has no per-batch -s lists).
bench_run_varlen_scenario() {
    local scenario="$1"
    local fwd_csv bwd_csv
    bench_check_build
    fwd_csv=$(bench_csv_path "$scenario" "fwd")
    bwd_csv=$(bench_csv_path "$scenario" "bwd")
    bench_csv_header "$fwd_csv"
    bench_csv_header "$bwd_csv"
    for B in $BATCHES; do
        for ((P = PAD_MIN; P <= PAD_MAX; P++)); do
            echo "======== scenario $scenario B=$B maxlen=$P ========" >&2
            LISTS=$(mktemp)
            "$PYTHON" "$GEN" "$scenario" "$B" "$P" "$LISTS"
            S_Q=$(sed -n '1p' "$LISTS")
            S_K=$(sed -n '2p' "$LISTS")
            rm -f "$LISTS"

            if [[ "$scenario" -eq 1 ]]; then
                sq_col=$P
                sk_col=$P
            else
                sq_col=1
                sk_col=$P
            fi

            ck_fwd=$(bench_ck_fwd_ms \
                -prec=bf16 "-b=$B" "-h=$NHEADS" "-d=$HDIM" \
                "-s=$S_Q" "-s_k=$S_K" \
                -iperm=1 -operm=1 -mask=0 -lse=1 \
                -mode=1 -fwd_v3=0 "-warmup=$WARMUP" "-repeat=$REPEAT" -v=0)

            ck_bwd=$(bench_ck_bwd_ms \
                -prec=bf16 "-b=$B" "-h=$NHEADS" "-d=$HDIM" \
                "-s=$sq_col" "-s_k=$sk_col" \
                -iperm=1 -operm=1 -mask=0 \
                -bwd_v3=0 -mode=0 "-warmup=$WARMUP" "-repeat=$REPEAT" -v=0)

            bench_append_row "$fwd_csv" "${B},${sq_col},${sk_col},${ck_fwd}"
            bench_append_row "$bwd_csv" "${B},${sq_col},${sk_col},${ck_bwd}"
        done
    done
    echo "wrote $fwd_csv" >&2
    echo "wrote $bwd_csv" >&2
}

# Scenario 3+4: fixed self-attn, batch mode, sq=skv sweep SEQ_MIN..SEQ_MAX (default 2..17).
bench_run_scenario_3_4() {
    local scenario=3_4
    local fwd_csv bwd_csv
    bench_check_build
    fwd_csv=$(bench_csv_path "$scenario" "fwd")
    bwd_csv=$(bench_csv_path "$scenario" "bwd")
    bench_csv_header "$fwd_csv"
    bench_csv_header "$bwd_csv"
    for B in $BATCHES; do
        for ((S = SEQ_MIN; S <= SEQ_MAX; S++)); do
            echo "======== scenario 3_4 B=$B s_q=s_kv=$S (fixed self-attn) ========" >&2

            ck_fwd=$(bench_ck_fwd_ms \
                -prec=bf16 "-b=$B" "-h=$NHEADS" "-d=$HDIM" \
                "-s=$S" "-s_k=$S" \
                -iperm=1 -operm=1 -mask=0 -lse=1 \
                -mode=0 -fwd_v3=0 "-warmup=$WARMUP" "-repeat=$REPEAT" -v=0)

            ck_bwd=$(bench_ck_bwd_ms \
                -prec=bf16 "-b=$B" "-h=$NHEADS" "-d=$HDIM" \
                "-s=$S" "-s_k=$S" \
                -iperm=1 -operm=1 -mask=0 \
                -bwd_v3=0 -mode=0 "-warmup=$WARMUP" "-repeat=$REPEAT" -v=0)

            bench_append_row "$fwd_csv" "${B},${S},${S},${ck_fwd}"
            bench_append_row "$bwd_csv" "${B},${S},${S},${ck_bwd}"
        done
    done
    echo "wrote $fwd_csv" >&2
    echo "wrote $bwd_csv" >&2
}

bench_run_scenario() {
    case "$1" in
        1 | 2) bench_run_varlen_scenario "$1" ;;
        *) echo "error: unknown scenario $1 (fixed self-attn: use ./scenario_3_4.sh)" >&2; return 1 ;;
    esac
}
