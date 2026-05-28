#!/usr/bin/env bash
# Regression test for unified-attention decode shapes.
#
# Sweeps the 4 (d, dtype) combos at TWO batch sizes that exercise the
# *attention kernel only* (b=128, light split-KV) vs the *attention +
# combine* pipeline (b=4, heavy split-KV).
#
#   b=128 hq=64 hk=8 → base_ctas = 8 * q_tiles ≈ 512 → num_splits = 2.
#       Split-KV is on technically (2 splits), but the combine kernel
#       is essentially free at this CTA count, and per-CTA work is
#       large. Use this row to A/B the attention kernel itself.
#
#   b=4   hq=64 hk=8 → base_ctas = 8 * 1            → num_splits = 64.
#       Heavy split-KV: 64-way per-token KV partition + a fan-in
#       combine kernel for every (token, head). Use this row to catch
#       regressions that *only* show up under heavy split-KV, where
#       the combine launch + workspace traffic + LSE merge can
#       dominate the per-split attention work.
#
# The two rows are tagged with `splitkv=N` so you can see at a glance
# which is which, and a regression that *only* affects the b=4 column
# almost certainly lives in the combine path (Triton
# `reduce_segments_ck_layout`) or the wrapper's workspace alloc, not
# in the CK attention kernel proper.
#
# Per-config the script runs `test_unified_attention_ck.py` in
# single-shape mode N times and reports per-run + median CK time / BW.
# CK-only by default (`--no-triton`) so the sweep is fast and we're
# not coupling regression detection to Triton perf; flip with
# `WITH_TRITON=1`.
#
# Usage:
#     ./regression_decode.sh                 # defaults: sk=16384, 5 runs/cfg, GPU 2
#     SK=32768 NUM_RUNS=7 GPU=3 ./regression_decode.sh
#     WITH_TRITON=1 ./regression_decode.sh   # include Triton comparison

set -euo pipefail

SK="${SK:-16384}"
NUM_RUNS="${NUM_RUNS:-5}"
GPU="${GPU:-2}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
WITH_TRITON="${WITH_TRITON:-0}"

export HIP_VISIBLE_DEVICES="$GPU"
cd "$(dirname "$0")/.."

# Triton flag passed through to the canonical script. Default off in
# regression mode (we're catching CK-side regressions, Triton numbers
# would add noise and double the runtime).
TRITON_FLAG="--no-triton"
[[ "$WITH_TRITON" == "1" ]] && TRITON_FLAG="--triton"

# (head_size, dtype) combos to sweep — same as before.
CONFIGS=(
    "64  bf16"
    "64  fp8"
    "128 bf16"
    "128 fp8"
)

# Batch sizes to sweep. b=128 is the "attention kernel" row; b=4 is the
# "attention + combine" row (see header for why).
BATCH_LIST=(128 4)

echo "=================================================================="
echo " UA decode regression  —  GPU=$GPU  sk=$SK  block_size=$BLOCK_SIZE"
echo "                          hq=64 hk=8 sq=1"
echo "                          $NUM_RUNS runs/config  (median reported)"
echo "                          Triton comparison: $( [[ "$WITH_TRITON" == "1" ]] && echo on || echo off )"
echo "=================================================================="

# Per-run table header — `splitkv` column makes attention-only vs
# attention+combine results unambiguous at a glance.
printf '%-4s %-5s %-5s %-4s %-12s %-14s %-8s %-9s\n' \
    "b" "d" "dtype" "run" "CK_time(ms)" "CK_BW(GB/s)" "correct" "splitkv"
echo "-------------------------------------------------------------------------"

declare -A TIMES BWS CORRS SPLITS

for b in "${BATCH_LIST[@]}"; do
    for cfg in "${CONFIGS[@]}"; do
        read -r d dt <<< "$cfg"

        # fp8 path doesn't support block_size < 32; skip the (fp8, 16)
        # combination cleanly instead of letting the kernel print a
        # "skipped" row that confuses the median calc.
        if [[ "$dt" == "fp8" && "$BLOCK_SIZE" -lt 32 ]]; then
            printf '%-4s %-5s %-5s %-4s %-12s %-14s %-8s %-9s\n' \
                "$b" "$d" "$dt" "—" "—" "—" "SKIP" "—"
            continue
        fi

        for ((i=1; i<=NUM_RUNS; i++)); do
            out=$(python3 op_tests/test_unified_attention_ck.py \
                  -b "$b" -sq 1 -sk "$SK" \
                  --num-heads 64,8 \
                  --head-size "$d" \
                  --block-size "$BLOCK_SIZE" \
                  --dtype "$dt" \
                  --num-blocks auto \
                  $TRITON_FLAG \
                  --seed 42 2>&1 || true)

            ck_time=$(echo "$out" | grep -E "^\s*CK time" | tail -1 | awk -F= '{print $2}' | awk '{print $1}')
            ck_bw=$(echo "$out"   | grep -E "^\s*CK Bandwidth" | tail -1 | awk -F= '{print $2}' | awk '{print $1}')
            splits=$(echo "$out"  | grep -E "^\s*split-KV" | tail -1 | sed -nE 's/.*num_splits=([0-9]+).*/\1/p')
            # Use the kernel's own correctness verdict (CK vs ref).
            if   echo "$out" | grep -qE "CK     vs ref:\s*PASS"; then corr="PASS"
            elif echo "$out" | grep -qE "CK     vs ref:\s*FAIL"; then corr="FAIL"
            else                                                       corr="???"
            fi

            printf '%-4s %-5s %-5s %-4s %-12s %-14s %-8s %-9s\n' \
                "$b" "$d" "$dt" "$i" "${ck_time:-—}" "${ck_bw:-—}" "$corr" "${splits:-?}"

            TIMES["$b-$d-$dt-$i"]="${ck_time:-0}"
            BWS["$b-$d-$dt-$i"]="${ck_bw:-0}"
            CORRS["$b-$d-$dt-$i"]="$corr"
            SPLITS["$b-$d-$dt"]="${splits:-?}"
        done
    done
done

echo
echo "=================================================================="
echo " Median per config  (rows tagged 'splitkv=N'; N>1 ⇒ combine kernel"
echo "                     is on the critical path — interpret regressions"
echo "                     there in the context of split-KV, not just the"
echo "                     attention kernel)"
echo "=================================================================="
printf '%-4s %-5s %-5s %-14s %-15s %-10s %-9s\n' \
    "b" "d" "dtype" "median time(ms)" "median BW(GB/s)" "all_pass" "splitkv"
echo "-------------------------------------------------------------------------"

for b in "${BATCH_LIST[@]}"; do
    for cfg in "${CONFIGS[@]}"; do
        read -r d dt <<< "$cfg"
        if [[ "$dt" == "fp8" && "$BLOCK_SIZE" -lt 32 ]]; then continue; fi

        times=()
        bws=()
        fail=0
        for ((i=1; i<=NUM_RUNS; i++)); do
            times+=("${TIMES[$b-$d-$dt-$i]:-0}")
            bws+=("${BWS[$b-$d-$dt-$i]:-0}")
            [[ "${CORRS[$b-$d-$dt-$i]:-}" != "PASS" ]] && fail=1
        done
        # Two-step join: ${arr[*]} joins by IFS (space); the substitution
        # then swaps spaces → commas. Doing it in one expansion is wrong
        # (bash applies the pattern per-element before joining, so no
        # spaces ever exist for it to match).
        times_str="${times[*]}"; times_csv="${times_str// /,}"
        bws_str="${bws[*]}";     bws_csv="${bws_str// /,}"
        med_time=$(python3 -c "import statistics; print(f'{statistics.median([${times_csv}]):.4f}')" \
                   2>/dev/null || echo "—")
        med_bw=$(python3 -c "import statistics; print(f'{statistics.median([${bws_csv}]):.2f}')" \
                 2>/dev/null || echo "—")
        status=$([[ $fail -eq 0 ]] && echo "PASS" || echo "FAIL")
        splits="${SPLITS[$b-$d-$dt]:-?}"
        splitkv_tag="splitkv=$splits"
        [[ "$splits" != "1" && "$splits" != "?" ]] && splitkv_tag="$splitkv_tag*"
        printf '%-4s %-5s %-5s %-14s %-15s %-10s %-9s\n' \
            "$b" "$d" "$dt" "$med_time" "$med_bw" "$status" "$splitkv_tag"
    done
done

echo "=================================================================="
echo " * suffix on splitkv= means split-KV is active (num_splits > 1),"
echo "   so the row's time/BW includes combine-kernel + workspace cost."
echo "=================================================================="
