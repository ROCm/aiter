#!/usr/bin/env bash
# Regression test for unified-attention decode shapes.
#
# Sweeps the 4 (d, dtype) combos for a fixed decode shape
# (b=128, sq=1, hq=64, hk=8) using `test_single_shape.py --test`
# (which also verifies correctness against the Triton reference).
# Each combo is run N times and the median CK bandwidth / time is
# reported; per-run numbers are kept so any fluctuation from
# colleagues sharing the GPU is visible.
#
# Usage:
#     ./regression_decode.sh                   # defaults: sk=8192, 5 runs, GPU 2
#     SK=16384 NUM_RUNS=7 GPU=3 ./regression_decode.sh
set -euo pipefail

SK="${SK:-16384}"
NUM_RUNS="${NUM_RUNS:-5}"
GPU="${GPU:-2}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"

export HIP_VISIBLE_DEVICES="$GPU"
cd "$(dirname "$0")/.."

# Common decode-shape args
COMMON=(-b 128 -sq 1 -hq 64 -hk 8 -sk "$SK" --block-size "$BLOCK_SIZE" --test)

# (head_size, dtype) combos to sweep
CONFIGS=(
    "64  bf16"
    "64  fp8"
    "128 bf16"
    "128 fp8"
)

echo "============================================================"
echo " UA decode regression  —  GPU=$GPU  sk=$SK  block_size=$BLOCK_SIZE"
echo "                          shape: b=128 sq=1 hq=64 hk=8"
echo "                          $NUM_RUNS runs/config  (median reported)"
echo "============================================================"

# Per-run table header
printf '%-5s %-5s %-4s %-12s %-14s %-8s\n' \
    "d" "dtype" "run" "CK_time(ms)" "CK_BW(GB/s)" "correct"
echo "------------------------------------------------------"

# Hold per-run results for median computation
declare -A TIMES BWS CORRS

for cfg in "${CONFIGS[@]}"; do
    read -r d dt <<< "$cfg"
    for ((i=1; i<=NUM_RUNS; i++)); do
        out=$(python3 ua-test-scripts/test_single_shape.py "${COMMON[@]}" \
              -d "$d" --dtype "$dt" --seed 42 2>&1 || true)
        ck_time=$(echo "$out" | grep -E "^\s*CK time:" | tail -1 | awk '{print $3}')
        ck_bw=$(echo "$out" | grep -E "^\s*CK Bandwidth:"   | awk '{print $3}')
        # Correctness line is "  Correctness: ✓ PASS" or "  Correctness: ✗ FAIL"
        if echo "$out" | grep -qE "Correctness:.*PASS"; then
            corr="PASS"
        elif echo "$out" | grep -qE "Correctness:.*FAIL"; then
            corr="FAIL"
        else
            corr="???"
        fi
        printf '%-5s %-5s %-4d %-12s %-14s %-8s\n' \
            "$d" "$dt" "$i" "${ck_time:-—}" "${ck_bw:-—}" "$corr"
        TIMES["$d-$dt-$i"]="${ck_time:-0}"
        BWS["$d-$dt-$i"]="${ck_bw:-0}"
        CORRS["$d-$dt-$i"]="$corr"
    done
done

echo
echo "============================================================"
echo " Median per config"
echo "============================================================"
printf '%-5s %-5s %-14s %-15s %-10s\n' \
    "d" "dtype" "median time(ms)" "median BW(GB/s)" "all_pass"
echo "------------------------------------------------------"

for cfg in "${CONFIGS[@]}"; do
    read -r d dt <<< "$cfg"
    times=()
    bws=()
    fail=0
    for ((i=1; i<=NUM_RUNS; i++)); do
        times+=("${TIMES[$d-$dt-$i]}")
        bws+=("${BWS[$d-$dt-$i]}")
        [[ "${CORRS[$d-$dt-$i]}" != "PASS" ]] && fail=1
    done
    # Two-step join: `${arr[*]}` joins with IFS (space); then `// /,` replaces
    # spaces with commas in the resulting string. Doing it in one expansion
    # (`${arr[*]// /,}`) is wrong — bash applies the pattern per-element
    # *before* joining, so no spaces ever exist for it to match.
    times_str="${times[*]}"; times_csv="${times_str// /,}"
    bws_str="${bws[*]}";     bws_csv="${bws_str// /,}"
    med_time=$(python3 -c "import statistics; print(f'{statistics.median([${times_csv}]):.4f}')" \
               2>/dev/null || echo "—")
    med_bw=$(python3 -c "import statistics; print(f'{statistics.median([${bws_csv}]):.2f}')" \
               2>/dev/null || echo "—")
    status=$([[ $fail -eq 0 ]] && echo "PASS" || echo "FAIL")
    printf '%-5s %-5s %-14s %-15s %-10s\n' "$d" "$dt" "$med_time" "$med_bw" "$status"
done

echo "============================================================"
