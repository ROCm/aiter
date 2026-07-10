#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# WP-G1 Phase B: Tune FlyDSL preshuffle GEMM across standard shapes.
#
# Usage:
#   bash scripts/wp_g1_tune_flydsl.sh [--mp NUM_PROCS] [--iters NUM]
#
# This runs the existing bpreshuffle tuner with --libtype flydsl
# on the standard untuned shape set, then merges results into the
# tuned CSV. Compare against CK with wp_g1_bench_flydsl_vs_ck.py.

set -euo pipefail
cd "$(dirname "$0")/.."

MP="${MP:-1}"
ITERS="${ITERS:-101}"
WARMUP="${WARMUP:-10}"

TUNE_SCRIPT="csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py"
UNTUNE_CSV="aiter/configs/a8w8_bpreshuffle_untuned_gemm.csv"
OUTPUT_CSV="aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv"

echo "=== WP-G1 FlyDSL Tuning ==="
echo "Tuner:    $TUNE_SCRIPT"
echo "Shapes:   $UNTUNE_CSV"
echo "Output:   $OUTPUT_CSV"
echo "MP:       $MP"
echo "Iters:    $ITERS"
echo ""

python "$TUNE_SCRIPT" \
    --libtype flydsl \
    --mp "$MP" \
    --iters "$ITERS" \
    --warmup "$WARMUP" \
    --untune_file "$UNTUNE_CSV" \
    --tune_file "$OUTPUT_CSV" \
    "$@"

echo ""
echo "=== Tuning complete ==="
echo "Results in: $OUTPUT_CSV"
echo "Run 'python scripts/wp_g1_bench_flydsl_vs_ck.py' to compare against CK."
