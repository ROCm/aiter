#!/bin/bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Description:
#   This script runs either the operator test or the tuning script for each
#   specified job, based on the input argument ("test" or "tune").
#
# Usage:
#   bash op_tune.sh test   # Run operator tests
#   bash op_tune.sh tune   # Run tuning scripts
# -----------------------------------------------------------------------------

if [ $# -ne 1 ]; then
    echo "Usage: $0 [test|tune]"
    exit 1
fi

mode="$1"

declare -a tune_jobs=(
  "csrc/ck_batched_gemm_a8w8:op_tests/test_batched_gemm_a8w8.py:python3 csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py -i aiter/configs/a8w8_untuned_batched_gemm.csv -o aiter/configs/a8w8_tuned_batched_gemm.csv"
  "csrc/ck_batched_gemm_bf16:op_tests/test_batched_gemm_bf16.py:python3 csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.py -i aiter/configs/bf16_untuned_batched_gemm.csv -o aiter/configs/bf16_tuned_batched_gemm.csv"
  "csrc/ck_gemm_a4w4_blockscale:op_tests/test_gemm_a4w4_blockscale.py:python3 csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py -i aiter/configs/a4w4_blockscale_untuned_gemm.csv -o aiter/configs/a4w4_blockscale_tuned_gemm.csv"
  "csrc/ck_gemm_a8w8:op_tests/test_gemm_a8w8.py:python3 csrc/ck_gemm_a8w8/gemm_a8w8_tune.py -i aiter/configs/a8w8_untuned_gemm.csv -o aiter/configs/a8w8_tuned_gemm.csv"
  "csrc/ck_gemm_a8w8_blockscale:op_tests/test_gemm_a8w8_blockscale.py:python3 csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py -i aiter/configs/a8w8_blockscale_untuned_gemm.csv -o aiter/configs/a8w8_blockscale_tuned_gemm.csv"
  "csrc/ck_gemm_a8w8_blockscale_bpreshuffle:op_tests/test_gemm_a8w8_blockscale_bpreshuffle.py:python3 csrc/ck_gemm_a8w8_blockscale_bpreshuffle/gemm_a8w8_blockscale_bpreshuffle_tune.py -i aiter/configs/a8w8_blockscale_bpreshuffle_untuned_gemm.csv -o aiter/configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv"
  "csrc/ck_gemm_a8w8_bpreshuffle:op_tests/test_gemm_a8w8.py:python3 csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py -i aiter/configs/a8w8_bpreshuffle_untuned_gemm.csv -o aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv"
)

for job in "${tune_jobs[@]}"; do
    IFS=':' read -r dir test_path tune_cmd <<< "$job"
    echo "============================================================"
    echo "🧪 Processing: $dir"
    echo "------------------------------------------------------------"
    if [ "$mode" == "test" ]; then
        echo "Running operator test: python3 $test_path"
        if python3 "$test_path"; then
            echo "✅ Test PASSED: $test_path"
        else
            echo "❌ Test FAILED: $test_path"
        fi
    elif [ "$mode" == "tune" ]; then
        echo "Running tuning script: $tune_cmd"
        if eval "$tune_cmd"; then
            echo "✅ Tuning PASSED: $tune_cmd"
        else
            echo "❌ Tuning FAILED: $tune_cmd"
        fi
    else
        echo "Unknown mode: $mode"
        exit 1
    fi
    echo "============================================================"
    echo
done