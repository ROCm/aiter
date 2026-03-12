#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 [test|tune] [shape_name (optional)] [tuning_arg (optional)]"
    exit 1
fi

mode="$1"
shape_filter="${2:-}"
tuning_arg="${3:-}"

tuneFailed=false
testFailed=false
tuneFailedCmds=()
testFailedFiles=()

declare -a tune_jobs=(
  "ck_batched_gemm_a8w8:csrc/ck_batched_gemm_a8w8:op_tests/test_batched_gemm_a8w8.py:python3 csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py -i aiter/configs/a8w8_untuned_batched_gemm.csv -o aiter/configs/a8w8_tuned_batched_gemm.csv"
  "ck_batched_gemm_bf16:csrc/ck_batched_gemm_bf16:op_tests/test_batched_gemm_bf16.py:python3 csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.py -i aiter/configs/bf16_untuned_batched_gemm.csv -o aiter/configs/bf16_tuned_batched_gemm.csv"
  "ck_gemm_a8w8:csrc/ck_gemm_a8w8:op_tests/test_gemm_a8w8.py:python3 csrc/ck_gemm_a8w8/gemm_a8w8_tune.py -i aiter/configs/a8w8_untuned_gemm.csv -o aiter/configs/a8w8_tuned_gemm.csv"
  "ck_gemm_a8w8_blockscale:csrc/ck_gemm_a8w8_blockscale:op_tests/test_gemm_a8w8_blockscale.py:python3 csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py -i aiter/configs/a8w8_blockscale_untuned_gemm.csv -o aiter/configs/a8w8_blockscale_tuned_gemm.csv"
  "ck_gemm_a8w8_blockscale_bpreshuffle:csrc/ck_gemm_a8w8_blockscale_bpreshuffle:op_tests/test_gemm_a8w8_blockscale.py:python3 csrc/ck_gemm_a8w8_blockscale_bpreshuffle/gemm_a8w8_blockscale_bpreshuffle_tune.py -i aiter/configs/a8w8_blockscale_bpreshuffle_untuned_gemm.csv -o aiter/configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv"
  "ck_gemm_a8w8_bpreshuffle:csrc/ck_gemm_a8w8_bpreshuffle:op_tests/test_gemm_a8w8.py:python3 csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py -i aiter/configs/a8w8_bpreshuffle_untuned_gemm.csv -o aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv"
  "ck_gemm_moe_2stages_codegen:csrc/ck_gemm_moe_2stages_codegen:op_tests/test_moe.py:python3 csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py -i aiter/configs/untuned_fmoe.csv -o aiter/configs/tuned_fmoe.csv"
  #"ck_gemm_a4w4_blockscale:csrc/ck_gemm_a4w4_blockscale:op_tests/test_gemm_a4w4_blockscale.py:python3 csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py -i aiter/configs/a4w4_blockscale_untuned_gemm.csv -o aiter/configs/a4w4_blockscale_tuned_gemm.csv"
)

for job in "${tune_jobs[@]}"; do
    IFS=':' read -r shape dir test_path tune_cmd <<< "$job"
    # If shape_filter is not empty, check if the current shape exists in the filter list.
    # shape_filter is a comma-separated list, e.g. "ck_gemm_a8w8,ck_batched_gemm_a8w8"
    if [ -n "$shape_filter" ]; then
        # Remove all whitespace from the shape_filter string
        shape_filter_no_space="${shape_filter//[[:space:]]/}"
        IFS=',' read -ra filter_shapes <<< "$shape_filter_no_space"
        found_match=false
        for filter_shape in "${filter_shapes[@]}"; do
            if [[ "$shape" == "$filter_shape" ]]; then
                found_match=true
                break
            fi
        done
        if [ "$found_match" = false ]; then
            continue
        fi
    fi
    echo "============================================================"
    echo "🧪 Processing shape: $shape under directory: $dir"
    echo "------------------------------------------------------------"
    if [ "$mode" == "test" ]; then
        echo "Running operator test: python3 $test_path"
        if python3 "$test_path"; then
            echo "✅ Test PASSED: $test_path"
        else
            echo "❌ Test FAILED: $test_path"
            testFailed=true
            testFailedFiles+=("$test_path")
        fi
    elif [ "$mode" == "tune" ]; then
        # Append tuning_arg if provided
        if [ -n "$tuning_arg" ]; then
            full_tune_cmd="$tune_cmd $tuning_arg"
        else
            full_tune_cmd="$tune_cmd"
        fi
        echo "Running tuning script: $full_tune_cmd"
        if eval "$full_tune_cmd"; then
            echo "✅ Tuning PASSED: $full_tune_cmd"
        else
            echo "❌ Tuning FAILED: $full_tune_cmd"
            tuneFailed=true
            tuneFailedCmds+=("$full_tune_cmd")
        fi
    else
        echo "Unknown mode: $mode"
        exit 1
    fi
    echo "==============================================="
    echo
done

if [ "$tuneFailed" = true ]; then
    echo "Failed tune commands:"
    for c in "${tuneFailedCmds[@]}"; do
        echo "  $c"
    done
    exit 1
elif [ "$testFailed" = true ]; then
    echo "Failed test files:"
    for f in "${testFailedFiles[@]}"; do
        echo "  $f"
    done
    exit 1
else
    echo "All tunes/tests passed." 
    exit 0
fi
