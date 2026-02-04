#!/bin/bash

# Default token value
TOKEN=432

# Parse command-line arguments
while getopts "t:" opt; do
  case $opt in
    t)
      TOKEN=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      echo "Usage: $0 [-t token_value]"
      exit 1
      ;;
  esac
done

rm -fr ~/.cache/flydsl/

echo "TEST Base: ASM mode"
python op_tests/test_moe_ep.py -t g1u1_int8smoothquant -m $TOKEN

echo "TEST 1: ATOMIC mode"
export AITER_USE_FLYDSL_MOE=1
export AITER_FLYDSL_USE_GEMM2_EX=1
export AITER_FLYDSL_GEMM2_MODE=ATOMIC
python op_tests/test_moe_ep.py -t g1u1_int8smoothquant -m $TOKEN

echo "TEST 2: REDUCE mode"
export AITER_USE_FLYDSL_MOE=1
export AITER_FLYDSL_USE_GEMM2_EX=1
export AITER_FLYDSL_GEMM2_MODE=REDUCE
AITER_LOG_MORE=1 python op_tests/test_moe_ep.py -t g1u1_int8smoothquant -m $TOKEN
