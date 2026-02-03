#!/bin/bash


rm -fr ~/.cache/flydsl/

echo "TEST Base: ASM mode"
python op_tests/test_moe_ep.py -t g1u1_int8smoothquant -m 10240

echo "TEST 1: ATOMIC mode"
export AITER_USE_FLYDSL_MOE=1
export AITER_FLYDSL_USE_GEMM2_EX=1
export AITER_FLYDSL_GEMM2_MODE=ATOMIC
python op_tests/test_moe_ep.py -t g1u1_int8smoothquant -m 10240

echo "TEST 2: REDUCE mode"
export AITER_USE_FLYDSL_MOE=1
export AITER_FLYDSL_USE_GEMM2_EX=1
export AITER_FLYDSL_GEMM2_MODE=REDUCE
python op_tests/test_moe_ep.py -t g1u1_int8smoothquant -m 10240
