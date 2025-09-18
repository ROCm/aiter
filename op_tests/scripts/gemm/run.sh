#!/bin/bash

export GEMM_A4W4_BLOCKWISE_HIP_CLANG_PATH=/home/jun_chen2_qle/llvm-project/build/bin/

AITER_HOME=$(pip show aiter | grep Location | awk '{ print $2 }')

rm gemm_*.log

# gemm bf16
python3 ${AITER_HOME}/op_tests/test_gemm.py         |& tee gemm_bf16.log

# gemm a8w8
python3 ${AITER_HOME}/op_tests/test_gemm_a8w8.py    |& tee gemm_a8w8.log

# gemm a4w4
python3 ${AITER_HOME}/op_tests/test_gemm_a4w4.py    |& tee gemm_a4w4.log
