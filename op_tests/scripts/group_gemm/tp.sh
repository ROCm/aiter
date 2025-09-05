#!/bin/bash

# export GEMM_A4W4_BLOCKWISE_HIP_CLANG_PATH=/home/jun_chen2_qle/llvm-project/build/bin/
export GEMM_A4W4_BLOCKWISE_HIP_CLANG_PATH=/home/zhimding/llvm-project/build/bin/

E=$1
topk=$2
N=$(( $3 / 2 ))
K=$4
Q=$5

AITER_HOME=$(pip show aiter | grep Location | awk '{ print $2 }')

for M in 128 256 512 1024 4096 8192 10240; do
    echo $M
    python3 ${AITER_HOME}/op_tests/test_moe_2stage_perf_v2.py -d bf16 -dim $K,$N -t $M -q $Q -a silu -s f -k $topk -e $E
done

