#!/bin/bash

# export GEMM_A4W4_BLOCKWISE_HIP_CLANG_PATH=/home/jun_chen2_qle/llvm-project/build/bin/
export GEMM_A4W4_BLOCKWISE_HIP_CLANG_PATH=/home/zhimding/llvm-project/build/bin/

E=$1
N=$(( $2 / 2 ))
K=$3
Q=$4

AITER_HOME=$(pip show aiter | grep Location | awk '{ print $2 }')

for tuple in "128 2" "256 4" "512 2" "1024 4" "4096 2" "8192 4" "10240 2"; do
    echo $tuple
    read M topk <<< "$tuple"
    python3 ${AITER_HOME}/op_tests/test_moe_2stage_perf_v2.py -d bf16 -dim $K,$N -t $M -q $Q -a silu -s f -k $topk -e $E
done

