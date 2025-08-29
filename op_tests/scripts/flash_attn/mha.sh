#!/bin/bash

q_nheads=$1
kv_nheads=$2
q_seq_len=$3
kv_seq_len=$4
head_dim=$5

AITER_HOME=$(pip show aiter | grep Location | awk '{ print $2 }')

for B in 1 8 16 32 64 128 256; do
    python3 ${AITER_HOME}/op_tests/test_mha_perf.py -b $B -n ${q_nheads} -nk ${kv_nheads} -q ${q_seq_len} -k ${kv_seq_len} -d ${head_dim} -v ${head_dim}
done
