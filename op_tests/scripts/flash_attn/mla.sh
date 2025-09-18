#!/bin/bash

q_nheads=$1
kv_seq_len=$2
kv_dtype=$3

AITER_HOME=$(pip show aiter | grep Location | awk '{ print $2 }')

for B in 1 8 16 32 64 128 256; do
    python3 ${AITER_HOME}/op_tests/test_mla.py -b $B -n ${q_nheads},1 -c ${kv_seq_len} -kvd ${kv_dtype}
done
