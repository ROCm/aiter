#!/bin/bash
# ab.sh SET NAME "ENV=.." script-args   SET 0: GPUs 0-3 port 29511, SET 1: GPUs 4-7 port 29512
# script defaults to bench/bench_layer_tp.py (override with SCRIPT=...)
SET=$1; NAME=$2; E=$3; shift 3
if [ "$SET" = 0 ]; then G=0,1,2,3; P=29511; else G=4,5,6,7; P=29512; fi
docker exec ix_dsv41 bash -c "cd /home/vigokhal/m3_monokernel && ulimit -c 0 && export HSA_ENABLE_COREDUMP=0 HIP_VISIBLE_DEVICES=$G AITER_JIT_DIR=/home/vigokhal/m3_bench/cache/aiter_jit TRITON_CACHE_DIR=/home/vigokhal/m3_bench/cache/triton OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 VLLM_M3_GLUON_HYBRID=1 $E; timeout ${TIMEOUT:-1500} torchrun --nproc_per_node 4 --master_port $P ${SCRIPT:-bench/bench_layer_tp.py} $* 2>&1" | grep --line-buffered -v "aiter\] import\|^W1\|OMP_NUM\|\*\*\*\*\|NUMA" > /home/vigokhal/m3_monokernel/bench/logs/$NAME.log
grep -i "rows=\|nrmse\|error\|mismatch\|exact\|PASS\|FAIL\|Traceback" /home/vigokhal/m3_monokernel/bench/logs/$NAME.log | grep -v "^ *File" | head -20
