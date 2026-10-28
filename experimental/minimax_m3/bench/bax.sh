#!/bin/bash
# bax.sh LOG "ENV=.." bench_agentx-args  (AgentX-shape bench on GPUs 4-7, dev container ix_dsv41)
NAME=$1; E=$2; shift 2
docker exec ix_dsv41 bash -c "cd /home/vigokhal/m3_monokernel && ulimit -c 0 && export HSA_ENABLE_COREDUMP=0 HIP_VISIBLE_DEVICES=4,5,6,7 AITER_JIT_DIR=/home/vigokhal/m3_bench/cache/aiter_jit TRITON_CACHE_DIR=/home/vigokhal/m3_bench/cache/triton OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 $E; timeout 1800 torchrun --standalone --nproc_per_node 4 bench/bench_agentx.py $* 2>&1" | grep -v "aiter\] import\|^W1\|OMP_NUM\|\*\*\*\*\|NUMA" > /home/vigokhal/m3_monokernel/bench/logs/$NAME.log
grep -iE "ctx=|layers |first step|nrmse|error|Traceback|watchdog|refused" /home/vigokhal/m3_monokernel/bench/logs/$NAME.log | grep -v "^ *File" | head -30
