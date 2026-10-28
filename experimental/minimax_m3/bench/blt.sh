#!/bin/bash
# blt.sh LOG "ENV=.." bench_layer_tp-args  (one 16-layer bench on GPUs 4-7)
NAME=$1; E=$2; shift 2
docker exec ix_dsv41 bash -c "cd /home/vigokhal/m3_monokernel && ulimit -c 0 && export HIP_VISIBLE_DEVICES=4,5,6,7 AITER_JIT_DIR=/home/vigokhal/m3_bench/cache/aiter_jit TRITON_CACHE_DIR=/home/vigokhal/m3_bench/cache/triton OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 $E; timeout 1200 torchrun --standalone --nproc_per_node 4 bench/bench_layer_tp.py $* 2>&1" | grep -v "aiter\] import\|^W1\|OMP_NUM\|\*\*\*\*\|NUMA" > /home/vigokhal/m3_monokernel/bench/logs/$NAME.log
grep -i "rows=\|nrmse\|error\|mismatch\|exact\|Traceback\|watchdog" /home/vigokhal/m3_monokernel/bench/logs/$NAME.log | grep -v "^ *File" | head -30
