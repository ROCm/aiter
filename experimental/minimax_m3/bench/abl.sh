#!/bin/bash
# abl.sh LOG "ENV=.. ENV2=.." ablate-args
NAME=$1; E=$2; shift 2
docker exec ix_dsv41 bash -c "cd /home/vigokhal/m3_monokernel && ulimit -c 0 && export HIP_VISIBLE_DEVICES=4,5,6,7 AITER_JIT_DIR=/home/vigokhal/m3_bench/cache/aiter_jit TRITON_CACHE_DIR=/home/vigokhal/m3_bench/cache/triton OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 $E; timeout 1800 torchrun --standalone --nproc_per_node 4 bench/ablate.py $* 2>&1" | grep -v "aiter\] import\|^W1\|OMP_NUM\|\*\*\*\*\|NUMA" > /home/vigokhal/m3_monokernel/bench/logs/$NAME.log
grep -A20 '^rows=' /home/vigokhal/m3_monokernel/bench/logs/$NAME.log
