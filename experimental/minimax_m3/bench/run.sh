#!/bin/bash
# Run the M3 mono microbenchmark in container ix_dsv41 on GPUs 4-7 (one job at a time).
#   bench/run.sh LOGNAME [bench_layer_tp.py args...]
set -o pipefail
NAME=$1; shift
LOG=/home/vigokhal/m3_monokernel/bench/logs/$NAME.log
docker exec ix_dsv41 bash -c "cd /home/vigokhal/m3_monokernel && ulimit -c 0 && export HIP_VISIBLE_DEVICES=4,5,6,7 \
  AITER_JIT_DIR=/home/vigokhal/m3_bench/cache/aiter_jit TRITON_CACHE_DIR=/home/vigokhal/m3_bench/cache/triton \
  OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 $ENVS; timeout ${TIMEOUT:-1500} torchrun --standalone --nproc_per_node 4 \
  bench/bench_layer_tp.py $* 2>&1" | grep --line-buffered -v "aiter\] import\|^W1\|OMP_NUM\|\*\*\*\*\|NUMA balancing" > "$LOG"
echo "exit ${PIPESTATUS[0]} log $LOG"
