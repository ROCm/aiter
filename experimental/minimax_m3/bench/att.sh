#!/bin/bash
# att.sh NAME CU "ENV" ITER-RANGE bench_agentx-args : rocprofv3 thread trace of rank 0's m3_step_kernel
# dispatches ITER-RANGE (e.g. "[346-347]"), CU CU of every shader engine, GPUs 4-7, container ix_dsv41
NAME=$1; CU=$2; E=$3; IT=$4; shift 4
OUT=/home/vigokhal/m3_monokernel/bench/logs/att/$NAME
docker exec ix_dsv41 bash -c "cd /home/vigokhal/m3_monokernel && ulimit -c 0 && export HSA_ENABLE_COREDUMP=0 HIP_VISIBLE_DEVICES=4,5,6,7 AITER_JIT_DIR=/home/vigokhal/m3_bench/cache/aiter_jit TRITON_CACHE_DIR=/home/vigokhal/m3_bench/cache/triton OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 M3_TL_SAVE=$OUT/tl VLLM_M3_GLUON_TRACE=3 $E; mkdir -p $OUT; timeout 2400 rocprofv3 --att --att-target-cu $CU --att-shader-engine-mask 0xffffffff --att-gpu-index 4 --att-buffer-size 1073741824 --kernel-include-regex m3_step_kernel --kernel-iteration-range $IT --kernel-trace --log-level warning -d $OUT/prof -o run_%pid% -f csv -- torchrun --standalone --nproc_per_node 4 bench/bench_agentx.py --timeline $* 2>&1" | grep -v "aiter\] import\|^W1\|OMP_NUM\|\*\*\*\*\|NUMA" > $OUT.log
tail -5 $OUT.log
