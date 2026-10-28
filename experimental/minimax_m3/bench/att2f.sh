#!/bin/bash
# att2.sh NAME CU "ENV" ITER-RANGE bench_agentx-args : rocprofv3 thread trace of rank 0's m3_step_kernel
# dispatches ITER-RANGE (e.g. "[346-347]"), CU CU of every shader engine, on one TP4 set (flock-serialized).
#   GPUSET=0|1 (GPUs 0-3 | 4-7, default 1)   ROOT=repo root (default /home/vigokhal/m3_monokernel)
# Output: bench/logs/att/NAME/{prof,tl_gluon_*.pt,tl_gluon_*_hw.pt}, NAME.log
NAME=$1; CU=$2; E=$3; IT=$4; shift 4
ROOT=${ROOT:-/home/vigokhal/m3_monokernel}
if [ "${GPUSET:-1}" = "0" ]; then DEVS=0,1,2,3; PORT=29600; GI=0; else DEVS=4,5,6,7; PORT=29700; GI=4; fi
if [ "${GPUSET:-1}" = "0" ] && ! $(dirname $0)/set0_idle.sh; then echo "GPUSET=0: GPUs 0-3 busy (another user?); using set 1" >&2; GPUSET=1; DEVS=4,5,6,7; PORT=29700; GI=4; fi
LOCK=/home/vigokhal/agentx/lock_set${GPUSET:-1}
OUT=/home/vigokhal/m3_monokernel/bench/logs/att/$NAME
mkdir -p $OUT /home/vigokhal/agentx
flock $LOCK docker exec ix_dsv41 bash -c "cd $ROOT && ulimit -c 0 && export HSA_ENABLE_COREDUMP=0 HIP_VISIBLE_DEVICES=$DEVS AITER_JIT_DIR=/home/vigokhal/m3_bench/cache/aiter_jit TRITON_CACHE_DIR=/home/vigokhal/m3_bench/cache/triton OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 M3_TL_SAVE=$OUT/tl VLLM_M3_GLUON_TRACE=3 $E; timeout 2400 rocprofv3 --att --att-target-cu $CU --att-shader-engine-mask 0xffffffff --att-gpu-index $GI --att-buffer-size 1073741824 --kernel-include-regex ${KREGEX:-m3_step_kernel} --kernel-iteration-range $IT --kernel-trace --log-level warning -d $OUT/prof -o run_%pid% -f csv -- torchrun --master_port $PORT --nproc_per_node 4 bench/bench_agentx.py --timeline $* 2>&1" | grep -v "aiter\] import\|^W1\|OMP_NUM\|\*\*\*\*\|NUMA" > $OUT.log
tail -3 $OUT.log
