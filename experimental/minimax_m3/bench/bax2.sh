#!/bin/bash
# bax2.sh LOG "ENV=.." bench_agentx-args : AgentX-shape bench in container ix_dsv41 on one TP4 set,
# serialized per set with flock (one job per GPU set at a time).
#   GPUSET=0|1 (GPUs 0-3 | 4-7, default 1)   ROOT=repo root to run from (default /home/vigokhal/m3_monokernel)
#   NOLOCK=1 when the caller already holds the set's lock (ab2.sh)
NAME=$1; E=$2; shift 2
ROOT=${ROOT:-/home/vigokhal/m3_monokernel}
if [ "${GPUSET:-1}" = "0" ]; then DEVS=0,1,2,3; PORT=29600; else DEVS=4,5,6,7; PORT=29700; fi
if [ "${GPUSET:-1}" = "0" ] && ! $(dirname $0)/set0_idle.sh; then echo "GPUSET=0: GPUs 0-3 busy (another user?); using set 1" >&2; GPUSET=1; DEVS=4,5,6,7; PORT=29700; GI=4; NOLOCK=0; fi  # a fallback takes set 1's own lock
LOCK=/home/vigokhal/agentx/lock_set${GPUSET:-1}
LOGS=/home/vigokhal/m3_monokernel/bench/logs
mkdir -p $LOGS /home/vigokhal/agentx
CMD="docker exec ix_dsv41 bash -c \"cd $ROOT && ulimit -c 0 && export HSA_ENABLE_COREDUMP=0 HIP_VISIBLE_DEVICES=$DEVS AITER_JIT_DIR=/home/vigokhal/m3_bench/cache/aiter_jit TRITON_CACHE_DIR=/home/vigokhal/m3_bench/cache/triton OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 $E; timeout ${TIMEOUT:-1800} torchrun --master_port $PORT --nproc_per_node 4 bench/bench_agentx.py $* 2>&1\" | grep -v 'aiter\] import\|^W1\|OMP_NUM\|\*\*\*\*\|NUMA' > $LOGS/$NAME.log"
if [ "$NOLOCK" = "1" ]; then bash -c "$CMD"; else flock $LOCK bash -c "$CMD"; fi
grep -iE "ctx=|layers |first step|nrmse|error|Traceback|watchdog|refused" $LOGS/$NAME.log | grep -v "^ *File" | head -30
