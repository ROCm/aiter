#!/bin/bash
set -euo pipefail
mode=${1:?mode}
cd /work/live-vector/ATOM
export PYTHONPATH=/work/live-vector/ATOM
ps -eo pid,lstart,args > /work/live-vector/${mode}-server-before.txt
bash /work/flash/run_bench.sh ${mode}_run1 1 0 > /work/live-vector/${mode}-client1.log 2>&1
bash /work/flash/run_bench.sh ${mode}_run2 1 0 > /work/live-vector/${mode}-client2.log 2>&1
ps -eo pid,lstart,args > /work/live-vector/${mode}-server-after.txt
