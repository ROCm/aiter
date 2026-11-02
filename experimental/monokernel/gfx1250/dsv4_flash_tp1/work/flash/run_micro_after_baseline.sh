#!/bin/bash
set -euo pipefail
root=/home/vigokhal/dsv4-monokernel/flash
while [ ! -f "$root/baseline.status" ]; do sleep 10; done
[ "$(cat "$root/baseline.status")" = 0 ]
docker stop -t 3 vigokhal_dsv4_tp4
docker start vigokhal_dsv4_tp4
docker exec vigokhal_dsv4_tp4 rocm-smi --showpids > "$root/pre-micro-processes.txt"
grep -q 'No KFD PIDs currently running' "$root/pre-micro-processes.txt"
docker exec -w /work/flash -e HIP_VISIBLE_DEVICES=0 -e ENABLE_CK=0 -e TRITON_CACHE_DIR=/work/flash/triton-cache vigokhal_dsv4_tp4 bash -c 'ulimit -c 0; python /work/flash/test_mhc_stage.py > /work/flash/mhc-stage.log 2>&1; echo $? > /work/flash/mhc-stage.status'
docker exec -w /work/flash -e HIP_VISIBLE_DEVICES=0 -e ENABLE_CK=0 -e TRITON_CACHE_DIR=/work/flash/triton-cache vigokhal_dsv4_tp4 bash -c 'ulimit -c 0; python /work/flash/tune_attention.py --phase screen > /work/flash/attention-screen.log 2>&1; echo $? > /work/flash/attention-screen.status'
