#!/bin/bash
set -euo pipefail
root=/home/vigokhal/dsv4-monokernel/flash
while [ ! -f "$root/attention-screen.status" ]; do sleep 3; done
docker exec -w /work/flash -e HIP_VISIBLE_DEVICES=0 -e ENABLE_CK=0 -e TRITON_CACHE_DIR=/work/flash/triton-cache vigokhal_dsv4_tp4 bash -c 'ulimit -c 0; python /work/flash/test_mhc_stage.py > /work/flash/mhc-stage.log 2>&1; echo $? > /work/flash/mhc-stage-retry.status'
docker exec -w /work/flash -e HIP_VISIBLE_DEVICES=0 -e ENABLE_CK=0 -e TRITON_CACHE_DIR=/work/flash/triton-cache vigokhal_dsv4_tp4 bash -c 'ulimit -c 0; python /work/flash/bench_finalists.py > /work/flash/attention-finalists.log 2>&1; echo $? > /work/flash/attention-finalists.status'
