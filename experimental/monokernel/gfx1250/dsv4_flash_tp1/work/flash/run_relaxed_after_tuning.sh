#!/bin/bash
set -euo pipefail
root=/home/vigokhal/dsv4-monokernel/flash
while [ ! -f "$root/attention-finalists-tuned.status" ]; do sleep 3; done
docker exec -w /work/flash -e HIP_VISIBLE_DEVICES=0 -e ENABLE_CK=0 -e TRITON_CACHE_DIR=/work/flash/triton-cache vigokhal_dsv4_tp4 bash -c 'ulimit -c 0; python /work/flash/test_mhc_relaxed_poll.py > /work/flash/mhc-relaxed-poll.log 2>&1; echo $? > /work/flash/mhc-relaxed-poll.status'
