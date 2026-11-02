#!/bin/bash
set -euo pipefail
container=vigokhal_dsv4_tp4
pids=$(docker exec "$container" rocm-smi --showpids)
printf '%s\n' "$pids"
grep -q 'No KFD PIDs currently running' <<< "$pids"
docker exec "$container" bash -c 'if [ -d /root/.cache/atom ]; then mv /root/.cache/atom /root/.cache/atom-before-upstream-integration-$(date +%s%N); fi'
docker exec -d -w /work/flash -e HIP_VISIBLE_DEVICES=0 \
 -e PYTHONPATH=/opt/atom-gfx1250-integration -e AITER_LOG_LEVEL=WARNING \
 -e ATOM_DSV4_MONO=1 -e ATOM_MONO_ENABLE=1 -e ATOM_MONO_CHECK=1 \
 -e ATOM_DSV4_FLASH_MOE=0 -e ATOM_DSV4_HYBRID=0 -e ATOM_DSV4_HYBRID_ATTN=0 \
 -e TRITON_CACHE_DIR=/work/flash/triton-cache \
 -e TORCHINDUCTOR_CACHE_DIR=/work/gfx1250-integration/inductor-flash \
 "$container" bash -c 'ulimit -c 0; bash /work/flash/serve.sh > /work/gfx1250-integration/flash-server.log 2>&1'
