#!/bin/bash
set -euo pipefail
mode=${1:?resumed_final, resumed_baseline, or resumed_no_moe}
case "$mode" in
 resumed_final) hybrid=1; moe=1;;
 resumed_baseline) hybrid=0; moe=0;;
 resumed_no_moe) hybrid=1; moe=0;;
 *) exit 2;;
esac
container=vigokhal_dsv4_tp4
docker stop -t 3 "$container" >/dev/null
docker start "$container" >/dev/null
pids=$(docker exec "$container" rocm-smi --showpids)
printf '%s\n' "$pids"
grep -q 'No KFD PIDs currently running' <<< "$pids"
docker exec "$container" bash -c 'if [ -d /root/.cache/atom ]; then mv /root/.cache/atom /root/.cache/atom-preserved-$(date +%s%N); fi'
docker exec -d -w /work/flash -e HIP_VISIBLE_DEVICES=0 \
 -e ATOM_DSV4_HYBRID="$hybrid" -e ATOM_DSV4_HYBRID_MAX_TOKENS=16 \
 -e ATOM_DSV4_HYBRID_GEMM=0 -e ATOM_DSV4_HYBRID_ATTN="$hybrid" \
 -e ATOM_DSV4_HYBRID_ATTN_CONFIG=/app/ATOM/atom/model_ops/dsv4_flash_final_configs.json \
 -e ATOM_DSV4_HYBRID_CHECK=1 -e ATOM_DSV4_HYBRID_ATTN_CHECK_LIMIT=16 \
 -e ATOM_DSV4_FLASH_MOE="$moe" -e ATOM_DSV4_FLASH_MOE_CHECK_LIMIT=16 \
 -e FAKE_EPLB=0 -e TRACE=1 -e TRACE_DIR="/work/flash/traces/$mode" \
 -e TRITON_CACHE_DIR=/work/flash/triton-cache \
 -e TORCHINDUCTOR_CACHE_DIR="/work/flash/inductor-$mode" \
 "$container" bash -c "ulimit -c 0; bash /work/flash/serve.sh > /work/flash/$mode-launch.log 2>&1"
