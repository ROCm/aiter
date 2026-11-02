#!/usr/bin/env bash
set -euo pipefail
profile=${1:?flash or pro}; mode=${2:?eager or compiled}; enabled=${3:-1}
case "$profile" in flash) gpus=0;; pro) gpus=0,1,2,3;; *) exit 2;; esac
case "$mode" in eager) extra=(--level 0 --enforce-eager);; compiled) extra=();; *) exit 2;; esac
container=vigokhal_dsv4_tp4
docker stop -t 3 "$container" >/dev/null
docker start "$container" >/dev/null
pids=$(docker exec "$container" rocm-smi --showpids)
printf '%s\n' "$pids"
grep -q 'No KFD PIDs currently running' <<< "$pids"
docker exec "$container" bash -c 'if [ -d /root/.cache/atom ]; then mv /root/.cache/atom /root/.cache/atom-before-upstream-integration-$(date +%s%N); fi'
docker cp /home/vigokhal/ATOM-gfx1250-integration/atom/models/deepseek_v4_mono "$container":/opt/atom-gfx1250-integration/atom/models/
docker cp /home/vigokhal/ATOM-gfx1250-integration/scripts/serve_dsv4_mono_gfx1250.sh "$container":/app/scripts/serve_dsv4_mono_gfx1250.sh
docker exec -d -w /work/flash -e HIP_VISIBLE_DEVICES="$gpus" \
 -e PYTHONPATH=/opt/atom-gfx1250-integration -e AITER_LOG_LEVEL=WARNING \
 -e ATOM_DSV4_MONO="$enabled" -e ATOM_MONO_ENABLE=1 -e ATOM_MONO_CHECK=1 \
 -e ATOM_MONO_FAILURE_DIR=/work/gfx1250-integration/failing-inputs \
 -e ATOM_DSV4_FLASH_MOE=0 -e ATOM_DSV4_HYBRID=0 -e ATOM_DSV4_HYBRID_ATTN=0 \
 -e TRITON_CACHE_DIR=/work/flash/triton-cache \
 -e TORCHINDUCTOR_CACHE_DIR="/work/gfx1250-integration/inductor-$profile-$mode-$enabled" \
 "$container" bash -c 'ulimit -c 0; profile=$1; mode=$2; enabled=$3; shift 3; exec bash /app/scripts/serve_dsv4_mono_gfx1250.sh "$profile" "$@" > "/work/gfx1250-integration/$profile-$mode-$enabled-server.log" 2>&1' bash "$profile" "$mode" "$enabled" "${extra[@]}"
