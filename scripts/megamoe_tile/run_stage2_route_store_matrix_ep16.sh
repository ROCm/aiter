#!/usr/bin/env bash
set -euo pipefail

node_rank="${1:?node rank required}"
base_port="${2:?base port required}"
order="${3:?forward or reverse required}"
tag_prefix="${4:?tag prefix required}"
warmup="${5:-5}"
iters="${6:-30}"
tail_iters="${7:-20}"

# Keep exactly 145 GMM2 CTAs while sweeping the dedicated node reducers.
# name|total grid|reducer CTAs|vector bytes
cases=(
  "r8v4|168|8|4"
  "r16v4|176|16|4"
  "r32v4|192|32|4"
  "r56v4|216|56|4"
  "r8v8|168|8|8"
  "r16v8|176|16|8"
  "r32v8|192|32|8"
  "r56v8|216|56|8"
)

if [[ "${order}" == "reverse" ]]; then
  reversed=()
  for ((idx=${#cases[@]}-1; idx>=0; --idx)); do
    reversed+=("${cases[idx]}")
  done
  cases=("${reversed[@]}")
elif [[ "${order}" != "forward" ]]; then
  printf 'order must be forward or reverse, got %s\n' "${order}" >&2
  exit 2
fi

idx=0
for spec in "${cases[@]}"; do
  IFS='|' read -r name workers reducers vec_bytes <<<"${spec}"
  port=$((base_port + idx))
  tag="${tag_prefix}_${order}_${name}"
  /home/hzm/aiter/scripts/megamoe_tile/run_stage2_breakdown_ep16.sh \
    "${node_rank}" "${port}" candidate full "${tag}" \
    "${warmup}" "${iters}" "${tail_iters}" \
    "${workers}" bf16 14 persistent_queue 8 buffer 0 \
    lockstep 0 lane32_meta 2 wave0 interleaved 2 a_double_buffer \
    route_store "${reducers}" "${vec_bytes}"
  idx=$((idx + 1))
done
