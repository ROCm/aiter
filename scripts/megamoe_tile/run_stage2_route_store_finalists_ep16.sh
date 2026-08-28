#!/usr/bin/env bash
set -euo pipefail

node_rank="${1:?node rank required}"
base_port="${2:?base port required}"
order="${3:?forward or reverse required}"
tag_prefix="${4:?tag prefix required}"
warmup="${5:-20}"
iters="${6:-100}"
tail_iters="${7:-50}"

# name|grid|accumulation mode|reducer CTAs|reducer vector bytes
cases=(
  "direct_adb|160|direct_atomic|32|4"
  "store_r16v4|176|route_store|16|4"
  "store_r16v8|176|route_store|16|8"
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
  IFS='|' read -r name workers accumulation reducers vec_bytes <<<"${spec}"
  port=$((base_port + idx))
  tag="${tag_prefix}_${order}_${name}"
  /home/hzm/aiter/scripts/megamoe_tile/run_stage2_breakdown_ep16.sh \
    "${node_rank}" "${port}" candidate full "${tag}" \
    "${warmup}" "${iters}" "${tail_iters}" \
    "${workers}" bf16 14 persistent_queue 8 buffer 0 \
    lockstep 0 lane32_meta 2 wave0 interleaved 2 a_double_buffer \
    "${accumulation}" "${reducers}" "${vec_bytes}" token interleaved
  idx=$((idx + 1))
done
