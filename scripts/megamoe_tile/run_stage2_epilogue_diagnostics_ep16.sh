#!/usr/bin/env bash
set -euo pipefail

node_rank="${1:?node rank required}"
base_port="${2:?base port required}"
order="${3:?forward or reverse required}"
tag_prefix="${4:?tag prefix required}"
warmup="${5:-20}"
iters="${6:-100}"
tail_iters="${7:-50}"

# name|diagnostic mode|epilogue schedule|N-tile group|group pipeline
cases=(
  "production_gmm2|gmm2_only|lane32|1|baseline"
  "production_gmm2_atomic|gmm2_atomic_only|lane32|1|baseline"
  "meta_ng1_gmm2_atomic|gmm2_atomic_only|lane32_meta|1|baseline"
  "meta_ng2_gmm2_atomic|gmm2_atomic_only|lane32_meta|2|baseline"
  "meta_ng2_hoist_gmm2_atomic|gmm2_atomic_only|lane32_meta|2|expert_meta_hoist"
  "meta_ng2_adb_gmm2|gmm2_only|lane32_meta|2|a_double_buffer"
  "meta_ng2_adb_gmm2_atomic|gmm2_atomic_only|lane32_meta|2|a_double_buffer"
  "meta_ng2_adb_atomic|atomic_only|lane32_meta|2|a_double_buffer"
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
  IFS='|' read -r name mode epilogue n_group pipeline <<<"${spec}"
  port=$((base_port + idx))
  tag="${tag_prefix}_${order}_${name}"
  /home/hzm/aiter/scripts/megamoe_tile/run_stage2_breakdown_ep16.sh \
    "${node_rank}" "${port}" candidate "${mode}" "${tag}" \
    "${warmup}" "${iters}" "${tail_iters}" \
    160 bf16 14 persistent_queue 8 buffer 0 \
    lockstep 0 "${epilogue}" "${n_group}" wave0 interleaved 2 "${pipeline}"
  idx=$((idx + 1))
done
