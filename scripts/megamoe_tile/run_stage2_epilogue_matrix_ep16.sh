#!/usr/bin/env bash
set -euo pipefail

node_rank="${1:?node rank required}"
base_port="${2:?base port required}"
order="${3:?forward or reverse required}"
tag_prefix="${4:?tag prefix required}"
warmup="${5:-5}"
iters="${6:-30}"
tail_iters="${7:-20}"

# name|rail schedule|epilogue schedule|N-tile group|group pipeline
cases=(
  "lane32_ng1|lockstep|lane32|1|baseline"
  "meta_ng1|lockstep|lane32_meta|1|baseline"
  "lane32_ng2|lockstep|lane32|2|baseline"
  "meta_ng2|lockstep|lane32_meta|2|baseline"
  "meta_ng2_hoist|lockstep|lane32_meta|2|expert_meta_hoist"
  "meta_ng2_loop|lockstep|lane32_meta|2|runtime_loop"
  "meta_ng2_adb|lockstep|lane32_meta|2|a_double_buffer"
  "meta_ng2_prepost|qp_prepost|lane32_meta|2|baseline"
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
  IFS='|' read -r name rail epilogue n_group pipeline <<<"${spec}"
  port=$((base_port + idx))
  tag="${tag_prefix}_${order}_${name}"
  bash /home/hzm/aiter/scripts/megamoe_tile/run_stage2_breakdown_ep16.sh \
    "${node_rank}" "${port}" candidate full "${tag}" \
    "${warmup}" "${iters}" "${tail_iters}" \
    160 bf16 14 persistent_queue 8 buffer 0 \
    "${rail}" 0 "${epilogue}" "${n_group}" wave0 interleaved 2 "${pipeline}"
  idx=$((idx + 1))
done
