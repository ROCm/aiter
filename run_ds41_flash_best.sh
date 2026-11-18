#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# DSV4.1 Flash: EP4, 16384 tokens/rank (65536 global), balanced S1F pull.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Preserve the public affinity switch; clear stale tuning overrides.
for name in $(compgen -e); do
  case "$name" in
    AITER_MOE_S1F_AFFINITY) ;;
    AITER_*|FLYDSL_*|MORI_*|MEGA_*|COMBINE_QUANT|DEBUG_HIP_GRAPH_CLASSIC_PATH)
      unset "$name" ;;
  esac
done
export PYTHONUNBUFFERED=1 ENABLE_CK=0 HIP_VISIBLE_DEVICES=0,1,2,3
export MORI_GPU_ARCHS=gfx1250 FLYDSL_GPU_ARCH=gfx1250
export MORI_AMD_SMI_PATH=/opt/rocm/lib/libamd_smi.so.27
export AITER_CONFIG_GROUPED_FMOE="$PWD/aiter/configs/model_configs/dsv41_flash_tuned_grouped_fmoe.csv"
export AITER_MEGA_S1F=1 AITER_MEGA_QUANT_FUSE=128 AITER_MEGA_PLAN_PREFETCH=0
export AITER_MEGA_DISPATCH_PULL=1 AITER_MEGA_PULL_TDM_GATHER=1
export AITER_MEGA_S1F_PULL=1 AITER_MEGA_S1F_PULL_PRODUCERS=${DS41_PRODUCERS:-96}
export AITER_MEGA_S1F_PULL_CLUSTER=${DS41_CLUSTER:-0}
export AITER_MOE_S1F_AFFINITY=${AITER_MOE_S1F_AFFINITY:-0}
export AITER_TDM_B_TH=${DS41_BTH:-0}
export AITER_MOE_EXPERT_BALANCE=true
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Use the benchmark host's MORI build when present, otherwise the installed one.
mori_root=${DS41_MORI_ROOT:-/workspace/mori-plan-describe}
if [[ -d "$mori_root/python/mori" ]]; then
  export PYTHONPATH="$mori_root/python:$PYTHONPATH"
  export MORI_V2_LIB_DIR="$mori_root/python/mori"
  export MORI_JIT_CACHE_DIR="$mori_root/jit-cache"
fi

command=(torchrun --standalone --nproc_per_node=4
  op_tests/multigpu_tests/bench_mega_moe.py
  -tpr 16384 --layers 8 -e 384 -k 6 -hd 5120 -id 2304
  --fused_shared_experts 0 --routing uniform
  -q a4w4_mxfp4 --dispatch_wire fp4
  --stage1_fused true --stage2_fused true
  --profile_table 0 --acc_verify 0
  --warmup 5 --iters 32 --seed 0 --data-init uniform)

echo "Waiting for idle GPUs; checking every second."
while true; do
  if state=$(rocm-smi --showpids) &&
      [[ "$state" == *"No KFD PIDs currently running"* ]]; then
    break
  fi
  sleep 1
done
if [[ -x /usr/local/bin/gpu-lock ]]; then
  exec /usr/local/bin/gpu-lock bash -c '
    state=$(rocm-smi --showpids) || exit 75
    [[ "$state" == *"No KFD PIDs currently running"* ]] || exit 75
    exec "$@"
  ' bash "${command[@]}"
fi
exec "${command[@]}"
