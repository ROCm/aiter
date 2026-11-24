#!/usr/bin/env bash
# Reproduce the fixed gfx1250 DSR1 EP4 global-token-9216 comparison.
set -euo pipefail
preset=${1:-selected}
case "$preset" in
  selected|control) ;;
  *) printf 'Usage: %s [selected|control]\n' "$0" >&2; exit 2 ;;
esac
preset_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
checkout=$(cd -- "$preset_dir/../../../.." && pwd)
python_bin=${DSR1_PYTHON:-python3}
cache_root=${DSR1_CACHE_ROOT:-/tmp/aiter-dsr1-global9216-${UID}}

# Lock once around the entire launch, including the immediate idle check.
if [[ -f /usr/local/bin/gpu-lock && -x /usr/local/bin/gpu-lock && ${DSR1_PRESET_LOCKED:-0} != 1 ]]; then
  export DSR1_PRESET_LOCKED=1
  exec /usr/local/bin/gpu-lock bash "$preset_dir/run.sh" "$preset"
fi
export PYTHONPATH="$checkout${PYTHONPATH:+:$PYTHONPATH}"
export AITER_META_DIR="$checkout"
export AITER_JIT_DIR="$cache_root/aiter"
export FLYDSL_CACHE_DIR="$cache_root/flydsl"
export TRITON_CACHE_DIR="$cache_root/triton"
export GPU_ARCHS=gfx1250 CU_NUM=256 FLYDSL_GPU_ARCH=gfx1250 MORI_GPU_ARCHS=gfx1250
export AITER_MOE_LOC=1 AITER_MOE_LOC_PW=512
export AITER_MEGA_S1F=0 AITER_GROUPED_DEBUG=1
export AITER_GROUPED_A4W4_A_PRESHUFFLE=1 AITER_FLYDSL_A2_PRESHUFFLE=1
export AITER_FLYDSL_EP_SCATTER_OVERLAP=0
export AITER_CONFIG_GROUPED_FMOE="$preset_dir/$preset.csv"
export DEBUG_HIP_GRAPH_CLASSIC_PATH=1 HIP_FORCE_DEV_KERNARG=1
export HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0 AITER_DISABLE_KERNARG_PRELOAD=1
export MAX_JOBS=16
if [[ $preset == selected ]]; then
  export AITER_TDM_COMPACT_BLOCKS=512 AITER_TDM_COMPACT_WARPS=16
  export AITER_TDM_COMPACT_ROUTE_PARALLEL=1
else
  unset AITER_TDM_COMPACT_BLOCKS AITER_TDM_COMPACT_WARPS
  unset AITER_TDM_COMPACT_ROUTE_PARALLEL
fi
mkdir -p "$AITER_JIT_DIR" "$FLYDSL_CACHE_DIR" "$TRITON_CACHE_DIR"
cd "$checkout"
gpu_state=$(rocm-smi --showpids)
printf '%s\n' "$gpu_state"
if ! grep -q 'No KFD PIDs currently running' <<< "$gpu_state"; then
  printf 'GPU busy or idleness uncertain; launch deferred.\n' >&2
  exit 75
fi
printf 'preset=%s global_tokens=9216 tpr=2304 checkout=%s\n' "$preset" "$checkout"
git rev-parse HEAD
"$python_bin" -c 'import aiter, flydsl; print("aiter:", aiter.__file__); print("flydsl:", flydsl.__version__, flydsl.__file__)'
env | sort | grep -E '^(AITER_|FLYDSL_|GPU_ARCHS=|CU_NUM=|DEBUG_HIP_|HIP_FORCE_|HSA_ENABLE_SCRATCH_)'
exec timeout --signal=TERM --kill-after=30s 15m "$python_bin" -m torch.distributed.run \
  --standalone --nproc_per_node=4 -m op_tests.multigpu_tests.bench_mega_moe \
  -q a4w4_mxfp4 -tpr 2304 -hd 7168 -id 2048 -e 256 -k 8 \
  --fused_shared_experts 1 --layers 1 \
  --stage1_fused true --stage2_fused true --dispatch_wire fp4 \
  --combine_quant none --data-init norm --seed 0 \
  --warmup 10 --iters 50 --prof_replays 10 --profile_table 1 --acc_verify 1
