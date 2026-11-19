#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$root/../../.." && pwd)
variant=${1:?affinity or current}
round=${2:?unique round label}
tokens=${CHECK_TPR:-16384}
bth=0
ipf=${BENCH_IPF:-1}
[[ "$ipf" == 0 || "$ipf" == 1 || "$ipf" == legacy ]] || exit 2
producers=${BENCH_PRODUCERS:-96}
[[ "$producers" =~ ^[0-9]+$ ]] || exit 2
(( producers > 0 && producers < 256 && producers % 4 == 0 )) || exit 2
[[ "$bth" =~ ^[0-6]$ ]] || exit 2
case "$variant" in
  affinity) checkout="$repo" ;;
  current) checkout="$repo" ;;
  *) exit 2 ;;
esac
out="$root/ipf${ipf}_p${producers}_${round}"
mkdir "$out"
cd "$checkout"
export BENCH_VARIANT=current
for name in $(compgen -e); do
  case "$name" in AITER_*|FLYDSL_*|MORI_*|MEGA_*|COMBINE_QUANT|DEBUG_HIP_GRAPH_CLASSIC_PATH) unset "$name" ;; esac
done
export PYTHONPATH="$checkout${PYTHONPATH:+:$PYTHONPATH}"
mori_root=${DS41_MORI_ROOT:-/workspace/mori-plan-describe}
if [[ -d "$mori_root/python/mori" ]]; then
  export PYTHONPATH="$mori_root/python:$PYTHONPATH"
  export MORI_V2_LIB_DIR="$mori_root/python/mori"
  export MORI_JIT_CACHE_DIR="$mori_root/jit-cache"
fi
export MORI_GPU_ARCHS=gfx1250 FLYDSL_GPU_ARCH=gfx1250 ENABLE_CK=0
export MORI_AMD_SMI_PATH=/opt/rocm/lib/libamd_smi.so.27
export HIP_VISIBLE_DEVICES=0,1,2,3 PYTHONUNBUFFERED=1
export AITER_MEGA_QUANT_FUSE=128 AITER_MEGA_PLAN_PREFETCH=0 AITER_MEGA_S1F=1
export AITER_MOE_EXPERT_BALANCE=${BENCH_BALANCED:-true}
export AITER_MOE_S1F_CLUSTER_LOCAL=0
export AITER_MOE_S1F_PRODUCER_GROUP=${BENCH_PRODUCER_GROUP:-1}
export CYCLE_TRACE=${CYCLE_TRACE:-1}
export AITER_MEGA_PULL_PLAN_PACK=${BENCH_PLAN_PACK:-0}
export AITER_MEGA_PULL_SRC_SHUFFLE=${BENCH_SRC_SHUFFLE:-0}
export AITER_MEGA_PULL_UNIQUE=${BENCH_UNIQUE_PULL:-0}
export AITER_MEGA_EP_LINEAR_STORE=${BENCH_EP_LINEAR_STORE:-0}
export AITER_MEGA_PULL_WAVE_LOAD=${BENCH_WAVE_LOAD:-1}
export AITER_MEGA_PULL_LOAD_K_BYTES=${BENCH_LOAD_K_BYTES:-1024}
export AITER_MOE_S1F_TILE_READY=${BENCH_TILE_READY:-0}
export AITER_MOE_S1F_READY_QUEUE=${BENCH_READY_QUEUE:-0}
export AITER_MOE_S1F_READY_FIRST=${BENCH_READY_FIRST:-0}
export AITER_CONFIG_GROUPED_FMOE="$checkout/aiter/configs/model_configs/dsv41_flash_tuned_grouped_fmoe.csv"
if [[ "$variant" == current || "$variant" == affinity ]]; then
  export AITER_MEGA_PULL_TDM_GATHER=1 AITER_MEGA_DISPATCH_PULL=1
  export AITER_MEGA_S1F_PULL=1 AITER_MEGA_S1F_PULL_PRODUCERS="$producers"
  export AITER_MEGA_S1F_PULL_CLUSTER=0
fi

if [[ "$variant" == affinity ]]; then
  export AITER_MOE_S1F_AFFINITY=1
fi
export FLYDSL_RUNTIME_CACHE_DIR=${BENCH_CACHE_DIR:-$root/cache/ipflegacy}
export AITER_TDM_B_TH="$bth"
export AITER_FLYDSL_INST_PREFETCH="$ipf"
if [[ "$ipf" == legacy ]]; then unset AITER_FLYDSL_INST_PREFETCH; fi
export AITER_MOE_S1F_CONSUMER_CLUSTER_N=1
export PROFILE_OUT="$out"
command=(torchrun --standalone --nproc_per_node=4 "$root/bench_cycles.py"
  -tpr "$tokens" --layers ${BENCH_LAYERS:-8} -e 384 -k 6 -hd 5120 -id 2304
  --fused_shared_experts 0 --routing uniform -q a4w4_mxfp4 --dispatch_wire fp4
  --stage1_fused true --stage2_fused true --profile_table 0 --prof_replays 32
  --acc_verify 0 --warmup 5 --iters 32 --seed 0 --data-init uniform)
if [[ "$round" == events* ]]; then
  command[3]="$root/bench_events.py"
elif [[ "$round" == check_src* ]]; then
  command=(python -u "$root/check_src.py")
else
  python "$root/build_overlay.py"
  if [[ "$round" == check* ]]; then
    command=(python -u "$root/check_cycles.py")
  fi
fi
printf '%q ' "${command[@]}" > "$out/command.sh"
printf '\n' >> "$out/command.sh"
git rev-parse HEAD > "$out/head.txt"
git diff HEAD > "$out/source.diff"
cp aiter/ops/flydsl/kernels/mega_moe_gfx1250/packed_wave_copy.py "$out/"
cp aiter/ops/flydsl/kernels/mega_moe_gfx1250/unique_pull.py "$out/"
cp aiter/ops/flydsl/kernels/mega_moe_gfx1250/linear_combine.py "$out/"
cp aiter/ops/flydsl/kernels/mega_moe_gfx1250/s1f_producer_group.py "$out/s1f_producer_group.py"
cp aiter/ops/flydsl/kernels/mega_moe_gfx1250/ready_queue.py "$out/ready_queue.py"
cp aiter/ops/flydsl/kernels/mega_moe_gfx1250/s1f_cluster_local.py "$out/s1f_cluster_local.py"
cat /proc/sys/kernel/random/boot_id > "$out/boot_before.txt"
date -Is > "$out/start.txt"
env | sort | rg '^(AITER_|FLYDSL_|MORI_|CHECK_|ENABLE_CK=|PYTHONPATH=|HIP_VISIBLE_DEVICES=)' > "$out/environment.txt"
sync -f "$out/start.txt"
cp "$root/REVIEW.md" "$out/REVIEW.md"
cp "$root/bench_cycles.py" "$out/profile_breakdown.py"
cp "$root"/*.py "$out/"
cp "$AITER_CONFIG_GROUPED_FMOE" "$out/model.csv"
while true; do
  if rocm-smi --showpids > "$out/idle.log" 2>&1 && rg -q 'No KFD PIDs currently running' "$out/idle.log"; then break; fi
  sleep 1
done
launch=(timeout --signal=TERM --kill-after=15s 240s "${command[@]}")
if [[ -x /usr/local/bin/gpu-lock ]]; then
  launch=(/usr/local/bin/gpu-lock bash -c '
    state=$(rocm-smi --showpids) || exit 75
    [[ "$state" == *"No KFD PIDs currently running"* ]] || exit 75
    exec "$@"
  ' bash "${launch[@]}")
fi
echo "Starting $variant tpr=$tokens producers=$producers"
set +e
"${launch[@]}" > "$out/benchmark.log" 2>&1
result=$?
set -e
printf '%s\n' "$result" > "$out/exitcode"
cat /proc/sys/kernel/random/boot_id > "$out/boot_after.txt"
date -Is > "$out/end.txt"
sync -f "$out/exitcode"
echo "Finished $variant tpr=$tokens producers=$producers exit=$result"
if [[ "$result" != 0 ]]; then tail -25 "$out/benchmark.log"; exit "$result"; fi
cmp "$out/boot_before.txt" "$out/boot_after.txt"
timeout 15s rocm-smi --showpids > "$out/gpu_after.log" 2>&1
