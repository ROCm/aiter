#!/usr/bin/env bash
# DSV4.1 Flash: native single-GPU GEMM timings and MegaMoE EP4 profiles.
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
script_path="$PWD/run_ds41_flash.sh"
mode=${1:-non-ep}
tokens=${2:-16384}
stage1=${3:-true}
result_dir=${DS41_RESULT_DIR:-/workspace/benchmarks/ds41_flash_prefill_20261113}
mkdir -p "$result_dir"

case "$mode" in
  compare)
    ep_tokens=${2:-63336}
    HIP_VISIBLE_DEVICES=${DS41_SINGLE_GPU:-2} \
      bash "$script_path" non-ep 16384
    bash "$script_path" ep "$ep_tokens" true
    bash "$script_path" ep "$ep_tokens" false
    exit
    ;;
  non-ep|sweep)
    token_list=("$tokens")
    if [[ "$mode" == sweep ]]; then
      token_list=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384)
    fi
    case_name="${mode}_t${tokens}"
    command=(python3 -u -m op_tests.flydsl_tests.test_flydsl_grouped_gemm
      --scenario kernel --data-format a4w4 --experts 96
      --tokens "${token_list[@]}" --topk 6 --model-dim 5120 --inter-dim 2304
      --act silu --no-bias --balanced-routing --warmup 5 --iters 32
      --seed 0 --no-check-aot-cache
      --data-init constant uniform --scale-init constant auto)
    if [[ -n "${DS41_NON_EP_RUNNER:-}" ]]; then
      command=(python3 -u "$DS41_NON_EP_RUNNER" "${command[@]:4}")
    fi
    ;;
  ep)
    case_name="ep4_t${tokens}_stage1_${stage1}"
    if [[ -z "${AITER_CONFIG_GROUPED_FMOE:-}" ]]; then
      export AITER_CONFIG_GROUPED_FMOE="$result_dir/ep4_t${tokens}_mapped.csv"
      # Transfer tuning parameters, never the source row's measured timings.
      python3 - "$tokens" "$AITER_CONFIG_GROUPED_FMOE" <<'PY'
import csv
import sys

source = "aiter/configs/model_configs/dsv41_tuned_grouped_fmoe.csv"
with open(source) as stream:
    reader = csv.DictReader(stream)
    fields = reader.fieldnames
    rows = list(reader)
matches = [
    row for row in rows
    if row["token"] == "16384" and row["topk"] == "6"
    and row["expert"] == "96" and row["model_dim"] == "5120"
    and row["inter_dim"] == "2304"
    and row["q_dtype_a"] == "torch.float4_e2m1fn_x2"
    and row["act_type"] == "ActivationType.Silu"
]
if len(matches) != 1:
    raise ValueError(f"Expected one non-EP source row, found {len(matches)}")
raw_tokens = int(sys.argv[1]) * 4
# Match grouped_moe_gfx1250._get_padded_m without initializing the GPU.
bucket = (1 << (raw_tokens - 1).bit_length()) if raw_tokens < 32768 else (
    131072 if raw_tokens >= 131072 else 32768
)
candidate = dict(matches[0])
candidate.update(token=str(bucket), topk="-1", ep_fused="1",
                 us="", tflops="", bw="")
with open(sys.argv[2], "w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    writer.writerow(candidate)
print(f"EP4 tpr={sys.argv[1]} raw_token={raw_tokens} csv_token={bucket}; "
      "using non-EP token=16384 tuning parameters")
PY
    fi
    command=(torchrun --standalone --nproc_per_node=4
      op_tests/multigpu_tests/bench_mega_moe.py
      -tpr "$tokens" --layers 8 -e 384 -k 6 -hd 5120 -id 2304
      --fused_shared_experts 0 --routing uniform
      -q a4w4_mxfp4 --dispatch_wire fp4
      --stage1_fused "$stage1" --stage2_fused true
      --profile_table 1 --prof_replays 32 --acc_verify 0
      --warmup 5 --iters 32 --seed 0 --data-init uniform)
    # Keep EP uniform routing independent of the single-GPU balancing flag.
    export AITER_MOE_EXPERT_BALANCE=false
    export AITER_MEGA_S1F=${AITER_MEGA_S1F:-1}
    export AITER_MEGA_QUANT_FUSE=128 AITER_MEGA_PLAN_PREFETCH=0
    ;;
  *)
    echo "Usage: $0 {non-ep|sweep|ep|compare} [tokens_per_rank] [stage1_fused]" >&2
    exit 2
    ;;
esac

export PYTHONUNBUFFERED=1 ENABLE_CK=0
export MORI_GPU_ARCHS=gfx1250 FLYDSL_GPU_ARCH=gfx1250
export MORI_AMD_SMI_PATH=/opt/rocm/lib/libamd_smi.so.27
export FLYDSL_RUNTIME_CACHE_DIR="$result_dir/cache/$case_name"

# Optional explicit dependency checkout; otherwise use the installed MORI.
if [[ -n "${DS41_MORI_ROOT:-}" ]]; then
  export PYTHONPATH="$DS41_MORI_ROOT/python:$PWD${PYTHONPATH:+:$PYTHONPATH}"
  export MORI_V2_LIB_DIR="$DS41_MORI_ROOT/python/mori"
  export MORI_JIT_CACHE_DIR="$DS41_MORI_ROOT/jit-cache"
fi

git rev-parse HEAD > "$result_dir/$case_name.head.txt"
git diff --stat > "$result_dir/$case_name.diff-stat.txt"
printf '%q ' "${command[@]}" > "$result_dir/$case_name.command.sh"
printf '\n' >> "$result_dir/$case_name.command.sh"
env | sort | rg '^(AITER_|FLYDSL_|MORI_|ENABLE_CK=|PYTHONPATH=|DS41_|HIP_VISIBLE_DEVICES=|ROCR_VISIBLE_DEVICES=|CUDA_VISIBLE_DEVICES=)' \
  > "$result_dir/$case_name.environment.txt"

wait_count=0
while true; do
  if rocm-smi --showpids > "$result_dir/$case_name.idle.log" 2>&1 &&
      rg -q 'No KFD PIDs currently running' "$result_dir/$case_name.idle.log"; then
    break
  fi
  if (( wait_count % 30 == 0 )); then
    echo "Waiting for idle GPUs: $case_name"
  fi
  wait_count=$((wait_count + 1))
  sleep 1
done

# Acquire a shared lock when available, then recheck within that lock.
launch=(timeout --signal=TERM --kill-after=15s 1200s "${command[@]}")
if [[ -x /usr/local/bin/gpu-lock ]]; then
  launch=(/usr/local/bin/gpu-lock bash -c '
    check=$(rocm-smi --showpids) || exit 75
    [[ "$check" == *"No KFD PIDs currently running"* ]] || exit 75
    exec "$@"
  ' bash "${launch[@]}")
fi
echo "Starting $case_name"
set +e
"${launch[@]}" 2>&1 | tee "$result_dir/$case_name.log"
status=${PIPESTATUS[0]}
set -e
if [[ "$mode" != ep && "$status" == 0 ]] &&
    rg -q 'timing unavailable|no grouped kernels captured' \
      "$result_dir/$case_name.log"; then
  echo "Accuracy completed, but kernel timing is unavailable." >&2
  status=65
fi
printf '%s\n' "$status" > "$result_dir/$case_name.exitcode"
exit "$status"
