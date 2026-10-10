#!/usr/bin/env bash
# M3 TP4 balanced Split/Mega e2e and pre/GEMM/post stage timing.
set -euo pipefail

if (($# < 1 || $# > 3)); then
    printf 'Usage: %s GLOBAL_M [GPU0,GPU1,GPU2,GPU3] [all|e2e|trace|stamp|analyze]\n' "$0" >&2
    exit 2
fi
M="$1"
gpus="${2:-0,1,2,3}"
mode="${3:-all}"
case "$M" in
    8|16|32|64|128|256|512|1024|2048|4096|8192) ;;
    *) printf 'Unsupported M: %s\n' "$M" >&2; exit 2 ;;
esac
if [[ ! "$gpus" =~ ^[0-7],[0-7],[0-7],[0-7]$ ]]; then
    printf 'Provide four comma-separated physical GPU IDs\n' >&2
    exit 2
fi
case "$mode" in
    all|e2e|trace|stamp|analyze) ;;
    *) printf 'Unknown mode: %s\n' "$mode" >&2; exit 2 ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"
cd -- "$repo_root"
mt_small="${AITER_MEGAMOE_TP_LB_MT_SMALL:-3}"
gpu_tag="${gpus//,/}"
out="${M3_TP4_OUTPUT_DIR:-/tmp/m3_tp4_balanced_M${M}_mt${mt_small}_gpu${gpu_tag}}"
mkdir -p -- "$out"

common_args=(--e2e --models m3 --tokens "$M" --tp 4 --lb-min "$M"
             --route balanced-routed-keep-shared --csv "$out/e2e.csv"
             --stage-output-dir "$out")
gpu_env=(env "HIP_VISIBLE_DEVICES=$gpus" HSA_ENABLE_IPC_MODE_LEGACY=1
         PYTHONUNBUFFERED=1 AITER_USE_SYSTEM_TRITON=1
         AITER_QUICK_REDUCE_QUANTIZATION=INT4 AITER_MEGAMOE_TP_LB_MT=5
         AITER_MEGAMOE_TP_LB_NPP=2 AITER_MEGAMOE_TP_LB_Q=3
         "AITER_MEGAMOE_TP_LB_MT_SMALL=$mt_small")

run_phase() {
    local phase="$1"
    local log="$out/$phase.log"
    printf '[RUN] M=%s mt_small=%s phase=%s GPUs=%s\n' "$M" "$mt_small" "$phase" "$gpus"
    if [[ "$phase" == analyze ]]; then
        if ! python3 "$script_dir/bench_mega_moe_TP.py" \
             "${common_args[@]}" --stage-mode analyze > "$log" 2>&1; then
            tail -n 60 "$log" >&2
            return 1
        fi
        grep -E '^\[STAGE\]|^Stage summary:' "$log"
    else
        if ! "${gpu_env[@]}" torchrun --standalone --nproc_per_node=4 \
             "$script_dir/bench_mega_moe_TP.py" \
             "${common_args[@]}" --stage-mode "$phase" > "$log" 2>&1; then
            tail -n 60 "$log" >&2
            return 1
        fi
        grep -E '^\[BALANCED\]|^\[E2E\]|^TP_MOE_E2E_OK|^\[TRACE\]|^\[STAMPS\]' "$log" || true
    fi
}

if [[ "$mode" == all ]]; then
    for phase in e2e trace stamp analyze; do
        run_phase "$phase"
    done
else
    run_phase "$mode"
fi

printf 'Results: %s\n' "$out"
