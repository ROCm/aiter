#!/usr/bin/env bash
# Regional gfx1250 hybrid; run from an environment with this ATOM checkout installed.
set -euo pipefail
# Prefer the checkout next to this script, or the isolated test installation
# when this launcher has been copied into /app/scripts.
source_root=${ATOM_SOURCE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
if [[ -d "$source_root/atom/models/deepseek_v4_mono" ]]; then
 export PYTHONPATH="$source_root${PYTHONPATH:+:$PYTHONPATH}"
elif [[ -d /opt/atom-gfx1250-integration/atom/models/deepseek_v4_mono ]]; then
 export PYTHONPATH="/opt/atom-gfx1250-integration${PYTHONPATH:+:$PYTHONPATH}"
fi
profile=${1:?usage: serve_dsv4_mono_gfx1250.sh flash|pro [server arguments]}
shift
case "$profile" in
 flash) tp=1; model=${MODEL_PATH:-/data/DeepSeek-V4-Flash}; gpus=0; a4w4=0;;
 pro) tp=4; model=${MODEL_PATH:-/data/DeepSeek-V4-Pro}; gpus=0,1,2,3; a4w4=1;;
 *) echo "Unknown profile: $profile" >&2; exit 2;;
esac
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-$gpus}
export ATOM_MONO_ENABLE=${ATOM_MONO_ENABLE:-1} ATOM_DSV4_MONO=${ATOM_DSV4_MONO:-1}
export ATOM_MONO_CHECK=${ATOM_MONO_CHECK:-0} AITER_LOG_LEVEL=${AITER_LOG_LEVEL:-WARNING}
export ATOM_USE_TRITON_MOE=1 ATOM_USE_TRITON_MOE_DECODE=1
export ATOM_USE_TRITON_MOE_A4W4=${ATOM_USE_TRITON_MOE_A4W4:-$a4w4}
export ATOM_USE_UNIFIED_ATTN=1 ATOM_FORCE_ATTN_TRITON=1
export ATOM_LOADER_USE_THREADPOOL=1 AITER_ROPE_TRITON_BACKEND=1
export ATOM_USE_TRITON_GEMM=1 ENABLE_CK=0
export AITER_USE_OPUS_MOE_SORTING=1 AITER_FORCE_A8W4=0 AITER_USE_GROUPED_GEMM=1
export ATOM_USE_AITER_TRITON_ATTN=1 AITER_TRITON_USE_HERD=1
export ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE=1 ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE=1
export ATOM_MOE_GU_ITLV=1 ATOM_DP_LM_HEAD_MODE=allgather
export ATOM_ENABLE_DS_QKNORM_FUSION=0 ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION=0
export ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=0
export AITER_CUSTOM_AR_MIN_SIZE=131072 AITER_CUSTOM_AR_MAX_SIZE=8388608
export HSA_ENABLE_SDMA=1 HSA_USE_SVM=1 HSA_XNACK=1 AITER_DISABLE_KERNARG_PRELOAD=0
ulimit -c 0
exec python -m atom.entrypoints.openai_server -tp "$tp" --model "$model" \
 --server-port "${PORT:-8000}" --kv_cache_dtype bf16 --index-cache-dtype fp8 \
 --no-enable_prefix_caching --disable_uvicorn_access_log --max-num-seqs 128 \
 --cudagraph-capture-sizes '[1,2,4,8,16,32,64,128]' "$@"
