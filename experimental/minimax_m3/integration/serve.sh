#!/bin/bash
# Serve MiniMax-M3-MXFP4 on vLLM, TP4 on GPUs 4-7 (run inside container ix_dsv41).
# Based on the InferenceX MI355X FP4 TP4 vLLM recipe (reference/minimaxm3-fp4-mi355x-vllm-agentx-mtp-20260812.sh),
# without the EAGLE3 draft (not on this host), no tool/reasoning parsers, no prefix caching.
# MONO=off (vLLM's own path) | flydsl | gluon (the M3 mono port, VLLM_M3_MONO_BACKEND).
set -eo pipefail
PORT=${PORT:-8343}
MONO=${MONO:-off}
MODEL=${MODEL:-/data/models/MiniMax-M3-MXFP4}
LOG=${LOG:-/home/vigokhal/m3_bench/logs/serve_${MONO}_$(date +%Y%m%d_%H%M%S).log}
C=/home/vigokhal/m3_bench/cache
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-4,5,6,7}
export HF_HOME=/data/huggingface-cache HF_HUB_OFFLINE=1
export VLLM_CACHE_ROOT=$C/vllm TRITON_CACHE_DIR=$C/triton AITER_JIT_DIR=${AITER_JIT_DIR:-$C/aiter_jit}
export VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export VLLM_USE_BREAKABLE_CUDAGRAPH=0 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=0
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB=256 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
case "$MONO" in
  off) export VLLM_M3_MONO=0 ;;
  flydsl|gluon) export VLLM_M3_MONO=1 VLLM_M3_MONO_BACKEND=$MONO ;;
  *) echo "MONO must be off|flydsl|gluon"; exit 1 ;;
esac
mkdir -p "$(dirname "$LOG")"
echo "log: $LOG"
exec vllm serve "$MODEL" --served-model-name minimax-m3 --port "$PORT" \
  --tensor-parallel-size 4 --trust-remote-code --block-size 128 --gpu-memory-utilization 0.85 \
  --enable-chunked-prefill --max-num-batched-tokens 32768 --language-model-only \
  --no-enable-prefix-caching --attention-backend TRITON_ATTN --moe-backend aiter --kv-cache-dtype fp8 \
  --max-num-seqs 2 --max-model-len 32768 \
  --hf-overrides '{"text_config": {"use_index_cache": true, "index_topk_freq": 4}}' \
  --disable-uvicorn-access-log ${EXTRA_ARGS} > "$LOG" 2>&1
