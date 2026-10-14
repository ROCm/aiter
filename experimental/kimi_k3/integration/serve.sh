#!/usr/bin/env bash
set -euo pipefail
model=${1:?Usage: bash integration/serve.sh /path/to/Kimi-K3 [vLLM arguments]}
shift
directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$directory/..:$directory${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_ROCM_USE_AITER=1 SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm
export VLLM_ROCM_USE_AITER_MOE_SITUV2=a16w4
unset AITER_SITUV2_A8W4 AITER_SITUV2_A4W4
export AITER_BF16_FP8_MOE_BOUND=0 VLLM_USE_BREAKABLE_CUDAGRAPH=0
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export VLLM_ENGINE_READY_TIMEOUT_S=7200 HF_HUB_OFFLINE=1
export VLLM_SERVER_DEV_MODE=1
exec python -m vllm.entrypoints.cli.main serve "$model" \
  --host 127.0.0.1 --port 8243 --served-model-name kimi-k3 \
  --trust-remote-code --moe-backend auto --tensor-parallel-size 8 \
  --load-format auto --gpu-memory-utilization 0.95 \
  --max-model-len 4096 --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --kv-cache-dtype fp8 --language-model-only --no-enable-prefix-caching \
  --worker-extension-cls worker_extension.BenchmarkWorkerExtension "$@"
