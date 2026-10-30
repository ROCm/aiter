#!/bin/bash
# MiniMax-M3 MXFP4 TP4 + EAGLE3 K3 per LLM-Booster #188 (lirui927 10-26), with a torch profiler dir.
export HSA_OVERRIDE_GFX_VERSION=12.5.0 PYTORCH_ROCM_ARCH=gfx1250 GPU_ARCHS=gfx1250
export AITER_USE_SYSTEM_TRITON=1 AITER_ROPE_TRITON_BACKEND=1 AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_PA_FLYDSL=1 ATOM_MOE_GU_ITLV=1 ATOM_FORCE_ATTN_TRITON=1 AITER_LOG_LEVEL=WARNING
export NCCL_IB_DISABLE=1 RCCL_IB_DISABLE=1
export ATOM_TORCH_PROFILER_DIR=${ATOM_TORCH_PROFILER_DIR:-/logs/prof}
cd /app/ATOM
exec python3 -m atom.entrypoints.openai_server \
  --model /models/MiniMax-M3-MXFP4 \
  --tensor-parallel-size 4 \
  --server-port 8000 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --block-size 128 \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --kv_cache_dtype fp8 \
  --max-num-batched-tokens 32768 \
  --attn-prefill-chunk-size 16384 \
  --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}' \
  --method eagle3 \
  --draft-model /models/MiniMax-M3-EAGLE3-GQA \
  --num-speculative-tokens 3 \
  --spec-decode-acceptance-rate 0.5933 \
  --torch-profiler-dir /logs/prof
