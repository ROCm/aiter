---
title: "vLLM Integration"
last_verified: 2026-04-06
source_files:
  - ../vllm-aiter-silu-mul/README.md
tags: [vllm, integration, inference, rocm]
---

# vLLM Integration

## Overview
aiter operators are integrated into vLLM for high-performance inference on AMD ROCm GPUs. The integration is gated by the `VLLM_ROCM_USE_AITER` environment variable and requires both aiter to be installed and a supported ROCm/gfx9 platform.

## Integration Points

### Attention
vLLM calls `unified_attention()` from aiter, which dispatches to the appropriate kernel (CK-UA, Triton 2D/3D). See [[operators/attention]] for the full selector logic.

### MoE Activation
Two MoE paths in vLLM:
1. **CK fused path**: GEMM1 + silu + quant + GEMM2 fused in a single CK kernel. Activation happens inside the GEMM.
2. **Triton MoE path** (fallback): Separate GEMM stages. `apply_moe_activation()` is called between them, where `torch.ops._C.silu_and_mul` can be replaced with `aiter.ops.triton.activation.act_mul` for better performance on ROCm.

### Gating
- Only activates when `VLLM_ROCM_USE_AITER=1` AND running on ROCm gfx9 with aiter installed
- Falls back to original torch.ops._C kernels otherwise
- No effect on CUDA/NVIDIA platforms

## Environment Variables
- `VLLM_ROCM_USE_AITER` -- master switch for aiter operators in vLLM

## Status
This page is a placeholder. Full vLLM integration documentation will be added when a complete vLLM checkout is available in the workspace.

## Related Pages
- [[operators/attention]] -- attention kernels used by vLLM
- [[operators/moe]] -- MoE operators used by vLLM
- [[concepts/backend-selection]] -- backend selection within vLLM context

## Source Files
- `../vllm-aiter-silu-mul/README.md` -- silu_mul integration notes and patches
