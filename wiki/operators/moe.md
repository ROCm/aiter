---
title: "Mixture of Experts (MoE) Operators"
last_verified: 2026-04-06
source_files:
  - aiter/ops/moe_op.py
  - aiter/ops/moe_sorting.py
  - aiter/ops/moe_sorting_opus.py
  - aiter/ops/activation.py
  - aiter/ops/triton/activation.py
  - aiter/configs/tuned_fmoe.csv
tags: [moe, triton, ck, asm, experts, routing]
---

# Mixture of Experts (MoE) Operators

## Overview
aiter provides a complete MoE pipeline: gating/routing (top-k selection), token sorting, expert GEMM computation, and activation functions. The implementation spans ASM, CK, and Triton backends. MoE is one of the most complex operator families due to the interaction between routing, sorting, and GEMM stages.

## MoE Pipeline Stages

### 1. Gating / Top-k Selection
Functions in `moe_op.py`:
- `topk_softmax(topk_weights, topk_indices, token_expert_indices, gating_output, need_renorm, num_shared_experts)` -- ASM-based top-k with softmax routing
- `topk_softmax_asm(...)` -- ctypes FFI variant
- `topk_sigmoid(topk_weights, topk_indices, gating_output)` -- sigmoid-based routing
- `topk_softmax_decode(...)` -- decode-optimized variant that outputs sorted results directly
- `grouped_topk_decode(...)` -- grouped expert selection with correction bias and routed scaling factor

### 2. Token Sorting
Functions in `moe_sorting.py` and `moe_sorting_opus.py`:
- `moe_sorting_fwd(...)` -- sort tokens by assigned expert for efficient batched GEMM
- `moe_sorting_opus_fwd(...)` -- "opus" variant of sorting

### 3. Expert Computation (GEMM)
Two main paths exist in vLLM integration:
1. **CK Fused Path** -- GEMM1 + activation + quant + GEMM2 fused into a single CK kernel. The `silu_and_mul` activation happens *inside* the GEMM. Used when tuned CK configs exist (e.g., from `tuned_fmoe.csv`).
2. **Triton MoE Path** -- GEMM1 and GEMM2 are separate. Activation (`apply_moe_activation()`) is called between them. Falls back to this when no tuned CK config exists.

### 4. Activation
- `activation.py` provides activation functions used between MoE GEMM stages
- `aiter.ops.triton.activation.act_mul(x, "silu", out=output)` -- Triton-based fused gated activation (`act(x[:d]) * x[d:]`) without quantization
- Supports silu and gelu activation types

## Configuration
- `aiter/configs/tuned_fmoe.csv` -- tuned configs for fused MoE kernels
- `aiter/configs/untuned_fmoe.csv` -- default/untuned configs

## Data Types
- Routing: FP32 gating logits, FP32 weights
- Expert GEMM: supports FP8 (A8W8), BF16, FP4 (A4W4 via FlyDSL)
- Intermediate activation: BF16/FP16

## Optional: FlyDSL
FlyDSL provides an alternative MoE kernel for mixed-precision A4W4. When installed (`pip install --pre flydsl`), aiter uses FlyDSL kernels for A4W4 MoE. Falls back to CK when not installed.

## Related Pages
- [[operators/gemm]] -- GEMM operators used for expert computation
- [[operators/quant]] -- quantization between MoE stages
- [[concepts/backend-selection]] -- CK vs Triton path selection
- [[integration/vllm]] -- vLLM integration of MoE operators

## Source Files
- `aiter/ops/moe_op.py` -- gating, top-k, grouped routing (~667 lines)
- `aiter/ops/moe_sorting.py` -- token sorting for expert dispatch
- `aiter/ops/moe_sorting_opus.py` -- opus sorting variant
- `aiter/ops/activation.py` -- activation functions
- `aiter/ops/triton/activation.py` -- Triton activation kernels (act_mul)
- `aiter/configs/tuned_fmoe.csv` -- tuned MoE configs
