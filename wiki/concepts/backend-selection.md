---
title: "Backend Selection"
last_verified: 2026-04-06
source_files:
  - aiter/__init__.py
  - aiter/ops/triton/attention/unified_attention.py
  - aiter/ops/gemm_op_a8w8.py
  - aiter/ops/gemm_op_a4w4.py
tags: [backend, triton, ck, asm, selection]
---

# Backend Selection

## Overview
aiter supports three kernel backends: hand-written Assembly (ASM), Composable Kernel (CK/HIP C++), and Triton. The backend selection varies per operator family and is driven by hardware capabilities, tuned configurations, and runtime conditions.

## Backends

### Assembly (ASM)
- Hand-tuned for specific shapes and GPU architectures
- Highest performance for matched shapes
- Limited coverage (specific GEMM shapes, MoE sorting)
- Example: `gemm_a16w16_asm()`, `topk_softmax_asm()`
- Selected via kernel names in tuned CSV configs

### Composable Kernel (CK)
- HIP C++ template library, vendored in `3rdparty/composable_kernel/`
- Broad coverage: GEMM, attention (FMHA), normalization, etc.
- Compiled via JIT using `compile_ops()` from `aiter/jit/core.py`
- Two generations: classic CK (template-based) and CK Tile (tile-based, newer)
- Selected by default when ASM doesn't have a matching tuned config

### Triton
- Python JIT-compiled GPU kernels
- Most flexible: supports features CK doesn't (sliding window attention, ALiBi, softcap, sinks)
- Available on all platforms including Windows (fallback when ROCm JIT unavailable)
- Lives in `aiter/ops/triton/`

## Selection by Operator Family

### Attention
Selection order in `unified_attention()`:
1. Try CK-UA (CK Unified Attention) -- decode only, no sliding window/softcap/ALiBi, moderate occupancy
2. If CK-UA doesn't apply, choose Triton 2D vs 3D based on workgroup count vs GPU size
3. Separate paths: `mha_varlen_fwd()` uses CK-SK (split-KV) or CK-Fwd depending on paged vs contiguous KV

CK-UA activation condition (MI300X with 8 KV-heads):
- Only decode (max_seqlen_q == 1)
- No sliding window, softcap, ALiBi, or sinks
- hdim=64 with GQA-8, or hdim=128 with MHA
- 128-256 sequences (occupancy zone: cu_count*4 <= num_kv_heads*num_seqs <= cu_count*8)

### GEMM
Priority: ASM > CK > CK Tile
- Tuned CSV configs (`aiter/configs/*.csv`) contain kernel names that determine the backend
- If the config specifies an ASM kernel, ASM is used
- Otherwise CK (with block-scale or pre-shuffle variants as applicable)
- M-padding applied automatically to improve tile efficiency

### MoE
Two paths:
1. CK fused path (GEMM1 + activation + quant + GEMM2 in one kernel) when tuned config exists
2. Triton fallback (separate GEMM + activation stages)
Optional FlyDSL backend for A4W4 mixed-precision

### Normalization
- CK-based implementations for RMSNorm, LayerNorm
- Selection between CK classic and CK tile based on hidden size (CK tile for >8192)

## Fallback Behavior
- On Windows: only Triton ops available (CK/HIP require ROCm)
- If ROCm JIT runtime unavailable: warning logged, CK/HIP ops disabled, Triton remains
- If Iris not installed: communication primitives unavailable, `IRIS_COMM_AVAILABLE = False`

## Environment Variables
- `AITER_LOG_TUNED_CONFIG` -- log which tuned config was selected
- `VLLM_ROCM_USE_AITER` -- enable aiter operators in vLLM

## Related Pages
- [[operators/attention]] -- attention kernel selector logic
- [[operators/gemm]] -- GEMM backend selection details
- [[concepts/autotuning]] -- how tuned configs are generated
- [[ck/architecture]] -- CK architecture overview

## Source Files
- `aiter/__init__.py` -- fallback logic (Windows, missing ROCm)
- `aiter/ops/triton/attention/unified_attention.py` -- attention selector (_try_ck_unified_attention, use_2d_kernel)
- `aiter/ops/gemm_op_a8w8.py` -- GEMM A8W8 backend selection
- `aiter/ops/gemm_op_a4w4.py` -- GEMM A4W4 config lookup and backend selection

---
