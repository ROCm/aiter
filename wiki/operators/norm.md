---
title: "Normalization Operators"
last_verified: 2026-04-06
source_files:
  - aiter/ops/norm.py
  - aiter/ops/rmsnorm.py
  - aiter/ops/groupnorm.py
tags: [normalization, rmsnorm, layernorm, groupnorm, ck]
---

# Normalization Operators

## Overview
aiter provides RMSNorm, LayerNorm, and GroupNorm operators with CK-based implementations. Many fused variants exist that combine normalization with residual addition, quantization, or smooth quantization in a single kernel.

## RMSNorm

### Basic Variants
- `rms_norm(input, weight, epsilon)` -- returns normalized tensor
- `rms_norm_cu(out, input, weight, epsilon)` -- in-place CUDA-style
- `rmsnorm2d_fwd(input, weight, epsilon, use_model_sensitive_rmsnorm)` -- auto-selects between CK classic (hidden_size > 8192) and CK tile

### Fused Variants
- `rmsnorm2d_fwd_with_add(out, input, residual_in, residual_out, weight, epsilon)` -- fused residual add + RMSNorm
- `rmsnorm2d_fwd_with_smoothquant(...)` -- fused RMSNorm + smooth quantization
- `rmsnorm2d_fwd_with_dynamicquant(...)` -- fused RMSNorm + dynamic FP8 quantization (with optional group_size and scale shuffle)
- `rmsnorm2d_fwd_with_add_smoothquant(...)` -- fused add + RMSNorm + smooth quant
- `rmsnorm2d_fwd_with_add_dynamicquant(...)` -- fused add + RMSNorm + dynamic quant

### Backend Selection
- Hidden size > 8192 or `use_model_sensitive_rmsnorm > 0`: uses CK classic (`rmsnorm2d_fwd_ck`)
- Otherwise: uses CK tile-based implementation (`rmsnorm_quant` from `module_rmsnorm_quant`)

## LayerNorm
- `layer_norm(input, weight, bias, epsilon, x_bias)` -- standard layer normalization
- `layernorm2d_fwd(input, weight, bias, epsilon, x_bias)` -- same, explicit 2D
- `layernorm2d_fwd_with_add(...)` -- fused residual add + LayerNorm
- `layernorm2d_fwd_with_smoothquant(...)` -- fused LayerNorm + smooth quant

## GroupNorm
- Module: `groupnorm.py`
- Group normalization for conv-style architectures

## Related Pages
- [[operators/quant]] -- quantization operators (used in fused norm+quant)
- [[concepts/backend-selection]] -- CK classic vs CK tile selection

## Source Files
- `aiter/ops/rmsnorm.py` -- RMSNorm implementations and fused variants (~276 lines)
- `aiter/ops/norm.py` -- LayerNorm implementations (~147 lines)
- `aiter/ops/groupnorm.py` -- GroupNorm
