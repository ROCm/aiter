# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Fused Gated RMSNorm + FP8 Per-Token Quantization

Operations:
1. Per-head Gated RMSNorm: norm(x) * silu(z) where:
   - norm(x) = x * weight / sqrt(variance + eps) (standard RMSNorm over head_dim)
   - silu(z) = z / (1 + exp(-z))
2. Flatten: [num_tokens, num_heads, head_dim] -> [num_tokens, num_heads*head_dim]
3. FP8 per-token quantization: ONE scale per token across the full flattened row.

This pairs with a per-output-channel weight scale a8w8 GEMM (per-token activation
scale x per-channel weight scale) with NO GEMM change -- only the activation
production (gated RMSNorm + dynamic per-token quant) is fused.

Constraint: ONLY supports head_dim=128 and num_heads <= 128.
"""

from torch import Tensor

from aiter.jit.core import compile_ops


# Shares the same JIT-compiled module as gated_rmsnorm_fp8_group_quant.
@compile_ops("module_gated_rmsnorm_quant")
def gated_rmsnorm_fp8_per_token_quant(
    out: Tensor,
    scale: Tensor,
    x: Tensor,
    z: Tensor,
    weight: Tensor,
    epsilon: float,
) -> None:
    """
    HIP kernel for fused Gated RMSNorm + FP8 per-token quantization.

    Args:
        out: [num_tokens, num_heads * head_dim] FP8 output (pre-allocated)
        scale: [num_tokens] fp32 per-token scales (pre-allocated)
        x: [num_tokens, num_heads, head_dim] tensor to normalize (bf16/fp16)
        z: [num_tokens, num_heads, head_dim] gating tensor (bf16/fp16)
        weight: [head_dim] RMSNorm weight (bf16/fp16)
        epsilon: numerical stability epsilon

    This is a JIT-compiled binding that will be replaced with the actual kernel.
    """
    ...
