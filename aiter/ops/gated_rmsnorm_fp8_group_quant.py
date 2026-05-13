# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Fused Gated RMSNorm + FP8 Group Quantization

Operations:
1. Per-head Gated RMSNorm: norm(x) * silu(z) where:
   - norm(x) = x * weight / sqrt(variance + eps) (standard RMSNorm)
   - silu(z) = z / (1 + exp(-z))
2. Flatten: [num_tokens, num_heads, head_dim] -> [num_tokens, num_heads*head_dim]
3. FP8 group quantization with group_size=128

Constraint: ONLY supports head_dim=128 and group_size=128
"""

from torch import Tensor

from aiter.jit.core import compile_ops


# This is the JIT-compiled binding to the C++ kernel
@compile_ops("module_gated_rmsnorm_quant")
def gated_rmsnorm_fp8_group_quant(
    out: Tensor,
    scale: Tensor,
    x: Tensor,
    z: Tensor,
    weight: Tensor,
    epsilon: float,
    group_size: int,
    transpose_scale: bool = False,
) -> None:
    """
    HIP kernel for fused Gated RMSNorm + FP8 group quantization.

    This is a JIT-compiled binding that will be replaced with the actual kernel.
    """
    ...
