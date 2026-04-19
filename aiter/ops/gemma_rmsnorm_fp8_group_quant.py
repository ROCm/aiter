# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Fused Gemma RMSNorm + FP8 Group Quantization

Operations:
1. Optional residual add: x = x + residual (written back inplace to residual)
2. Gemma RMSNorm: out = x * rsqrt(mean(x^2) + eps) * (1 + weight)
   - Variance computed over full hidden_size
   - Gemma-style weight: (1 + weight) instead of weight
3. FP8 group quantization with group_size=128

Constraint: hidden_size must be a multiple of 128, group_size must be 128
"""

from typing import Optional

from torch import Tensor

from aiter.jit.core import compile_ops


@compile_ops("module_gemma_rmsnorm_quant")
def gemma_rmsnorm_fp8_group_quant(
    out: Tensor,
    scale: Tensor,
    x: Tensor,
    weight: Tensor,
    epsilon: float,
    group_size: int,
    transpose_scale: bool = False,
    residual: Optional[Tensor] = None,
) -> None:
    """
    HIP kernel for fused Gemma RMSNorm + FP8 group quantization.

    This is a JIT-compiled binding that will be replaced with the actual kernel.
    """
    ...
