# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl


@triton.jit
def _compute_fp8_scaling_factors(x, fp8_max: tl.constexpr):
    # compute fp8 scaling and descaling factor for a block
    x_amax = tl.max(tl.abs(x))  # NOTE: abs deals with negative values
    x_amax = tl.where(x_amax <= 1e-9, 1e-9, x_amax)
    scale_x = fp8_max / x_amax
    descale_x = x_amax / fp8_max
    return scale_x, descale_x

# TODO: a Triton version of this function
import math
import torch
from typing import Tuple
def int8_per_block_quantize_bshd(
    x: torch.Tensor,
    int8_dtype: torch.dtype,
    clamp_val: float = 1e-9,
    block_size: int = 128,
    include_sqrt_scale: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a tensor to INT8 format, returning an INT8 tensor and a descale factor.
    Quantization is done per block on the seqlen dimension.
    x: [batch, seqlen, heads, dim] (bshd)
    include_sqrt_scale: if True, include 1/sqrt(head_dim) in descale for attention's Q
    Returns:
        x_int8: same shape as x, stored as int8_dtype
        descale_factor: [batch, heads, num_blocks, 1] scale to dequantize:
                        x_fp32 ≈ x_int8 * descale_factor
    """
    if len(x.shape) != 4:
        raise ValueError(
            f"'bshd' tensor should have shape [batch, seqlen, heads, dim], got {x.shape}"
        )
    batch, seqlen, num_heads, head_dim = x.shape
    if seqlen % block_size != 0:
        raise ValueError(
            f"seqlen={seqlen} must be divisible by block_size={block_size} for per-block quantization"
        )
    # Reshape to expose blocks along seqlen: [b, n_blocks, block_size, h, d]
    n_blocks = seqlen // block_size
    x_reshaped = x.view(batch, n_blocks, block_size, num_heads, head_dim)
    # Compute max absolute value per block (reduce over block_size and dim)
    # Shape: [b, n_blocks, num_heads, 1]
    max_abs = x_reshaped.abs().amax(dim=2)        # [b, n_blocks, h, d]
    max_abs = max_abs.amax(dim=-1, keepdim=True)  # [b, n_blocks, h, 1]
    # Avoid division by zero
    max_abs = torch.clamp(max_abs, min=clamp_val)
    # Symmetric INT8 range
    qmax = torch.iinfo(torch.int8).max  # 127
    # Scale used for quantization (fp32 -> int8)
    scale = qmax / max_abs  # [b, n_blocks, h, 1]
    # Apply scale per block
    # Broadcast scale over block_size and dim
    x_scaled = x_reshaped * scale.unsqueeze(2)  # [b, n_blocks, block_size, h, d]
    # Quantize and clamp to valid INT8 range
    x_int8 = torch.round(x_scaled).clamp(-qmax - 1, qmax).to(int8_dtype)
    # Reshape back to [b, s, h, d]
    x_int8 = x_int8.view(batch, seqlen, num_heads, head_dim)
    # Descale factor for dequantization: x_fp32 ≈ x_int8 * descale_factor
    descale_factor = 1.0 / scale
    # Include 1/sqrt(head_dim) for attention scaling (applied to Q's descale)
    if include_sqrt_scale:
        descale_factor = descale_factor / math.sqrt(head_dim)
    # Kernel expects scale in [b, h, n_blocks, 1] format, so permute from [b, n_blocks, h, 1]
    # Must call contiguous() after permute so kernel's pointer arithmetic (+= 1) works correctly
    descale_factor = descale_factor.permute(0, 2, 1, 3).contiguous()  # [b, h, n_blocks, 1]
    return x_int8, descale_factor