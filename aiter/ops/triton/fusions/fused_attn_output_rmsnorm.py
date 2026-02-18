# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Python interface for Fused Attention Output + RMSNorm

This module provides a fused kernel that combines:
1. Attention output
2. Residual connection (optional)
3. RMSNorm
4. Output padding (optional, for MoE compatibility)

into a single GPU kernel for better performance.
"""

import torch
import triton
from typing import Tuple, Union
from aiter.ops.triton._triton_kernels.fusions.fused_attn_output_rmsnorm import (
    _fused_attn_output_rmsnorm_kernel,
    _get_config,
)


def fused_attn_output_rmsnorm(
    attn_output: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-5,
    residual: torch.Tensor = None,
    x_pad_to_multiple: int = 0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Fused Attention Output + Residual + RMSNorm (with optional padding)
    
    This function fuses three operations:
    1. Add residual connection (optional)
    2. Apply RMSNorm
    3. Pad output to multiple (optional, for MoE compatibility)
    
    Args:
        attn_output: [M, N] - attention output tensor
        weight: [N] - RMSNorm weight
        epsilon: small value for numerical stability (default: 1e-5)
        residual: [M, N] - optional residual to add before normalization
        x_pad_to_multiple: if > 0, pad output N dimension to this multiple
        
    Returns:
        If residual is None:
            output: [M, N_out] - normalized output (N_out may be padded)
        If residual is provided:
            (output, residual_out): tuple of normalized output and updated residual
        
    Example:
        >>> # Without residual
        >>> attn_out = attention(q, k, v)  # [batch*seq, hidden]
        >>> norm_out = fused_attn_output_rmsnorm(attn_out, weight)
        
        >>> # With residual connection
        >>> norm_out, residual = fused_attn_output_rmsnorm(
        ...     attn_out, weight, residual=x
        ... )
        
        >>> # With padding for MoE (pad to 256)
        >>> norm_out, residual = fused_attn_output_rmsnorm(
        ...     attn_out, weight, residual=x, x_pad_to_multiple=256
        ... )
    """
    M, N = attn_output.shape
    assert weight.shape[0] == N, f"Weight shape mismatch: {weight.shape[0]} != {N}"
    
    # Calculate output dimension with padding
    if x_pad_to_multiple > 0:
        N_out = triton.cdiv(N, x_pad_to_multiple) * x_pad_to_multiple
    else:
        N_out = N
    
    # Allocate output
    output = torch.empty((M, N_out), dtype=attn_output.dtype, device=attn_output.device)
    
    # Determine if we have residual
    has_residual = residual is not None
    
    # Allocate residual output if needed
    residual_out = None
    if has_residual:
        residual_out = torch.empty((M, N), dtype=residual.dtype, device=residual.device)
    
    # Get configuration
    config = _get_config(N)
    BLOCK_SIZE_N = config["BLOCK_SIZE_N"]
    num_warps = config["num_warps"]
    num_stages = config["num_stages"]
    
    # Ensure BLOCK_SIZE_N is power of 2 and >= N_out
    BLOCK_SIZE_N = triton.next_power_of_2(max(BLOCK_SIZE_N, N_out))
    
    # Grid: one program per row
    grid = (M,)
    
    # Launch kernel
    _fused_attn_output_rmsnorm_kernel[grid](
        # Input pointers
        attn_output,
        residual if has_residual else attn_output,  # dummy if no residual
        weight,
        # Output pointers
        output,
        residual_out if has_residual else output,  # dummy if no residual
        # Dimensions
        M,
        N,
        # Strides
        attn_output.stride(0),
        attn_output.stride(1),
        residual.stride(0) if has_residual else 0,
        residual.stride(1) if has_residual else 0,
        output.stride(0),
        output.stride(1),
        residual_out.stride(0) if has_residual else 0,
        residual_out.stride(1) if has_residual else 0,
        # Parameters
        epsilon,
        # Meta-parameters
        HAS_RESIDUAL=has_residual,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    
    # Return based on whether residual was provided
    if has_residual:
        return output, residual_out
    return output


__all__ = ["fused_attn_output_rmsnorm"]
