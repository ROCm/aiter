# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Pool a BSHD tensor along the sequence axis into one row per block of tokens, for Sol-Attn's pooled
Q/K/V: the block mean, plus optionally the block's population variance or its mean of squares,
from a single read of the input.
"""

from __future__ import annotations

import torch
import triton

from aiter.ops.triton._triton_kernels.attention.sol_attn_pool import (
    POOL_AUX_NONE,
    POOL_AUX_SQUARE_MEAN,
    POOL_AUX_VARIANCE,
    _sol_attn_pool_kernel,
)

__all__ = [
    "POOL_AUX_NONE",
    "POOL_AUX_SQUARE_MEAN",
    "POOL_AUX_VARIANCE",
    "sol_attn_pool",
]


def _pool_tile(block: int, element_size: int, aux: int) -> tuple[int, int, int]:
    """(ROWS, HEADS_PER_PROGRAM, num_warps) for one launch.

    Measured on MI355X at hd128 and 40 heads, 16K and 32K tokens: the mean alone runs at 5.4-5.7
    TB/s for bf16 and 5.0-5.2 TB/s for fp8 in 8-row chunks of one head over four warps, within
    10% of the best tile found in every case but an 8-bit 64-token block, which is too small a
    program and wants two heads on one warp. With a second statistic, 32-row chunks and one warp
    per 128 bytes of block height stay within 7%, except that an 8-bit input's two fp32
    accumulators want shorter chunks as the block grows.
    """
    if aux == POOL_AUX_NONE:
        return (16, 2, 1) if element_size == 1 and block <= 64 else (8, 1, 4)
    if element_size == 1:
        return max(8, 2048 // block), 1, 4 if block >= 256 else 1
    return min(block, 32), 1, max(1, min(8, block * element_size // 128))


def sol_attn_pool(
    x: torch.Tensor,
    block: int,
    mean_dtype: torch.dtype,
    aux: int = POOL_AUX_NONE,
    aux_dtype: torch.dtype | None = None,
    aux_multiplier: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Pool BSHD `x` over each `block` tokens, returning (mean, aux), both contiguous
    (batch, ceil(seqlen / block), nheads, d).

    mean is stored in `mean_dtype`, rounded to nearest-even first when that is an integer dtype
    (a plain cast would truncate). aux is None for POOL_AUX_NONE; otherwise the per-block
    population variance (POOL_AUX_VARIANCE) or mean of squares (POOL_AUX_SQUARE_MEAN), times
    `aux_multiplier`, in `aux_dtype`. Everything accumulates in fp32.
    """
    batch, seqlen, nheads, head_dim = x.shape
    if head_dim & (head_dim - 1):
        raise ValueError(f"sol_attn_pool needs a power-of-two head dim, got {head_dim}")
    num_blocks = triton.cdiv(seqlen, block)
    mean = torch.empty(
        (batch, num_blocks, nheads, head_dim), dtype=mean_dtype, device=x.device
    )
    aux_out = None
    if aux != POOL_AUX_NONE:
        aux_out = torch.empty(
            (batch, num_blocks, nheads, head_dim), dtype=aux_dtype, device=x.device
        )

    rows, heads_per_program, num_warps = _pool_tile(block, x.element_size(), aux)
    rows = 1 << (rows.bit_length() - 1)
    # The kernel only masks the sequence end, so a chunk must not straddle two blocks.
    while block % rows:
        rows //= 2
    grid = (batch * num_blocks * triton.cdiv(nheads, heads_per_program),)
    _sol_attn_pool_kernel[grid](
        x,
        mean,
        aux_out if aux_out is not None else mean,
        seqlen,
        nheads,
        num_blocks,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        aux_multiplier,
        BLOCK=block,
        ROWS=rows,
        HEAD_DIM=head_dim,
        HEADS_PER_PROGRAM=heads_per_program,
        AUX=aux,
        ROUND=not mean_dtype.is_floating_point,
        num_warps=num_warps,
    )
    return mean, aux_out
