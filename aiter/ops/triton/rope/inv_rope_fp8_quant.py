# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused inverse RoPE + FP8 block-scaled quantization for DeepSeek-V4 MLA.

Public API wrapping the Triton JIT kernel. Automatically detects the platform
FP8 type (fn on gfx950/CDNA4, fnuz on gfx942/CDNA3) to ensure hardware-
accelerated FP8 conversion intrinsics are used.

Usage:
    from aiter.ops.triton.rope.inv_rope_fp8_quant import inv_rope_fp8_quant

    o_fp8, o_scale = inv_rope_fp8_quant(
        o, positions, cos_sin_cache,
        n_groups=4, heads_per_group=4,
    )
"""

from __future__ import annotations

import torch

from aiter.ops.triton._triton_kernels.rope.inv_rope_fp8_quant import (
    _inv_rope_fp8_quant_kernel,
)


def inv_rope_fp8_quant(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    rope_head_dim: int = 64,
    quant_group_size: int = 128,
    fp8_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused inverse RoPE + FP8 block-scaled quantization.

    Args:
        o: Attention output [num_tokens, num_heads, head_dim] bf16.
        positions: Token positions [num_tokens] int64.
        cos_sin_cache: Precomputed [max_pos, rope_dim] fp32 with cos||sin.
        n_groups: Number of KV groups.
        heads_per_group: Q heads per KV group.
        rope_head_dim: RoPE dimensions per head (default 64).
        quant_group_size: FP8 quantization block size (default 128).
        fp8_dtype: Override FP8 dtype (default: auto-detect from platform).

    Returns:
        o_fp8:   (n_groups, T, D_per_group) platform fp8 dtype
        o_scale: (n_groups, T, num_k_blocks) float32
    """
    assert o.dtype == torch.bfloat16
    assert o.dim() == 3

    T, num_heads, head_dim = o.shape
    nope_dim = head_dim - rope_head_dim
    assert num_heads == n_groups * heads_per_group
    assert head_dim % quant_group_size == 0
    assert rope_head_dim % 2 == 0

    if positions.dtype != torch.int64:
        positions = positions.to(torch.int64)

    d_per_group = heads_per_group * head_dim
    num_k_blocks = d_per_group // quant_group_size
    chunks_per_head = head_dim // quant_group_size

    if fp8_dtype is None:
        from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype

        fp8_dtype = get_fp8_e4m3_dtype()

    is_fnuz = fp8_dtype == torch.float8_e4m3fnuz
    fp8_max = torch.finfo(fp8_dtype).max

    fp8_out = torch.empty(n_groups, T, d_per_group, dtype=fp8_dtype, device=o.device)
    scale_out = torch.empty(
        n_groups, T, num_k_blocks, dtype=torch.float32, device=o.device
    )

    if T == 0:
        return fp8_out, scale_out

    grid = (T, n_groups * heads_per_group)

    _inv_rope_fp8_quant_kernel[grid](
        o,
        positions,
        cos_sin_cache,
        fp8_out,
        scale_out,
        T,
        heads_per_group=heads_per_group,
        o_stride_token=o.stride(0),
        o_stride_head=o.stride(1),
        cache_stride_pos=cos_sin_cache.stride(0),
        fp8_stride_group=fp8_out.stride(0),
        fp8_stride_token=fp8_out.stride(1),
        scale_stride_group=scale_out.stride(0),
        scale_stride_token=scale_out.stride(1),
        scale_stride_k=scale_out.stride(2),
        fp8_max=fp8_max,
        eps=1e-10,
        QUANT_GROUP_SIZE=quant_group_size,
        CHUNKS_PER_HEAD=chunks_per_head,
        ROPE_START=nope_dim % quant_group_size,
        HALF_ROPE=rope_head_dim // 2,
        IS_FNUZ=is_fnuz,
        num_warps=1,
        num_stages=1,
    )
    return fp8_out, scale_out
