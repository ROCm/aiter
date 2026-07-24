# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused inverse RoPE + FP8 block-scaled quantization for DeepSeek-V4 MLA.

Public API wrapping the Triton JIT kernel.  Scale output uses MN-major
layout (token stride = 1) so downstream fp8_einsum / DeepGEMM can consume
it directly without ``transform_sf_into_required_layout``, matching the
vllm ``fused_inv_rope_fp8_quant`` output contract.

Returns::

    o_fp8:   (T, n_groups, D_per_group)  platform fp8, strides (D, T*D, 1)
    o_scale: (T, n_groups, num_k_blocks) fp32,  strides (1, K*T_a, T_a)

Usage::

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


def _tma_align(n: int, alignment: int = 4) -> int:
    return (n + alignment - 1) // alignment * alignment


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
        o: Attention output ``[T, num_heads, head_dim]`` bf16.
        positions: Token positions ``[T]`` int64.
        cos_sin_cache: ``[max_pos, rope_dim]`` fp32, cos||sin concatenated.
        n_groups: Number of KV groups.
        heads_per_group: Q heads per KV group.
        rope_head_dim: RoPE dimensions per head (default 64).
        quant_group_size: FP8 quantization block size (default 128).
        fp8_dtype: Override FP8 dtype (default: auto-detect from platform).

    Returns:
        ``(o_fp8, o_scale)`` where

        * **o_fp8** has shape ``(T, G, D_per_group)`` and dtype *fp8_dtype*.
          Memory layout: transposed view of an internal ``(G, T, D)`` buffer
          — strides ``(D, T*D, 1)``.
        * **o_scale** has shape ``(T, G, num_k_blocks)`` with **MN-major**
          strides ``(1, K*T_aligned, T_aligned)`` where
          ``T_aligned = ceil(T/4)*4``.  The token dimension has stride 1,
          matching what ``fp8_einsum`` expects.
    """
    assert o.dtype == torch.bfloat16
    assert o.dim() == 3

    T, num_heads, head_dim = o.shape
    nope_dim = head_dim - rope_head_dim
    assert num_heads == n_groups * heads_per_group
    assert head_dim % quant_group_size == 0
    assert rope_head_dim % 2 == 0

    d_per_group = heads_per_group * head_dim
    num_k_blocks = d_per_group // quant_group_size
    chunks_per_head = head_dim // quant_group_size

    if fp8_dtype is None:
        from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype

        fp8_dtype = get_fp8_e4m3_dtype()

    if T == 0:
        fp8_out = torch.empty(0, n_groups, d_per_group, dtype=fp8_dtype, device=o.device)
        scale_out = torch.empty(
            0, n_groups, num_k_blocks, dtype=torch.float32, device=o.device
        )
        return fp8_out, scale_out

    if positions.dtype != torch.int64:
        positions = positions.to(torch.int64)

    is_fnuz = fp8_dtype == torch.float8_e4m3fnuz
    fp8_max = torch.finfo(fp8_dtype).max

    fp8_out = torch.empty(n_groups, T, d_per_group, dtype=fp8_dtype, device=o.device)

    T_aligned = _tma_align(T)
    scale_out = torch.empty(
        n_groups * num_k_blocks * T_aligned,
        dtype=torch.float32,
        device=o.device,
    ).as_strided(
        (n_groups, T, num_k_blocks),
        (num_k_blocks * T_aligned, 1, T_aligned),
    )

    grid = (T_aligned, n_groups * heads_per_group)

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
    return fp8_out.transpose(0, 1), scale_out.transpose(0, 1)
