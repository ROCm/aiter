# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused QK RMSNorm, RoPE, and Sage INT8/FP8 preparation for gfx1201 (7/14/28/56 heads)."""

import torch
from torch import Tensor

from aiter.jit.core import compile_ops

HEADS = (7, 14, 28, 56)


@compile_ops(
    "module_gfx1201_norm_rope_prepare",
    fc_name="gfx1201_norm_rope_prepare_hip",
    develop=True,
)
def gfx1201_norm_rope_prepare_hip(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cosine: Tensor,
    sine: Tensor,
    maximum: Tensor,
    query_weight: Tensor,
    key_weight: Tensor,
    query_out: Tensor,
    key_out: Tensor,
    value_out: Tensor,
    query_scale: Tensor,
    key_scale: Tensor,
    value_scale: Tensor,
    rows: int,
    padded_rows: int,
    sm_scale: float,
    parts: int,
    heads: int,
) -> None:
    """Fused QK RMSNorm, RoPE, and Sage quantization. Writes the prepared tensors."""


def norm_rope_prepare_sage(query, key, value, query_weight, key_weight, cosine, sine, parts=3):
    """query/key/value: pre-norm BF16 ``[S, H, 128]`` with H in ``HEADS`` (the TP2 shard is 28 heads, Ulysses
    SP2/4/8 shards 28/14/7); norm weights BF16 ``[128]``, eps ``1e-5``; cosine/sine FP32 ``[S, 96]``.
    """
    rows, heads = query.shape[0], query.shape[1]
    if heads not in HEADS or query.shape != (rows, heads, 128) or key.shape != query.shape or value.shape != query.shape:
        raise ValueError(f"Expected Q/K/V [S,H,128] with H in {HEADS}, got {tuple(query.shape)}")
    if any(t.dtype != torch.bfloat16 or not t.is_contiguous() for t in (query, key, value)):
        raise ValueError("Expected contiguous BF16 Q/K/V")
    if any(t.shape != (rows, 96) or t.dtype != torch.float32 or not t.is_contiguous() for t in (cosine, sine)):
        raise ValueError("Expected contiguous FP32 cosine/sine [S,96]")
    if any(t.shape != (128,) or t.dtype != torch.bfloat16 or not t.is_contiguous() for t in (query_weight, key_weight)):
        raise ValueError("Expected contiguous BF16 norm weights [128]")
    if query.numel() >= 2**31:
        raise ValueError("Expected Q/K/V within the 32-bit index range")
    padded = (rows + 31) // 32 * 32
    device = query.device
    query_int8 = torch.empty((1, padded, heads, 128), device=device, dtype=torch.int8)
    key_int8 = torch.empty_like(query_int8)
    value_fp8 = torch.empty((1, heads, 128, padded), device=device, dtype=torch.float8_e4m3fn)
    query_scale = torch.empty((1, heads, padded // 32), device=device, dtype=torch.float32)
    key_scale = torch.empty_like(query_scale)
    value_scale = torch.empty((1, heads, 128), device=device, dtype=torch.float32)
    maximum = torch.empty((heads * 128,), device=device, dtype=torch.int32)
    gfx1201_norm_rope_prepare_hip(
        query,
        key,
        value,
        cosine,
        sine,
        maximum,
        query_weight,
        key_weight,
        query_int8,
        key_int8,
        value_fp8,
        query_scale,
        key_scale,
        value_scale,
        rows,
        padded,
        128**-0.5 * 1.4426950408889634,
        parts,
        heads,
    )
    return query_int8, query_scale, key_int8, key_scale, value_fp8, value_scale
