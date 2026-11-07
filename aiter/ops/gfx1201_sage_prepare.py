# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import math

import torch
from torch import Tensor

from ..jit.core import compile_ops

_HEAD_DIM = 128
_BLOCK_ROWS = 32
_LOG2E = 1.4426950408889634


@compile_ops("module_gfx1201_sage_prepare", develop=True)
def gfx1201_sage_prepare_hip(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    query_weight: Tensor | None,
    key_weight: Tensor | None,
    cosine: Tensor | None,
    sine: Tensor | None,
    maximum: Tensor,
    query_out: Tensor,
    key_out: Tensor,
    value_out: Tensor,
    query_scale: Tensor,
    key_scale: Tensor,
    value_scale: Tensor,
    rope_dim: int,
    eps: float,
    sm_scale: float,
) -> None: ...


def gfx1201_sage_prepare(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    softmax_scale: float | None = None,
    query_weight: Tensor | None = None,
    key_weight: Tensor | None = None,
    cosine: Tensor | None = None,
    sine: Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Quantize BF16 ``[B, S, H, 128]`` Q/K/V for gfx1201 SageAttention.

    Returns ``(q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale)``:
    Q/K as INT8 ``[B, S_pad, H, 128]`` with FP32 scales ``[B, H, S_pad / 32]``
    (one per 32-row block; Q carries ``softmax_scale * log2(e)``), V as FP8 e4m3
    ``[B, H, 128, S_pad]`` with per-channel FP32 scales ``[B, H, 128]``.
    ``S_pad`` is ``S`` rounded up to 32; padded rows are zero.

    If ``query_weight``/``key_weight`` (BF16 ``[128]``) and ``cosine``/``sine``
    (FP32 ``[S, rope_dim]``, rope_dim in 32/64/96/128) are given, Q and K first go
    through RMSNorm and rotate-half RoPE on their leading ``rope_dim`` channels,
    rounded like the eager PyTorch BF16 sequence.
    """
    if query.ndim != 4 or query.shape[-1] != _HEAD_DIM:
        raise ValueError(
            f"Expected Q/K/V [B, S, H, {_HEAD_DIM}], got {tuple(query.shape)}"
        )
    if key.shape != query.shape or value.shape != query.shape:
        raise ValueError("Q/K/V must have equal shapes")
    if any(size <= 0 for size in query.shape):
        raise ValueError("Expected positive dimensions")
    for tensor in (query, key, value):
        if (
            tensor.dtype != torch.bfloat16
            or not tensor.is_contiguous()
            or tensor.device != query.device
        ):
            raise ValueError("Q/K/V must be contiguous BF16 tensors on the same device")
    batch, rows, heads, _ = query.shape
    padded = (rows + _BLOCK_ROWS - 1) // _BLOCK_ROWS * _BLOCK_ROWS
    if batch * padded * heads * _HEAD_DIM >= 2**31:
        raise ValueError("Q/K/V exceed the 32-bit index range")

    fused = (query_weight, key_weight, cosine, sine)
    rope_dim = 0
    if any(t is not None for t in fused):
        if any(t is None for t in fused):
            raise ValueError(
                "query_weight, key_weight, cosine and sine must be given together"
            )
        for weight in (query_weight, key_weight):
            if (
                weight.shape != (_HEAD_DIM,)
                or weight.dtype != torch.bfloat16
                or not weight.is_contiguous()
            ):
                raise ValueError(f"Norm weights must be contiguous BF16 [{_HEAD_DIM}]")
        rope_dim = cosine.shape[-1]
        if rope_dim not in (32, 64, 96, 128):
            raise ValueError(f"rope_dim must be 32, 64, 96 or 128, got {rope_dim}")
        for table in (cosine, sine):
            if (
                table.shape != (rows, rope_dim)
                or table.dtype != torch.float32
                or not table.is_contiguous()
            ):
                raise ValueError(
                    f"cosine/sine must be contiguous FP32 [{rows}, {rope_dim}]"
                )

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(_HEAD_DIM)
    device = query.device
    query_int8 = torch.empty(
        (batch, padded, heads, _HEAD_DIM), device=device, dtype=torch.int8
    )
    key_int8 = torch.empty_like(query_int8)
    value_fp8 = torch.empty(
        (batch, heads, _HEAD_DIM, padded), device=device, dtype=torch.float8_e4m3fn
    )
    query_scale = torch.empty(
        (batch, heads, padded // _BLOCK_ROWS), device=device, dtype=torch.float32
    )
    key_scale = torch.empty_like(query_scale)
    value_scale = torch.empty(
        (batch, heads, _HEAD_DIM), device=device, dtype=torch.float32
    )
    maximum = torch.empty((batch, heads * _HEAD_DIM), device=device, dtype=torch.int32)
    gfx1201_sage_prepare_hip(
        query,
        key,
        value,
        query_weight,
        key_weight,
        cosine,
        sine,
        maximum,
        query_int8,
        key_int8,
        value_fp8,
        query_scale,
        key_scale,
        value_scale,
        rope_dim,
        eps,
        softmax_scale * _LOG2E,
    )
    return query_int8, query_scale, key_int8, key_scale, value_fp8, value_scale
