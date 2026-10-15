# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor

from ..jit.core import compile_ops

_HEAD_DIM = 128
_K_GROUP_ROWS = 64
_V_PAD_ROWS = 16


@compile_ops("module_gfx1201_sage_prepare_h3", develop=True)
def gfx1201_sage_prepare_h3_hip(
    key: Tensor,
    value: Tensor,
    key_mean: Tensor,
    key_out: Tensor,
    key_scale: Tensor,
    value_out: Tensor,
    value_scale: Tensor,
) -> None: ...


def gfx1201_sage_prepare_h3(
    key: Tensor,
    value: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Quantize BF16 ``[B, S, H, 128]`` K/V for the H3 SageAttention core.

    Returns ``(k_int8, k_scale, v_fp8, v_scale)``: K as INT8 ``[B, S, H, 128]`` with FP32
    scales ``[B, H, ceil(S / 64)]`` taken after subtracting the per-channel K mean, and V as
    FP8 e4m3 ``[B, H, 128, S_pad]`` with per-channel FP32 scales ``[B, H, 128]``, where
    ``S_pad`` rounds ``S`` up to 16 and the padded columns are zero.

    Q is not quantized here: the H3 core reads BF16 Q and quantizes it per 32-row block
    internally, which is what distinguishes it from ``gfx1201_sage_prepare``.
    """
    if key.ndim != 4 or key.shape[-1] != _HEAD_DIM:
        raise ValueError(f"Expected K/V [B, S, H, {_HEAD_DIM}], got {tuple(key.shape)}")
    if value.shape != key.shape:
        raise ValueError("K and V must have equal shapes")
    if any(size <= 0 for size in key.shape):
        raise ValueError("Expected positive dimensions")
    for tensor in (key, value):
        if (
            tensor.dtype != torch.bfloat16
            or not tensor.is_contiguous()
            or tensor.device != key.device
        ):
            raise ValueError("K/V must be contiguous BF16 tensors on the same device")
    batch, rows, heads, _ = key.shape
    padded = (rows + _V_PAD_ROWS - 1) // _V_PAD_ROWS * _V_PAD_ROWS
    if batch * padded * heads * _HEAD_DIM >= 2**31:
        raise ValueError("K/V exceed the 32-bit index range")

    device = key.device
    groups = (rows + _K_GROUP_ROWS - 1) // _K_GROUP_ROWS
    key_mean = torch.empty(
        (batch, heads, _HEAD_DIM), device=device, dtype=torch.bfloat16
    )
    key_int8 = torch.empty_like(key, dtype=torch.int8)
    key_scale = torch.empty((batch, heads, groups), device=device, dtype=torch.float32)
    value_fp8 = torch.empty(
        (batch, heads, _HEAD_DIM, padded), device=device, dtype=torch.float8_e4m3fn
    )
    value_scale = torch.empty(
        (batch, heads, _HEAD_DIM), device=device, dtype=torch.float32
    )
    gfx1201_sage_prepare_h3_hip(
        key, value, key_mean, key_int8, key_scale, value_fp8, value_scale
    )
    return key_int8, key_scale, value_fp8, value_scale
