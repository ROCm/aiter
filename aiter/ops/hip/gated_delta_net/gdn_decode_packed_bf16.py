# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import torch
from torch import Tensor

from aiter.jit.core import compile_ops

NUM_QK_HEADS = 8
NUM_V_HEADS = 32
HEAD_K_DIM = 128
HEAD_V_DIM = 128
QKV_DIM = 6144


@compile_ops("module_gdn_decode_packed_bf16", fc_name="gdn_decode_packed_bf16")
def _gdn_decode_packed_bf16(
    mixed_qkv: Tensor,
    a: Tensor,
    b: Tensor,
    dt_bias: Tensor,
    A_log: Tensor,
    indices: Tensor,
    state: Tensor,
    out: Tensor,
    scale: float,
) -> None: ...


def gdn_decode_packed_bf16_hip(
    mixed_qkv: Tensor,
    a: Tensor,
    b: Tensor,
    dt_bias: Tensor,
    A_log: Tensor,
    indices: Tensor,
    state: Tensor,
    out: Tensor,
    *,
    scale: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Decode one token from packed QKV and update BF16 V-major state in place.

    Every non-negative entry in ``indices`` must be smaller than the state-pool
    size and unique within the batch. Negative sentinel entries may repeat.
    Duplicate valid indices cause unsynchronized concurrent state updates and
    therefore have undefined behavior.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("packed BF16 GDN decode requires an available ROCm GPU")

    device = torch.cuda.current_device()
    arch = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")
    if "gfx950" not in arch:
        raise RuntimeError(f"packed BF16 GDN decode requires gfx950, got {arch!r}")

    expected_scale = HEAD_K_DIM**-0.5
    if scale is None:
        scale = expected_scale
    if abs(float(scale) - expected_scale) > 1e-12:
        raise ValueError(f"expected scale={expected_scale}, got {scale}")

    if mixed_qkv.ndim != 2 or mixed_qkv.shape[1] != QKV_DIM:
        raise ValueError(f"mixed_qkv must have shape [B,{QKV_DIM}]")
    batch = mixed_qkv.shape[0]
    if mixed_qkv.stride(1) != 1 or mixed_qkv.stride(0) * mixed_qkv.element_size() % 16:
        raise ValueError("mixed_qkv must have a contiguous, 16B-aligned row layout")
    if mixed_qkv.data_ptr() % 16:
        raise ValueError("mixed_qkv base must be 16B aligned")

    for name, tensor in (("a", a), ("b", b)):
        if tensor.ndim != 2 or tensor.shape != (batch, NUM_V_HEADS):
            raise ValueError(f"{name} must have shape [B,{NUM_V_HEADS}]")
        if tensor.stride(1) != 1:
            raise ValueError(f"{name} must be contiguous in its last dimension")

    if dt_bias.shape != (NUM_V_HEADS,) or dt_bias.stride(0) != 1:
        raise ValueError("dt_bias must be contiguous [32]")
    if A_log.shape != (NUM_V_HEADS,) or A_log.stride(0) != 1:
        raise ValueError("A_log must be contiguous [32]")
    if indices.shape != (batch,) or indices.stride(0) < 1:
        raise ValueError("indices must have shape [B] and a positive stride")

    if state.ndim != 4 or state.shape[1:] != (
        NUM_V_HEADS,
        HEAD_V_DIM,
        HEAD_K_DIM,
    ):
        raise ValueError("state must have shape [pool,32,128,128]")
    if state.stride()[1:] != (HEAD_V_DIM * HEAD_K_DIM, HEAD_K_DIM, 1):
        raise ValueError("state must use V-major [slot,HV,V,K] inner strides")
    if state.data_ptr() % 16 or state.stride(0) * state.element_size() % 16:
        raise ValueError("state base and slot stride must be 16B aligned")

    if out.shape != (batch, 1, NUM_V_HEADS, HEAD_V_DIM) or not out.is_contiguous():
        raise ValueError("out must be contiguous [B,1,32,128]")

    tensors = (mixed_qkv, a, b, dt_bias, A_log, indices, state, out)
    if any(tensor.device != mixed_qkv.device for tensor in tensors):
        raise ValueError("all tensors must be on the same device")
    if any(
        tensor.dtype != torch.bfloat16
        for tensor in (mixed_qkv, a, b, dt_bias, state, out)
    ):
        raise ValueError("mixed_qkv/a/b/dt_bias/state/out must be BF16")
    if A_log.dtype != torch.float32 or indices.dtype != torch.int32:
        raise ValueError("A_log must be FP32 and indices must be INT32")

    _gdn_decode_packed_bf16(
        mixed_qkv,
        a,
        b,
        dt_bias,
        A_log,
        indices,
        state,
        out,
        float(scale),
    )
    return out, state
