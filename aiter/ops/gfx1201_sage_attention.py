# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor

from csrc.cpp_itfs.torch_utils import direct_register_custom_op

from ..jit.core import compile_ops
from .gfx1201_sage_prepare import gfx1201_sage_prepare


@compile_ops("module_gfx1201_sage_attention", develop=True)
def gfx1201_sage_attention_fwd_hip(
    q_int8: Tensor,
    q_scale: Tensor,
    k_int8: Tensor,
    k_scale: Tensor,
    v_fp8: Tensor,
    v_scale: Tensor,
    out: Tensor,
    seq_len: int,
) -> None: ...


def gfx1201_sage_attention_fwd(
    q_int8: Tensor,
    q_scale: Tensor,
    k_int8: Tensor,
    k_scale: Tensor,
    v_fp8: Tensor,
    v_scale: Tensor,
    seq_len: int,
) -> Tensor:
    """Attention over the tensors returned by ``gfx1201_sage_prepare``.

    Returns BF16 ``[B, S_pad, H, 128]``; rows at or beyond ``seq_len`` are left
    uninitialized.
    """
    batch, padded, heads, dim = q_int8.shape
    if not 0 < seq_len <= padded or padded % 32:
        raise ValueError(f"seq_len {seq_len} does not fit padded length {padded}")
    out = torch.empty(
        (batch, padded, heads, dim), device=q_int8.device, dtype=torch.bfloat16
    )
    gfx1201_sage_attention_fwd_hip(
        q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale, out, seq_len
    )
    return out


def _check(query, key, value):
    if query.requires_grad or key.requires_grad or value.requires_grad:
        raise ValueError("gfx1201_sage_attention is inference-only")
    if query.device.type != "cuda":
        raise ValueError("gfx1201_sage_attention expects GPU tensors")
    arch = torch.cuda.get_device_properties(query.device).gcnArchName.split(":")[0]
    if arch != "gfx1201":
        raise RuntimeError(f"gfx1201_sage_attention requires gfx1201, got {arch}")


def gfx1201_sage_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    softmax_scale: float | None = None,
    query_weight: Tensor | None = None,
    key_weight: Tensor | None = None,
    cosine: Tensor | None = None,
    sine: Tensor | None = None,
    eps: float = 1e-6,
) -> Tensor:
    """Non-causal SageAttention for gfx1201: INT8 QK^T, FP8 PV, FP32 accumulation.

    ``query``/``key``/``value``: contiguous BF16 ``[B, S, H, 128]``; returns the same
    shape. Lossy by design (Q/K quantized per 32-row block, V per channel).
    The optional norm/RoPE arguments are applied to Q and K first, see
    ``gfx1201_sage_prepare``.
    """
    _check(query, key, value)
    with torch.cuda.device(query.device):
        prepared = gfx1201_sage_prepare(
            query,
            key,
            value,
            softmax_scale,
            query_weight,
            key_weight,
            cosine,
            sine,
            eps,
        )
        out = gfx1201_sage_attention_fwd(*prepared, query.shape[1])
    return out[:, : query.shape[1]].contiguous()


def _gfx1201_sage_attention_fake(
    query,
    key,
    value,
    softmax_scale=None,
    query_weight=None,
    key_weight=None,
    cosine=None,
    sine=None,
    eps=1e-6,
):
    return torch.empty_like(query)


direct_register_custom_op(
    op_name="gfx1201_sage_attention",
    op_func=gfx1201_sage_attention,
    mutates_args=[],
    fake_impl=_gfx1201_sage_attention_fake,
)
