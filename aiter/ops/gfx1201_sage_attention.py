# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor

from csrc.cpp_itfs.torch_utils import direct_register_custom_op

from ..jit.core import compile_ops
from .gfx1201_sage_prepare import gfx1201_sage_prepare
from .gfx1201_sage_prepare_h3 import gfx1201_sage_prepare_h3


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


@compile_ops(
    "module_gfx1201_sage_attention_asm",
    fc_name="gfx1201_sage_attention_fwd_asm",
    ffi_type="ctypes",
)
def gfx1201_sage_attention_fwd_asm(
    q_int8: Tensor,
    q_scale: Tensor,
    k_int8: Tensor,
    k_scale: Tensor,
    v_fp8: Tensor,
    v_scale: Tensor,
    out: Tensor,
    seq_len: int,
) -> None: ...


@compile_ops(
    "module_gfx1201_sage_attention_h3_asm",
    fc_name="gfx1201_sage_attention_h3_fwd_asm",
    ffi_type="ctypes",
)
def gfx1201_sage_attention_h3_fwd_asm(
    query: Tensor,
    k_int8: Tensor,
    k_scale: Tensor,
    v_fp8: Tensor,
    v_scale: Tensor,
    v_residual: Tensor,
    out: Tensor,
    softmax_scale: float,
) -> None: ...


_BACKENDS = {
    "asm": gfx1201_sage_attention_fwd_asm,
    "hip": gfx1201_sage_attention_fwd_hip,
}


def gfx1201_sage_attention_fwd(
    q_int8: Tensor,
    q_scale: Tensor,
    k_int8: Tensor,
    k_scale: Tensor,
    v_fp8: Tensor,
    v_scale: Tensor,
    seq_len: int,
    backend: str = "asm",
) -> Tensor:
    """Attention over the tensors returned by ``gfx1201_sage_prepare``.

    Returns BF16 ``[B, S_pad, H, 128]``; rows at or beyond ``seq_len`` are left
    uninitialized.

    ``backend="asm"`` runs the hand-scheduled code object, ``"hip"`` the HIP kernel.
    They differ only in how the INT8 Q K^T scores reach the softmax: on (32 query rows,
    32 key rows) tiles with ``q_scale * k_scale <= 1/384`` the code object reads the int32
    scores through a 1.5 * 2^23 bias and folds scale and row maximum into one FMA, which
    adds one rounding of at most 2^-9 in log2 of P. Other tiles use the exact conversion;
    if no tile qualifies, the output is bitwise equal to the HIP kernel.
    """
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {sorted(_BACKENDS)}, got {backend!r}")
    batch, padded, heads, dim = q_int8.shape
    if not 0 < seq_len <= padded or padded % 32:
        raise ValueError(f"seq_len {seq_len} does not fit padded length {padded}")
    out = torch.empty(
        (batch, padded, heads, dim), device=q_int8.device, dtype=torch.bfloat16
    )
    _BACKENDS[backend](q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale, out, seq_len)
    return out


def gfx1201_sage_attention_h3_fwd(
    query: Tensor,
    k_int8: Tensor,
    k_scale: Tensor,
    v_fp8: Tensor,
    v_scale: Tensor,
    v_residual: Tensor,
    softmax_scale: float,
) -> Tensor:
    """H3 attention over raw BF16 ``query`` and the tensors from ``gfx1201_sage_prepare_h3``.

    Returns BF16 ``[B, S, H, 128]``. The code object quantizes Q internally, so unlike
    ``gfx1201_sage_attention_fwd`` no q_int8/q_scale is involved and the output is not padded.
    """
    out = torch.empty_like(query)
    gfx1201_sage_attention_h3_fwd_asm(
        query, k_int8, k_scale, v_fp8, v_scale, v_residual, out, softmax_scale
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
    backend: str = "asm",
) -> Tensor:
    """Non-causal SageAttention for gfx1201: INT8 QK^T, FP8 PV, FP32 accumulation.

    ``query``/``key``/``value``: contiguous BF16 ``[B, S, H, 128]``; returns the same
    shape. Lossy by design (Q/K quantized per 32-row block, V per channel).
    The optional norm/RoPE arguments are applied to Q and K first, see
    ``gfx1201_sage_prepare``. ``backend`` selects the attention kernel, see
    ``gfx1201_sage_attention_fwd``; ``"asm_h3"`` runs the bm512 code object, which quantizes
    Q inside the kernel and does not support the fused norm/RoPE arguments.
    """
    _check(query, key, value)
    if backend == "asm_h3":
        if key.shape != query.shape or value.shape != query.shape:
            raise ValueError("Q/K/V must have equal shapes")
        if any(t is not None for t in (query_weight, key_weight, cosine, sine)):
            raise ValueError(
                "backend='asm_h3' does not implement the fused norm/RoPE path"
            )
        scale = query.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
        with torch.cuda.device(query.device):
            prepared = gfx1201_sage_prepare_h3(key, value)
            return gfx1201_sage_attention_h3_fwd(query, *prepared, scale)
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
        out = gfx1201_sage_attention_fwd(*prepared, query.shape[1], backend)
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
    backend="asm",
):
    return torch.empty_like(query)


direct_register_custom_op(
    op_name="gfx1201_sage_attention",
    op_func=gfx1201_sage_attention,
    mutates_args=[],
    fake_impl=_gfx1201_sage_attention_fake,
)
