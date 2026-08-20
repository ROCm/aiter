# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li

"""
Gate activation kernels for chunk_delta_attn.

Provides:
  - beta_sigmoid_fwd: elementwise sigmoid for the beta gate (forward only).
  - chunk_delta_attn_gate_fwd: per-token fused gate (-exp(A)*softplus or
    lower_bound*sigmoid(exp(A)*g)) without cumsum (forward only).
"""

import torch
import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.chunk_delta_attn.chunk_delta_attn_utils import (
    autotune_cache_kwargs,
    exp,
    input_guard,
    softplus,
)

_BETA_SIGMOID_BLOCK_SIZE = 2048
_BETA_SIGMOID_NUM_WARPS = 8

BT_LIST = [32, 64, 128]
NUM_WARPS_AUTOTUNE = [2, 4, 8, 16]


@triton.jit
def beta_sigmoid_fwd_kernel(
    x,
    y,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < n_elements
    b_x = tl.load(x + offs, mask=mask, other=0).to(tl.float32)
    b_y = tl.sigmoid(b_x)
    tl.store(y + offs, b_y.to(y.dtype.element_ty), mask=mask)


@input_guard
def beta_sigmoid_fwd(x: torch.Tensor) -> torch.Tensor:
    """Elementwise sigmoid of ``x``, output in float32."""
    y = torch.empty_like(x, dtype=torch.float32)
    n = x.numel()
    grid = (triton.cdiv(n, _BETA_SIGMOID_BLOCK_SIZE),)
    beta_sigmoid_fwd_kernel[grid](
        x,
        y,
        n,
        BLOCK_SIZE=_BETA_SIGMOID_BLOCK_SIZE,
        num_warps=_BETA_SIGMOID_NUM_WARPS,
    )
    return y


@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["dt_bias"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=nw, num_stages=ns)
        for BT in BT_LIST
        for nw in NUM_WARPS_AUTOTUNE
        for ns in [2, 3]
    ],
    key=["H", "D"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_delta_attn_gate_fwd_kernel(
    g,
    A_log,
    dt_bias,
    yg,
    lower_bound,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    """Per-token gate: yg = -exp(A) * softplus(g + bias)  OR  lb * sigmoid(exp(A) * g)."""
    i_t, i_h = tl.program_id(0).to(tl.int64), tl.program_id(1)

    b_A = tl.load(A_log + i_h).to(tl.float32)

    o_t = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m_t = o_t < T
    m_g = m_t[:, None] & (o_d[None, :] < D)
    p_g = g + i_h * D + o_t[:, None] * (H * D) + o_d[None, :]
    p_yg = yg + i_h * D + o_t[:, None] * (H * D) + o_d[None, :]

    b_g = tl.load(p_g, mask=m_g, other=0.0).to(tl.float32)

    if HAS_BIAS:
        o_b = i_h * D + tl.arange(0, BD)
        b_g = b_g + tl.load(dt_bias + o_b, mask=o_b < H * D, other=0.0).to(tl.float32)

    if not USE_LOWER_BOUND:
        b_yg = -exp(b_A) * softplus(b_g)
    else:
        b_yg = lower_bound * tl.sigmoid(exp(b_A) * b_g)

    tl.store(p_yg, b_yg.to(p_yg.dtype.element_ty), mask=m_g)


@input_guard
def chunk_delta_attn_gate_fwd(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Per-token gate (no cumsum).

    Args:
        g:            Gate input ``[..., H, D]``.
        A_log:        Per-head log-scale ``[H]``.
        dt_bias:      Optional per-head bias ``[H * D]``.
        lower_bound:  If set, use sigmoid gating; else softplus gating.
        output_dtype: Output dtype (default float32).

    Returns:
        Gated tensor, same shape as ``g``.
    """
    H, D = g.shape[-2:]
    T = g.numel() // (H * D)

    yg = torch.empty_like(g, dtype=output_dtype)

    def grid(meta):
        return (triton.cdiv(T, meta["BT"]), H)

    chunk_delta_attn_gate_fwd_kernel[grid](
        g=g,
        A_log=A_log,
        dt_bias=dt_bias,
        yg=yg,
        lower_bound=lower_bound,
        T=T,
        H=H,
        D=D,
        BD=triton.next_power_of_2(D),
    )
    return yg
