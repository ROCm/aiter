# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Varlen packing for the FlyDSL KDA kernel.

The FlyDSL KDA kernel is a dense ``[B, H, T, D]`` chunkwise forward: it has no
``cu_seqlens`` support and needs ``T`` to be a multiple of the chunk size.  A
serving batch arrives packed instead -- ``[total_tokens, H, D]`` plus a
``cu_seqlens`` -- so the two kernels here bridge the two layouts.

``kda_pack_prepare`` scatters the packed batch into a right-padded
``[N, H, Tp, D]`` buffer and, in the same pass, applies the preprocessing that
``fla.ops.kda.chunk_kda`` would otherwise fuse into its own kernels: the q/k
L2 norm, the KDA gate activation, and the beta sigmoid.  Padding slots are
written as zeros, which is an exact no-op for the recurrence: ``beta = 0``
means no write and ``g = 0`` means no decay, so
``S_t = (I - 0) Diag(1) S_{t-1} = S_{t-1}``.

``kda_unpack_output`` reverses the layout change on the output, writing
straight into the caller's packed ``[total_tokens, H, V]`` buffer.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = [
    "kda_pack_prepare",
    "kda_unpack_output",
]


@triton.jit
def _softplus(x):
    return tl.where(x < 20.0, tl.log(1 + tl.exp(x)), x)


@triton.jit
def _kda_prepare_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    b_ptr,
    qo_ptr,
    ko_ptr,
    vo_ptr,
    go_ptr,
    bo_ptr,
    cu_ptr,
    alog_ptr,
    dtb_ptr,
    lower_bound,
    eps,
    beta_scale,
    s_qt,
    s_kt,
    s_vt,
    s_gt,
    s_bt,
    Tp,
    H: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BT: tl.constexpr,
    USE_L2NORM: tl.constexpr,
    USE_GATE: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_BETA_SIGMOID: tl.constexpr,
):
    i_n, i_t, i_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    bos = tl.load(cu_ptr + i_n).to(tl.int64)
    eos = tl.load(cu_ptr + i_n + 1).to(tl.int64)
    seqlen = eos - bos

    t = i_t * BT + tl.arange(0, BT)
    in_pad = t < Tp
    valid = t < seqlen
    # Rows past this sequence's length are the right-padding: they are written
    # as zeros so the recurrence steps over them without touching the state.
    src_t = (bos + t).to(tl.int64)
    dst_t = (i_n * H + i_h).to(tl.int64) * Tp + t

    o_k = tl.arange(0, DK)
    o_v = tl.arange(0, DV)

    # ---- q / k: optional L2 norm over the head dim -----------------------
    # A padded (all-zero) row normalizes to 0 * rsqrt(eps) = 0, never NaN.
    b_q = tl.load(
        q_ptr + src_t[:, None] * s_qt + i_h * DK + o_k[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    if USE_L2NORM:
        b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q, 1)[:, None] + eps)
    tl.store(
        qo_ptr + dst_t[:, None] * DK + o_k[None, :],
        b_q.to(qo_ptr.dtype.element_ty),
        mask=in_pad[:, None],
    )

    b_k = tl.load(
        k_ptr + src_t[:, None] * s_kt + i_h * DK + o_k[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    if USE_L2NORM:
        b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k, 1)[:, None] + eps)
    tl.store(
        ko_ptr + dst_t[:, None] * DK + o_k[None, :],
        b_k.to(ko_ptr.dtype.element_ty),
        mask=in_pad[:, None],
    )

    # ---- v: layout change only -------------------------------------------
    b_v = tl.load(
        v_ptr + src_t[:, None] * s_vt + i_h * DV + o_v[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    tl.store(
        vo_ptr + dst_t[:, None] * DV + o_v[None, :],
        b_v.to(vo_ptr.dtype.element_ty),
        mask=in_pad[:, None],
    )

    # ---- g: KDA gate activation, emitted as per-token log decay ----------
    b_g = tl.load(
        g_ptr + src_t[:, None] * s_gt + i_h * DK + o_k[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    if USE_GATE:
        if HAS_BIAS:
            b_g = b_g + tl.load(dtb_ptr + i_h * DK + o_k).to(tl.float32)[None, :]
        b_a = tl.exp(tl.load(alog_ptr + i_h).to(tl.float32))
        if USE_LOWER_BOUND:
            b_g = lower_bound * tl.sigmoid(b_a * b_g)
        else:
            b_g = -b_a * _softplus(b_g)
        # The activation of a zeroed pad row is not zero, so re-zero it here;
        # a nonzero decay on a pad row would shrink the carried state.
        b_g = tl.where(valid[:, None], b_g, 0.0)
    tl.store(
        go_ptr + dst_t[:, None] * DK + o_k[None, :],
        b_g.to(go_ptr.dtype.element_ty),
        mask=in_pad[:, None],
    )

    # ---- beta: optional sigmoid ------------------------------------------
    b_b = tl.load(b_ptr + src_t * s_bt + i_h, mask=valid, other=0.0).to(tl.float32)
    if USE_BETA_SIGMOID:
        b_b = beta_scale * tl.sigmoid(b_b)
        b_b = tl.where(valid, b_b, 0.0)
    tl.store(bo_ptr + dst_t, b_b.to(bo_ptr.dtype.element_ty), mask=in_pad)


@triton.jit
def _kda_unpack_kernel(
    o_pad_ptr,
    o_ptr,
    cu_ptr,
    s_ot,
    Tp,
    H: tl.constexpr,
    DV: tl.constexpr,
    BT: tl.constexpr,
):
    i_n, i_t, i_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    bos = tl.load(cu_ptr + i_n).to(tl.int64)
    eos = tl.load(cu_ptr + i_n + 1).to(tl.int64)
    seqlen = eos - bos

    t = i_t * BT + tl.arange(0, BT)
    valid = t < seqlen
    src_t = (i_n * H + i_h).to(tl.int64) * Tp + t
    dst_t = (bos + t).to(tl.int64)
    o_v = tl.arange(0, DV)

    b_o = tl.load(
        o_pad_ptr + src_t[:, None] * DV + o_v[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    tl.store(
        o_ptr + dst_t[:, None] * s_ot + i_h * DV + o_v[None, :],
        b_o.to(o_ptr.dtype.element_ty),
        mask=valid[:, None],
    )


def _launch_grid(num_seqs, t_pad, num_heads, block_t):
    return (num_seqs, triton.cdiv(t_pad, block_t), num_heads)


def kda_pack_prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_seqs: int,
    t_pad: int,
    *,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    use_qk_l2norm: bool = True,
    use_gate: bool = True,
    use_beta_sigmoid: bool = True,
    beta_scale: float = 1.0,
    eps: float = 1e-6,
    block_t: int = 32,
):
    """Scatter a packed KDA batch ([total_tokens,H,D], head-contiguous) into
    padded [N,H,Tp,D] FlyDSL inputs, with q/k/v in-dtype and g/beta in fp32."""
    num_heads, head_k = q.shape[-2], q.shape[-1]
    head_v = v.shape[-1]
    device = q.device

    shape_k = (num_seqs, num_heads, t_pad, head_k)
    q_pad = torch.empty(shape_k, dtype=q.dtype, device=device)
    k_pad = torch.empty(shape_k, dtype=k.dtype, device=device)
    v_pad = torch.empty(
        (num_seqs, num_heads, t_pad, head_v), dtype=v.dtype, device=device
    )
    g_pad = torch.empty(shape_k, dtype=torch.float32, device=device)
    beta_pad = torch.empty(
        (num_seqs, num_heads, t_pad), dtype=torch.float32, device=device
    )

    _kda_prepare_kernel[_launch_grid(num_seqs, t_pad, num_heads, block_t)](
        q,
        k,
        v,
        g,
        beta,
        q_pad,
        k_pad,
        v_pad,
        g_pad,
        beta_pad,
        cu_seqlens,
        A_log,
        dt_bias,
        lower_bound if lower_bound is not None else 0.0,
        eps,
        beta_scale,
        q.stride(-3),
        k.stride(-3),
        v.stride(-3),
        g.stride(-3),
        beta.stride(-2),
        t_pad,
        H=num_heads,
        DK=head_k,
        DV=head_v,
        BT=block_t,
        USE_L2NORM=use_qk_l2norm,
        USE_GATE=use_gate,
        USE_LOWER_BOUND=lower_bound is not None,
        HAS_BIAS=dt_bias is not None,
        USE_BETA_SIGMOID=use_beta_sigmoid,
        num_warps=4,
    )
    return q_pad, k_pad, v_pad, g_pad, beta_pad


def kda_unpack_output(
    o_pad: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_seqs: int,
    *,
    block_t: int = 32,
):
    """Gather ``[N, H, Tp, V]`` back into a packed ``[total_tokens, H, V]``."""
    _, num_heads, t_pad, head_v = o_pad.shape
    _kda_unpack_kernel[_launch_grid(num_seqs, t_pad, num_heads, block_t)](
        o_pad,
        out,
        cu_seqlens,
        out.stride(-3),
        t_pad,
        H=num_heads,
        DV=head_v,
        BT=block_t,
        num_warps=4,
    )
    return out
