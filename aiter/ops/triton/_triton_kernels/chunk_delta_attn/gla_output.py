# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2026, Songlin Yang, Yu Zhang

"""
GLA output computation for chunk_delta_attn forward pass.

``chunk_gla_fwd_o`` computes the final attention output:
    o = (q * exp2(gk)) @ h * scale  +  A @ v_new
where h is the recurrent hidden state and A is the local intra-chunk attention
matrix (Aqk).
"""

import torch
import triton
import triton.language as tl

from .utils import (
    autotune_cache_kwargs,
    chunk_delta_attn_autotune_configs,
    exp,
    exp2,
    input_guard,
    prepare_chunk_indices,
)


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=chunk_delta_attn_autotune_configs(
        [
            triton.Config({"BK": BK, "BV": BV}, num_warps=nw, num_stages=ns)
            for BK in [32, 64]
            for BV in [64, 128]
            for nw in [2, 4, 8]
            for ns in [2, 3, 4]
        ],
        default_config=triton.Config({"BK": 64, "BV": 128}, num_warps=8, num_stages=3),
    ),
    key=["BT", "HV", "TRANSPOSE_STATE"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_gla_fwd_kernel_o(
    q,
    v,
    g,
    h,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_EXP2: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = (
        tl.program_id(0),
        tl.program_id(1).to(tl.int64),
        tl.program_id(2).to(tl.int64),
    )
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        i_tg = i_t.to(tl.int64)
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = (i_b * NT + i_t).to(tl.int64)
        bos = (i_b * T).to(tl.int64)

    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    q += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    v += (bos * HV + i_hv) * V
    o += (bos * HV + i_hv) * V
    h += (i_tg * HV + i_hv).to(tl.int64) * K * V
    A += (bos * HV + i_hv) * BT

    b_o = tl.zeros([BT, BV], dtype=tl.float32)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_tk = m_t[:, None] & m_k[None, :]
        p_q = q + o_t[:, None] * (H * K) + o_k[None, :]
        p_g = g + o_t[:, None] * (HV * K) + o_k[None, :]
        if TRANSPOSE_STATE:
            p_h = h + o_v[:, None] * K + o_k[None, :]
            m_h = m_v[:, None] & m_k[None, :]
        else:
            p_h = h + o_k[:, None] * V + o_v[None, :]
            m_h = m_k[:, None] & m_v[None, :]

        b_q = tl.load(p_q, mask=m_tk, other=0.0)
        b_g = tl.load(p_g, mask=m_tk, other=0.0).to(tl.float32)
        if USE_EXP2:
            b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
        else:
            b_qg = (b_q * exp(b_g)).to(b_q.dtype)
        b_h = tl.load(p_h, mask=m_h, other=0.0)
        if TRANSPOSE_STATE:
            b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))
        else:
            b_o += tl.dot(b_qg, b_h.to(b_qg.dtype))

    b_o *= scale

    o_A = tl.arange(0, BT)
    m_tv = m_t[:, None] & m_v[None, :]
    m_A = m_t[:, None] & (o_A[None, :] < BT)
    p_v = v + o_t[:, None] * (HV * V) + o_v[None, :]
    p_o = o + o_t[:, None] * (HV * V) + o_v[None, :]
    p_A = A + o_t[:, None] * (HV * BT) + o_A[None, :]

    b_v = tl.load(p_v, mask=m_tv, other=0.0)
    b_A = tl.load(p_A, mask=m_A, other=0.0)
    b_A = tl.where(m_s, b_A, 0.0).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_tv)


@input_guard
def chunk_gla_fwd_o(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.Tensor | None = None,
    use_exp2: bool = True,
    transpose_state: bool = False,
) -> torch.Tensor:
    """
    Compute the final output for chunk_delta_attn (Triton path).

    Args:
        q:              Query ``[B, T, H, K]``.
        v:              (Modified) value ``[B, T, HV, V]``.
        g:              Gate cumsum ``[B, T, HV, K]``.
        A:              Aqk attention matrix ``[B, T, HV, BT]``.
        h:              Recurrent hidden state ``[NT, B, HV, K, V]`` (or transposed).
        scale:          Attention scale.
        cu_seqlens:     Variable-length cumulative sequence lengths.
        chunk_size:     Chunk size BT.
        chunk_indices:  Pre-computed chunk index pairs for varlen mode.
        use_exp2:       Whether g is in log2 space (True) or natural log (False).
        transpose_state: Whether h has shape ``[..., V, K]`` (True) or ``[..., K, V]``.

    Returns:
        Output tensor ``[B, T, HV, V]``.
    """
    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    o = torch.zeros_like(v)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * HV)

    chunk_gla_fwd_kernel_o[grid](
        q=q,
        v=v,
        g=g,
        h=h,
        o=o,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        USE_EXP2=use_exp2,
        TRANSPOSE_STATE=transpose_state,
    )
    return o
