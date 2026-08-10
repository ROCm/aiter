# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li

"""
Recompute W and U tensors for chunk_delta_attn forward pass.

``recompute_w_u_fwd`` takes the chunk Akk inverse matrix (A), key (k), value
(v), gate-cumsum (gk) and beta scalar to produce:
  - w = A @ (k * beta * exp2(gk))   [chunk-gated key correction]
  - u = A @ (v * beta)               [chunk-gated value target]
  - qg = q * exp2(gk)               [gated query, optional]
  - kg = k * exp2(gk_last - gk)     [boundary-normalised key]
"""

import torch
import triton
import triton.language as tl

from .utils import (
    autotune_cache_kwargs,
    chunk_delta_attn_autotune_configs,
    exp2,
    input_guard,
    prepare_chunk_indices,
)

_BK_DEFAULT = 64
_BV_DEFAULT = 64


@triton.heuristics(
    {
        "STORE_QG": lambda args: args["qg"] is not None,
        "STORE_KG": lambda args: args["kg"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=chunk_delta_attn_autotune_configs(
        [
            triton.Config({}, num_warps=nw, num_stages=ns)
            for nw in [2, 4, 8]
            for ns in [2, 3, 4]
        ],
        default_config=triton.Config({}, num_warps=4, num_stages=3),
    ),
    key=["H", "HV", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(
    q,
    k,
    qg,
    kg,
    v,
    beta,
    w,
    u,
    A,
    gk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_QG: tl.constexpr,
    STORE_KG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    k += (bos * H + i_h) * K
    v += (bos * HV + i_hv) * V
    u += (bos * HV + i_hv) * V
    w += (bos * HV + i_hv) * K
    gk += (bos * HV + i_hv) * K
    beta += bos * HV + i_hv
    A += (bos * HV + i_hv) * BT
    if STORE_QG:
        q += (bos * H + i_h) * K
        qg += (bos * HV + i_hv) * K
    if STORE_KG:
        kg += (bos * HV + i_hv) * K

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    p_b = beta + o_t * HV
    b_b = tl.load(p_b, mask=m_t, other=0.0)

    o_A = tl.arange(0, BT)
    m_A = m_t[:, None] & (o_A[None, :] < BT)
    p_A = A + o_t[:, None] * (HV * BT) + o_A[None, :]
    b_A = tl.load(p_A, mask=m_A, other=0.0)

    # u = A @ (v * beta)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = m_t[:, None] & (o_v[None, :] < V)
        p_v = v + o_t[:, None] * (HV * V) + o_v[None, :]
        p_u = u + o_t[:, None] * (HV * V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_v, other=0.0)
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), mask=m_v)

    # w = A @ (k * beta * exp2(gk));  optionally qg, kg
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_tk = m_t[:, None] & m_k[None, :]
        p_w = w + o_t[:, None] * (HV * K) + o_k[None, :]
        p_k = k + o_t[:, None] * (H * K) + o_k[None, :]
        p_gk = gk + o_t[:, None] * (HV * K) + o_k[None, :]

        b_k = tl.load(p_k, mask=m_tk, other=0.0)
        b_gk = tl.load(p_gk, mask=m_tk, other=0.0).to(tl.float32)
        b_kb = b_k * b_b[:, None]
        b_kb *= exp2(b_gk)

        if STORE_QG:
            p_q = q + o_t[:, None] * (H * K) + o_k[None, :]
            p_qg = qg + o_t[:, None] * (HV * K) + o_k[None, :]
            b_q = tl.load(p_q, mask=m_tk, other=0.0)
            b_qg = b_q * exp2(b_gk)
            tl.store(p_qg, b_qg.to(p_qg.dtype.element_ty), mask=m_tk)

        if STORE_KG:
            last_idx = min(i_t * BT + BT, T) - 1
            b_gn = tl.load(gk + last_idx * HV * K + o_k, mask=m_k, other=0.0).to(
                tl.float32
            )
            b_kg = b_k * tl.where(
                (i_t * BT + tl.arange(0, BT) < T)[:, None],
                exp2(b_gn[None, :] - b_gk),
                0,
            )
            p_kg = kg + o_t[:, None] * (HV * K) + o_k[None, :]
            tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), mask=m_tk)

        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), mask=m_tk)


@input_guard
def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    gk: torch.Tensor,
    q: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """
    Recompute W, U (and optionally QG) from the chunk Akk inverse.

    Args:
        k:             Key tensor ``[B, T, H, K]``.
        v:             Value tensor ``[B, T, HV, V]``.
        beta:          (Sigmoided) beta gate ``[B, T, HV]``.
        A:             Chunk Akk inverse ``[B, T, HV, BT]``.
        gk:            Gate cumsum (exp2 space) ``[B, T, HV, K]``.
        q:             Optional query ``[B, T, H, K]`` to produce QG.
        cu_seqlens:    Variable-length cumulative sequence lengths.
        chunk_indices: Pre-computed chunk index pairs for varlen mode.

    Returns:
        (w, u, qg, kg):
          w   ``[B, T, HV, K]``
          u   ``[B, T, HV, V]``
          qg  ``[B, T, HV, K]`` or None
          kg  ``[B, T, HV, K]``
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    HV = v.shape[2]
    BT = A.shape[-1]

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = torch.empty(B, T, HV, K, device=k.device, dtype=k.dtype)
    u = torch.empty_like(v)
    qg = (
        torch.empty(B, T, HV, K, device=k.device, dtype=k.dtype)
        if q is not None
        else None
    )
    kg = torch.empty(B, T, HV, K, device=k.device, dtype=k.dtype)

    recompute_w_u_fwd_kernel[(NT, B * HV)](
        q=q,
        k=k,
        qg=qg,
        kg=kg,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=_BK_DEFAULT,
        BV=_BV_DEFAULT,
    )
    return w, u, qg, kg
