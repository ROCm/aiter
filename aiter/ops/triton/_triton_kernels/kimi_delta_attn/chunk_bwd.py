# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li

"""
Backward pass for chunk_delta_attn.

The backward saves only the forward's inputs and recomputes the default
pipeline's intermediates (``g`` cumsum in log2 space, ``Aqk``, ``Akk``,
``w``/``u``/``qg``/``kg``, ``h``, ``v_new``), as fla does with
``disable_recompute=False``. The gradient is that of the default pipeline,
whichever path ran the forward: FlashKDA computes the same function up to
rounding. Stages:

1. ``dAqk = do @ v_new^T * scale``, ``dv = Aqk^T @ do``
   (``chunk_delta_attn_bwd_kernel_dav``).
2. The reverse-time state gradient ``dh`` per chunk, ``dh0``, and ``dv``
   through the state (``chunk_delta_attn_bwd_kernel_dhu``). With too few
   states to fill the GPU it shares a launch with the recomputed forward
   recurrence (``chunk_delta_attn_bwd_kernel_fwd_h_dhu``), after the ``dv``
   half of stage 1 and before its ``dAqk`` half.
3. ``dq``/``dk``/``dg``/``dbeta``/``dv`` through the WY representation and the
   inter-chunk terms, plus ``dAkk`` (``chunk_delta_attn_bwd_kernel_wy_dqkg``).
4. The intra-chunk terms of ``dq``/``dk``/``dg``/``dbeta`` from ``dAqk`` and
   ``dAkk`` (``chunk_delta_attn_bwd_kernel_intra``). With the in-kernel l2norm,
   no GVA and ``K <= 128`` it also takes ``dq``/``dk`` back through the l2norm
   and finishes ``dbeta`` (beta sigmoid included).
5. A reverse chunk-local cumsum turns the gradient w.r.t. the cumulative gate
   into one w.r.t. the per-token gate, and the same kernel undoes the gate
   activation (``chunk_delta_attn_bwd_kernel_gate_cumsum``). l2norm and beta
   sigmoid are undone elementwise when stage 4 did not.

Kernel bodies follow fla's ``chunk_kda_bwd``. The state gradient keeps ``dh``
in the ``[K, V]`` layout of the forward's ``h``; ``TRANSPOSE_STATE`` only
describes ``dht`` / ``dh0``. The forward recurrence is recomputed with this
module's own copy of it (``chunk_delta_attn_bwd_kernel_fwd_h``) rather than the
shared GDN ``chunk_gated_delta_rule_fwd_h``, so the two recurrences can share a
launch.
"""

import functools

import torch
import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.gated_delta_net.utils import chunk_local_cumsum
from aiter.ops.triton._triton_kernels.gated_delta_net.utils.index import (
    prepare_chunk_offsets,
)
from aiter.ops.triton._triton_kernels.kimi_delta_attn.chunk_delta_attn_utils import (
    RCP_LN2,
    chunk_delta_attn_tuned_config,
    exp,
    exp2,
    softplus,
)
from aiter.ops.triton._triton_kernels.kimi_delta_attn.fast_launch import fast_launch
from aiter.ops.triton._triton_kernels.kimi_delta_attn.gate import beta_sigmoid_fwd
from aiter.ops.triton._triton_kernels.kimi_delta_attn.intra_attn import (
    chunk_delta_attn_fwd_intra,
)
from aiter.ops.triton._triton_kernels.kimi_delta_attn.utils.cumsum import (
    chunk_gate_cumsum,
)
from aiter.ops.triton._triton_kernels.kimi_delta_attn.utils.index import (
    prepare_chunk_indices,
)
from aiter.ops.triton._triton_kernels.kimi_delta_attn.utils.l2norm import l2norm_fwd
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils._triton.pid_preprocessing import remap_xcd


@triton.jit
def _l2norm_bwd_rows(x, dy, eps):
    """``dx`` for ``y = x / sqrt(sum(x^2) + eps)`` per row of fp32 ``[rows, D]`` tiles."""
    rstd = tl.rsqrt(tl.sum(x * x, axis=1) + eps)[:, None]
    y = x * rstd
    return dy * rstd - tl.sum(dy * y, axis=1)[:, None] * y * rstd


@triton.jit
def chunk_delta_attn_bwd_kernel_dav(
    v,
    A,
    do,
    cu_seqlens,
    chunk_indices,
    dA,
    dv,
    scale,
    T,
    HV: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    COMPUTE_DA: tl.constexpr,
    COMPUTE_DV: tl.constexpr,
):
    """dA = tril(do @ v^T) * scale and dv = tril(A)^T @ do, per chunk (``v`` is ``v_new``).

    ``COMPUTE_DA`` / ``COMPUTE_DV`` select the outputs, so ``dv`` (which needs no
    ``v_new``) can be produced ahead of ``chunk_delta_attn_bwd_kernel_fwd_h_dhu``.
    """
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_hv = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = eos - bos
        if i_t * BT >= T:
            return
    else:
        bos, eos = i_b * T, i_b * T + T

    do += (bos * HV + i_hv) * V
    if COMPUTE_DA:
        v += (bos * HV + i_hv) * V
        dA += (bos * HV + i_hv) * BT
    if COMPUTE_DV:
        dv += (bos * HV + i_hv) * V

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_A = tl.arange(0, BT)
    if COMPUTE_DV:
        m_AT = (o_A[:, None] < BT) & m_t[None, :]
        # b_A[s, t] = A[t, s]
        p_A = A + (bos * HV + i_hv) * BT + o_A[:, None] + o_t[None, :] * (HV * BT)
        b_A = tl.load(p_A, mask=m_AT, other=0.0)
        m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)
        b_A = tl.where(m_A, b_A, 0).to(do.dtype.element_ty)

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        m_vT = m_v[:, None] & m_t[None, :]
        m_tv = m_t[:, None] & m_v[None, :]
        p_do = do + o_t[:, None] * (HV * V) + o_v[None, :]
        b_do = tl.load(p_do, mask=m_tv, other=0.0)
        if COMPUTE_DA:
            p_v = v + o_v[:, None] + o_t[None, :] * (HV * V)
            b_v = tl.load(p_v, mask=m_vT, other=0.0)
            b_dA = tl.dot(b_do, b_v, b_dA)
        if COMPUTE_DV:
            p_dv = dv + o_t[:, None] * (HV * V) + o_v[None, :]
            b_dv = tl.dot(b_A.to(b_do.dtype), b_do)
            tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_tv)

    if COMPUTE_DA:
        m_dA = m_t[:, None] & (o_A[None, :] < BT)
        p_dA = dA + o_t[:, None] * (HV * BT) + o_A[None, :]
        b_dA = tl.where(o_t[:, None] >= o_t, b_dA * scale, 0.0)
        tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), mask=m_dA)


@triton.jit
def _bwd_dhu_body(
    i_v,
    i_nh,
    qg,
    g,
    k,
    w,
    dht,
    do,
    dv,
    cu_seqlens,
    chunk_offsets,
    dh,
    dh0,
    dv2,
    scale,
    T,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
):
    """Body of ``chunk_delta_attn_bwd_kernel_dhu`` for program ``(i_v, i_nh)``."""
    i_nh = i_nh.to(tl.int64)
    i_n, i_hv = i_nh // HV, i_nh % HV
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int64)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_dh4 = tl.zeros([64, BV], dtype=tl.float32)

    qg += (bos * HV + i_hv) * K
    g += (bos * HV + i_hv) * K
    k += (bos * HV + i_hv) * K
    w += (bos * HV + i_hv) * K
    do += (bos * HV + i_hv) * V
    dv += (bos * HV + i_hv) * V
    dv2 += (bos * HV + i_hv) * V
    dh += (boh * HV + i_hv) * K * V
    if USE_INITIAL_STATE:
        dh0 += i_nh * K * V
    if USE_FINAL_STATE_GRADIENT:
        dht += i_nh * K * V

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    o_k2 = 64 + o_k1
    m_k2 = o_k2 < K
    o_k3 = 128 + o_k1
    m_k3 = o_k3 < K
    o_k4 = 192 + o_k1
    m_k4 = o_k4 < K
    m_h1 = m_k1[:, None] & m_v[None, :]
    m_h2 = m_k2[:, None] & m_v[None, :]
    m_h3 = m_k3[:, None] & m_v[None, :]
    m_h4 = m_k4[:, None] & m_v[None, :]

    if USE_FINAL_STATE_GRADIENT:
        if TRANSPOSE_STATE:
            p_dht = dht + o_k1[:, None] + o_v[None, :] * K
        else:
            p_dht = dht + o_k1[:, None] * V + o_v[None, :]
        b_dh1 += tl.load(p_dht, mask=m_h1, other=0.0).to(tl.float32)
        if K > 64:
            if TRANSPOSE_STATE:
                p_dht = dht + o_k2[:, None] + o_v[None, :] * K
            else:
                p_dht = dht + o_k2[:, None] * V + o_v[None, :]
            b_dh2 += tl.load(p_dht, mask=m_h2, other=0.0).to(tl.float32)
        if K > 128:
            if TRANSPOSE_STATE:
                p_dht = dht + o_k3[:, None] + o_v[None, :] * K
            else:
                p_dht = dht + o_k3[:, None] * V + o_v[None, :]
            b_dh3 += tl.load(p_dht, mask=m_h3, other=0.0).to(tl.float32)
        if K > 192:
            if TRANSPOSE_STATE:
                p_dht = dht + o_k4[:, None] + o_v[None, :] * K
            else:
                p_dht = dht + o_k4[:, None] * V + o_v[None, :]
            b_dh4 += tl.load(p_dht, mask=m_h4, other=0.0).to(tl.float32)

    for i_t in range(NT - 1, -1, -1):
        i_t64 = i_t.to(tl.int64)
        o_t = i_t64 * BT + tl.arange(0, BT)
        m_t = o_t < T
        m_tv = m_t[:, None] & m_v[None, :]

        dh_t = dh + i_t64 * HV * K * V
        tl.store(
            dh_t + o_k1[:, None] * V + o_v[None, :],
            b_dh1.to(dh.dtype.element_ty),
            mask=m_h1,
        )
        if K > 64:
            tl.store(
                dh_t + o_k2[:, None] * V + o_v[None, :],
                b_dh2.to(dh.dtype.element_ty),
                mask=m_h2,
            )
        if K > 128:
            tl.store(
                dh_t + o_k3[:, None] * V + o_v[None, :],
                b_dh3.to(dh.dtype.element_ty),
                mask=m_h3,
            )
        if K > 192:
            tl.store(
                dh_t + o_k4[:, None] * V + o_v[None, :],
                b_dh4.to(dh.dtype.element_ty),
                mask=m_h4,
            )

        last_idx = min((i_t64 + 1) * BT, T) - 1
        b_do = tl.load(
            do + o_t[:, None] * (HV * V) + o_v[None, :], mask=m_tv, other=0.0
        )

        # dv2 = dv + kg @ dh
        b_k = tl.load(
            k + o_t[:, None] * (HV * K) + o_k1[None, :],
            mask=m_t[:, None] & m_k1[None, :],
            other=0.0,
        )
        b_gl1 = tl.load(g + last_idx * HV * K + o_k1, mask=m_k1, other=0.0).to(
            tl.float32
        )
        b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype))
        if K > 64:
            b_k = tl.load(
                k + o_t[:, None] * (HV * K) + o_k2[None, :],
                mask=m_t[:, None] & m_k2[None, :],
                other=0.0,
            )
            b_gl2 = tl.load(g + last_idx * HV * K + o_k2, mask=m_k2, other=0.0).to(
                tl.float32
            )
            b_dv = tl.dot(b_k, b_dh2.to(b_k.dtype), b_dv)
        if K > 128:
            b_k = tl.load(
                k + o_t[:, None] * (HV * K) + o_k3[None, :],
                mask=m_t[:, None] & m_k3[None, :],
                other=0.0,
            )
            b_gl3 = tl.load(g + last_idx * HV * K + o_k3, mask=m_k3, other=0.0).to(
                tl.float32
            )
            b_dv = tl.dot(b_k, b_dh3.to(b_k.dtype), b_dv)
        if K > 192:
            b_k = tl.load(
                k + o_t[:, None] * (HV * K) + o_k4[None, :],
                mask=m_t[:, None] & m_k4[None, :],
                other=0.0,
            )
            b_gl4 = tl.load(g + last_idx * HV * K + o_k4, mask=m_k4, other=0.0).to(
                tl.float32
            )
            b_dv = tl.dot(b_k, b_dh4.to(b_k.dtype), b_dv)
        p_dv = dv + o_t[:, None] * (HV * V) + o_v[None, :]
        b_dv += tl.load(p_dv, mask=m_tv, other=0.0)
        p_dv2 = dv2 + o_t[:, None] * (HV * V) + o_v[None, :]
        tl.store(p_dv2, b_dv.to(p_dv2.dtype.element_ty), mask=m_tv)

        # dh = dh * exp2(g_last) + qg^T @ do * scale - w^T @ dv2
        m_kt = m_k1[:, None] & m_t[None, :]
        b_qg = tl.load(
            qg + o_k1[:, None] + o_t[None, :] * (HV * K), mask=m_kt, other=0.0
        )
        b_w = tl.load(w + o_k1[:, None] + o_t[None, :] * (HV * K), mask=m_kt, other=0.0)
        b_dh1 *= exp2(b_gl1)[:, None]
        b_dh1 += tl.dot(b_qg, b_do.to(b_qg.dtype)) * scale - tl.dot(
            b_w, b_dv.to(b_w.dtype)
        )
        if K > 64:
            m_kt = m_k2[:, None] & m_t[None, :]
            b_qg = tl.load(
                qg + o_k2[:, None] + o_t[None, :] * (HV * K), mask=m_kt, other=0.0
            )
            b_w = tl.load(
                w + o_k2[:, None] + o_t[None, :] * (HV * K), mask=m_kt, other=0.0
            )
            b_dh2 *= exp2(b_gl2)[:, None]
            b_dh2 += tl.dot(b_qg, b_do.to(b_qg.dtype)) * scale - tl.dot(
                b_w, b_dv.to(b_w.dtype)
            )
        if K > 128:
            m_kt = m_k3[:, None] & m_t[None, :]
            b_qg = tl.load(
                qg + o_k3[:, None] + o_t[None, :] * (HV * K), mask=m_kt, other=0.0
            )
            b_w = tl.load(
                w + o_k3[:, None] + o_t[None, :] * (HV * K), mask=m_kt, other=0.0
            )
            b_dh3 *= exp2(b_gl3)[:, None]
            b_dh3 += tl.dot(b_qg, b_do.to(b_qg.dtype)) * scale - tl.dot(
                b_w, b_dv.to(b_w.dtype)
            )
        if K > 192:
            m_kt = m_k4[:, None] & m_t[None, :]
            b_qg = tl.load(
                qg + o_k4[:, None] + o_t[None, :] * (HV * K), mask=m_kt, other=0.0
            )
            b_w = tl.load(
                w + o_k4[:, None] + o_t[None, :] * (HV * K), mask=m_kt, other=0.0
            )
            b_dh4 *= exp2(b_gl4)[:, None]
            b_dh4 += tl.dot(b_qg, b_do.to(b_qg.dtype)) * scale - tl.dot(
                b_w, b_dv.to(b_w.dtype)
            )

    if USE_INITIAL_STATE:
        if TRANSPOSE_STATE:
            p_dh0 = dh0 + o_k1[:, None] + o_v[None, :] * K
        else:
            p_dh0 = dh0 + o_k1[:, None] * V + o_v[None, :]
        tl.store(p_dh0, b_dh1.to(p_dh0.dtype.element_ty), mask=m_h1)
        if K > 64:
            if TRANSPOSE_STATE:
                p_dh0 = dh0 + o_k2[:, None] + o_v[None, :] * K
            else:
                p_dh0 = dh0 + o_k2[:, None] * V + o_v[None, :]
            tl.store(p_dh0, b_dh2.to(p_dh0.dtype.element_ty), mask=m_h2)
        if K > 128:
            if TRANSPOSE_STATE:
                p_dh0 = dh0 + o_k3[:, None] + o_v[None, :] * K
            else:
                p_dh0 = dh0 + o_k3[:, None] * V + o_v[None, :]
            tl.store(p_dh0, b_dh3.to(p_dh0.dtype.element_ty), mask=m_h3)
        if K > 192:
            if TRANSPOSE_STATE:
                p_dh0 = dh0 + o_k4[:, None] + o_v[None, :] * K
            else:
                p_dh0 = dh0 + o_k4[:, None] * V + o_v[None, :]
            tl.store(p_dh0, b_dh4.to(p_dh0.dtype.element_ty), mask=m_h4)


@triton.jit
def chunk_delta_attn_bwd_kernel_dhu(
    qg,
    g,
    k,
    w,
    dht,
    do,
    dv,
    cu_seqlens,
    chunk_offsets,
    dh,
    dh0,
    dv2,
    scale,
    T,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    """Reverse-time recurrence of the per-chunk state gradient.

    ``dh[t]`` is the gradient w.r.t. ``h[t]`` (the state entering chunk ``t``);
    ``dv2 = dv + kg @ dh[t]`` adds the path through the state to ``v_new``'s
    gradient. ``qg`` (``q * exp2(g)``) and ``k`` (``kg``) are per value head and
    ``g`` is the chunk-local log2 cumsum.
    """
    # The V blocks of one state share their qg/kg/w/g loads: keep them on one XCD.
    NV = tl.num_programs(0)
    pid = remap_xcd(
        tl.program_id(1) * NV + tl.program_id(0), NV * tl.num_programs(1), NUM_XCDS
    )
    _bwd_dhu_body(
        pid % NV,
        pid // NV,
        qg,
        g,
        k,
        w,
        dht,
        do,
        dv,
        cu_seqlens,
        chunk_offsets,
        dh,
        dh0,
        dv2,
        scale,
        T,
        HV,
        K,
        V,
        BT,
        BV,
        USE_INITIAL_STATE,
        USE_FINAL_STATE_GRADIENT,
        IS_VARLEN,
        TRANSPOSE_STATE,
    )


@triton.jit
def _fwd_h_body(
    i_v,
    i_nh,
    k,
    v,
    w,
    gk,
    h0,
    cu_seqlens,
    chunk_offsets,
    h,
    v_new,
    ht,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
):
    """Body of ``chunk_delta_attn_bwd_kernel_fwd_h`` for program ``(i_v, i_nh)``."""
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # Widen to int64 before scaling by K/V: a 64-head K=V=128 model overflows
    # int32 at ~131k tokens.
    h += (boh * H + i_h).to(tl.int64) * K * V
    v += (bos * H + i_h).to(tl.int64) * V
    k += (bos * H + i_h).to(tl.int64) * K
    w += (bos * H + i_h).to(tl.int64) * K
    v_new += (bos * H + i_h).to(tl.int64) * V
    stride_v = H * V
    stride_h = H * K * V
    stride_k = H * K
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh.to(tl.int64) * K * V
    if STORE_FINAL_STATE:
        ht = ht + i_nh.to(tl.int64) * K * V

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    o_k2 = 64 + o_k1
    m_k2 = o_k2 < K
    o_k3 = 128 + o_k1
    m_k3 = o_k3 < K
    o_k4 = 192 + o_k1
    m_k4 = o_k4 < K
    m_h1 = m_k1[:, None] & m_v[None, :]
    m_h2 = m_k2[:, None] & m_v[None, :]
    m_h3 = m_k3[:, None] & m_v[None, :]
    m_h4 = m_k4[:, None] & m_v[None, :]

    # TRANSPOSE_STATE describes the state buffers as V-first ``[V, K]``.
    if USE_INITIAL_STATE:
        if TRANSPOSE_STATE:
            p_h0_1 = h0 + o_k1[:, None] + o_v[None, :] * K
        else:
            p_h0_1 = h0 + o_k1[:, None] * V + o_v[None, :]
        b_h1 += tl.load(p_h0_1, mask=m_h1, other=0.0).to(tl.float32)
        if K > 64:
            if TRANSPOSE_STATE:
                p_h0_2 = h0 + o_k2[:, None] + o_v[None, :] * K
            else:
                p_h0_2 = h0 + o_k2[:, None] * V + o_v[None, :]
            b_h2 += tl.load(p_h0_2, mask=m_h2, other=0.0).to(tl.float32)
        if K > 128:
            if TRANSPOSE_STATE:
                p_h0_3 = h0 + o_k3[:, None] + o_v[None, :] * K
            else:
                p_h0_3 = h0 + o_k3[:, None] * V + o_v[None, :]
            b_h3 += tl.load(p_h0_3, mask=m_h3, other=0.0).to(tl.float32)
        if K > 192:
            if TRANSPOSE_STATE:
                p_h0_4 = h0 + o_k4[:, None] + o_v[None, :] * K
            else:
                p_h0_4 = h0 + o_k4[:, None] * V + o_v[None, :]
            b_h4 += tl.load(p_h0_4, mask=m_h4, other=0.0).to(tl.float32)

    for i_t in range(NT):
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        m_tk1 = m_t[:, None] & m_k1[None, :]
        m_tv = m_t[:, None] & m_v[None, :]

        # The state entering chunk i_t and v_new are stored at the end of the
        # iteration: stores issued ahead of the chunk's loads hold up the waits
        # on those loads, and they sit on the serial chain (~1.5x per chunk).
        b_hs1 = b_h1.to(h.dtype.element_ty)
        if K > 64:
            b_hs2 = b_h2.to(h.dtype.element_ty)
        if K > 128:
            b_hs3 = b_h3.to(h.dtype.element_ty)
        if K > 192:
            b_hs4 = b_h4.to(h.dtype.element_ty)

        p_w = w + o_t[:, None] * stride_k + o_k1[None, :]
        b_w = tl.load(p_w, mask=m_tk1, other=0.0)
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = w + o_t[:, None] * stride_k + o_k2[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k2[None, :], other=0.0)
            b_v = tl.dot(b_w, b_h2.to(b_w.dtype), acc=b_v)
        if K > 128:
            p_w = w + o_t[:, None] * stride_k + o_k3[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k3[None, :], other=0.0)
            b_v = tl.dot(b_w, b_h3.to(b_w.dtype), acc=b_v)
        if K > 192:
            p_w = w + o_t[:, None] * stride_k + o_k4[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k4[None, :], other=0.0)
            b_v = tl.dot(b_w, b_h4.to(b_w.dtype), acc=b_v)
        p_v = v + o_t[:, None] * stride_v + o_v[None, :]
        b_v = tl.load(p_v, mask=m_tv, other=0.0) - b_v

        b_vs = b_v.to(v_new.dtype.element_ty)

        last_idx = min((i_t + 1) * BT, T) - 1
        b_gk_last1 = tl.load(
            gk + (bos + last_idx) * H * K + i_h * K + o_k1,
            mask=m_k1,
            other=0.0,
        )
        b_h1 *= tl.math.exp2(b_gk_last1)[:, None]
        if K > 64:
            b_gk_last2 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k2,
                mask=m_k2,
                other=0.0,
            )
            b_h2 *= tl.math.exp2(b_gk_last2)[:, None]
        if K > 128:
            b_gk_last3 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k3,
                mask=m_k3,
                other=0.0,
            )
            b_h3 *= tl.math.exp2(b_gk_last3)[:, None]
        if K > 192:
            b_gk_last4 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k4,
                mask=m_k4,
                other=0.0,
            )
            b_h4 *= tl.math.exp2(b_gk_last4)[:, None]
        b_v = b_v.to(k.dtype.element_ty)

        p_k = k + o_k1[:, None] + o_t[None, :] * stride_k
        b_k = tl.load(p_k, mask=m_k1[:, None] & m_t[None, :], other=0.0)
        b_h1 = tl.dot(b_k, b_v, acc=b_h1)
        if K > 64:
            p_k = k + o_k2[:, None] + o_t[None, :] * stride_k
            b_k = tl.load(p_k, mask=m_k2[:, None] & m_t[None, :], other=0.0)
            b_h2 = tl.dot(b_k, b_v, acc=b_h2)
        if K > 128:
            p_k = k + o_k3[:, None] + o_t[None, :] * stride_k
            b_k = tl.load(p_k, mask=m_k3[:, None] & m_t[None, :], other=0.0)
            b_h3 = tl.dot(b_k, b_v, acc=b_h3)
        if K > 192:
            p_k = k + o_k4[:, None] + o_t[None, :] * stride_k
            b_k = tl.load(p_k, mask=m_k4[:, None] & m_t[None, :], other=0.0)
            b_h4 = tl.dot(b_k, b_v, acc=b_h4)

        h_t = h + i_t.to(tl.int64) * stride_h
        tl.store(h_t + o_k1[:, None] * V + o_v[None, :], b_hs1, mask=m_h1)
        if K > 64:
            tl.store(h_t + o_k2[:, None] * V + o_v[None, :], b_hs2, mask=m_h2)
        if K > 128:
            tl.store(h_t + o_k3[:, None] * V + o_v[None, :], b_hs3, mask=m_h3)
        if K > 192:
            tl.store(h_t + o_k4[:, None] * V + o_v[None, :], b_hs4, mask=m_h4)
        tl.store(v_new + o_t[:, None] * stride_v + o_v[None, :], b_vs, mask=m_tv)

    if STORE_FINAL_STATE:
        if TRANSPOSE_STATE:
            p_ht = ht + o_k1[:, None] + o_v[None, :] * K
        else:
            p_ht = ht + o_k1[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), mask=m_h1)
        if K > 64:
            if TRANSPOSE_STATE:
                p_ht = ht + o_k2[:, None] + o_v[None, :] * K
            else:
                p_ht = ht + o_k2[:, None] * V + o_v[None, :]
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), mask=m_h2)
        if K > 128:
            if TRANSPOSE_STATE:
                p_ht = ht + o_k3[:, None] + o_v[None, :] * K
            else:
                p_ht = ht + o_k3[:, None] * V + o_v[None, :]
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), mask=m_h3)
        if K > 192:
            if TRANSPOSE_STATE:
                p_ht = ht + o_k4[:, None] + o_v[None, :] * K
            else:
                p_ht = ht + o_k4[:, None] * V + o_v[None, :]
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), mask=m_h4)


@triton.jit
def chunk_delta_attn_bwd_kernel_fwd_h(
    k,
    v,
    w,
    gk,
    h0,
    cu_seqlens,
    chunk_offsets,
    h,
    v_new,
    ht,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
):
    """Per-chunk hidden states ``h`` and ``v_new = u - w @ h`` for a per-K (vector) log2 gate.

    ``H`` here is the value-head count. ``h`` is indexed by global chunk,
    ``chunk_offsets[n]`` being sequence ``n``'s first one.
    """
    _fwd_h_body(
        tl.program_id(0),
        tl.program_id(1),
        k,
        v,
        w,
        gk,
        h0,
        cu_seqlens,
        chunk_offsets,
        h,
        v_new,
        ht,
        T,
        H,
        K,
        V,
        BT,
        BV,
        USE_INITIAL_STATE,
        STORE_FINAL_STATE,
        IS_VARLEN,
        TRANSPOSE_STATE,
    )


# ---------------------------------------------------------------------------
# Default pipeline: output
# ---------------------------------------------------------------------------
@triton.jit
def chunk_delta_attn_bwd_kernel_fwd_h_dhu(
    qg,
    g,
    k,
    v,
    w,
    h0,
    dht,
    do,
    dv,
    cu_seqlens,
    chunk_offsets,
    h,
    v_new,
    dh,
    dh0,
    dv2,
    scale,
    T,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    """The ``fwd_h`` recompute and ``chunk_delta_attn_bwd_kernel_dhu`` in one launch.

    The two recurrences are independent (``dhu`` needs ``dv = Aqk^T @ do`` but
    not ``v_new``) and each is a serial chain over chunks that leaves most CUs
    idle when there are few states, so they share the grid: program ``i_v <
    cdiv(V, BV)`` runs ``fwd_h`` on V block ``i_v``, the rest run ``dhu``.
    ``qg``, ``k`` (``kg``), ``v`` (``u``) and ``g`` (the log2 gate cumsum) are as
    for the two kernels; the final state is not stored.
    """
    NV: tl.constexpr = (V + BV - 1) // BV
    # A state's fwd_h (and dhu) V blocks share their loads: keep them on one XCD.
    n_v = tl.num_programs(0)
    pid = remap_xcd(
        tl.program_id(1) * n_v + tl.program_id(0), n_v * tl.num_programs(1), NUM_XCDS
    )
    i_v, i_nh = pid % n_v, pid // n_v
    if i_v < NV:
        _fwd_h_body(
            i_v,
            i_nh,
            k,
            v,
            w,
            g,
            h0,
            cu_seqlens,
            chunk_offsets,
            h,
            v_new,
            None,
            T,
            HV,
            K,
            V,
            BT,
            BV,
            USE_INITIAL_STATE,
            False,
            IS_VARLEN,
            TRANSPOSE_STATE,
        )
    else:
        _bwd_dhu_body(
            i_v - NV,
            i_nh,
            qg,
            g,
            k,
            w,
            dht,
            do,
            dv,
            cu_seqlens,
            chunk_offsets,
            dh,
            dh0,
            dv2,
            scale,
            T,
            HV,
            K,
            V,
            BT,
            BV,
            USE_INITIAL_STATE,
            USE_FINAL_STATE_GRADIENT,
            IS_VARLEN,
            TRANSPOSE_STATE,
        )


@triton.jit
def chunk_delta_attn_bwd_kernel_wy_dqkg(
    q,
    k,
    v,
    v_new,
    g,
    beta,
    A,
    h,
    do,
    dh,
    dv,
    cu_seqlens,
    chunk_indices,
    dq,
    dk,
    dv2,
    dg,
    db,
    dA,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Inter-chunk dq/dk/dg, the WY-representation gradients dv/dbeta/dg, and dAkk.

    ``A`` is ``Akk`` (the inverted WY matrix); ``dv`` is ``v_new``'s gradient
    from ``chunk_delta_attn_bwd_kernel_dhu``. ``dq``/``dk``/``dg`` are per value head, fp32.
    ``dA`` is the gradient w.r.t. the strictly lower ``beta * k k^T`` block.
    """
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        i_tg = i_t.to(tl.int64)
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = (eos - bos).to(tl.int32)
        if i_t * BT >= T:
            return
    else:
        NT = tl.cdiv(T, BT)
        i_tg = (i_b * NT + i_t).to(tl.int64)
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_last = o_t == min(T, i_t * BT + BT) - 1

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * HV + i_hv) * V
    v_new += (bos * HV + i_hv) * V
    g += (bos * HV + i_hv) * K
    beta += bos * HV + i_hv
    A += (bos * HV + i_hv) * BT
    h += (i_tg * HV + i_hv) * K * V
    do += (bos * HV + i_hv) * V
    dh += (i_tg * HV + i_hv) * K * V
    dq += (bos * HV + i_hv) * K
    dk += (bos * HV + i_hv) * K
    dv += (bos * HV + i_hv) * V
    dv2 += (bos * HV + i_hv) * V
    dg += (bos * HV + i_hv) * K
    db += bos * HV + i_hv
    dA += (bos * HV + i_hv) * BT

    p_beta = beta + o_t * HV
    b_beta = tl.load(p_beta, mask=m_t, other=0.0)

    o_A = tl.arange(0, BT)
    m_AT = (o_A[:, None] < BT) & m_t[None, :]
    # b_A[s, t] = A[t, s]
    p_A = A + o_A[:, None] + o_t[None, :] * (HV * BT)
    b_A = tl.load(p_A, mask=m_AT, other=0.0)

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    b_db = tl.zeros([BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_tk = m_t[:, None] & m_k[None, :]

        p_k = k + o_t[:, None] * (H * K) + o_k[None, :]
        p_g = g + o_t[:, None] * (HV * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_tk, other=0.0)
        b_g = tl.load(p_g, mask=m_tk, other=0.0).to(tl.float32)

        p_gn = g + (min(T, i_t * BT + BT) - 1).to(tl.int64) * HV * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)

        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        b_dk = tl.zeros([BT, BK], dtype=tl.float32)
        b_dw = tl.zeros([BT, BK], dtype=tl.float32)
        b_dgk = tl.zeros([BK], dtype=tl.float32)

        for i_v in range(tl.cdiv(V, BV)):
            o_v = i_v * BV + tl.arange(0, BV)
            m_tv = m_t[:, None] & (o_v[None, :] < V)
            m_h = (o_v[:, None] < V) & m_k[None, :]
            p_v_new = v_new + o_t[:, None] * (HV * V) + o_v[None, :]
            p_do = do + o_t[:, None] * (HV * V) + o_v[None, :]
            # [BV, BK] views of the [K, V] states.
            p_h = h + o_v[:, None] + o_k[None, :] * V
            p_dh = dh + o_v[:, None] + o_k[None, :] * V
            p_dv = dv + o_t[:, None] * (HV * V) + o_v[None, :]
            b_v_new = tl.load(p_v_new, mask=m_tv, other=0.0)
            b_do = tl.load(p_do, mask=m_tv, other=0.0)
            b_h = tl.load(p_h, mask=m_h, other=0.0)
            b_dh = tl.load(p_dh, mask=m_h, other=0.0)
            b_dv = tl.load(p_dv, mask=m_tv, other=0.0)

            b_dgk += tl.sum(b_h * b_dh, axis=0)
            b_dq = tl.dot(b_do, b_h.to(b_do.dtype), b_dq)
            b_dk = tl.dot(b_v_new, b_dh.to(b_v_new.dtype), b_dk)
            b_dw = tl.dot(b_dv.to(b_v_new.dtype), b_h.to(b_v_new.dtype), b_dw)
            tl.debug_barrier()  # fla: required for correctness
            if i_k == 0:
                p_v = v + o_t[:, None] * (HV * V) + o_v[None, :]
                p_dv2 = dv2 + o_t[:, None] * (HV * V) + o_v[None, :]
                b_v = tl.load(p_v, mask=m_tv, other=0.0)
                b_dA = tl.dot(b_dv, tl.trans(b_v), b_dA)
                b_dvb = tl.dot(b_A, b_dv)
                b_dv2 = b_dvb * b_beta[:, None]
                b_db += tl.sum(b_dvb * b_v, 1)
                tl.store(p_dv2, b_dv2.to(p_dv2.dtype.element_ty), mask=m_tv)

        b_gk_exp = exp2(b_g)
        b_gb = b_gk_exp * b_beta[:, None]
        b_dgk *= exp2(b_gn)
        b_dq = b_dq * b_gk_exp * scale
        b_dk = b_dk * tl.where(m_t[:, None], exp2(b_gn[None, :] - b_g), 0)

        b_kg = b_k * b_gk_exp

        b_dw = -b_dw.to(b_A.dtype)
        b_dA = tl.dot(b_dw, tl.trans(b_kg.to(b_A.dtype)), b_dA)

        b_dkgb = tl.dot(b_A, b_dw)
        b_db += tl.sum(b_dkgb * b_kg, 1)

        p_q = q + o_t[:, None] * (H * K) + o_k[None, :]
        b_q = tl.load(p_q, mask=m_tk, other=0.0)
        b_kdk = b_k * b_dk
        b_dgk += tl.sum(b_kdk, axis=0)
        b_dg = (
            b_q * b_dq
            - b_kdk
            + m_last[:, None] * b_dgk
            + b_kg * b_dkgb * b_beta[:, None]
        )
        b_dk = b_dk + b_dkgb * b_gb

        p_dq = dq + o_t[:, None] * (HV * K) + o_k[None, :]
        p_dk = dk + o_t[:, None] * (HV * K) + o_k[None, :]
        p_dg = dg + o_t[:, None] * (HV * K) + o_k[None, :]
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=m_tk)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_tk)
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=m_tk)

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, b_dA * b_beta[None, :], 0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))
    b_dA = tl.where(m_A, -b_dA, 0)

    m_dA = m_t[:, None] & (o_A[None, :] < BT)
    p_dA = dA + o_t[:, None] * (HV * BT) + o_A[None, :]
    p_db = db + o_t * HV
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), mask=m_dA)
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), mask=m_t)


@triton.jit
def chunk_delta_attn_bwd_kernel_intra(
    q,
    k,
    g,
    beta,
    dAqk,
    dAkk,
    dq,
    dk,
    dg,
    q_raw,
    k_raw,
    beta_raw,
    db,
    cu_seqlens,
    chunk_indices,
    dq2,
    dk2,
    dg2,
    db2,
    eps,
    B,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    SAFE_GATE: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    L2NORM_QK: tl.constexpr,
    FINISH_DB: tl.constexpr,
    BETA_SIGMOID: tl.constexpr,
):
    """Intra-chunk dq/dk/dg/dbeta from dAqk and dAkk, one ``BC`` sub-chunk and ``BK`` slice each.

    ``dq2 = dq + ...`` etc. accumulate onto the inter-chunk terms; ``db2`` is
    ``[NK, B*T, HV]`` partials, one per ``BK`` slice. With a single slice
    (``BK >= K``) the epilogue can also finish the gradients:

    * ``L2NORM_QK`` (``H == HV``): ``dq2``/``dk2`` are taken back through the
      in-kernel l2norm of ``q_raw``/``k_raw``, as ``chunk_delta_attn_bwd_kernel_l2norm``.
    * ``FINISH_DB``: ``db2`` is the whole ``dbeta``: it adds the inter-chunk
      ``db`` and, with ``BETA_SIGMOID``, goes back through ``sigmoid(beta_raw)``.
    """
    if L2NORM_QK or FINISH_DB:
        tl.static_assert(BK >= K)
    # The NK * NC programs of a chunk re-read each other's k/g/q tiles; launched
    # round-robin they would sit on different XCDs (L2s), so keep them on one.
    n_kc, n_t = tl.num_programs(0), tl.num_programs(1)
    pid = remap_xcd(
        (tl.program_id(2) * n_t + tl.program_id(1)) * n_kc + tl.program_id(0),
        n_kc * n_t * tl.num_programs(2),
        NUM_XCDS,
    )
    i_kc = pid % n_kc
    i_t, i_bh = (pid // n_kc % n_t).to(tl.int64), (pid // (n_kc * n_t)).to(tl.int64)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)
    i_k, i_i = i_kc // NC, i_kc % NC

    n_all = B * T
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    beta += bos * HV + i_hv

    dAqk += (bos * HV + i_hv) * BT
    dAkk += (bos * HV + i_hv) * BT
    dq += (bos * HV + i_hv) * K
    dq2 += (bos * HV + i_hv) * K
    dk += (bos * HV + i_hv) * K
    dk2 += (bos * HV + i_hv) * K
    dg += (bos * HV + i_hv) * K
    dg2 += (bos * HV + i_hv) * K
    db2 += (i_k * n_all + bos) * HV + i_hv
    if L2NORM_QK:
        q_raw += (bos * H + i_h) * K
        k_raw += (bos * H + i_h) * K
    if FINISH_DB:
        db += bos * HV + i_hv
        if BETA_SIGMOID:
            beta_raw += bos * HV + i_hv

    o_i = tl.arange(0, BC)
    o_c = i_ti + o_i
    m_c = o_c < T
    m_ck = m_c[:, None] & m_k[None, :]
    m_dAf = m_c[:, None] & (o_i[None, :] < BT)
    m_dAt = (o_i[:, None] < BT) & m_c[None, :]
    p_g = g + o_c[:, None] * (HV * K) + o_k[None, :]
    b_g = tl.load(p_g, mask=m_ck, other=0.0).to(tl.float32)

    p_b = beta + o_c * HV
    b_b = tl.load(p_b, mask=m_c, other=0.0)

    # Rows of this sub-chunk against earlier sub-chunks.
    b_dq2 = tl.zeros([BC, BK], dtype=tl.float32)
    b_dk2 = tl.zeros([BC, BK], dtype=tl.float32)
    if i_i > 0:
        p_gn = g + i_ti * HV * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
        for i_j in range(i_i):
            o_j = i_t * BT + i_j * BC + o_i
            m_jk = (o_j < T)[:, None] & m_k[None, :]
            p_k = k + o_j[:, None] * (H * K) + o_k[None, :]
            p_gk = g + o_j[:, None] * (HV * K) + o_k[None, :]
            p_dAqk = dAqk + o_c[:, None] * (HV * BT) + (i_j * BC + o_i)[None, :]
            p_dAkk = dAkk + o_c[:, None] * (HV * BT) + (i_j * BC + o_i)[None, :]
            b_k = tl.load(p_k, mask=m_jk, other=0.0)
            b_gk = tl.load(p_gk, mask=m_jk, other=0.0)
            b_kg = b_k * exp2(b_gn - b_gk)
            b_dAqk = tl.load(p_dAqk, mask=m_dAf, other=0.0)
            b_dAkk = tl.load(p_dAkk, mask=m_dAf, other=0.0)
            b_dq2 = tl.dot(b_dAqk, b_kg, b_dq2)
            b_dk2 = tl.dot(b_dAkk, b_kg, b_dk2)
        b_gqn = exp2(b_g - b_gn)
        b_dq2 *= b_gqn
        b_dk2 *= b_gqn

    # Rows of this sub-chunk against its own diagonal block.
    o_dA = (i_ti + o_i) * HV * BT + i_i * BC
    m_dA = (i_ti + o_i) < T
    p_kj = k + i_ti * H * K + o_k
    p_gkj = g + i_ti * HV * K + o_k

    p_q = q + o_c[:, None] * (H * K) + o_k[None, :]
    p_k = k + o_c[:, None] * (H * K) + o_k[None, :]
    b_q = tl.load(p_q, mask=m_ck, other=0.0)
    b_k = tl.load(p_k, mask=m_ck, other=0.0)

    if SAFE_GATE:
        # Pivot on the sub-chunk midpoint; the bounded gate keeps both exponents in range.
        p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * HV * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]

        p_dAqk = dAqk + o_c[:, None] * (HV * BT) + (i_i * BC + o_i)[None, :]
        p_dAkk = dAkk + o_c[:, None] * (HV * BT) + (i_i * BC + o_i)[None, :]
        b_dAqk_d = tl.load(p_dAqk, mask=m_dAf, other=0.0).to(tl.float32)
        b_dAkk_d = tl.load(p_dAkk, mask=m_dAf, other=0.0).to(tl.float32)

        m_i_d = (
            (o_i[:, None] >= o_i[None, :])
            & ((i_ti + o_i[:, None]) < T)
            & ((i_ti + o_i[None, :]) < T)
        )
        m_j_d = (i_ti + o_i[:, None]) < T
        b_dAqk_d = tl.where(m_i_d, b_dAqk_d, 0.0)
        b_dAkk_d = tl.where(m_i_d, b_dAkk_d, 0.0)
        b_g_d = tl.where(m_j_d, b_g - b_gn, 0.0)
        exp_g_d = tl.where(m_j_d, exp2(b_g_d), 0.0)
        exp_neg_g_d = tl.where(m_j_d, exp2(-b_g_d), 0.0)

        b_k_exp = b_k * exp_neg_g_d
        b_dq2 += tl.dot(b_dAqk_d, b_k_exp) * exp_g_d
        b_dk2 += tl.dot(b_dAkk_d, b_k_exp) * exp_g_d
    else:
        for j in range(min(BC, T - i_t * BT - i_i * BC)):
            b_dAqk = tl.load(dAqk + o_dA + j, mask=m_dA, other=0)
            b_dAkk = tl.load(dAkk + o_dA + j, mask=m_dA, other=0)
            b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
            b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
            m_i = o_i[:, None] >= j
            b_gqk = exp2(b_g - b_gkj[None, :])
            b_dq2 += tl.where(m_i, b_dAqk[:, None] * b_kj[None, :] * b_gqk, 0.0)
            b_dk2 += tl.where(m_i, b_dAkk[:, None] * b_kj[None, :] * b_gqk, 0.0)
            p_kj += H * K
            p_gkj += HV * K

    b_db = tl.sum(b_dk2 * b_k, 1)
    b_dk2 *= b_b[:, None]

    p_dq = dq + o_c[:, None] * (HV * K) + o_k[None, :]
    p_dq2 = dq2 + o_c[:, None] * (HV * K) + o_k[None, :]
    p_db2 = db2 + o_c * HV

    b_dg2 = b_q * b_dq2
    b_dq2 = b_dq2 + tl.load(p_dq, mask=m_ck, other=0.0)
    if L2NORM_QK:
        p_qr = q_raw + o_c[:, None] * (H * K) + o_k[None, :]
        b_qr = tl.load(p_qr, mask=m_ck, other=0.0).to(tl.float32)
        b_dq2 = _l2norm_bwd_rows(b_qr, b_dq2, eps)
    tl.store(p_dq2, b_dq2.to(p_dq2.dtype.element_ty), mask=m_ck)
    if FINISH_DB:
        b_db += tl.load(db + o_c * HV, mask=m_c, other=0.0)
        if BETA_SIGMOID:
            b_s = tl.sigmoid(
                tl.load(beta_raw + o_c * HV, mask=m_c, other=0).to(tl.float32)
            )
            b_db = b_db * b_s * (1.0 - b_s)
    tl.store(p_db2, b_db.to(p_db2.dtype.element_ty), mask=m_c)

    tl.debug_barrier()
    # Columns of this sub-chunk against later sub-chunks.
    b_dkt = tl.zeros([BC, BK], dtype=tl.float32)

    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
        p_gn = g + (min(i_ti + BC, T) - 1) * HV * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
        for i_j in range(i_i + 1, NC):
            o_j = i_t * BT + i_j * BC + o_i
            m_j = o_j < T
            m_jk = m_j[:, None] & m_k[None, :]
            m_dAj = (o_i[:, None] < BT) & m_j[None, :]
            p_q = q + o_j[:, None] * (H * K) + o_k[None, :]
            p_k = k + o_j[:, None] * (H * K) + o_k[None, :]
            p_gk = g + o_j[:, None] * (HV * K) + o_k[None, :]
            p_b = beta + o_j * HV
            p_dAqk = dAqk + (i_i * BC + o_i)[:, None] + o_j[None, :] * (HV * BT)
            p_dAkk = dAkk + (i_i * BC + o_i)[:, None] + o_j[None, :] * (HV * BT)
            b_bj = tl.load(p_b, mask=m_j, other=0.0)
            b_qj = tl.load(p_q, mask=m_jk, other=0.0)
            b_kb = tl.load(p_k, mask=m_jk, other=0.0) * b_bj[:, None]
            b_gk = tl.load(p_gk, mask=m_jk, other=0.0).to(tl.float32)
            b_dAqk = tl.load(p_dAqk, mask=m_dAj, other=0.0)
            b_dAkk = tl.load(p_dAkk, mask=m_dAj, other=0.0)
            b_gkn = exp2(b_gk - b_gn)
            b_qg = b_qj * tl.where(m_j[:, None], b_gkn, 0)
            b_kbg = b_kb * tl.where(m_j[:, None], b_gkn, 0)
            # fp32 operands: bf16 loses too much here (fla).
            b_dkt = tl.dot(b_dAqk, b_qg, b_dkt)
            b_dkt = tl.dot(b_dAkk, b_kbg, b_dkt)
        b_dkt *= exp2(b_gn - b_g)

    o_dA = i_ti * HV * BT + i_i * BC + o_i
    p_qj = q + i_ti * H * K + o_k
    p_kj = k + i_ti * H * K + o_k
    p_gkj = g + i_ti * HV * K + o_k
    p_bj = beta + i_ti * HV

    if SAFE_GATE:
        p_gn = g + (i_ti + min(BC // 2, T - i_ti - 1)) * HV * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]

        p_dAqk = dAqk + (i_i * BC + o_i)[:, None] + o_c[None, :] * (HV * BT)
        p_dAkk = dAkk + (i_i * BC + o_i)[:, None] + o_c[None, :] * (HV * BT)
        b_dAqk_d = tl.load(p_dAqk, mask=m_dAt, other=0.0).to(tl.float32)
        b_dAkk_d = tl.load(p_dAkk, mask=m_dAt, other=0.0).to(tl.float32)

        m_i_d = (
            (o_i[:, None] <= o_i[None, :])
            & ((i_ti + o_i[:, None]) < T)
            & ((i_ti + o_i[None, :]) < T)
        )
        m_j_d = (i_ti + o_i[:, None]) < T
        b_dAqk_d = tl.where(m_i_d, b_dAqk_d, 0.0)
        b_dAkk_d = tl.where(m_i_d, b_dAkk_d, 0.0)
        b_g_d = tl.where(m_j_d, b_g - b_gn, 0.0)
        exp_g_d = tl.where(m_j_d, exp2(b_g_d), 0.0)
        exp_neg_g_d = tl.where(m_j_d, exp2(-b_g_d), 0.0)

        b_q_exp = b_q * exp_g_d
        b_kb_exp = b_k * b_b[:, None] * exp_g_d
        b_dkt += tl.dot(b_dAqk_d, b_q_exp) * exp_neg_g_d
        b_dkt += tl.dot(b_dAkk_d, b_kb_exp) * exp_neg_g_d
    else:
        for j in range(min(BC, T - i_t * BT - i_i * BC)):
            b_dAqk = tl.load(dAqk + o_dA + j * HV * BT)
            b_dAkk = tl.load(dAkk + o_dA + j * HV * BT)
            b_qj = tl.load(p_qj, mask=m_k, other=0).to(tl.float32)
            b_kbj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32) * tl.load(p_bj)
            b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
            m_i = o_i[:, None] <= j
            b_gkq = exp2(b_gkj[None, :] - b_g)
            b_dkt += tl.where(m_i, b_dAqk[:, None] * b_qj[None, :] * b_gkq, 0.0)
            b_dkt += tl.where(m_i, b_dAkk[:, None] * b_kbj[None, :] * b_gkq, 0.0)
            p_qj += H * K
            p_kj += H * K
            p_gkj += HV * K
            p_bj += HV

    p_dk = dk + o_c[:, None] * (HV * K) + o_k[None, :]
    p_dk2 = dk2 + o_c[:, None] * (HV * K) + o_k[None, :]
    p_dg = dg + o_c[:, None] * (HV * K) + o_k[None, :]
    p_dg2 = dg2 + o_c[:, None] * (HV * K) + o_k[None, :]

    b_dg2 += (b_dk2 - b_dkt) * b_k + tl.load(p_dg, mask=m_ck, other=0.0)
    b_dk2 += tl.load(p_dk, mask=m_ck, other=0.0)
    b_dk2 += b_dkt
    if L2NORM_QK:
        p_kr = k_raw + o_c[:, None] * (H * K) + o_k[None, :]
        b_kr = tl.load(p_kr, mask=m_ck, other=0.0).to(tl.float32)
        b_dk2 = _l2norm_bwd_rows(b_kr, b_dk2, eps)

    tl.store(p_dk2, b_dk2.to(p_dk2.dtype.element_ty), mask=m_ck)
    tl.store(p_dg2, b_dg2.to(p_dg2.dtype.element_ty), mask=m_ck)


@triton.jit
def chunk_delta_attn_bwd_kernel_gate_cumsum(
    s,
    g,
    A_log,
    dt_bias,
    cu_seqlens,
    chunk_indices,
    o,
    dA,
    dbias,
    lower_bound,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_GATE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    """Chunk-local reverse cumsum of ``s`` ``[B, T, H, S]`` (the adjoint of the forward's
    gate cumsum), then with ``USE_GATE`` the backward of the fused gate activation.

    The activation is ``-exp(A_log) * softplus(g + bias)``, or ``lower_bound *
    sigmoid(exp(A_log) * (g + bias))``. ``o`` is written in its own dtype.
    ``dA`` ``[NT, B * H, cdiv(S, BS)]`` and ``dbias`` ``[NT, B * H, S]`` receive
    per-program partial sums of the ``A_log`` and ``dt_bias`` gradients (zeros
    from programs on padded varlen chunks).
    """
    i_s, i_t, i_bh = (
        tl.program_id(0),
        tl.program_id(1).to(tl.int64),
        tl.program_id(2).to(tl.int64),
    )
    i_p = (i_t * tl.num_programs(2) + i_bh) * tl.num_programs(0) + i_s
    i_b, i_h = i_bh // H, i_bh % H
    o_s = i_s * BS + tl.arange(0, BS)
    o_db = (i_p - i_s) // tl.num_programs(0) * S + o_s
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = eos - bos
        if i_t * BT >= T:
            if USE_GATE:
                tl.store(dA + i_p, 0.0)
                if HAS_BIAS:
                    tl.store(
                        dbias + o_db, tl.zeros([BS], dtype=tl.float32), mask=o_s < S
                    )
            return
    else:
        bos, eos = i_b * T, i_b * T + T

    o_t = i_t * BT + tl.arange(0, BT)
    m_s = (o_t[:, None] < T) & (o_s[None, :] < S)
    offs = (bos * H + i_h) * S + o_t[:, None] * (H * S) + o_s[None, :]
    b_s = tl.load(s + offs, mask=m_s, other=0.0).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0, reverse=True)

    if USE_GATE:
        b_A = tl.load(A_log + i_h).to(tl.float32)
        b_g = tl.load(g + offs, mask=m_s, other=0.0).to(tl.float32)
        if HAS_BIAS:
            b_bias = tl.load(dt_bias + i_h * S + o_s, mask=o_s < S, other=0.0).to(
                tl.float32
            )
            b_g = b_g + b_bias[None, :]
        if not USE_LOWER_BOUND:
            b_A = -exp(b_A)
            b_yg = b_A * softplus(b_g)
            b_dg = b_A * (b_o * tl.sigmoid(b_g))
            b_dA = tl.sum(tl.sum(tl.where(m_s, b_o * b_yg, 0.0), 1), 0)
        else:
            b_A = exp(b_A)
            b_sig = tl.sigmoid(b_A * b_g)
            b_dg = b_o * (lower_bound * b_sig * (1.0 - b_sig)) * b_A
            b_dA = tl.sum(tl.sum(tl.where(m_s, b_dg * b_g, 0.0), 1), 0)
        b_o = b_dg
        tl.store(dA + i_p, b_dA)
        if HAS_BIAS:
            tl.store(dbias + o_db, tl.sum(tl.where(m_s, b_dg, 0.0), 0), mask=o_s < S)

    tl.store(o + offs, b_o.to(o.dtype.element_ty), mask=m_s)


@triton.jit
def chunk_delta_attn_bwd_kernel_l2norm(
    X,
    DY,
    DX,
    eps,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
):
    """Backward of the row L2 normalization, recomputing the norm from ``X``."""
    xoffset = tl.program_id(0).to(tl.int64) * BT
    row_idx = xoffset + tl.arange(0, BT)[:, None]
    col_idx = tl.arange(0, BD)[None, :]
    mask = (row_idx < T) & (col_idx < D)
    x = tl.load(X + col_idx + D * row_idx, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + col_idx + D * row_idx, mask=mask, other=0.0).to(tl.float32)
    dx = _l2norm_bwd_rows(x, dy, eps)
    tl.store(DX + col_idx + D * row_idx, dx.to(DX.dtype.element_ty), mask=mask)


@triton.jit
def chunk_delta_attn_bwd_kernel_beta_sigmoid(
    x,
    dy,
    dx,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """dx = dy * sigmoid(x) * (1 - sigmoid(x))."""
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < n_elements
    b_x = tl.load(x + offs, mask=mask, other=0).to(tl.float32)
    b_dy = tl.load(dy + offs, mask=mask, other=0).to(tl.float32)
    b_y = tl.sigmoid(b_x)
    tl.store(dx + offs, (b_dy * b_y * (1.0 - b_y)).to(dx.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

# Launchable anywhere; the measured tiles are published per arch under the
# CHUNK_DELTA_ATTN config, keyed by kernel name. The recurrences have a narrow
# and a wide tile: see `_recurrence_config`.
_FALLBACK_CONFIGS = {
    "chunk_delta_attn_bwd_kernel_dav": triton.Config(
        {"BV": 64}, num_warps=4, num_stages=2
    ),
    "chunk_delta_attn_bwd_kernel_fwd_h": triton.Config(
        {"BV": 32}, num_warps=2, num_stages=2
    ),
    "chunk_delta_attn_bwd_kernel_fwd_h_wide": triton.Config(
        {"BV": 64}, num_warps=4, num_stages=2
    ),
    "chunk_delta_attn_bwd_kernel_dhu": triton.Config(
        {"BV": 32}, num_warps=2, num_stages=2
    ),
    "chunk_delta_attn_bwd_kernel_dhu_wide": triton.Config(
        {"BV": 64}, num_warps=4, num_stages=2
    ),
    "chunk_delta_attn_bwd_kernel_wy_dqkg": triton.Config(
        {"BK": 64, "BV": 32}, num_warps=2, num_stages=1
    ),
    "chunk_delta_attn_bwd_kernel_intra": triton.Config(
        {"BK": 64}, num_warps=2, num_stages=2
    ),
    "chunk_delta_attn_bwd_kernel_gate_cumsum": triton.Config(
        {"BS": 32}, num_warps=4, num_stages=2
    ),
    "chunk_delta_attn_bwd_kernel_l2norm": triton.Config(
        {"BT": 32}, num_warps=2, num_stages=2
    ),
    "chunk_delta_attn_bwd_kernel_beta_sigmoid": triton.Config(
        {"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2
    ),
}

_dav_k = fast_launch(chunk_delta_attn_bwd_kernel_dav)
_fwd_h_k = fast_launch(chunk_delta_attn_bwd_kernel_fwd_h)
_dhu_k = fast_launch(chunk_delta_attn_bwd_kernel_dhu)
_fwd_h_dhu_k = fast_launch(chunk_delta_attn_bwd_kernel_fwd_h_dhu)
_wy_dqkg_k = fast_launch(chunk_delta_attn_bwd_kernel_wy_dqkg)
_intra_k = fast_launch(chunk_delta_attn_bwd_kernel_intra)
_gate_cumsum_k = fast_launch(chunk_delta_attn_bwd_kernel_gate_cumsum)
_l2norm_k = fast_launch(chunk_delta_attn_bwd_kernel_l2norm)
_beta_sigmoid_k = fast_launch(chunk_delta_attn_bwd_kernel_beta_sigmoid)

# Program ids are renumbered so programs sharing loads land on one XCD (L2).
# The XCD count cannot be queried at runtime; 1 disables the remap.
_NUM_XCDS: int = 8 if arch_info.get_arch() in ("gfx942", "gfx950") else 1


@functools.cache
def _config(name: str) -> triton.Config:
    return chunk_delta_attn_tuned_config(name, _FALLBACK_CONFIGS[name])


@functools.cache
def _num_cus(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _recurrence_config(name: str, n_states: int, V: int, num_cus: int):
    """Tile of a chunk recurrence (``fwd_h`` or ``dhu``) over ``n_states`` states.

    Both run one program per (V block, state), each looping over every chunk in
    turn. With few states the narrow tile's extra V blocks are most of the
    parallelism; once the wide tile alone fills the CUs its larger tiles win
    (1.4-1.65x per kernel on gfx950 from ``N * HV = 128`` up).
    """
    wide = _config(name + "_wide")
    if triton.cdiv(V, wide.kwargs["BV"]) * n_states >= num_cus:
        return wide
    return _config(name)


def _fused_recurrences_config(n_states: int, V: int, num_cus: int):
    """Tile of ``chunk_delta_attn_bwd_kernel_fwd_h_dhu``, or None to launch the two apart.

    They share a launch when both take their narrow tile -- neither fills the
    CUs on its own, and together they still fit about one wave per SIMD -- and
    the two narrow tiles agree, since the launch has one tile and warp count.
    """
    name_h = "chunk_delta_attn_bwd_kernel_fwd_h"
    name_dh = "chunk_delta_attn_bwd_kernel_dhu"
    cfg_h = _recurrence_config(name_h, n_states, V, num_cus)
    cfg_dh = _recurrence_config(name_dh, n_states, V, num_cus)
    narrow = cfg_h is _config(name_h) and cfg_dh is _config(name_dh)
    same = (cfg_h.kwargs, cfg_h.num_warps) == (cfg_dh.kwargs, cfg_dh.num_warps)
    return cfg_dh if narrow and same else None


def _recompute_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    chunk_size,
    safe_gate,
    lower_bound,
    use_gate_in_kernel,
    A_log,
    dt_bias,
    use_qk_l2norm_in_kernel,
    use_beta_sigmoid_in_kernel,
    state_v_first,
    run_fwd_h,
    num_cus,
):
    """The default forward pipeline up to (not including) the output kernel.

    Returns the activated ``q``/``k``/``beta``, ``g_cumsum``, ``Aqk``, ``Akk``,
    ``w``, ``u``, ``qg``, ``kg``, and ``h``/``v_new``, which are left unwritten
    unless ``run_fwd_h`` (the caller then fills them alongside ``dhu``).
    """
    B, T, _, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    BT = chunk_size
    is_varlen = cu_seqlens is not None
    N = cu_seqlens.numel() - 1 if is_varlen else B
    NT = triton.cdiv(T, BT) if not is_varlen else len(chunk_indices)

    if use_qk_l2norm_in_kernel:
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)
    if use_beta_sigmoid_in_kernel:
        beta = beta_sigmoid_fwd(beta)

    if use_gate_in_kernel:
        g_cumsum = chunk_gate_cumsum(
            g=g,
            A_log=A_log,
            chunk_size=BT,
            scale=RCP_LN2,
            dt_bias=dt_bias,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            lower_bound=lower_bound,
        )
    else:
        g_cumsum = chunk_local_cumsum(
            g=g,
            chunk_size=BT,
            scale=RCP_LN2,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

    # disable_recompute=True also returns qg = q * exp2(g_cumsum), which dhu
    # reads instead of gating q on its serial chunk chain.
    w, u, qg, kg, Aqk, Akk = chunk_delta_attn_fwd_intra(
        q=q,
        k=k,
        v=v,
        gk=g_cumsum,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=BT,
        chunk_indices=chunk_indices,
        safe_gate=safe_gate,
        disable_recompute=True,
    )

    h = k.new_empty(B, NT, HV, K, V)
    v_new = torch.empty_like(u)
    if run_fwd_h:
        cfg = _recurrence_config(
            "chunk_delta_attn_bwd_kernel_fwd_h", N * HV, V, num_cus
        )
        BV = cfg.kwargs["BV"]
        _fwd_h_k[(triton.cdiv(V, BV), N * HV)](
            k=kg,
            v=u,
            w=w,
            gk=g_cumsum,
            h0=initial_state,
            cu_seqlens=cu_seqlens,
            chunk_offsets=chunk_offsets,
            h=h,
            v_new=v_new,
            ht=None,
            T=T,
            H=HV,
            K=K,
            V=V,
            BT=BT,
            BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=False,
            IS_VARLEN=is_varlen,
            TRANSPOSE_STATE=state_v_first,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )
    return q, k, beta, g_cumsum, Aqk, Akk, w, u, qg, kg, h, v_new


def _l2norm_bwd(x: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    D = x.shape[-1]
    x2 = x.reshape(-1, D)
    dx = torch.empty_like(x2)
    T = x2.shape[0]
    cfg = _config("chunk_delta_attn_bwd_kernel_l2norm")
    BT = cfg.kwargs["BT"]
    _l2norm_k[(triton.cdiv(T, BT),)](
        X=x2,
        DY=dy.reshape(-1, D),
        DX=dx,
        eps=1e-6,
        T=T,
        D=D,
        BD=triton.next_power_of_2(D),
        BT=BT,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    return dx.view(x.shape)


def _beta_sigmoid_bwd(x: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    dx = torch.empty_like(x)
    n = x.numel()
    cfg = _config("chunk_delta_attn_bwd_kernel_beta_sigmoid")
    bs = cfg.kwargs["BLOCK_SIZE"]
    _beta_sigmoid_k[(triton.cdiv(n, bs),)](
        x=x,
        dy=dy,
        dx=dx,
        n_elements=n,
        BLOCK_SIZE=bs,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    return dx


def chunk_delta_attn_bwd(
    do: torch.Tensor,
    dht: torch.Tensor | None,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    state_v_first: bool = False,
) -> tuple:
    """
    Backward pass for chunk_delta_attn, recomputing the forward's intermediates.

    Args:
        do:   Gradient of the output ``[B, T, HV, V]``.
        dht:  Gradient of the final state, laid out like ``initial_state``, or None.
        The rest are the forward's inputs, contiguous, as ``chunk_delta_attn_fwd``
        takes them. ``chunk_size`` is the default pipeline's (32 or 64), whatever
        the forward ran with.

    Returns:
        ``(dq, dk, dv, dg, dbeta, dA_log, ddt_bias, dh0)``, each in the dtype of
        the input it belongs to; entries for absent inputs are None.
    """
    if chunk_size not in (32, 64):
        raise ValueError(
            f"`chunk_size` must be either 32 or 64 for chunk_delta_attn, got {chunk_size}."
        )
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    BT = chunk_size
    is_varlen = cu_seqlens is not None
    N = cu_seqlens.numel() - 1 if is_varlen else B
    if is_varlen:
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
        NT = len(chunk_indices)
    else:
        chunk_offsets = None
        NT = triton.cdiv(T, BT)
    num_cus = _num_cus(q.device.index or 0)
    rec_cfg = _fused_recurrences_config(N * HV, V, num_cus)

    qn, kn, bn, gc, Aqk, Akk, w, u, qg, kg, h, v_new = _recompute_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        chunk_size=BT,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        state_v_first=state_v_first,
        run_fwd_h=rec_cfg is None,
        num_cus=num_cus,
    )
    do = do.contiguous()
    dht = dht.contiguous() if dht is not None else None

    # dAqk = do @ v_new^T, dv = Aqk^T @ do.
    dAqk = torch.empty(B, T, HV, BT, device=q.device, dtype=torch.float32)
    dv = torch.empty_like(v)
    cfg_dav = _config("chunk_delta_attn_bwd_kernel_dav")

    def dav(compute_da, compute_dv):
        _dav_k[(NT, B * HV)](
            v=v_new,
            A=Aqk,
            do=do,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            dA=dAqk,
            dv=dv,
            scale=scale,
            T=T,
            HV=HV,
            V=V,
            BT=BT,
            BV=cfg_dav.kwargs["BV"],
            IS_VARLEN=is_varlen,
            COMPUTE_DA=compute_da,
            COMPUTE_DV=compute_dv,
            num_warps=cfg_dav.num_warps,
            num_stages=cfg_dav.num_stages,
        )

    # Reverse-time state gradient.
    dh = q.new_empty(B, NT, HV, K, V)
    dh0 = (
        torch.empty_like(initial_state, dtype=torch.float32)
        if initial_state is not None
        else None
    )
    dv2 = torch.empty_like(dv)
    dhu_args = {
        "qg": qg,
        "g": gc,
        "k": kg,
        "w": w,
        "dht": dht,
        "do": do,
        "dv": dv,
        "cu_seqlens": cu_seqlens,
        "chunk_offsets": chunk_offsets,
        "dh": dh,
        "dh0": dh0,
        "dv2": dv2,
        "scale": scale,
        "T": T,
        "HV": HV,
        "K": K,
        "V": V,
        "BT": BT,
        "USE_INITIAL_STATE": initial_state is not None,
        "USE_FINAL_STATE_GRADIENT": dht is not None,
        "IS_VARLEN": is_varlen,
        "TRANSPOSE_STATE": state_v_first,
        "NUM_XCDS": _NUM_XCDS,
    }
    if rec_cfg is None:
        dav(True, True)
        cfg = _recurrence_config("chunk_delta_attn_bwd_kernel_dhu", N * HV, V, num_cus)
        BV = cfg.kwargs["BV"]
        _dhu_k[(triton.cdiv(V, BV), N * HV)](
            **dhu_args, BV=BV, num_warps=cfg.num_warps, num_stages=cfg.num_stages
        )
    else:
        # dhu needs dv but not v_new, so it runs alongside the recomputed
        # fwd_h; dAqk, which does need v_new, follows.
        dav(False, True)
        BV = rec_cfg.kwargs["BV"]
        _fwd_h_dhu_k[(2 * triton.cdiv(V, BV), N * HV)](
            **dhu_args,
            v=u,
            h0=initial_state,
            h=h,
            v_new=v_new,
            BV=BV,
            num_warps=rec_cfg.num_warps,
            num_stages=rec_cfg.num_stages,
        )
        dav(True, False)

    # Inter-chunk and WY-representation gradients.
    # Varlen tokens past cu_seqlens[-1] are never written: zero those outputs.
    alloc = torch.zeros if is_varlen else torch.empty
    dq = torch.empty(B, T, HV, K, device=q.device, dtype=torch.float32)
    dk = torch.empty_like(dq)
    dg = torch.empty_like(dq)
    dv = alloc(v.shape, device=v.device, dtype=v.dtype)
    db = alloc(B, T, HV, device=q.device, dtype=torch.float32)
    dAkk = torch.empty_like(dAqk)
    cfg = _config("chunk_delta_attn_bwd_kernel_wy_dqkg")
    _wy_dqkg_k[(NT, B * HV)](
        q=qn,
        k=kn,
        v=v,
        v_new=v_new,
        g=gc,
        beta=bn,
        A=Akk,
        h=h,
        do=do,
        dh=dh,
        dv=dv2,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        dq=dq,
        dk=dk,
        dv2=dv,
        dg=dg,
        db=db,
        dA=dAkk,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=cfg.kwargs["BK"],
        BV=cfg.kwargs["BV"],
        IS_VARLEN=is_varlen,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )

    # Intra-chunk gradients.
    # With the in-kernel l2norm and no GVA (no head-group sum pending), one
    # program per whole K row (K <= 128) finishes dq/dk through the l2norm
    # and dbeta. Without the l2norm to fold in, the wider tile costs more
    # than the small dbeta kernels it would save.
    cfg = _config("chunk_delta_attn_bwd_kernel_intra")
    BC = min(16, BT)
    fuse = K <= 128 and use_qk_l2norm_in_kernel and HV == H
    sigmoid = fuse and use_beta_sigmoid_in_kernel
    BK = triton.next_power_of_2(K)
    if not fuse:
        BK = min(cfg.kwargs["BK"], BK)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)
    dq2 = alloc(dq.shape, device=q.device, dtype=q.dtype if fuse else torch.float32)
    dk2 = alloc(dk.shape, device=q.device, dtype=k.dtype if fuse else torch.float32)
    dg2 = torch.empty_like(dg)
    if fuse:
        db2 = alloc(B, T, HV, device=q.device, dtype=beta.dtype)
    else:
        db2 = alloc(NK, B, T, HV, device=q.device, dtype=torch.float32)
    _intra_k[(NK * NC, NT, B * HV)](
        q=qn,
        k=kn,
        g=gc,
        beta=bn,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dk=dk,
        dg=dg,
        q_raw=q if fuse else None,
        k_raw=k if fuse else None,
        beta_raw=beta if sigmoid else None,
        db=db if fuse else None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        dq2=dq2,
        dk2=dk2,
        dg2=dg2,
        db2=db2,
        eps=1e-6,
        B=B,
        T=T,
        H=H,
        HV=HV,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        NC=NC,
        IS_VARLEN=is_varlen,
        SAFE_GATE=safe_gate,
        NUM_XCDS=_NUM_XCDS,
        L2NORM_QK=fuse,
        FINISH_DB=fuse,
        BETA_SIGMOID=sigmoid,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    dq, dk, db = dq2, dk2, db2 if fuse else db2.sum(0).add_(db)
    if HV > H:
        dq = dq.view(B, T, H, HV // H, K).sum(3)
        dk = dk.view(B, T, H, HV // H, K).sum(3)

    # Gradient w.r.t. the per-token gate, then through its activation, in one
    # pass that also leaves per-chunk partial sums of dA_log and ddt_bias.
    cfg = _config("chunk_delta_attn_bwd_kernel_gate_cumsum")
    BS = cfg.kwargs["BS"]
    NS = triton.cdiv(K, BS)
    use_bias = use_gate_in_kernel and dt_bias is not None
    f32 = {"device": q.device, "dtype": torch.float32}
    dA_part = torch.empty(NT, B, HV, NS, **f32) if use_gate_in_kernel else None
    db_part = torch.empty(NT * B, HV * K, **f32) if use_bias else None
    dg = alloc(dg2.shape, device=q.device, dtype=g.dtype)
    _gate_cumsum_k[(NS, NT, B * HV)](
        s=dg2,
        g=g,
        A_log=A_log,
        dt_bias=dt_bias,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        o=dg,
        dA=dA_part,
        dbias=db_part,
        lower_bound=0.0 if lower_bound is None else float(lower_bound),
        T=T,
        H=HV,
        S=K,
        BT=BT,
        BS=BS,
        IS_VARLEN=is_varlen,
        USE_GATE=use_gate_in_kernel,
        HAS_BIAS=use_bias,
        USE_LOWER_BOUND=lower_bound is not None,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    dA_log = dA_part.sum((0, 1, 3)).to(A_log.dtype) if use_gate_in_kernel else None
    ddt_bias = db_part.sum(0).to(dt_bias.dtype) if use_bias else None

    if use_qk_l2norm_in_kernel and not fuse:
        dq = _l2norm_bwd(q, dq)
        dk = _l2norm_bwd(k, dk)
    if use_beta_sigmoid_in_kernel and not sigmoid:
        db = _beta_sigmoid_bwd(beta, db)
    return (
        dq.to(q.dtype),
        dk.to(k.dtype),
        dv,
        dg.to(g.dtype),
        db.to(beta.dtype),
        dA_log,
        ddt_bias,
        dh0.to(initial_state.dtype) if dh0 is not None else None,
    )
