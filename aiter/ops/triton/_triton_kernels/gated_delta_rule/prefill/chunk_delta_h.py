# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""
Chunk-based hidden state computation for gated delta rule (Forward only).

This module computes the per-chunk hidden states and the recomputed value
tensor (`v_new`) consumed by the chunked gated delta rule, supporting both
the `[K, V]` and the transposed `[V, K]` hidden-state layouts.
"""

import torch
import triton
import triton.language as tl

from ..gated_delta_rule_utils import (
    IS_AMD,
    IS_NVIDIA_HOPPER,
    RCP_LN2,
    USE_CUDA_GRAPH,
    autotune_cache_kwargs,
    check_shared_mem,
    gated_delta_rule_autotune_configs,
)
from ..utils import (
    GatedDeltaRulePrefillMetadata,
    prepare_chunk_indices,
    prepare_chunk_offsets,
    prepare_rebased_cu_seqlens,
)
from ..utils.op import exp

NUM_WARPS = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8, 16]
# Workaround: AMD ROCm Triton compiler fails with num_stages=4 in stream pipeline
NUM_STAGES_FWD = [2, 3] if IS_AMD else [2, 3, 4]


@triton.jit
def _gate_exp(x, USE_EXP2: tl.constexpr):
    """Exponentiate a cumulative gate held in log2 space (USE_EXP2) or natural log."""
    return tl.math.exp2(x) if USE_EXP2 else exp(x)


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=gated_delta_rule_autotune_configs(
        [
            triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
            for num_warps in [2, 4]
            for num_stages in NUM_STAGES_FWD
            for BV in [32, 64]
        ]
    ),
    key=["H", "K", "V", "BT", "TRANSPOSE_STATE"],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
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

    # [BK, BV]
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    #
    # Widen to int64 *before* scaling by K/V: a 64-head K=V=128 model overflows
    # int32 at ~131k tokens, and the wrapped negative offset corrupts memory in
    # front of the buffer rather than failing loudly.
    h += (boh * H + i_h).to(tl.int64) * K * V
    v += (bos * H + i_h).to(tl.int64) * V
    k += (bos * H + i_h).to(tl.int64) * K
    w += (bos * H + i_h).to(tl.int64) * K
    if SAVE_NEW_VALUE:
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

    # load initial state
    #
    # TRANSPOSE_STATE describes the state buffer as V-first ``[V, K]`` instead of
    # ``[K, V]``.
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

    # main recurrence
    for i_t in range(NT):
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        m_tk1 = m_t[:, None] & m_k1[None, :]
        m_tv = m_t[:, None] & m_v[None, :]

        h_t = h + i_t.to(tl.int64) * stride_h
        p_h1 = h_t + o_k1[:, None] * V + o_v[None, :]
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), mask=m_h1)
        if K > 64:
            p_h2 = h_t + o_k2[:, None] * V + o_v[None, :]
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), mask=m_h2)
        if K > 128:
            p_h3 = h_t + o_k3[:, None] * V + o_v[None, :]
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), mask=m_h3)
        if K > 192:
            p_h4 = h_t + o_k4[:, None] * V + o_v[None, :]
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), mask=m_h4)

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

        if SAVE_NEW_VALUE:
            p_v = v_new + o_t[:, None] * stride_v + o_v[None, :]
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), mask=m_tv)

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = g + bos * H + i_h + o_t * H
            b_g = tl.load(p_g, mask=m_t, other=0.0)
            b_v = b_v * tl.where(m_t, _gate_exp(b_g_last - b_g, USE_EXP2), 0)[:, None]
            b_g_last = _gate_exp(b_g_last, USE_EXP2)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last

        if USE_GK:
            b_gk_last1 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k1,
                mask=m_k1,
                other=0.0,
            )
            b_h1 *= _gate_exp(b_gk_last1, USE_EXP2)[:, None]
            if K > 64:
                b_gk_last2 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k2,
                    mask=m_k2,
                    other=0.0,
                )
                b_h2 *= _gate_exp(b_gk_last2, USE_EXP2)[:, None]
            if K > 128:
                b_gk_last3 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k3,
                    mask=m_k3,
                    other=0.0,
                )
                b_h3 *= _gate_exp(b_gk_last3, USE_EXP2)[:, None]
            if K > 192:
                b_gk_last4 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k4,
                    mask=m_k4,
                    other=0.0,
                )
                b_h4 *= _gate_exp(b_gk_last4, USE_EXP2)[:, None]
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
    # epilogue
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


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_INITIAL_STATE": lambda args: args["dh0"] is not None,
        "USE_FINAL_STATE_GRADIENT": lambda args: args["dht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=gated_delta_rule_autotune_configs(
        [
            triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
            for num_warps in [2, 4]
            for num_stages in (
                [3, 2] if IS_AMD else ([4, 3, 2] if check_shared_mem("ampere") else [1])
            )
            for BV in [64, 32]
        ]
    ),
    key=["H", "K", "V", "BT", "BV", "USE_G"],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64(
    q,
    k,
    w,
    g,
    gk,
    dht,
    dh0,
    do,
    dh,
    dv,
    dv2,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
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

    # [BK, BV]
    b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_dh4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    q += (bos * H + i_h).to(tl.int64) * K
    k += (bos * H + i_h).to(tl.int64) * K
    w += (bos * H + i_h).to(tl.int64) * K
    do += (bos * H + i_h).to(tl.int64) * V
    dv += (bos * H + i_h).to(tl.int64) * V
    dv2 += (bos * H + i_h).to(tl.int64) * V
    dh += (boh * H + i_h).to(tl.int64) * K * V
    if USE_GK:
        gk += (bos * H + i_h).to(tl.int64) * K

    stride_v = H * V
    stride_h = H * K * V
    stride_k = H * K
    if USE_INITIAL_STATE:
        dh0 += i_nh.to(tl.int64) * K * V
    if USE_FINAL_STATE_GRADIENT:
        dht += i_nh.to(tl.int64) * K * V

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
        p_dht1 = dht + o_k1[:, None] * V + o_v[None, :]
        b_dh1 += tl.load(p_dht1, mask=m_h1, other=0.0)
        if K > 64:
            p_dht2 = dht + o_k2[:, None] * V + o_v[None, :]
            b_dh2 += tl.load(p_dht2, mask=m_h2, other=0.0)
        if K > 128:
            p_dht3 = dht + o_k3[:, None] * V + o_v[None, :]
            b_dh3 += tl.load(p_dht3, mask=m_h3, other=0.0)
        if K > 192:
            p_dht4 = dht + o_k4[:, None] * V + o_v[None, :]
            b_dh4 += tl.load(p_dht4, mask=m_h4, other=0.0)

    for i_t in range(NT - 1, -1, -1):
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        m_tv = m_t[:, None] & m_v[None, :]

        dh_t = dh + i_t.to(tl.int64) * stride_h
        p_dh1 = dh_t + o_k1[:, None] * V + o_v[None, :]
        tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), mask=m_h1)
        if K > 64:
            p_dh2 = dh_t + o_k2[:, None] * V + o_v[None, :]
            tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), mask=m_h2)
        if K > 128:
            p_dh3 = dh_t + o_k3[:, None] * V + o_v[None, :]
            tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), mask=m_h3)
        if K > 192:
            p_dh4 = dh_t + o_k4[:, None] * V + o_v[None, :]
            tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), mask=m_h4)

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            bg_last = tl.load(g + (bos + last_idx) * H + i_h)
            bg_last_exp = exp(bg_last)
            p_g = g + bos * H + i_h + o_t * H
            b_g = tl.load(p_g, mask=m_t, other=0.0)
            b_g_exp = exp(b_g)

        p_dv = dv + o_t[:, None] * stride_v + o_v[None, :]
        p_dv2 = dv2 + o_t[:, None] * stride_v + o_v[None, :]
        p_do = do + o_t[:, None] * stride_v + o_v[None, :]

        b_do = tl.load(p_do, mask=m_tv, other=0.0)

        # Update dv
        p_k = k + o_t[:, None] * stride_k + o_k1[None, :]
        b_k = tl.load(p_k, mask=m_t[:, None] & m_k1[None, :], other=0.0)
        if USE_GK:
            b_gk_last1 = tl.load(gk + last_idx * H * K + o_k1, mask=m_k1, other=0.0)
        b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype))

        if K > 64:
            p_k = k + o_t[:, None] * stride_k + o_k2[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & m_k2[None, :], other=0.0)
            if USE_GK:
                b_gk_last2 = tl.load(gk + last_idx * H * K + o_k2, mask=m_k2, other=0.0)
            b_dv = tl.dot(b_k, b_dh2.to(b_k.dtype), acc=b_dv)

        if K > 128:
            p_k = k + o_t[:, None] * stride_k + o_k3[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & m_k3[None, :], other=0.0)
            if USE_GK:
                b_gk_last3 = tl.load(gk + last_idx * H * K + o_k3, mask=m_k3, other=0.0)
            b_dv = tl.dot(b_k, b_dh3.to(b_k.dtype), acc=b_dv)

        if K > 192:
            p_k = k + o_t[:, None] * stride_k + o_k4[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & m_k4[None, :], other=0.0)
            if USE_GK:
                b_gk_last4 = tl.load(gk + last_idx * H * K + o_k4, mask=m_k4, other=0.0)
            b_dv = tl.dot(b_k, b_dh4.to(b_k.dtype), acc=b_dv)

        if USE_G:
            b_dv *= tl.where(m_t, exp(bg_last - b_g), 0)[:, None]
        b_dv += tl.load(p_dv, mask=m_tv, other=0.0)

        tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), mask=m_tv)
        # Update dh
        p_w = w + o_k1[:, None] + o_t[None, :] * stride_k
        p_q = q + o_k1[:, None] + o_t[None, :] * stride_k
        m_kt1 = m_k1[:, None] & m_t[None, :]
        b_w = tl.load(p_w, mask=m_kt1, other=0.0)
        b_q = tl.load(p_q, mask=m_kt1, other=0.0)
        if USE_G:
            b_dh1 *= bg_last_exp
            b_q = b_q * b_g_exp[None, :]
        if USE_GK:
            b_dh1 *= exp(b_gk_last1[:, None])
        b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(
            b_w, b_dv.to(b_w.dtype)
        )
        if K > 64:
            p_q = q + o_k2[:, None] + o_t[None, :] * stride_k
            p_w = w + o_k2[:, None] + o_t[None, :] * stride_k
            m_kt2 = m_k2[:, None] & m_t[None, :]
            b_q = tl.load(p_q, mask=m_kt2, other=0.0)
            b_w = tl.load(p_w, mask=m_kt2, other=0.0)
            if USE_G:
                b_dh2 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                b_dh2 *= exp(b_gk_last2[:, None])
            b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(
                b_w, b_dv.to(b_w.dtype)
            )
        if K > 128:
            p_q = q + o_k3[:, None] + o_t[None, :] * stride_k
            p_w = w + o_k3[:, None] + o_t[None, :] * stride_k
            m_kt3 = m_k3[:, None] & m_t[None, :]
            b_q = tl.load(p_q, mask=m_kt3, other=0.0)
            b_w = tl.load(p_w, mask=m_kt3, other=0.0)
            if USE_G:
                b_dh3 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                b_dh3 *= exp(b_gk_last3[:, None])
            b_dh3 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(
                b_w, b_dv.to(b_w.dtype)
            )
        if K > 192:
            p_q = q + o_k4[:, None] + o_t[None, :] * stride_k
            p_w = w + o_k4[:, None] + o_t[None, :] * stride_k
            m_kt4 = m_k4[:, None] & m_t[None, :]
            b_q = tl.load(p_q, mask=m_kt4, other=0.0)
            b_w = tl.load(p_w, mask=m_kt4, other=0.0)
            if USE_G:
                b_dh4 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                b_dh4 *= exp(b_gk_last4[:, None])
            b_dh4 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(
                b_w, b_dv.to(b_w.dtype)
            )

    if USE_INITIAL_STATE:
        p_dh0 = dh0 + o_k1[:, None] * V + o_v[None, :]
        tl.store(p_dh0, b_dh1.to(p_dh0.dtype.element_ty), mask=m_h1)
        if K > 64:
            p_dh1 = dh0 + o_k2[:, None] * V + o_v[None, :]
            tl.store(p_dh1, b_dh2.to(p_dh1.dtype.element_ty), mask=m_h2)
        if K > 128:
            p_dh2 = dh0 + o_k3[:, None] * V + o_v[None, :]
            tl.store(p_dh2, b_dh3.to(p_dh2.dtype.element_ty), mask=m_h3)
        if K > 192:
            p_dh3 = dh0 + o_k4[:, None] * V + o_v[None, :]
            tl.store(p_dh3, b_dh4.to(p_dh3.dtype.element_ty), mask=m_h4)


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    transpose_state: bool = False,
    use_exp2: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the per-chunk hidden states and the recomputed value tensor.

    ``transpose_state`` selects the V-first ``[N, H, V, K]`` layout for
    ``initial_state`` / ``final_state`` instead of the default ``[N, H, K, V]``,
    matching fla's ``state_v_first``. Only the recurrent state boundary is
    affected; the per-chunk ``h`` stays ``[B, NT, H, K, V]``.

    ``use_exp2`` must match the space the cumulative gate was accumulated in:
    callers that scale the cumsum by ``RCP_LN2`` (log2 space) need it set,
    otherwise the inter-chunk decay is applied as ``exp(g/ln2)`` instead of
    ``exp(g)``, over-decaying the state carried across every chunk boundary.
    """
    B, T, H, K, V = *k.shape, u.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = (
            len(cu_seqlens) - 1,
            len(chunk_indices),
            prepare_chunk_offsets(cu_seqlens, BT),
        )
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    state_shape = (N, H, V, K) if transpose_state else (N, H, K, V)
    if initial_state is not None and tuple(initial_state.shape) != state_shape:
        raise ValueError(
            f"`initial_state` must have shape {state_shape} for "
            f"transpose_state={transpose_state}, got {tuple(initial_state.shape)}."
        )

    h = k.new_empty(B, NT, H, K, V)
    final_state = (
        k.new_empty(*state_shape, dtype=torch.float32) if output_final_state else None
    )

    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        TRANSPOSE_STATE=transpose_state,
        USE_EXP2=use_exp2,
    )
    return h, v_new, final_state


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=gated_delta_rule_autotune_configs(
        [
            triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
            for num_warps in [2, 4]
            for num_stages in NUM_STAGES_FWD
            for BV in [16, 32, 64]
        ]
    ),
    key=["H", "K", "V", "BT", "IS_VARLEN"],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T", "T_flat"])
def chunk_gated_delta_rule_fwd_kernel_h_opt(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    T_flat,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
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

    h += (boh * H + i_h).to(tl.int64) * K * V
    k += (bos * Hg + i_h // (H // Hg)).to(tl.int64) * K
    if IS_VARLEN:
        v += (i_h * T_flat + bos).to(tl.int64) * V
        w += (i_h * T_flat + bos).to(tl.int64) * K
    else:
        v += ((i_n * H + i_h) * T_flat).to(tl.int64) * V
        w += ((i_n * H + i_h) * T_flat).to(tl.int64) * K
    stride_v = V
    stride_w = K
    if SAVE_NEW_VALUE:
        if IS_VARLEN:
            v_new += (i_h * T_flat + bos).to(tl.int64) * V
        else:
            v_new += ((i_n * H + i_h) * T_flat).to(tl.int64) * V
    stride_h = H * K * V
    stride_k = Hg * K
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

    if USE_INITIAL_STATE:
        p_h0_1 = h0 + o_k1[:, None] * V + o_v[None, :]
        b_h1 += tl.load(p_h0_1, mask=m_h1, other=0.0).to(tl.float32)
        if K > 64:
            p_h0_2 = h0 + o_k2[:, None] * V + o_v[None, :]
            b_h2 += tl.load(p_h0_2, mask=m_h2, other=0.0).to(tl.float32)
        if K > 128:
            p_h0_3 = h0 + o_k3[:, None] * V + o_v[None, :]
            b_h3 += tl.load(p_h0_3, mask=m_h3, other=0.0).to(tl.float32)
        if K > 192:
            p_h0_4 = h0 + o_k4[:, None] * V + o_v[None, :]
            b_h4 += tl.load(p_h0_4, mask=m_h4, other=0.0).to(tl.float32)

    for i_t in range(NT):
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        m_tv = m_t[:, None] & m_v[None, :]

        h_t = h + i_t.to(tl.int64) * stride_h
        p_h1 = h_t + o_k1[:, None] * V + o_v[None, :]
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), mask=m_h1)
        if K > 64:
            p_h2 = h_t + o_k2[:, None] * V + o_v[None, :]
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), mask=m_h2)
        if K > 128:
            p_h3 = h_t + o_k3[:, None] * V + o_v[None, :]
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), mask=m_h3)
        if K > 192:
            p_h4 = h_t + o_k4[:, None] * V + o_v[None, :]
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), mask=m_h4)

        p_w = w + o_t[:, None] * stride_w + o_k1[None, :]
        b_w = tl.load(p_w, mask=m_t[:, None] & m_k1[None, :], other=0.0)
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = w + o_t[:, None] * stride_w + o_k2[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k2[None, :], other=0.0)
            b_v = tl.dot(b_w, b_h2.to(b_w.dtype), acc=b_v)
        if K > 128:
            p_w = w + o_t[:, None] * stride_w + o_k3[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k3[None, :], other=0.0)
            b_v = tl.dot(b_w, b_h3.to(b_w.dtype), acc=b_v)
        if K > 192:
            p_w = w + o_t[:, None] * stride_w + o_k4[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k4[None, :], other=0.0)
            b_v = tl.dot(b_w, b_h4.to(b_w.dtype), acc=b_v)
        p_v = v + o_t[:, None] * stride_v + o_v[None, :]
        b_v = tl.load(p_v, mask=m_tv, other=0.0) - b_v

        if SAVE_NEW_VALUE:
            p_vn = v_new + o_t[:, None] * V + o_v[None, :]
            tl.store(p_vn, b_v.to(p_vn.dtype.element_ty), mask=m_tv)

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = g + bos * H + i_h + o_t * H
            b_g = tl.load(p_g, mask=m_t, other=0.0)
            b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
            b_g_last = exp(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last

        if USE_GK:
            b_gk_last1 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k1,
                mask=m_k1,
                other=0.0,
            )
            b_h1 *= exp(b_gk_last1)[:, None]
            if K > 64:
                b_gk_last2 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k2,
                    mask=m_k2,
                    other=0.0,
                )
                b_h2 *= exp(b_gk_last2)[:, None]
            if K > 128:
                b_gk_last3 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k3,
                    mask=m_k3,
                    other=0.0,
                )
                b_h3 *= exp(b_gk_last3)[:, None]
            if K > 192:
                b_gk_last4 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k4,
                    mask=m_k4,
                    other=0.0,
                )
                b_h4 *= exp(b_gk_last4)[:, None]
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

    if STORE_FINAL_STATE:
        p_ht = ht + o_k1[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), mask=m_h1)
        if K > 64:
            p_ht = ht + o_k2[:, None] * V + o_v[None, :]
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), mask=m_h2)
        if K > 128:
            p_ht = ht + o_k3[:, None] * V + o_v[None, :]
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), mask=m_h3)
        if K > 192:
            p_ht = ht + o_k4[:, None] * V + o_v[None, :]
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), mask=m_h4)


def chunk_gated_delta_rule_fwd_h_opt(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Optimized hidden state forward with Hg-aware k strides.

    w and u are expected in head-major contiguous layout [B, H, T, K] / [B, H, T, V].
    v_new output is [B, H, T_flat, V].
    """
    B, T, Hg, K = k.shape
    BT = chunk_size

    H = w.shape[1]
    V = u.shape[-1]
    T_flat = w.shape[2]

    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        N = len(cu_seqlens) - 1
        NT = len(chunk_indices)
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
    else:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None

    assert K <= 256, "current kernel does not support head dimension larger than 256."

    h = k.new_empty(B, NT, H, K, V)
    final_state = (
        k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    )
    v_new = k.new_empty(B, H, T_flat, V, dtype=u.dtype) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    chunk_gated_delta_rule_fwd_kernel_h_opt[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        T_flat=T_flat,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
    )
    return h, v_new, final_state


# =====================================================================
# opt_vk variant: h layout [V, K] (transposed from opt's [K, V])
# All other layouts (k, w, u, v_new) are identical to opt.
# =====================================================================


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_STATE_INDICES": lambda args: args["state_indices"] is not None,
    }
)
@triton.autotune(
    configs=gated_delta_rule_autotune_configs(
        [
            triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
            for num_warps in [2, 4]
            for num_stages in NUM_STAGES_FWD
            for BV in [16, 32, 64]
        ]
    ),
    key=["H", "K", "V", "BT", "IS_VARLEN"],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T", "T_flat"])
def chunk_gated_delta_rule_fwd_kernel_h_opt_vk(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    state_indices,
    cu_seqlens,
    chunk_offsets,
    T,
    T_flat,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_STATE_INDICES: tl.constexpr,
    USE_EXP2: tl.constexpr = False,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    # Indexed pool: each sequence's state slot is gathered from `state_indices`;
    # otherwise the slot is the dense sequence index (slot == i_n).
    if USE_STATE_INDICES:
        i_ss = tl.load(state_indices + i_n).to(tl.int32)
    else:
        i_ss = i_n
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BV, 64] — h in [V, K] layout (transposed from opt's [64, BV])
    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([BV, 64], dtype=tl.float32)

    h += (boh * H + i_h).to(tl.int64) * V * K
    k += (bos * Hg + i_h // (H // Hg)).to(tl.int64) * K
    if IS_VARLEN:
        v += (i_h * T_flat + bos).to(tl.int64) * V
        w += (i_h * T_flat + bos).to(tl.int64) * K
    else:
        v += ((i_n * H + i_h) * T_flat).to(tl.int64) * V
        w += ((i_n * H + i_h) * T_flat).to(tl.int64) * K
    stride_v = V
    stride_w = K
    if SAVE_NEW_VALUE:
        if IS_VARLEN:
            v_new += (i_h * T_flat + bos).to(tl.int64) * V
        else:
            v_new += ((i_n * H + i_h) * T_flat).to(tl.int64) * V
    stride_h = H * V * K
    stride_k = Hg * K
    # `i_ss * H + i_h` == `i_nh` on the dense path; the int64 cast happens before
    # the `V * K` scale so pool offsets never overflow int32.
    if USE_INITIAL_STATE:
        h0 = h0 + (i_ss * H + i_h).to(tl.int64) * V * K
    if STORE_FINAL_STATE:
        ht = ht + (i_ss * H + i_h).to(tl.int64) * V * K

    if USE_G:
        if IS_VARLEN:
            g += (i_h * T_flat + bos).to(tl.int64)
        else:
            g += ((i_n * H + i_h) * T_flat).to(tl.int64)

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
    m_h1 = m_v[:, None] & m_k1[None, :]
    m_h2 = m_v[:, None] & m_k2[None, :]
    m_h3 = m_v[:, None] & m_k3[None, :]
    m_h4 = m_v[:, None] & m_k4[None, :]

    if USE_INITIAL_STATE:
        p_h0_1 = h0 + o_v[:, None] * K + o_k1[None, :]
        b_h1 += tl.load(p_h0_1, mask=m_h1, other=0.0).to(tl.float32)
        if K > 64:
            p_h0_2 = h0 + o_v[:, None] * K + o_k2[None, :]
            b_h2 += tl.load(p_h0_2, mask=m_h2, other=0.0).to(tl.float32)
        if K > 128:
            p_h0_3 = h0 + o_v[:, None] * K + o_k3[None, :]
            b_h3 += tl.load(p_h0_3, mask=m_h3, other=0.0).to(tl.float32)
        if K > 192:
            p_h0_4 = h0 + o_v[:, None] * K + o_k4[None, :]
            b_h4 += tl.load(p_h0_4, mask=m_h4, other=0.0).to(tl.float32)

    for i_t in range(NT):
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        m_tv = m_t[:, None] & m_v[None, :]

        # Store h snapshot [V, K]
        h_t = h + i_t.to(tl.int64) * stride_h
        p_h1 = h_t + o_v[:, None] * K + o_k1[None, :]
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), mask=m_h1)
        if K > 64:
            p_h2 = h_t + o_v[:, None] * K + o_k2[None, :]
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), mask=m_h2)
        if K > 128:
            p_h3 = h_t + o_v[:, None] * K + o_k3[None, :]
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), mask=m_h3)
        if K > 192:
            p_h4 = h_t + o_v[:, None] * K + o_k4[None, :]
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), mask=m_h4)

        # b_v = u - w @ h^T  (h is [BV,64], need [64,BV] for dot with w[BT,64])
        p_w = w + o_t[:, None] * stride_w + o_k1[None, :]
        b_w = tl.load(p_w, mask=m_t[:, None] & m_k1[None, :], other=0.0)
        b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        if K > 64:
            p_w = w + o_t[:, None] * stride_w + o_k2[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k2[None, :], other=0.0)
            b_v = tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype), acc=b_v)
        if K > 128:
            p_w = w + o_t[:, None] * stride_w + o_k3[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k3[None, :], other=0.0)
            b_v = tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype), acc=b_v)
        if K > 192:
            p_w = w + o_t[:, None] * stride_w + o_k4[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k4[None, :], other=0.0)
            b_v = tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype), acc=b_v)
        p_v = v + o_t[:, None] * stride_v + o_v[None, :]
        b_v = tl.load(p_v, mask=m_tv, other=0.0) - b_v

        if SAVE_NEW_VALUE:
            p_vn = v_new + o_t[:, None] * V + o_v[None, :]
            tl.store(p_vn, b_v.to(p_vn.dtype.element_ty), mask=m_tv)

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + last_idx)
            p_g = g + o_t
            b_g = tl.load(p_g, mask=m_t, other=0.0)
            if USE_EXP2:
                b_v = b_v * tl.where(m_t, tl.math.exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = tl.math.exp2(b_g_last)
            else:
                b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
                b_g_last = exp(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last

        if USE_GK:
            b_gk_last1 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k1,
                mask=m_k1,
                other=0.0,
            )
            b_h1 *= (tl.math.exp2(b_gk_last1) if USE_EXP2 else exp(b_gk_last1))[None, :]
            if K > 64:
                b_gk_last2 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k2,
                    mask=m_k2,
                    other=0.0,
                )
                b_h2 *= (tl.math.exp2(b_gk_last2) if USE_EXP2 else exp(b_gk_last2))[
                    None, :
                ]
            if K > 128:
                b_gk_last3 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k3,
                    mask=m_k3,
                    other=0.0,
                )
                b_h3 *= (tl.math.exp2(b_gk_last3) if USE_EXP2 else exp(b_gk_last3))[
                    None, :
                ]
            if K > 192:
                b_gk_last4 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k4,
                    mask=m_k4,
                    other=0.0,
                )
                b_h4 *= (tl.math.exp2(b_gk_last4) if USE_EXP2 else exp(b_gk_last4))[
                    None, :
                ]
        b_v = b_v.to(k.dtype.element_ty)

        # h[V,K] += v_new^T @ k  →  [BV,64] += trans(dot(k[64,BT], v[BT,BV]))
        p_k = k + o_k1[:, None] + o_t[None, :] * stride_k
        b_k = tl.load(p_k, mask=m_k1[:, None] & m_t[None, :], other=0.0)
        b_h1 += tl.trans(tl.dot(b_k, b_v))
        if K > 64:
            p_k = k + o_k2[:, None] + o_t[None, :] * stride_k
            b_k = tl.load(p_k, mask=m_k2[:, None] & m_t[None, :], other=0.0)
            b_h2 += tl.trans(tl.dot(b_k, b_v))
        if K > 128:
            p_k = k + o_k3[:, None] + o_t[None, :] * stride_k
            b_k = tl.load(p_k, mask=m_k3[:, None] & m_t[None, :], other=0.0)
            b_h3 += tl.trans(tl.dot(b_k, b_v))
        if K > 192:
            p_k = k + o_k4[:, None] + o_t[None, :] * stride_k
            b_k = tl.load(p_k, mask=m_k4[:, None] & m_t[None, :], other=0.0)
            b_h4 += tl.trans(tl.dot(b_k, b_v))

    if STORE_FINAL_STATE:
        p_ht = ht + o_v[:, None] * K + o_k1[None, :]
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), mask=m_h1)
        if K > 64:
            p_ht = ht + o_v[:, None] * K + o_k2[None, :]
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), mask=m_h2)
        if K > 128:
            p_ht = ht + o_v[:, None] * K + o_k3[None, :]
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), mask=m_h3)
        if K > 192:
            p_ht = ht + o_v[:, None] * K + o_k4[None, :]
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), mask=m_h4)


def chunk_gated_delta_rule_fwd_h_opt_vk(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    use_exp2: bool = True,
    state_dtype: torch.dtype | None = None,
    num_decodes: int = 0,
    num_decode_tokens: int = 0,
    initial_state_indices: torch.Tensor | None = None,
    inplace_final_state: bool | None = None,
    prefill_metadata: GatedDeltaRulePrefillMetadata | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Optimized hidden state forward with h layout [V, K].

    w and u are expected in head-major contiguous layout [B, H, T, K] / [B, H, T, V].
    initial_state/final_state: [N, H, V, K].
    h snapshots: [B, NT, H, V, K].
    v_new output is [B, H, T_flat, V].
    `g` is expected in head-major layout [B, H, T].
    use_exp2 selects whether cumulative gates are interpreted in log2 space.
    state_dtype selects the initial/final hidden-state dtype (`fp32` or `bf16`);
    defaults to fp32. The kernel accumulates in fp32 and casts on store.
    num_decodes / num_decode_tokens skip a leading decode-only prefix in the
    ORIGINAL cu_seqlens (data tensors are expected pre-sliced); offsets are
    rebased internally via the cached prologue helpers so the chunk-index /
    offset build stays cache-warm across forward calls.

    State handling:
      * Dense (default): ``initial_state`` is ``[N, H, V, K]`` (slot == i_n) and
        ``final_state`` is a freshly allocated ``[N, H, V, K]`` tensor.
      * Indexed pool: pass ``initial_state`` as the pool ``[pool_size, H, V, K]``
        plus ``initial_state_indices`` ``[N]``; each sequence's slot is gathered
        from the index array.
      * ``inplace_final_state`` (default: ``True`` when ``initial_state_indices``
        is given) writes the final state back into ``initial_state`` in place and
        returns that same buffer as ``final_state`` (no extra allocation).
    """
    B, T, Hg, K = k.shape
    BT = chunk_size

    H = w.shape[1]
    V = u.shape[-1]
    T_flat = w.shape[2]

    if cu_seqlens is not None:
        if prefill_metadata is not None:
            prefill_metadata.validate(
                cu_seqlens=cu_seqlens,
                chunk_size=BT,
                num_decodes=num_decodes,
                num_decode_tokens=num_decode_tokens,
                total_prefill_tokens=T,
                num_sequences=len(cu_seqlens) - 1,
            )
            schedule = prefill_metadata.get_chunk_schedule(
                BT,
                num_decodes=num_decodes,
                num_decode_tokens=num_decode_tokens,
            )
            chunk_indices = schedule.sequence_ids
            chunk_offsets = schedule.chunk_offsets
            kernel_cu_seqlens = schedule.kernel_cu_seqlens
        else:
            # Pass the ORIGINAL (cache-stable) cu_seqlens + decode ints into the
            # cached prologue helpers so repeated calls avoid host synchronization.
            chunk_indices = prepare_chunk_indices(
                cu_seqlens, chunk_size, num_decodes, num_decode_tokens
            )
            chunk_offsets = prepare_chunk_offsets(
                cu_seqlens, BT, num_decodes, num_decode_tokens
            )
            kernel_cu_seqlens = prepare_rebased_cu_seqlens(
                cu_seqlens, num_decodes, num_decode_tokens
            )
        N = len(kernel_cu_seqlens) - 1
        NT = (
            schedule.total_chunks
            if prefill_metadata is not None
            else len(chunk_indices)
        )
    else:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
        kernel_cu_seqlens = None

    assert K <= 256, "current kernel does not support head dimension larger than 256."

    if state_dtype is not None and state_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"`state_dtype` must be fp32 or bf16, got {state_dtype}.")
    _state_dtype = state_dtype if state_dtype is not None else torch.float32
    if (
        state_dtype is not None
        and initial_state is not None
        and initial_state.dtype != _state_dtype
    ):
        raise ValueError(
            f"`initial_state.dtype` ({initial_state.dtype}) must match "
            f"`state_dtype` ({_state_dtype})."
        )

    has_indices = initial_state_indices is not None
    inplace = has_indices if inplace_final_state is None else inplace_final_state
    if inplace and initial_state is None:
        raise ValueError("`inplace_final_state` requires `initial_state`.")
    # Indexed slots address the shared pool, so the final state must be written
    # back into that pool in place; a dense `[N, ...]` buffer cannot hold them.
    if has_indices and not inplace:
        raise ValueError(
            "`initial_state_indices` requires in-place update; "
            "leave `inplace_final_state` unset or set it to True."
        )
    if inplace and not initial_state.is_contiguous():
        raise ValueError("`initial_state` must be contiguous for in-place update.")
    state_indices = (
        initial_state_indices.to(torch.int32).contiguous() if has_indices else None
    )

    if gk is not None:
        gk = gk.contiguous()
        if use_exp2:
            # gk is expressed in natural-log space, so pre-scale it for exp2 kernels.
            gk = gk * RCP_LN2

    h = k.new_empty(B, NT, H, V, K)
    if not output_final_state:
        final_state = None
    elif inplace:
        # Alias the caller's pool: the kernel loads h0 fully before storing ht,
        # so writing the final state back into `initial_state` is safe.
        final_state = initial_state
    else:
        final_state = k.new_empty(N, H, V, K, dtype=_state_dtype)
    v_new = k.new_empty(B, H, T_flat, V, dtype=u.dtype) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    chunk_gated_delta_rule_fwd_kernel_h_opt_vk[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final_state,
        state_indices=state_indices,
        cu_seqlens=kernel_cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        T_flat=T_flat,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        USE_EXP2=use_exp2,
    )
    return h, v_new, final_state


def chunk_gated_delta_rule_bwd_dhu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *q.shape, do.shape[-1]
    # N: the actual number of sequences in the batch with either equal or variable lengths
    BT = 64
    assert (
        K <= 256
    ), "current kernel does not support head dimension being larger than 256."

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = (
            len(cu_seqlens) - 1,
            len(chunk_indices),
            prepare_chunk_offsets(cu_seqlens, BT),
        )

    dh = q.new_empty(B, NT, H, K, V)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.empty_like(dv)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64[grid](
        q=q,
        k=k,
        w=w,
        g=g,
        gk=gk,
        dht=dht,
        dh0=dh0,
        do=do,
        dh=dh,
        dv=dv,
        dv2=dv2,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
    )
    return dh, dh0, dv2
