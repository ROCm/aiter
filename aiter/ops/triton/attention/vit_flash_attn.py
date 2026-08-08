# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Head-dim-tiled Triton flash attention for Vision Transformer encoders.

Non-causal, per-image variable-length (varlen) attention for ViT towers whose
head_dim is small and not a power of two (e.g. Qwen3.x ViT: head_dim=72). Stock
flash kernels pad head_dim up to the next power of two (128), wasting ~1.6-1.8x
of the QK/AV contraction. This kernel instead tiles head_dim into 5x16=80 lanes
(the WMMA k-tile is 16), so the contraction is only 80-deep with masked loads for
d >= head_dim. Online-softmax, one workgroup per (image, head, query-block).

Constraint: head_dim <= 80 (5 tiles of 16). Portable Triton; validated on
gfx1151 (RDNA3.5), where torch SDPA falls back to the unfused math backend.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _vit_attn_varlen(
    Q,
    K,
    V,
    Out,
    B_Start,
    B_Seqlen,
    sm_scale,
    sq_n,
    sq_h,
    sk_n,
    sk_h,
    sv_n,
    sv_h,
    so_n,
    so_h,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    seqlen = tl.load(B_Seqlen + cur_b)
    start = tl.load(B_Start + cur_b)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_t = tl.arange(0, 16)
    m_mask = offs_m < seqlen
    qb = Q + (start + offs_m)[:, None] * sq_n + pid_h * sq_h
    d0 = 0 * 16 + offs_t
    q0 = tl.load(
        qb + d0[None, :], mask=m_mask[:, None] & (d0[None, :] < HEAD_DIM), other=0.0
    )
    a0 = tl.zeros([BLOCK_M, 16], tl.float32)
    d1 = 1 * 16 + offs_t
    q1 = tl.load(
        qb + d1[None, :], mask=m_mask[:, None] & (d1[None, :] < HEAD_DIM), other=0.0
    )
    a1 = tl.zeros([BLOCK_M, 16], tl.float32)
    d2 = 2 * 16 + offs_t
    q2 = tl.load(
        qb + d2[None, :], mask=m_mask[:, None] & (d2[None, :] < HEAD_DIM), other=0.0
    )
    a2 = tl.zeros([BLOCK_M, 16], tl.float32)
    d3 = 3 * 16 + offs_t
    q3 = tl.load(
        qb + d3[None, :], mask=m_mask[:, None] & (d3[None, :] < HEAD_DIM), other=0.0
    )
    a3 = tl.zeros([BLOCK_M, 16], tl.float32)
    d4 = 4 * 16 + offs_t
    q4 = tl.load(
        qb + d4[None, :], mask=m_mask[:, None] & (d4[None, :] < HEAD_DIM), other=0.0
    )
    a4 = tl.zeros([BLOCK_M, 16], tl.float32)
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    for n0 in range(0, seqlen, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        nmask = offs_n < seqlen
        kb = K + (start + offs_n)[None, :] * sk_n + pid_h * sk_h
        vb = V + (start + offs_n)[:, None] * sv_n + pid_h * sv_h
        qk = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
        k0 = tl.load(
            kb + d0[:, None], mask=nmask[None, :] & (d0[:, None] < HEAD_DIM), other=0.0
        )
        qk += tl.dot(q0, k0)
        k1 = tl.load(
            kb + d1[:, None], mask=nmask[None, :] & (d1[:, None] < HEAD_DIM), other=0.0
        )
        qk += tl.dot(q1, k1)
        k2 = tl.load(
            kb + d2[:, None], mask=nmask[None, :] & (d2[:, None] < HEAD_DIM), other=0.0
        )
        qk += tl.dot(q2, k2)
        k3 = tl.load(
            kb + d3[:, None], mask=nmask[None, :] & (d3[:, None] < HEAD_DIM), other=0.0
        )
        qk += tl.dot(q3, k3)
        k4 = tl.load(
            kb + d4[:, None], mask=nmask[None, :] & (d4[:, None] < HEAD_DIM), other=0.0
        )
        qk += tl.dot(q4, k4)
        qk = qk * sm_scale + tl.where(nmask[None, :], 0.0, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        p = p.to(V.dtype.element_ty)
        v0 = tl.load(
            vb + d0[None, :], mask=nmask[:, None] & (d0[None, :] < HEAD_DIM), other=0.0
        )
        a0 = a0 * alpha[:, None] + tl.dot(p, v0)
        v1 = tl.load(
            vb + d1[None, :], mask=nmask[:, None] & (d1[None, :] < HEAD_DIM), other=0.0
        )
        a1 = a1 * alpha[:, None] + tl.dot(p, v1)
        v2 = tl.load(
            vb + d2[None, :], mask=nmask[:, None] & (d2[None, :] < HEAD_DIM), other=0.0
        )
        a2 = a2 * alpha[:, None] + tl.dot(p, v2)
        v3 = tl.load(
            vb + d3[None, :], mask=nmask[:, None] & (d3[None, :] < HEAD_DIM), other=0.0
        )
        a3 = a3 * alpha[:, None] + tl.dot(p, v3)
        v4 = tl.load(
            vb + d4[None, :], mask=nmask[:, None] & (d4[None, :] < HEAD_DIM), other=0.0
        )
        a4 = a4 * alpha[:, None] + tl.dot(p, v4)
        m_i = m_new
    ob = Out + (start + offs_m)[:, None] * so_n + pid_h * so_h
    tl.store(
        ob + d0[None, :],
        (a0 / l_i[:, None]).to(Out.dtype.element_ty),
        mask=m_mask[:, None] & (d0[None, :] < HEAD_DIM),
    )
    tl.store(
        ob + d1[None, :],
        (a1 / l_i[:, None]).to(Out.dtype.element_ty),
        mask=m_mask[:, None] & (d1[None, :] < HEAD_DIM),
    )
    tl.store(
        ob + d2[None, :],
        (a2 / l_i[:, None]).to(Out.dtype.element_ty),
        mask=m_mask[:, None] & (d2[None, :] < HEAD_DIM),
    )
    tl.store(
        ob + d3[None, :],
        (a3 / l_i[:, None]).to(Out.dtype.element_ty),
        mask=m_mask[:, None] & (d3[None, :] < HEAD_DIM),
    )
    tl.store(
        ob + d4[None, :],
        (a4 / l_i[:, None]).to(Out.dtype.element_ty),
        mask=m_mask[:, None] & (d4[None, :] < HEAD_DIM),
    )


def vit_flash_attn(
    q, k, v, b_start_loc, b_seq_len, max_seqlen, BLOCK_M=128, BLOCK_N=32
):
    # q,k,v: (total_tokens, num_heads, head_dim) contiguous. Per-image varlen via cu_seqlens.
    N, H, D = q.shape
    assert D <= 80, "this kernel tiles head_dim into 5x16=80"
    o = torch.empty_like(q)
    batch = b_seq_len.shape[0]
    grid = (batch, H, triton.cdiv(max_seqlen, BLOCK_M))
    _vit_attn_varlen[grid](
        q,
        k,
        v,
        o,
        b_start_loc,
        b_seq_len,
        1.0 / math.sqrt(D),
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return o
