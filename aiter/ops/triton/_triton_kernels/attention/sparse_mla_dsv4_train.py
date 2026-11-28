# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# DSV4 sparse-MLA training kernels (forward with LSE + backward).
# Optimised for MI308X (gfx942, 64 KB LDS).
#
# Forward: online-softmax sparse attention with autotuning.
# Backward, for each top-k chunk: dQ (stores dS and P) → dKV-interm. The wrapper
# reduces dKV-interm into dKV with the CSR gather of sparse_attention_dsv4_bwd.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_FWD_FALLBACK = triton.Config({"BLOCK_H": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2)

# =====================================================================
# Forward kernel — sparse MLA with LSE output
# =====================================================================


_sparse_mla_fwd_kernel_repr = make_kernel_repr(
    "_sparse_mla_fwd_kernel",
    ["HAS_ATTN_SINK", "BLOCK_D", "BLOCK_H", "BLOCK_K"],
)


@triton.jit(repr=_sparse_mla_fwd_kernel_repr)
def _sparse_mla_fwd_kernel(
    q_ptr,
    kv_ptr,
    indices_ptr,
    attn_sink_ptr,
    out_ptr,
    lse_ptr,
    q_stride_t: tl.int64,
    q_stride_h: tl.int64,
    kv_stride_n: tl.int64,
    out_stride_t: tl.int64,
    out_stride_h: tl.int64,
    lse_stride_t: tl.int64,
    idx_stride_t: tl.int64,
    num_heads: tl.int32,
    head_dim: tl.int32,
    num_kv: tl.int32,
    topk: tl.int32,
    scale,
    HAS_ATTN_SINK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query_idx = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < head_dim

    q = tl.load(
        q_ptr
        + query_idx * q_stride_t
        + head_offsets[:, None] * q_stride_h
        + dim_offsets[None, :],
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    m_i = tl.full((BLOCK_H,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    k_offsets = tl.arange(0, BLOCK_K)

    for k_start in tl.range(0, topk, BLOCK_K):
        k_pos = k_start + k_offsets
        in_range = k_pos < topk
        slot = tl.load(
            indices_ptr + query_idx * idx_stride_t + k_pos, mask=in_range, other=-1
        ).to(tl.int64)
        valid = in_range & (slot >= 0) & (slot < num_kv)

        kv = tl.load(
            kv_ptr + slot[:, None] * kv_stride_n + dim_offsets[None, :],
            mask=valid[:, None] & dim_mask[None, :],
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(kv)) * scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, float("-inf"))

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.where(m_new == float("-inf"), 0.0, tl.exp(m_i - m_new))
        p = tl.where(
            m_new[:, None] == float("-inf"), 0.0, tl.exp(scores - m_new[:, None])
        )
        p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new

    if HAS_ATTN_SINK:
        sink = tl.load(
            attn_sink_ptr + head_offsets, mask=head_mask, other=float("-inf")
        ).to(tl.float32)
        m_final = tl.maximum(m_i, sink)
        alpha = tl.where(m_final == float("-inf"), 0.0, tl.exp(m_i - m_final))
        exp_sink = tl.where(sink == float("-inf"), 0.0, tl.exp(sink - m_final))
        l_final = l_i * alpha + exp_sink
        denom = tl.maximum(l_final, 1e-30)
        out = tl.where(
            l_final[:, None] > 0.0, (acc * alpha[:, None]) / denom[:, None], 0.0
        )
        lse = tl.where(l_final > 0.0, tl.log(l_final) + m_final, float("-inf"))
    else:
        denom = tl.maximum(l_i, 1e-30)
        out = tl.where(l_i[:, None] > 0.0, acc / denom[:, None], 0.0)
        lse = tl.where(l_i > 0.0, tl.log(l_i) + m_i, float("-inf"))

    tl.store(
        out_ptr
        + query_idx * out_stride_t
        + head_offsets[:, None] * out_stride_h
        + dim_offsets[None, :],
        out,
        mask=head_mask[:, None] & dim_mask[None, :],
    )
    tl.store(lse_ptr + query_idx * lse_stride_t + head_offsets, lse, mask=head_mask)


# =====================================================================
# Backward — Kernel 1: dQ for one top-k chunk, store dS and P
# =====================================================================
# Grid: (N, num_hg). dQ is fp32 and accumulates over the chunks.


_bwd_dq_kernel_repr = make_kernel_repr(
    "_bwd_dq_kernel",
    ["R_CHUNK", "IS_FIRST_CHUNK", "BLOCK_H", "BLOCK_D", "BLOCK_K"],
)


@triton.jit(repr=_bwd_dq_kernel_repr)
def _bwd_dq_kernel(
    q_ptr,
    kv_ptr,
    do_ptr,
    indices_ptr,
    lse_ptr,
    delta_ptr,
    dq_ptr,
    ds_ptr,
    p_ptr,
    q_stride_t: tl.int64,
    q_stride_h: tl.int64,
    kv_stride_n: tl.int64,
    do_stride_t: tl.int64,
    do_stride_h: tl.int64,
    idx_stride_t: tl.int64,
    lse_stride_t: tl.int64,
    delta_stride_t: tl.int64,
    dq_stride_t: tl.int64,
    dq_stride_h: tl.int64,
    ds_stride_t: tl.int64,
    ds_stride_h: tl.int64,
    num_heads: tl.int32,
    num_kv: tl.int32,
    r_start: tl.int32,
    scale,
    R_CHUNK: tl.constexpr,
    IS_FIRST_CHUNK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tok_idx = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1)

    # dQ is kept in two halves of D. A full fp32 [BLOCK_H, BLOCK_D] tile read back from
    # memory changes layout through LDS: 128 KB at BLOCK_H 64, D 512, but gfx942 has 64 KB.
    HALF_D: tl.constexpr = BLOCK_D // 2
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    d_lo = tl.arange(0, HALF_D)
    d_hi = HALF_D + tl.arange(0, HALF_D)
    h_mask = h_offsets < num_heads

    q_rows = q_ptr + tok_idx * q_stride_t + h_offsets[:, None] * q_stride_h
    q_lo = tl.load(q_rows + d_lo[None, :], mask=h_mask[:, None], other=0.0)
    q_hi = tl.load(q_rows + d_hi[None, :], mask=h_mask[:, None], other=0.0)
    do_rows = do_ptr + tok_idx * do_stride_t + h_offsets[:, None] * do_stride_h
    do_lo = tl.load(do_rows + d_lo[None, :], mask=h_mask[:, None], other=0.0)
    do_hi = tl.load(do_rows + d_hi[None, :], mask=h_mask[:, None], other=0.0)
    lse = tl.load(lse_ptr + tok_idx * lse_stride_t + h_offsets, mask=h_mask, other=0.0)
    delta = tl.load(
        delta_ptr + tok_idx * delta_stride_t + h_offsets, mask=h_mask, other=0.0
    )

    dq_rows = dq_ptr + tok_idx * dq_stride_t + h_offsets[:, None] * dq_stride_h
    if IS_FIRST_CHUNK:
        acc_lo = tl.zeros((BLOCK_H, HALF_D), dtype=tl.float32)
        acc_hi = tl.zeros((BLOCK_H, HALF_D), dtype=tl.float32)
    else:
        acc_lo = tl.load(dq_rows + d_lo[None, :], mask=h_mask[:, None], other=0.0)
        acc_hi = tl.load(dq_rows + d_hi[None, :], mask=h_mask[:, None], other=0.0)

    k_offsets = tl.arange(0, BLOCK_K)
    NUM_TILES: tl.constexpr = (R_CHUNK + BLOCK_K - 1) // BLOCK_K
    for tile in range(NUM_TILES):
        k_local = tile * BLOCK_K + k_offsets
        in_chunk = k_local < R_CHUNK
        slot = tl.load(
            indices_ptr + tok_idx * idx_stride_t + r_start + k_local,
            mask=in_chunk,
            other=-1,
        ).to(tl.int64)
        valid = (slot >= 0) & (slot < num_kv)

        # [HALF_D, BLOCK_K]: K for the scores, and its transpose is V for dQ (K == V).
        kv_cols = kv_ptr + slot[None, :] * kv_stride_n
        kv_lo = tl.load(kv_cols + d_lo[:, None], mask=valid[None, :], other=0.0)
        kv_hi = tl.load(kv_cols + d_hi[:, None], mask=valid[None, :], other=0.0)

        hk_mask = h_mask[:, None] & valid[None, :]
        scores = tl.dot(q_hi, kv_hi, tl.dot(q_lo, kv_lo)) * scale
        p = tl.where(hk_mask, tl.exp(scores - lse[:, None]), 0.0)
        dp = tl.dot(do_hi, kv_hi, tl.dot(do_lo, kv_lo))
        ds = tl.where(hk_mask, p * (dp - delta[:, None]) * scale, 0.0)
        ds_in = ds.to(kv_lo.dtype)
        acc_lo = tl.dot(ds_in, tl.trans(kv_lo), acc_lo)
        acc_hi = tl.dot(ds_in, tl.trans(kv_hi), acc_hi)

        # Invalid slots store zeros, so the next kernel reads defined values.
        out_offsets = (
            tok_idx * ds_stride_t + h_offsets[:, None] * ds_stride_h + k_local[None, :]
        )
        out_mask = h_mask[:, None] & in_chunk[None, :]
        tl.store(ds_ptr + out_offsets, ds.to(ds_ptr.dtype.element_ty), mask=out_mask)
        tl.store(p_ptr + out_offsets, p.to(p_ptr.dtype.element_ty), mask=out_mask)

    tl.store(dq_rows + d_lo[None, :], acc_lo, mask=h_mask[:, None])
    tl.store(dq_rows + d_hi[None, :], acc_hi, mask=h_mask[:, None])


# =====================================================================
# Backward — Kernel 2: dKV intermediate for one top-k chunk
# =====================================================================
# Grid: (N,). interm[t, k, :] = sum_h (Q^T dS + dO^T P): K == V, so one gradient
# per selected slot. The CSR gather in sparse_attention_dsv4_bwd reduces it into dKV.


_bwd_dkv_interm_kernel_repr = make_kernel_repr(
    "_bwd_dkv_interm_kernel",
    ["R_CHUNK", "BLOCK_H", "BLOCK_D", "BLOCK_K", "NUM_HG"],
)


@triton.jit(repr=_bwd_dkv_interm_kernel_repr)
def _bwd_dkv_interm_kernel(
    q_t_ptr,
    do_t_ptr,
    ds_ptr,
    p_ptr,
    interm_ptr,
    qt_stride_t: tl.int64,
    qt_stride_d: tl.int64,
    dot_stride_t: tl.int64,
    dot_stride_d: tl.int64,
    ds_stride_t: tl.int64,
    ds_stride_h: tl.int64,
    interm_stride_t: tl.int64,
    interm_stride_k: tl.int64,
    num_heads: tl.int32,
    R_CHUNK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_HG: tl.constexpr,
):
    tok_idx = tl.program_id(0).to(tl.int64)
    d_offsets = tl.arange(0, BLOCK_D)
    k_offsets = tl.arange(0, BLOCK_K)

    NUM_TILES: tl.constexpr = (R_CHUNK + BLOCK_K - 1) // BLOCK_K
    for tile in range(NUM_TILES):
        k_local = tile * BLOCK_K + k_offsets
        k_mask = k_local < R_CHUNK
        acc = tl.zeros((BLOCK_D, BLOCK_K), dtype=tl.float32)

        for hg in range(NUM_HG):
            h_offsets = hg * BLOCK_H + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < num_heads
            q_t = tl.load(
                q_t_ptr
                + tok_idx * qt_stride_t
                + d_offsets[:, None] * qt_stride_d
                + h_offsets[None, :],
                mask=h_mask[None, :],
                other=0.0,
            )
            do_t = tl.load(
                do_t_ptr
                + tok_idx * dot_stride_t
                + d_offsets[:, None] * dot_stride_d
                + h_offsets[None, :],
                mask=h_mask[None, :],
                other=0.0,
            )
            hk_offsets = (
                tok_idx * ds_stride_t
                + h_offsets[:, None] * ds_stride_h
                + k_local[None, :]
            )
            hk_mask = h_mask[:, None] & k_mask[None, :]
            ds = tl.load(ds_ptr + hk_offsets, mask=hk_mask, other=0.0)
            p = tl.load(p_ptr + hk_offsets, mask=hk_mask, other=0.0)
            acc += tl.dot(q_t, ds.to(q_t.dtype))
            acc += tl.dot(do_t, p.to(do_t.dtype))

        tl.store(
            interm_ptr
            + tok_idx * interm_stride_t
            + k_local[None, :] * interm_stride_k
            + d_offsets[:, None],
            acc.to(interm_ptr.dtype.element_ty),
            mask=k_mask[None, :],
        )
