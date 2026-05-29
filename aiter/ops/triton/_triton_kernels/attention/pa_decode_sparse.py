# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sparse paged-decode attention kernels (split-K + per-token paged indices).

Two-kernel decomposition of a flash-decode whose K range for each token is a
gathered subset of a unified KV pool:

  ``_pa_decode_sparse``        : split-K main kernel. Grid (T, ceil(H/BLOCK_H),
                                 KV_SPLITS). Each program owns one token, one
                                 head-block, and one slice of the token's
                                 sparse K range; writes pre-sink
                                 (m, l, acc) partials.
  ``_pa_decode_sparse_reduce`` : combines KV_SPLITS partials per (token, head)
                                 via log-sum-exp, folds in the per-head
                                 ``attn_sink`` as a virtual K, writes the
                                 final output.

Both kernels follow the ``tiles_per_segment`` pattern from aiter's
``kernel_unified_attention_3d``: the split-axis grid dim ``KV_SPLITS`` is
constexpr; ``tiles_per_segment`` is computed at runtime per token; trailing
segments past the end exit early; the reduce derives ``act_num_segments``
from ``kv_indptr`` and masks the stale partial-buffer slots out of its load.

Caller contract:
  unified_kv:       [total_pages, D] bf16/fp16  (page_size = 1)
  kv_indices:       [total_indices] int32 — per-token slot lists, flat. ``-1``
                    entries are skipped (sentinel for unused tail).
  kv_indptr:        [N+1] int32 — true prefix sum (variable per-token len).
  attn_sink:        [H] fp32 per-head learnable softmax-denom bias.
"""

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_pa_decode_sparse_repr = make_kernel_repr(
    "_pa_decode_sparse",
    [
        "BLOCK_H",
        "BLOCK_D",
        "BLOCK_K",
        "H",
        "D",
        "KV_SPLITS",
    ],
)


@triton.jit(repr=_pa_decode_sparse_repr)
def _pa_decode_sparse(
    q_ptr,  # [N, H, D]
    unified_kv_ptr,  # [total_pages, D]
    kv_indices_ptr,  # [total_indices] int32
    kv_indptr_ptr,  # [N+1] int32
    m_partial_ptr,  # [N, KV_SPLITS, H_padded] fp32
    l_partial_ptr,  # [N, KV_SPLITS, H_padded] fp32
    acc_partial_ptr,  # [N, KV_SPLITS, H_padded, D] fp32
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    kv_stride_n: tl.constexpr,
    kv_stride_d: tl.constexpr,
    mp_stride_t: tl.constexpr,
    mp_stride_k: tl.constexpr,
    mp_stride_h: tl.constexpr,
    lp_stride_t: tl.constexpr,
    lp_stride_k: tl.constexpr,
    lp_stride_h: tl.constexpr,
    ap_stride_t: tl.constexpr,
    ap_stride_k: tl.constexpr,
    ap_stride_h: tl.constexpr,
    ap_stride_d: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    softmax_scale: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """3D split-K sparse paged-decode. Grid: (N, ceil(H/BLOCK_H), KV_SPLITS).

    Each program owns one token, one head-block, and one slice of the token's
    sparse K range. The attn_sink fold-in lives in the reduce kernel — splits
    only emit (m_i, l_i, acc) in pre-sink form. ``BLOCK_H`` is widened so a
    single head-block program can cover many heads, killing the MLA-style KV
    re-fetch across head-block programs.
    """
    t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_k = tl.program_id(2)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    d_offs = tl.arange(0, BLOCK_D)
    h_mask = h_offs < H
    d_mask = d_offs < D

    q = tl.load(
        q_ptr
        + t * q_stride_t
        + h_offs[:, None] * q_stride_h
        + d_offs[None, :] * q_stride_d,
        mask=h_mask[:, None] & d_mask[None, :],
        other=0.0,
    )

    kv_start = tl.load(kv_indptr_ptr + t)
    kv_end = tl.load(kv_indptr_ptr + t + 1)
    kv_len = kv_end - kv_start

    # aiter unified_attention 3d pattern: fixed grid axis (KV_SPLITS), runtime
    # tiles_per_segment, early-return for trailing segments past the end.
    tiles_per_segment = tl.cdiv(kv_len, KV_SPLITS * BLOCK_K)
    if pid_k * tiles_per_segment * BLOCK_K >= kv_len:
        return

    num_tiles = tl.cdiv(kv_len, BLOCK_K)
    tile_start = pid_k * tiles_per_segment
    tile_end = tl.minimum((pid_k + 1) * tiles_per_segment, num_tiles)

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    k_offs = tl.arange(0, BLOCK_K)
    for j in tl.range(tile_start, tile_end, num_stages=2):
        k_start = j * BLOCK_K
        k_pos = k_start + k_offs
        in_range = k_pos < kv_len
        slot = tl.load(
            kv_indices_ptr + kv_start + k_pos,
            mask=in_range,
            other=-1,
        )
        valid = in_range & (slot >= 0)

        kv = tl.load(
            unified_kv_ptr
            + slot[:, None] * kv_stride_n
            + d_offs[None, :] * kv_stride_d,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(kv)) * softmax_scale
        scores = tl.where(h_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(h_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new

    # Emit partials. The reduce reads (m, l, acc) per split and folds in the
    # sink there, so we do *not* touch attn_sink here.
    m_base = t * mp_stride_t + pid_k * mp_stride_k
    tl.store(
        m_partial_ptr + m_base + h_offs * mp_stride_h,
        m_i,
        mask=h_mask,
    )
    l_base = t * lp_stride_t + pid_k * lp_stride_k
    tl.store(
        l_partial_ptr + l_base + h_offs * lp_stride_h,
        l_i,
        mask=h_mask,
    )
    a_base = t * ap_stride_t + pid_k * ap_stride_k
    tl.store(
        acc_partial_ptr
        + a_base
        + h_offs[:, None] * ap_stride_h
        + d_offs[None, :] * ap_stride_d,
        acc,
        mask=h_mask[:, None] & d_mask[None, :],
    )


_pa_decode_sparse_reduce_repr = make_kernel_repr(
    "_pa_decode_sparse_reduce",
    [
        "BLOCK_H",
        "BLOCK_D",
        "BLOCK_K",
        "H",
        "D",
        "KV_SPLITS",
    ],
)


@triton.jit(repr=_pa_decode_sparse_reduce_repr)
def _pa_decode_sparse_reduce(
    m_partial_ptr,  # [N, KV_SPLITS, H_padded] fp32
    l_partial_ptr,  # [N, KV_SPLITS, H_padded] fp32
    acc_partial_ptr,  # [N, KV_SPLITS, H_padded, D] fp32
    attn_sink_ptr,  # [H]
    kv_indptr_ptr,  # [N+1] int32 — used to derive per-token kv_len
    out_ptr,  # [N, H, D]
    mp_stride_t: tl.constexpr,
    mp_stride_k: tl.constexpr,
    mp_stride_h: tl.constexpr,
    lp_stride_t: tl.constexpr,
    lp_stride_k: tl.constexpr,
    lp_stride_h: tl.constexpr,
    ap_stride_t: tl.constexpr,
    ap_stride_k: tl.constexpr,
    ap_stride_h: tl.constexpr,
    ap_stride_d: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Combine KV_SPLITS partials, fold in attn_sink, write final output.

    The split kernel uses aiter's tiles_per_segment pattern and early-returns
    for trailing splits whose tile range is past kv_len. Their slot of the
    partial buffer is therefore stale (torch.empty contents). We derive
    ``act_num_segments = cdiv(kv_len, tiles_per_segment * BLOCK_K)`` and mask
    those stale slots out of the load."""
    t = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    d_offs = tl.arange(0, BLOCK_D)
    k_offs = tl.arange(0, KV_SPLITS)
    h_mask = h_offs < H
    d_mask = d_offs < D

    kv_start = tl.load(kv_indptr_ptr + t)
    kv_end = tl.load(kv_indptr_ptr + t + 1)
    kv_len = kv_end - kv_start
    tiles_per_segment = tl.cdiv(kv_len, KV_SPLITS * BLOCK_K)
    # Only the first ``act_num_segments`` slots of the partial buffer were
    # actually written by the split kernel; the rest are stale.
    act_num_segments = tl.cdiv(kv_len, tiles_per_segment * BLOCK_K)
    segm_mask = k_offs < act_num_segments

    neg_large = -3.4028234663852886e38

    m_p = tl.load(
        m_partial_ptr
        + t * mp_stride_t
        + k_offs[:, None] * mp_stride_k
        + h_offs[None, :] * mp_stride_h,
        mask=segm_mask[:, None] & h_mask[None, :],
        other=neg_large,
    )  # [KV_SPLITS, BLOCK_H]
    l_p = tl.load(
        l_partial_ptr
        + t * lp_stride_t
        + k_offs[:, None] * lp_stride_k
        + h_offs[None, :] * lp_stride_h,
        mask=segm_mask[:, None] & h_mask[None, :],
        other=0.0,
    )  # [KV_SPLITS, BLOCK_H]
    a_p = tl.load(
        acc_partial_ptr
        + t * ap_stride_t
        + k_offs[:, None, None] * ap_stride_k
        + h_offs[None, :, None] * ap_stride_h
        + d_offs[None, None, :] * ap_stride_d,
        mask=segm_mask[:, None, None] & h_mask[None, :, None] & d_mask[None, None, :],
        other=0.0,
    )  # [KV_SPLITS, BLOCK_H, BLOCK_D]

    # Pre-sink combine across splits.
    m_max = tl.max(m_p, axis=0)  # [BLOCK_H]
    alpha_split = tl.exp(m_p - m_max[None, :])  # [KV_SPLITS, BLOCK_H]
    l_combined = tl.sum(l_p * alpha_split, axis=0)  # [BLOCK_H]
    acc_combined = tl.sum(a_p * alpha_split[:, :, None], axis=0)  # [BLOCK_H, BLOCK_D]

    # Fold attn_sink as a virtual K of weight 1.
    sink = tl.load(attn_sink_ptr + h_offs, mask=h_mask, other=neg_large).to(tl.float32)
    m_final = tl.maximum(m_max, sink)
    alpha_kv = tl.exp(m_max - m_final)
    alpha_sink = tl.exp(sink - m_final)
    l_final = l_combined * alpha_kv + alpha_sink
    acc_final = acc_combined * alpha_kv[:, None]

    denom = tl.maximum(l_final, 1.0e-30)
    out = tl.where(l_final[:, None] > 0.0, acc_final / denom[:, None], 0.0)
    tl.store(
        out_ptr
        + t * out_stride_t
        + h_offs[:, None] * out_stride_h
        + d_offs[None, :] * out_stride_d,
        out.to(out_ptr.dtype.element_ty),
        mask=h_mask[:, None] & d_mask[None, :],
    )
