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

A second front-end kernel, ``_pa_decode_sparse_shuffled``, keeps that split-K
skeleton but reads a *paged* SHUFFLE K/V cache pair addressed by a per-token
block table instead of a page_size=1 unified pool (MiniMax-M3 block-sparse
decode). It shares ``_pa_decode_sparse_reduce`` via ``USE_CTX_LENS``.

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
    unified_kv_ptr,  # [total_pages, D] bf16/fp16, or fp8 when QUANT_KV
    kv_scales_ptr,  # [total_pages, NUM_GROUPS] fp32 when QUANT_KV (dummy otherwise)
    kv_indices_ptr,  # [total_indices] int32
    kv_indptr_ptr,  # [N+1] int32
    m_partial_ptr,  # [N, KV_SPLITS, H_padded] fp32 (unused when KV_SPLITS==1)
    l_partial_ptr,  # [N, KV_SPLITS, H_padded] fp32 (unused when KV_SPLITS==1)
    acc_partial_ptr,  # [N, KV_SPLITS, H_padded, D] fp32 (unused when KV_SPLITS==1)
    attn_sink_ptr,  # [H] (only used when KV_SPLITS==1)
    out_ptr,  # [N, H, D] (only used when KV_SPLITS==1)
    total_pages,
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    kv_stride_n: tl.constexpr,
    kv_stride_d: tl.constexpr,
    ks_stride_n: tl.constexpr,
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
    softmax_scale: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_INVALID: tl.constexpr,
    QUANT_KV: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    USE_EXP2: tl.constexpr,
    num_warps: tl.constexpr,
):
    """3D split-K sparse paged-decode. Grid: (N, ceil(H/BLOCK_H), KV_SPLITS).

    Each program owns one token, one head-block, and one slice of the token's
    sparse K range. ``BLOCK_H`` is widened so a single head-block program can
    cover many heads, killing the MLA-style KV re-fetch across head-block
    programs.

    When ``KV_SPLITS > 1``: only emits pre-sink (m_i, l_i, acc) partials; the
    reduce kernel folds in ``attn_sink`` and normalises.

    When ``KV_SPLITS == 1``: the whole softmax happens in one CTA, so we fold
    the sink in as the initial running max (virtual K of weight 1) and divide
    by L inline before writing the final result to ``out``. No partial
    buffers, no reduce.
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
    # When USE_EXP2, fold log2(e) into q so QK scores land in the base-2
    # domain and per-element softmax uses the bare exp2 HW instruction.
    LOG2E = 1.4426950408889634
    qk_scale = softmax_scale * LOG2E if USE_EXP2 else softmax_scale
    q = (q.to(tl.float32) * qk_scale).to(q_ptr.dtype.element_ty)

    kv_start = tl.load(kv_indptr_ptr + t)
    kv_end = tl.load(kv_indptr_ptr + t + 1)
    kv_len = kv_end - kv_start

    # aiter unified_attention 3d pattern: fixed grid axis (KV_SPLITS), runtime
    # tiles_per_segment, early-return for trailing segments past the end.
    tiles_per_segment = tl.maximum(tl.cdiv(kv_len, KV_SPLITS * BLOCK_K), 1)
    # tl.maximum(kv_len, 1): with kv_len == 0 the plain ``>= kv_len`` test is
    # true even for pid_k 0, so every segment would bail and leave ``out`` at
    # its torch.empty contents. Letting segment 0 through emits the zeros.
    if pid_k * tiles_per_segment * BLOCK_K >= tl.maximum(kv_len, 1):
        return

    num_tiles = tl.cdiv(kv_len, BLOCK_K)
    tile_start = pid_k * tiles_per_segment
    tile_end = tl.minimum((pid_k + 1) * tiles_per_segment, num_tiles)

    if KV_SPLITS == 1:
        sink_scale = LOG2E if USE_EXP2 else 1.0
        sink = (
            tl.load(attn_sink_ptr + h_offs, mask=h_mask, other=float("-inf")).to(
                tl.float32
            )
            * sink_scale
        )
        m_i = sink
        if USE_EXP2:
            l_i = tl.exp2(sink - m_i)
        else:
            l_i = tl.full((BLOCK_H,), 1.0, dtype=tl.float32)
    else:
        m_i = tl.full((BLOCK_H,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    k_offs = tl.arange(0, BLOCK_K)
    if QUANT_KV:
        g_idx_per_d = d_offs // GROUP_SIZE
    for j in tl.range(tile_start, tile_end, num_stages=2):
        k_start = j * BLOCK_K
        k_pos = k_start + k_offs
        in_range = k_pos < kv_len
        slot = tl.load(
            kv_indices_ptr + kv_start + k_pos,
            mask=in_range,
            other=-1,
        )
        # in_range masks the partial final tile (always needed); the slot >= 0
        # term skips -1 sentinels and is dropped when the caller guarantees none.
        if HAS_INVALID:
            valid = in_range & (slot >= 0)
        else:
            valid = in_range

        # 64-bit before the multiply: a slot index fits 32 bits (the index
        # buffer is int32 by ABI) but `slot * kv_stride_n` does not. A unified
        # V4 pool runs to ~150M rows, so at a 512-element row the product
        # passes 2^31 only 2.8% of the way in and every row above that is read
        # from a wrapped address -- silently, since the wrapped offset still
        # lands inside the same allocation.
        slot_off = slot.to(tl.int64)
        kv_raw = tl.load(
            unified_kv_ptr
            + slot_off[:, None] * kv_stride_n
            + d_offs[None, :] * kv_stride_d,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )
        if QUANT_KV:
            scales_full = tl.load(
                kv_scales_ptr + slot_off[:, None] * ks_stride_n + g_idx_per_d[None, :],
                mask=valid[:, None] & d_mask[None, :],
                other=0.0,
            )
            kv = (kv_raw.to(tl.float32) * scales_full).to(q_ptr.dtype.element_ty)
        else:
            kv = kv_raw

        scores = tl.dot(q, tl.trans(kv))
        scores = tl.where(h_mask[:, None] & valid[None, :], scores, float("-inf"))

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        # A tile with no valid key (all sentinels / out-of-range) leaves
        # m_new == -inf, so exp(m_i - m_new) = exp(-inf + inf) = NaN. With
        # l_i/acc still 0 that NaN survives as 0*NaN = NaN and poisons the
        # split's partials (and the reduce). Treat such a tile as a no-op.
        if USE_EXP2:
            alpha = tl.where(m_new == float("-inf"), 1.0, tl.exp2(m_i - m_new))
            p = tl.exp2(scores - m_new[:, None])
        else:
            alpha = tl.where(m_new == float("-inf"), 1.0, tl.exp(m_i - m_new))
            p = tl.exp(scores - m_new[:, None])
        p = tl.where(h_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new

    if KV_SPLITS == 1:
        denom = tl.maximum(l_i, 1.0e-30)
        out = tl.where(l_i[:, None] > 0.0, acc / denom[:, None], 0.0)
        tl.store(
            out_ptr
            + t * out_stride_t
            + h_offs[:, None] * out_stride_h
            + d_offs[None, :] * out_stride_d,
            out.to(out_ptr.dtype.element_ty),
            mask=h_mask[:, None] & d_mask[None, :],
        )
    else:
        # Emit partials. The reduce reads (m, l, acc) per split and folds in
        # the sink there, so we do *not* touch attn_sink here.
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
    attn_sink_ptr,  # [H] (unused when HAS_SINK is False)
    kv_indptr_ptr,  # [N+1] int32 prefix sum, or [N] ctx lens when USE_CTX_LENS
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
    USE_EXP2: tl.constexpr,
    USE_CTX_LENS: tl.constexpr = False,
    HAS_SINK: tl.constexpr = True,
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

    # The split kernel derives its per-token K range either from a prefix-sum
    # indptr (unified pool) or from a context length (paged block table); mirror
    # whichever the caller used so ``act_num_segments`` below matches exactly.
    if USE_CTX_LENS:
        kv_len = tl.load(kv_indptr_ptr + t)
    else:
        kv_start = tl.load(kv_indptr_ptr + t)
        kv_end = tl.load(kv_indptr_ptr + t + 1)
        kv_len = kv_end - kv_start
    # The clamp mirrors the split kernels: kv_len == 0 leaves cdiv at 0, which
    # would divide by zero deriving act_num_segments below. With it an empty
    # token falls through to act_num_segments == 0 -> every slot masked -> the
    # zero output the split kernel also produces.
    tiles_per_segment = tl.maximum(tl.cdiv(kv_len, KV_SPLITS * BLOCK_K), 1)
    # Only the first ``act_num_segments`` slots of the partial buffer were
    # actually written by the split kernel; the rest are stale.
    act_num_segments = tl.cdiv(kv_len, tiles_per_segment * BLOCK_K)
    segm_mask = k_offs < act_num_segments

    m_p = tl.load(
        m_partial_ptr
        + t * mp_stride_t
        + k_offs[:, None] * mp_stride_k
        + h_offs[None, :] * mp_stride_h,
        mask=segm_mask[:, None] & h_mask[None, :],
        other=float("-inf"),
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

    # The main kernel's domain must match here: the triton main scales scores by
    # softmax_scale * log2(e) and emits base-2 partials (USE_EXP2=True, exp2 +
    # sink lifted by log2(e)); the gluon main stays in the natural-exp domain
    # (USE_EXP2=False, exp + sink unscaled).
    LOG2E = 1.4426950408889634
    sink_scale = LOG2E if USE_EXP2 else 1.0

    # Pre-sink combine across splits.
    m_max = tl.max(m_p, axis=0)  # [BLOCK_H]
    # Empty/stale splits carry m_p == -inf. Force their weight to 0 rather than
    # evaluating exp(m_p - m_max): when a token has *no* valid key at all,
    # m_max is also -inf and exp(-inf + inf) = NaN would corrupt the output.
    if USE_EXP2:
        alpha_split = tl.where(
            m_p == float("-inf"), 0.0, tl.exp2(m_p - m_max[None, :])
        )  # [KV_SPLITS, BLOCK_H]
    else:
        alpha_split = tl.where(m_p == float("-inf"), 0.0, tl.exp(m_p - m_max[None, :]))
    # A split with zero valid keys (all -inf / sentinels) carries m_p == -inf and
    # may have written NaN l/acc partials (the gluon main kernel does, since it
    # has no dead-tile guard). alpha_split is 0 there, but NaN * 0 == NaN would
    # still leak through the sum, zeroing the whole token's output. Mask the full
    # term (not just the weight) so dead splits contribute exactly 0.
    is_dead = m_p == float("-inf")  # [KV_SPLITS, BLOCK_H]
    l_combined = tl.sum(tl.where(is_dead, 0.0, l_p * alpha_split), axis=0)  # [BLOCK_H]
    acc_combined = tl.sum(
        tl.where(is_dead[:, :, None], 0.0, a_p * alpha_split[:, :, None]), axis=0
    )  # [BLOCK_H, BLOCK_D]

    # Fold attn_sink as a virtual K of weight 1 (lifted into the main kernel's
    # domain: base-2 for triton, natural for gluon). Without a sink the combine
    # already sits in the m_max domain, so there is nothing to rescale.
    if HAS_SINK:
        sink = (
            tl.load(attn_sink_ptr + h_offs, mask=h_mask, other=float("-inf")).to(
                tl.float32
            )
            * sink_scale
        )
        m_final = tl.maximum(m_max, sink)
        if USE_EXP2:
            alpha_kv = tl.exp2(m_max - m_final)
            alpha_sink = tl.exp2(sink - m_final)
        else:
            alpha_kv = tl.exp(m_max - m_final)
            alpha_sink = tl.exp(sink - m_final)
        l_final = l_combined * alpha_kv + alpha_sink
        acc_final = acc_combined * alpha_kv[:, None]
    else:
        l_final = l_combined
        acc_final = acc_combined

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


_pa_decode_sparse_shuffled_repr = make_kernel_repr(
    "_pa_decode_sparse_shuffled",
    [
        "BLOCK_H",
        "BLOCK_D",
        "PAGE_SIZE",
        "PAGES_PER_TILE",
        "H",
        "D",
        "KV_SPLITS",
    ],
)


@triton.jit(repr=_pa_decode_sparse_shuffled_repr)
def _pa_decode_sparse_shuffled(
    q_ptr,  # [N, H, D]
    k_ptr,  # SHUFFLE K [P, 1, D//X, PAGE_SIZE, X]
    v_ptr,  # SHUFFLE V [P, 1, PAGE_SIZE//X, D, X]
    k_scales_ptr,  # [P, 1, PAGE_SIZE] fp32 when QUANT_KV (dummy otherwise)
    v_scales_ptr,  # [P, 1, PAGE_SIZE] fp32 when QUANT_KV (dummy otherwise)
    block_table_ptr,  # [N, MAX_PAGES] int32 — per-token page list
    ctx_lens_ptr,  # [N] int32 — per-token valid key count
    m_partial_ptr,  # [N, KV_SPLITS, H_padded] fp32 (unused when KV_SPLITS==1)
    l_partial_ptr,  # [N, KV_SPLITS, H_padded] fp32 (unused when KV_SPLITS==1)
    acc_partial_ptr,  # [N, KV_SPLITS, H_padded, D] fp32 (unused when KV_SPLITS==1)
    attn_sink_ptr,  # [H] (only read when KV_SPLITS==1 and HAS_SINK)
    out_ptr,  # [N, H, D] (only written when KV_SPLITS==1)
    max_pages,
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_p: tl.constexpr,
    k_stride_dg: tl.constexpr,
    k_stride_pos: tl.constexpr,
    v_stride_p: tl.constexpr,
    v_stride_pg: tl.constexpr,
    v_stride_d: tl.constexpr,
    s_stride_p: tl.constexpr,
    bt_stride_n: tl.constexpr,
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
    softmax_scale: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGES_PER_TILE: tl.constexpr,
    X: tl.constexpr,
    HAS_SINK: tl.constexpr,
    QUANT_KV: tl.constexpr,
    USE_EXP2: tl.constexpr,
    num_warps: tl.constexpr,
):
    """3D split-K sparse paged-decode over a SHUFFLE K/V cache pair.

    Same skeleton as ``_pa_decode_sparse`` -- grid (N, ceil(H/BLOCK_H),
    KV_SPLITS), widened BLOCK_H so one CTA owns all of a token's heads, split-K
    over the third axis -- but the keys come from a *paged* cache addressed by a
    per-token block table (``block_table``/``ctx_lens``, the compacted top-k
    selection) rather than a page_size=1 unified pool, and K and V live in
    separate tensors in AMD's SHUFFLE (asm) layout:

        K[page, pos, d] at  page*k_stride_p + (d//X)*k_stride_dg
                            + pos*k_stride_pos + (d%X)
        V[page, pos, d] at  page*v_stride_p + (pos//X)*v_stride_pg
                            + d*v_stride_d + (pos%X)

    That interleaving is what makes the tiles 3-D here. Addressing K as a plain
    [BLOCK_K, D] tile makes the offset non-affine in ``d`` (it steps by 1 inside
    an X-group and by k_stride_dg between groups), so Triton cannot prove
    contiguity and falls back to per-element loads -- ~5x slower end to end.
    Loading [BLOCK_K, D//X, X] instead keeps ``X`` as its own affine axis with
    unit stride, which vectorises to 16-byte ``buffer_load_b128``; the trailing
    ``tl.reshape`` back to [BLOCK_K, D] is a pure re-index of the same registers
    (d == (d//X)*X + d%X). V is loaded pre-transposed, [D, BLOCK_K//X, X] ->
    [D, BLOCK_K], for the same reason.

    ``QUANT_KV`` carries MiniMax-M3's *per-token* fp8 descale (one fp32 per
    cached token, not per D-group): K's folds onto the score columns and V's
    onto the softmax probabilities, both exact because the factor is constant
    along D.
    """
    BLOCK_K: tl.constexpr = PAGE_SIZE * PAGES_PER_TILE
    DG: tl.constexpr = D // X  # d-groups per head-dim row
    VG: tl.constexpr = BLOCK_K // X  # pos-groups per KV tile
    PPG: tl.constexpr = PAGE_SIZE // X  # pos-groups per page
    LOG2E = 1.4426950408889634

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
    # When USE_EXP2, fold log2(e) into q so QK scores land in the base-2 domain
    # and per-element softmax uses the bare exp2 HW instruction.
    qk_scale = softmax_scale * LOG2E if USE_EXP2 else softmax_scale
    q = (q.to(tl.float32) * qk_scale).to(q_ptr.dtype.element_ty)

    kv_len = tl.load(ctx_lens_ptr + t)

    # aiter unified_attention 3d pattern: fixed grid axis (KV_SPLITS), runtime
    # tiles_per_segment, early-return for trailing segments past the end.
    tiles_per_segment = tl.maximum(tl.cdiv(kv_len, KV_SPLITS * BLOCK_K), 1)
    # tl.maximum(kv_len, 1): with kv_len == 0 the plain ``>= kv_len`` test is
    # true even for pid_k 0, so every segment would bail and leave ``out`` at
    # its torch.empty contents. Letting segment 0 through emits the zeros.
    if pid_k * tiles_per_segment * BLOCK_K >= tl.maximum(kv_len, 1):
        return

    num_tiles = tl.cdiv(kv_len, BLOCK_K)
    tile_start = pid_k * tiles_per_segment
    tile_end = tl.minimum((pid_k + 1) * tiles_per_segment, num_tiles)

    if KV_SPLITS == 1 and HAS_SINK:
        sink_scale = LOG2E if USE_EXP2 else 1.0
        sink = (
            tl.load(attn_sink_ptr + h_offs, mask=h_mask, other=float("-inf")).to(
                tl.float32
            )
            * sink_scale
        )
        m_i = sink
        if USE_EXP2:
            l_i = tl.exp2(sink - m_i)
        else:
            l_i = tl.full((BLOCK_H,), 1.0, dtype=tl.float32)
    else:
        m_i = tl.full((BLOCK_H,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    # Per-tile K offsets, tile shape [BLOCK_K, DG, X] -> reshape [BLOCK_K, D].
    k_off = (
        (tl.arange(0, BLOCK_K)[:, None, None] % PAGE_SIZE) * k_stride_pos
        + tl.arange(0, DG)[None, :, None] * k_stride_dg
        + tl.arange(0, X)[None, None, :]
    )
    k_page_slot = (tl.arange(0, BLOCK_K) // PAGE_SIZE)[:, None, None]
    # Per-tile V offsets, tile shape [D, VG, X] -> reshape [D, BLOCK_K] (V^T).
    v_off = (
        d_offs[:, None, None] * v_stride_d
        + (tl.arange(0, VG)[None, :, None] % PPG) * v_stride_pg
        + tl.arange(0, X)[None, None, :]
    )
    v_page_slot = (tl.arange(0, VG) // PPG)[None, :, None]

    k_pos = tl.arange(0, BLOCK_K)
    bt_row = block_table_ptr + t * bt_stride_n

    for j in tl.range(tile_start, tile_end, num_stages=2):
        page_base = j * PAGES_PER_TILE
        valid = (j * BLOCK_K + k_pos) < kv_len
        # A trailing partial tile can reach past the block table row (max_pages
        # need not divide by PAGES_PER_TILE), so the page-id gathers are masked
        # and fall back to page 0 -- an in-bounds page whose keys ``valid`` drops
        # anyway. That keeps the far larger K/V gathers mask-free.
        k_slot = page_base + k_page_slot
        k_pages = tl.load(bt_row + k_slot, mask=k_slot < max_pages, other=0).to(
            tl.int64
        )
        k_raw = tl.load(k_ptr + k_pages * k_stride_p + k_off)

        v_slot = page_base + v_page_slot
        v_pages = tl.load(bt_row + v_slot, mask=v_slot < max_pages, other=0).to(
            tl.int64
        )
        v_raw = tl.load(v_ptr + v_pages * v_stride_p + v_off)

        if QUANT_KV:
            k_raw = k_raw.to(q_ptr.dtype.element_ty)
            v_raw = v_raw.to(q_ptr.dtype.element_ty)
        k_tile = tl.reshape(k_raw, (BLOCK_K, D))
        v_tile = tl.reshape(v_raw, (D, BLOCK_K))

        scores = tl.dot(q, tl.trans(k_tile))
        if QUANT_KV:
            # ``k_pages`` already carries one page id per token of the tile.
            s_pages = tl.reshape(k_pages, (BLOCK_K,))
            s_pos = k_pos % PAGE_SIZE
            k_dq = tl.load(k_scales_ptr + s_pages * s_stride_p + s_pos)
            scores = scores * k_dq[None, :]
        scores = tl.where(h_mask[:, None] & valid[None, :], scores, float("-inf"))

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        # A tile with no valid key leaves m_new == -inf, so exp(m_i - m_new)
        # = exp(-inf + inf) = NaN; with l_i/acc still 0 that NaN survives as
        # 0*NaN and poisons the split. Treat such a tile as a no-op.
        if USE_EXP2:
            alpha = tl.where(m_new == float("-inf"), 1.0, tl.exp2(m_i - m_new))
            p = tl.exp2(scores - m_new[:, None])
        else:
            alpha = tl.where(m_new == float("-inf"), 1.0, tl.exp(m_i - m_new))
            p = tl.exp(scores - m_new[:, None])
        p = tl.where(h_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        if QUANT_KV:
            v_dq = tl.load(v_scales_ptr + s_pages * s_stride_p + s_pos)
            p = p * v_dq[None, :]

        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(k_tile.dtype), tl.trans(v_tile), acc)
        m_i = m_new
        l_i = l_new

    if KV_SPLITS == 1:
        denom = tl.maximum(l_i, 1.0e-30)
        out = tl.where(l_i[:, None] > 0.0, acc / denom[:, None], 0.0)
        tl.store(
            out_ptr
            + t * out_stride_t
            + h_offs[:, None] * out_stride_h
            + d_offs[None, :] * out_stride_d,
            out.to(out_ptr.dtype.element_ty),
            mask=h_mask[:, None] & d_mask[None, :],
        )
    else:
        # Emit partials. The reduce reads (m, l, acc) per split and folds in the
        # sink there, so we do *not* touch attn_sink here.
        tl.store(
            m_partial_ptr
            + t * mp_stride_t
            + pid_k * mp_stride_k
            + h_offs * mp_stride_h,
            m_i,
            mask=h_mask,
        )
        tl.store(
            l_partial_ptr
            + t * lp_stride_t
            + pid_k * lp_stride_k
            + h_offs * lp_stride_h,
            l_i,
            mask=h_mask,
        )
        tl.store(
            acc_partial_ptr
            + t * ap_stride_t
            + pid_k * ap_stride_k
            + h_offs[:, None] * ap_stride_h
            + d_offs[None, :] * ap_stride_d,
            acc,
            mask=h_mask[:, None] & d_mask[None, :],
        )
