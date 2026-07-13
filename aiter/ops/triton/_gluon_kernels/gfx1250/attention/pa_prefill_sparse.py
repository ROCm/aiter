# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon (gfx1250) sparse prefill attention over two KV sources with sink.

Grid: ``(T, cdiv(H, BLOCK_H))`` — each CTA handles one query token and
BLOCK_H query heads. Same grid as the decode kernel, but iterates two KV
regions (prefix + extend) sequentially sharing the online softmax
accumulator. No split-K: prefill has enough tokens to fill the GPU.

Derived from ``pa_decode_sparse`` with KV_SPLITS=1, minus FP8/reduce,
plus the second KV region.
"""

import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_pa_prefill_sparse_repr = make_kernel_repr(
    "_pa_prefill_sparse",
    [
        "BLOCK_H",
        "BLOCK_D",
        "BLOCK_K",
        "H",
        "D",
    ],
)


def _region_prologue(
    slot_desc, kv_desc, slot_bufs, kv_bufs, slot_reg_layout, BLOCK_K, NUM_SLOT_BUFFERS
):
    """Issue first two slot async_loads, wait for slot[0], gather KV[0]."""
    gl.amd.gfx1250.tdm.async_load(slot_desc, [0, 0], slot_bufs.index(0))
    gl.amd.gfx1250.tdm.async_load(slot_desc, [0, 1 * BLOCK_K], slot_bufs.index(1))
    gl.amd.gfx1250.tdm.async_wait(1)
    slot_reg = slot_bufs.index(0).reshape([BLOCK_K]).load(layout=slot_reg_layout)
    cur_valid = slot_reg >= 0
    safe_slot_cur = gl.where(cur_valid, slot_reg, 0)
    gl.amd.gfx1250.tdm.async_gather(kv_desc, safe_slot_cur, 0, kv_bufs.index(0))
    return cur_valid, safe_slot_cur


@gluon.jit(repr=_pa_prefill_sparse_repr)
def _pa_prefill_sparse(
    q_ptr,  # [T, H, D]
    unified_kv_ptr,  # [total_pages, D]    — prefix source
    kv_indices_prefix_ptr,  # [total_prefix_indices] int32
    kv_indptr_prefix_ptr,  # [T+1] int32
    kv_ptr,  # [total_tokens, D]     — extend source
    kv_indices_extend_ptr,  # [total_extend_indices] int32
    kv_indptr_extend_ptr,  # [T+1] int32
    attn_sink_ptr,  # [H]
    out_ptr,  # [T, H, D]
    total_prefix_pages,
    total_extend_tokens,
    q_stride_t: gl.constexpr,
    q_stride_h: gl.constexpr,
    q_stride_d: gl.constexpr,
    pkv_stride_n: gl.constexpr,
    pkv_stride_d: gl.constexpr,
    ekv_stride_n: gl.constexpr,
    ekv_stride_d: gl.constexpr,
    out_stride_t: gl.constexpr,
    out_stride_h: gl.constexpr,
    out_stride_d: gl.constexpr,
    H: gl.constexpr,
    D: gl.constexpr,
    softmax_scale: gl.constexpr,
    BLOCK_H: gl.constexpr,
    BLOCK_D: gl.constexpr,
    BLOCK_K: gl.constexpr,
    USE_EXP2: gl.constexpr,
    num_warps: gl.constexpr,
):
    WARP_SIZE: gl.constexpr = 32
    LOG2E: gl.constexpr = 1.4426950408889634

    if num_warps == 1:
        pv_warp_bases: gl.constexpr = []
        qk_warp_bases: gl.constexpr = []
    elif num_warps == 2:
        pv_warp_bases: gl.constexpr = [[0, 1]]
        qk_warp_bases: gl.constexpr = [[1, 0]]
    elif num_warps == 4:
        pv_warp_bases: gl.constexpr = [[0, 1], [0, 2]]
        qk_warp_bases: gl.constexpr = [[1, 0], [2, 0]]
    else:
        pv_warp_bases: gl.constexpr = [[0, 1], [0, 2], [0, 4]]
        qk_warp_bases: gl.constexpr = [[1, 0], [2, 0], [4, 0]]

    QK_WMMA_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        instr_shape=[16, 16, 32],
        warp_bases=qk_warp_bases,
    )
    PV_WMMA_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        instr_shape=[16, 16, 32],
        warp_bases=pv_warp_bases,
    )
    K_WIDTH: gl.constexpr = 8
    dot_q_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=QK_WMMA_LAYOUT, k_width=K_WIDTH
    )
    dot_k_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=QK_WMMA_LAYOUT, k_width=K_WIDTH
    )
    dot_p_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=PV_WMMA_LAYOUT, k_width=K_WIDTH
    )
    dot_v_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=PV_WMMA_LAYOUT, k_width=K_WIDTH
    )

    D_INNER: gl.constexpr = BLOCK_D // 8
    QKV_WARPS_H: gl.constexpr = 2 if num_warps >= 2 else 1
    QKV_WARPS_D: gl.constexpr = num_warps // QKV_WARPS_H
    Q_BLOCKED_LAYOUT: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[WARP_SIZE // (D_INNER // 2), D_INNER // 2],
        warps_per_cta=[QKV_WARPS_H, QKV_WARPS_D],
        order=[1, 0],
    )
    SLOT_BLOCKED_LAYOUT: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[BLOCK_K],
        threads_per_warp=[32],
        warps_per_cta=[num_warps],
        order=[0],
    )
    slot_reg_layout: gl.constexpr = SLOT_BLOCKED_LAYOUT

    kv_shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_D, 8]], [BLOCK_K, BLOCK_D], [1, 0]
    )
    slot_shared: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[1, 0]
    )
    valid_col_mma: gl.constexpr = gl.SliceLayout(0, QK_WMMA_LAYOUT)

    t = gl.program_id(0)
    pid_h = gl.program_id(1)

    h_off_base = pid_h * BLOCK_H

    # ---- Q load: [BLOCK_H, D] for one token ----
    h_offs_q = gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, Q_BLOCKED_LAYOUT))
    d_offs_q = gl.arange(0, BLOCK_D, layout=gl.SliceLayout(0, Q_BLOCKED_LAYOUT))
    h_offs_q_eff = h_off_base + h_offs_q
    h_mask_q = h_offs_q_eff < H
    q = gl.amd.cdna4.buffer_load(
        ptr=q_ptr + t * q_stride_t,
        offsets=(
            h_offs_q_eff[:, None] * q_stride_h + d_offs_q[None, :] * q_stride_d
        ).to(gl.int32),
        mask=h_mask_q[:, None],
        other=0.0,
    )
    qk_scale = softmax_scale * LOG2E if USE_EXP2 else softmax_scale
    mfma_q = gl.convert_layout(q, dot_q_layout)
    mfma_q = mfma_q.to(gl.float32) * qk_scale
    mfma_q = mfma_q.to(q_ptr.dtype.element_ty)

    # ---- Sink init (no split-K, fold sink into m_i upfront) ----
    h_offs_mma_row = gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, PV_WMMA_LAYOUT))
    h_offs_mma_row_eff = h_off_base + h_offs_mma_row
    h_mask_mma_row = h_offs_mma_row_eff < H

    sink = gl.amd.cdna4.buffer_load(
        ptr=attn_sink_ptr,
        offsets=h_offs_mma_row_eff.to(gl.int32),
        mask=h_mask_mma_row,
        other=float("-inf"),
    ).to(gl.float32)
    if USE_EXP2:
        sink = sink * LOG2E
    sink = gl.convert_layout(sink, gl.SliceLayout(1, QK_WMMA_LAYOUT))
    m_i = sink
    if USE_EXP2:
        l_i = gl.exp2(sink - m_i)
    else:
        l_i = gl.full(
            [BLOCK_H],
            1.0,
            dtype=gl.float32,
            layout=gl.SliceLayout(1, QK_WMMA_LAYOUT),
        )

    acc = gl.zeros([BLOCK_H, BLOCK_D], dtype=gl.float32, layout=PV_WMMA_LAYOUT)

    # ---- Shared memory (2-stage pipeline, reused across regions) ----
    NUM_BUFFERS: gl.constexpr = 2
    kv_bufs = gl.allocate_shared_memory(
        unified_kv_ptr.dtype.element_ty,
        [NUM_BUFFERS, BLOCK_K, BLOCK_D],
        kv_shared,
    )
    NUM_SLOT_BUFFERS: gl.constexpr = 2
    slot_bufs = gl.allocate_shared_memory(
        kv_indices_prefix_ptr.dtype.element_ty,
        [NUM_SLOT_BUFFERS, 1, BLOCK_K],
        slot_shared,
    )

    k_offs_slot = gl.arange(0, BLOCK_K, layout=slot_reg_layout)

    # ===== Region 1: prefix from unified_kv =====
    p_start = gl.load(kv_indptr_prefix_ptr + t)
    p_end = gl.load(kv_indptr_prefix_ptr + t + 1)
    p_len = p_end - p_start

    pkv_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=unified_kv_ptr,
        shape=[total_prefix_pages, BLOCK_D],
        strides=[pkv_stride_n, 1],
        block_shape=[BLOCK_K, BLOCK_D],
        layout=kv_shared,
    )

    if p_len > 0:
        p_slot_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=kv_indices_prefix_ptr + p_start,
            shape=[1, p_len],
            strides=[p_len, 1],
            block_shape=[1, BLOCK_K],
            layout=slot_shared,
        )
        p_num_tiles = gl.cdiv(p_len, BLOCK_K)

        # Prologue
        gl.amd.gfx1250.tdm.async_load(p_slot_desc, [0, 0], slot_bufs.index(0))
        gl.amd.gfx1250.tdm.async_load(p_slot_desc, [0, 1 * BLOCK_K], slot_bufs.index(1))
        gl.amd.gfx1250.tdm.async_wait(1)
        slot_reg = slot_bufs.index(0).reshape([BLOCK_K]).load(layout=slot_reg_layout)
        cur_valid = slot_reg >= 0
        safe_slot_cur = gl.where(cur_valid, slot_reg, 0)
        gl.amd.gfx1250.tdm.async_gather(pkv_desc, safe_slot_cur, 0, kv_bufs.index(0))

        buf_idx: gl.int32 = 0
        p_iters = p_num_tiles

        gl.assume(p_iters >= 1)
        for i in tl.range(0, p_iters - 1):
            async_idx = (buf_idx + 1) % NUM_BUFFERS

            gl.amd.gfx1250.tdm.async_load(
                p_slot_desc,
                [0, (i + 2) * BLOCK_K],
                slot_bufs.index(i % NUM_SLOT_BUFFERS),
            )
            gl.amd.gfx1250.tdm.async_wait(2)
            slot_reg = (
                slot_bufs.index((i + 1) % NUM_SLOT_BUFFERS)
                .reshape([BLOCK_K])
                .load(layout=slot_reg_layout)
            )
            next_valid = slot_reg >= 0
            safe_next_slot = gl.where(next_valid, slot_reg, 0)
            gl.amd.gfx1250.tdm.async_gather(
                pkv_desc, safe_next_slot, 0, kv_bufs.index(async_idx)
            )

            gl.amd.gfx1250.tdm.async_wait(2)

            kv_t = kv_bufs.index(buf_idx).permute([1, 0]).load(dot_k_layout)
            scores = gl.amd.gfx1250.wmma(
                mfma_q,
                kv_t,
                gl.zeros([BLOCK_H, BLOCK_K], dtype=gl.float32, layout=QK_WMMA_LAYOUT),
            )
            valid_col = gl.convert_layout(cur_valid, valid_col_mma)
            score_bias = gl.where(valid_col, 0.0, float("-inf"))
            scores = scores + score_bias[None, :]

            m_block = gl.max(scores, axis=1)
            m_new = gl.maximum(m_i, m_block)
            if USE_EXP2:
                alpha = gl.exp2(m_i - m_new)
                p = gl.exp2(scores - m_new[:, None])
            else:
                alpha = gl.exp(m_i - m_new)
                p = gl.exp(scores - m_new[:, None])
            l_new = l_i * alpha + gl.sum(p, axis=1)

            kv_for_acc = kv_bufs.index(buf_idx).load(dot_v_layout)
            p_dot = gl.convert_layout(p.to(q_ptr.dtype.element_ty), dot_p_layout)
            acc = acc * gl.convert_layout(alpha[:, None], layout=PV_WMMA_LAYOUT)
            acc = gl.amd.gfx1250.wmma(p_dot, kv_for_acc, acc)

            m_i = m_new
            l_i = l_new
            cur_valid = next_valid
            buf_idx = async_idx

        # Epilogue: final prefix tile
        gl.amd.gfx1250.tdm.async_wait(0)

        final_in_range = ((p_iters - 1) * BLOCK_K + k_offs_slot) < p_len
        final_valid = final_in_range & cur_valid

        kv_t = kv_bufs.index(buf_idx).permute([1, 0]).load(dot_k_layout)
        scores = gl.amd.gfx1250.wmma(
            mfma_q,
            kv_t,
            gl.zeros([BLOCK_H, BLOCK_K], dtype=gl.float32, layout=QK_WMMA_LAYOUT),
        )
        valid_col = gl.convert_layout(final_valid, valid_col_mma)
        score_bias = gl.where(valid_col, 0.0, float("-inf"))
        scores = scores + score_bias[None, :]

        m_block = gl.max(scores, axis=1)
        m_new = gl.maximum(m_i, m_block)
        if USE_EXP2:
            alpha = gl.exp2(m_i - m_new)
            p = gl.exp2(scores - m_new[:, None])
            p = gl.where(valid_col[None, :], p, 0.0)
        else:
            alpha = gl.exp(m_i - m_new)
            p = gl.exp(scores - m_new[:, None])
        l_new = l_i * alpha + gl.sum(p, axis=1)

        kv_for_acc = kv_bufs.index(buf_idx).load(dot_v_layout)
        p_dot = gl.convert_layout(p.to(q_ptr.dtype.element_ty), dot_p_layout)
        acc = acc * gl.convert_layout(alpha[:, None], layout=PV_WMMA_LAYOUT)
        acc = gl.amd.gfx1250.wmma(p_dot, kv_for_acc, acc)

        m_i = m_new
        l_i = l_new

    # ===== Region 2: extend from kv (per-fwd flat) =====
    e_start = gl.load(kv_indptr_extend_ptr + t)
    e_end = gl.load(kv_indptr_extend_ptr + t + 1)
    e_len = e_end - e_start

    ekv_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=kv_ptr,
        shape=[total_extend_tokens, BLOCK_D],
        strides=[ekv_stride_n, 1],
        block_shape=[BLOCK_K, BLOCK_D],
        layout=kv_shared,
    )

    if e_len > 0:
        e_slot_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=kv_indices_extend_ptr + e_start,
            shape=[1, e_len],
            strides=[e_len, 1],
            block_shape=[1, BLOCK_K],
            layout=slot_shared,
        )
        e_num_tiles = gl.cdiv(e_len, BLOCK_K)

        # Prologue
        gl.amd.gfx1250.tdm.async_load(e_slot_desc, [0, 0], slot_bufs.index(0))
        gl.amd.gfx1250.tdm.async_load(e_slot_desc, [0, 1 * BLOCK_K], slot_bufs.index(1))
        gl.amd.gfx1250.tdm.async_wait(1)
        slot_reg = slot_bufs.index(0).reshape([BLOCK_K]).load(layout=slot_reg_layout)
        cur_valid = slot_reg >= 0
        safe_slot_cur = gl.where(cur_valid, slot_reg, 0)
        gl.amd.gfx1250.tdm.async_gather(ekv_desc, safe_slot_cur, 0, kv_bufs.index(0))

        buf_idx: gl.int32 = 0
        e_iters = e_num_tiles

        gl.assume(e_iters >= 1)
        for i in tl.range(0, e_iters - 1):
            async_idx = (buf_idx + 1) % NUM_BUFFERS

            gl.amd.gfx1250.tdm.async_load(
                e_slot_desc,
                [0, (i + 2) * BLOCK_K],
                slot_bufs.index(i % NUM_SLOT_BUFFERS),
            )
            gl.amd.gfx1250.tdm.async_wait(2)
            slot_reg = (
                slot_bufs.index((i + 1) % NUM_SLOT_BUFFERS)
                .reshape([BLOCK_K])
                .load(layout=slot_reg_layout)
            )
            next_valid = slot_reg >= 0
            safe_next_slot = gl.where(next_valid, slot_reg, 0)
            gl.amd.gfx1250.tdm.async_gather(
                ekv_desc, safe_next_slot, 0, kv_bufs.index(async_idx)
            )

            gl.amd.gfx1250.tdm.async_wait(2)

            kv_t = kv_bufs.index(buf_idx).permute([1, 0]).load(dot_k_layout)
            scores = gl.amd.gfx1250.wmma(
                mfma_q,
                kv_t,
                gl.zeros([BLOCK_H, BLOCK_K], dtype=gl.float32, layout=QK_WMMA_LAYOUT),
            )
            valid_col = gl.convert_layout(cur_valid, valid_col_mma)
            score_bias = gl.where(valid_col, 0.0, float("-inf"))
            scores = scores + score_bias[None, :]

            m_block = gl.max(scores, axis=1)
            m_new = gl.maximum(m_i, m_block)
            if USE_EXP2:
                alpha = gl.exp2(m_i - m_new)
                p = gl.exp2(scores - m_new[:, None])
            else:
                alpha = gl.exp(m_i - m_new)
                p = gl.exp(scores - m_new[:, None])
            l_new = l_i * alpha + gl.sum(p, axis=1)

            kv_for_acc = kv_bufs.index(buf_idx).load(dot_v_layout)
            p_dot = gl.convert_layout(p.to(q_ptr.dtype.element_ty), dot_p_layout)
            acc = acc * gl.convert_layout(alpha[:, None], layout=PV_WMMA_LAYOUT)
            acc = gl.amd.gfx1250.wmma(p_dot, kv_for_acc, acc)

            m_i = m_new
            l_i = l_new
            cur_valid = next_valid
            buf_idx = async_idx

        # Epilogue: final extend tile
        gl.amd.gfx1250.tdm.async_wait(0)

        final_in_range = ((e_iters - 1) * BLOCK_K + k_offs_slot) < e_len
        final_valid = final_in_range & cur_valid

        kv_t = kv_bufs.index(buf_idx).permute([1, 0]).load(dot_k_layout)
        scores = gl.amd.gfx1250.wmma(
            mfma_q,
            kv_t,
            gl.zeros([BLOCK_H, BLOCK_K], dtype=gl.float32, layout=QK_WMMA_LAYOUT),
        )
        valid_col = gl.convert_layout(final_valid, valid_col_mma)
        score_bias = gl.where(valid_col, 0.0, float("-inf"))
        scores = scores + score_bias[None, :]

        m_block = gl.max(scores, axis=1)
        m_new = gl.maximum(m_i, m_block)
        if USE_EXP2:
            alpha = gl.exp2(m_i - m_new)
            p = gl.exp2(scores - m_new[:, None])
            p = gl.where(valid_col[None, :], p, 0.0)
        else:
            alpha = gl.exp(m_i - m_new)
            p = gl.exp(scores - m_new[:, None])
        l_new = l_i * alpha + gl.sum(p, axis=1)

        kv_for_acc = kv_bufs.index(buf_idx).load(dot_v_layout)
        p_dot = gl.convert_layout(p.to(q_ptr.dtype.element_ty), dot_p_layout)
        acc = acc * gl.convert_layout(alpha[:, None], layout=PV_WMMA_LAYOUT)
        acc = gl.amd.gfx1250.wmma(p_dot, kv_for_acc, acc)

        m_i = m_new
        l_i = l_new

    # ===== Output: normalize by l_i =====
    one_over_L = 1.0 / l_i[:, None]
    one_over_L = gl.convert_layout(one_over_L, layout=PV_WMMA_LAYOUT)
    out_val = acc * one_over_L

    h_offs_out = gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, Q_BLOCKED_LAYOUT))
    d_offs_out = gl.arange(0, BLOCK_D, layout=gl.SliceLayout(0, Q_BLOCKED_LAYOUT))
    h_offs_out_eff = h_off_base + h_offs_out
    h_mask_out = h_offs_out_eff < H

    out_blocked = gl.convert_layout(
        out_val.to(out_ptr.dtype.element_ty), Q_BLOCKED_LAYOUT
    )
    gl.amd.cdna4.buffer_store(
        out_blocked,
        ptr=out_ptr + t * out_stride_t,
        offsets=(
            h_offs_out_eff[:, None] * out_stride_h + d_offs_out[None, :] * out_stride_d
        ).to(gl.int32),
        mask=h_mask_out[:, None],
    )
