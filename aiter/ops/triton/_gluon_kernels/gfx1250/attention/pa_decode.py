# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon GQA paged-attention decode for gfx1250.

TDM copies each K/V page into a 2-deep LDS ring, so the next page loads while the
current page is computed.
"""

import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_pa_decode_gqa_tdm_repr = make_kernel_repr(
    "_pa_decode_gqa_tdm_kernel",
    [
        "NUM_KV_HEADS",
        "G",
        "HEAD_SZ",
        "BLOCK_H",
        "BLOCK_D",
        "BLOCK_K",
        "SEQ_PARTITION_SZ",
        "SPLIT",
        "num_warps",
    ],
)


@gluon.jit(repr=_pa_decode_gqa_tdm_repr)
def _pa_decode_gqa_tdm_kernel(
    out_ptr,  # [num_seqs, num_kv_heads * G, HEAD_SZ]
    q_ptr,  # [num_seqs, num_kv_heads * G, HEAD_SZ]
    k_ptr,  # [num_blks, num_kv_heads, BLOCK_K, HEAD_SZ]
    v_ptr,  # [num_blks, num_kv_heads, BLOCK_K, HEAD_SZ]
    blk_tables_ptr,  # [num_seqs, max_num_blks_per_seq], int32
    seq_lens_ptr,  # [num_seqs]
    num_rows,  # num_blks * num_kv_heads * BLOCK_K
    softmax_scale,
    stride_o_s,
    stride_o_h,
    stride_q_s,
    stride_q_h,
    stride_bt_s,
    exp_sums_ptr,  # SPLIT only: [num_seqs, num_kv_heads, max_parts, G]
    max_logits_ptr,  # SPLIT only: same shape, natural-log units
    logits_ptr,  # SPLIT only: [num_seqs, num_kv_heads, max_parts, G, HEAD_SZ]
    stride_ml_s,
    stride_ml_h,
    stride_ml_p,
    stride_lg_s,
    stride_lg_h,
    stride_lg_p,
    stride_lg_g,
    NUM_KV_HEADS: gl.constexpr,
    G: gl.constexpr,
    HEAD_SZ: gl.constexpr,
    BLOCK_H: gl.constexpr,
    BLOCK_D: gl.constexpr,  # HEAD_SZ padded to a power of two
    BLOCK_K: gl.constexpr,
    SEQ_PARTITION_SZ: gl.constexpr,  # tokens per program; a multiple of BLOCK_K
    SPLIT: gl.constexpr,  # write partials for the v2 reduce kernel
    num_warps: gl.constexpr,
):
    WARP_SIZE: gl.constexpr = 32
    LOG2E: gl.constexpr = 1.4426950408889634
    LN2: gl.constexpr = 0.6931471805599453
    PAGES_PER_PART: gl.constexpr = SEQ_PARTITION_SZ // BLOCK_K

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
    WARPS_H: gl.constexpr = 2 if num_warps >= 2 else 1
    WARPS_D: gl.constexpr = num_warps // WARPS_H
    Q_BLOCKED_LAYOUT: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[WARP_SIZE // (D_INNER // 2), D_INNER // 2],
        warps_per_cta=[WARPS_H, WARPS_D],
        order=[1, 0],
    )
    kv_shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_D, 8]], [BLOCK_K, BLOCK_D], [1, 0]
    )

    seq_idx = gl.program_id(0)
    kv_head_idx = gl.program_id(1)
    part_idx = gl.program_id(2)
    seq_len = gl.load(seq_lens_ptr + seq_idx).to(gl.int32)
    num_pages = (seq_len + BLOCK_K - 1) // BLOCK_K
    # Pages of this partition; zero when the partition lies past the sequence.
    page_start = part_idx * PAGES_PER_PART
    num_loc = gl.maximum(gl.minimum(num_pages - page_start, PAGES_PER_PART), 0)

    # q: [BLOCK_H, BLOCK_D]; rows >= G are padding.
    h_offs = gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, Q_BLOCKED_LAYOUT))
    d_offs = gl.arange(0, BLOCK_D, layout=gl.SliceLayout(0, Q_BLOCKED_LAYOUT))
    q_head = kv_head_idx * G + h_offs
    q = gl.amd.cdna4.buffer_load(
        ptr=q_ptr + seq_idx * stride_q_s,
        offsets=(q_head[:, None] * stride_q_h + d_offs[None, :]).to(gl.int32),
        mask=(h_offs < G)[:, None] & (d_offs < HEAD_SZ)[None, :],
        other=0.0,
    )
    q_dot = gl.convert_layout(q, dot_q_layout)

    m_i = gl.full(
        [BLOCK_H], float("-inf"), gl.float32, layout=gl.SliceLayout(1, QK_WMMA_LAYOUT)
    )
    l_i = gl.zeros(
        [BLOCK_H], dtype=gl.float32, layout=gl.SliceLayout(1, QK_WMMA_LAYOUT)
    )
    acc = gl.zeros([BLOCK_H, BLOCK_D], dtype=gl.float32, layout=PV_WMMA_LAYOUT)
    col_offs = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, QK_WMMA_LAYOUT))
    row_offs = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, dot_v_layout))

    k_bufs = gl.allocate_shared_memory(
        k_ptr.dtype.element_ty, [2, BLOCK_K, BLOCK_D], kv_shared
    )
    v_bufs = gl.allocate_shared_memory(
        v_ptr.dtype.element_ty, [2, BLOCK_K, BLOCK_D], kv_shared
    )
    # The tile is BLOCK_D wide but rows are HEAD_SZ wide: TDM zero-fills the padding.
    k_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=k_ptr,
        shape=[num_rows, HEAD_SZ],
        strides=[HEAD_SZ, 1],
        block_shape=[BLOCK_K, BLOCK_D],
        layout=kv_shared,
    )
    v_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=v_ptr,
        shape=[num_rows, HEAD_SZ],
        strides=[HEAD_SZ, 1],
        block_shape=[BLOCK_K, BLOCK_D],
        layout=kv_shared,
    )

    blk_tbl = blk_tables_ptr + seq_idx * stride_bt_s
    head_row = kv_head_idx * BLOCK_K

    # Prologue: pages 0 and 1 are in flight.
    if num_loc > 0:
        row = gl.load(blk_tbl + page_start) * (NUM_KV_HEADS * BLOCK_K) + head_row
        gl.amd.gfx1250.tdm.async_load(k_desc, [row, 0], k_bufs.index(0))
        gl.amd.gfx1250.tdm.async_load(v_desc, [row, 0], v_bufs.index(0))
    if num_loc > 1:
        row = gl.load(blk_tbl + page_start + 1) * (NUM_KV_HEADS * BLOCK_K) + head_row
        gl.amd.gfx1250.tdm.async_load(k_desc, [row, 0], k_bufs.index(1))
        gl.amd.gfx1250.tdm.async_load(v_desc, [row, 0], v_bufs.index(1))

    buf: gl.int32 = 0
    for i in tl.range(0, num_loc):
        # Same-type TDM ops finish in issue order: <= 2 left means page i has landed.
        if i + 1 < num_loc:
            gl.amd.gfx1250.tdm.async_wait(2)
        else:
            gl.amd.gfx1250.tdm.async_wait(0)

        k_t = k_bufs.index(buf).permute([1, 0]).load(dot_k_layout)
        scores = gl.amd.gfx1250.wmma(
            q_dot,
            k_t,
            gl.zeros([BLOCK_H, BLOCK_K], dtype=gl.float32, layout=QK_WMMA_LAYOUT),
        )
        scores = scores * (softmax_scale * LOG2E)
        # Select, not add: unused rows of the last page may hold NaN or Inf.
        valid = ((page_start + i) * BLOCK_K + col_offs) < seq_len
        scores = gl.where(valid[None, :], scores, float("-inf"))

        m_new = gl.maximum(m_i, gl.max(scores, axis=1))
        alpha = gl.exp2(m_i - m_new)
        p = gl.exp2(scores - m_new[:, None])
        l_i = l_i * alpha + gl.sum(p, axis=1)

        v_t = v_bufs.index(buf).load(dot_v_layout)
        if page_start + i == num_pages - 1:
            v_valid = ((page_start + i) * BLOCK_K + row_offs) < seq_len
            v_t = gl.where(v_valid[:, None], v_t, 0.0)
        p_dot = gl.convert_layout(p.to(q_ptr.dtype.element_ty), dot_p_layout)
        acc = acc * gl.convert_layout(alpha[:, None], layout=PV_WMMA_LAYOUT)
        acc = gl.amd.gfx1250.wmma(p_dot, v_t, acc)
        m_i = m_new

        # Every warp has read this slot; only then may TDM refill it with page i + 2.
        gl.barrier()
        if i + 2 < num_loc:
            row = (
                gl.load(blk_tbl + page_start + i + 2) * (NUM_KV_HEADS * BLOCK_K)
                + head_row
            )
            gl.amd.gfx1250.tdm.async_load(k_desc, [row, 0], k_bufs.index(buf))
            gl.amd.gfx1250.tdm.async_load(v_desc, [row, 0], v_bufs.index(buf))
        buf = 1 - buf

    # l_i >= 1 for any non-empty sequence; the floor makes an empty one write zeros.
    inv_l = gl.convert_layout(
        1.0 / gl.maximum(l_i, 1e-30)[:, None], layout=PV_WMMA_LAYOUT
    )
    out_val = gl.convert_layout(
        (acc * inv_l).to(out_ptr.dtype.element_ty), Q_BLOCKED_LAYOUT
    )
    if SPLIT:
        # The reduce kernel expects normalized partials and natural-log maxima.
        in_range = num_loc > 0
        ml_base = (
            seq_idx * stride_ml_s + kv_head_idx * stride_ml_h + part_idx * stride_ml_p
        )
        h_layout: gl.constexpr = gl.SliceLayout(1, Q_BLOCKED_LAYOUT)
        m_out = gl.convert_layout(m_i * LN2, h_layout)
        l_out = gl.convert_layout(l_i, h_layout)
        h_mask = (h_offs < G) & in_range
        gl.amd.cdna4.buffer_store(
            m_out, ptr=max_logits_ptr + ml_base, offsets=h_offs, mask=h_mask
        )
        gl.amd.cdna4.buffer_store(
            l_out, ptr=exp_sums_ptr + ml_base, offsets=h_offs, mask=h_mask
        )
        lg_base = (
            seq_idx * stride_lg_s + kv_head_idx * stride_lg_h + part_idx * stride_lg_p
        )
        gl.amd.cdna4.buffer_store(
            out_val,
            ptr=logits_ptr + lg_base,
            offsets=(h_offs[:, None] * stride_lg_g + d_offs[None, :]).to(gl.int32),
            mask=h_mask[:, None] & (d_offs < HEAD_SZ)[None, :],
        )
    else:
        gl.amd.cdna4.buffer_store(
            out_val,
            ptr=out_ptr + seq_idx * stride_o_s,
            offsets=(q_head[:, None] * stride_o_h + d_offs[None, :]).to(gl.int32),
            mask=(h_offs < G)[:, None] & (d_offs < HEAD_SZ)[None, :],
        )
