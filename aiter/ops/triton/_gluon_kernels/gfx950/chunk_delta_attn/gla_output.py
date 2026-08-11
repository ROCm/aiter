# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._gluon_kernels.gfx950.chunk_delta_attn._varlen import (
    async_load,
    reg_load,
    reg_store,
)

exp2 = tl.math.exp2


@gluon.jit
def chunk_gla_fwd_kernel_o_gluon(
    q,
    v,
    g,
    h,
    o,
    A,
    scale,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    T,
    H: gl.constexpr,
    HV: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BT: gl.constexpr,
    BLOCK_T: gl.constexpr,
    BK: gl.constexpr,
    BV: gl.constexpr,
    USE_EXP2: gl.constexpr,
    TRANSPOSE_STATE: gl.constexpr,
    IS_VARLEN: gl.constexpr,
    BUFFERED: gl.constexpr,
    LOAD_CACHE: gl.constexpr,
    STORE_CACHE: gl.constexpr,
):
    # Unified Gluon kernel for gla_o.
    #   BUFFERED=2 + NT_PER_BLOCK>1 → persistent, double-buffered software pipeline (large H).
    #   BUFFERED=1 + NT_PER_BLOCK==1 → non-persistent, single chunk per CTA (small H):
    #     main loop range(0) is skipped, so only prologue (prefetch) + epilogue (compute) run,
    #     which is exactly the old chunk_gla_fwd_kernel_o_gluon_small_h behavior.
    gl.static_assert(USE_EXP2, "pipelined path: USE_EXP2=True only")
    gl.static_assert(not TRANSPOSE_STATE, "pipelined path: TRANSPOSE_STATE=False only")
    gl.static_assert(K == 128, "pipelined path: K=128 only (i_k=0,1)")
    gl.static_assert(BT == 64, "pipelined path: BT=64 only")
    gl.static_assert(BK == 64, "pipelined path: BK=64 only")
    gl.static_assert(BV == 128, "pipelined path: BV=128 only")
    # Varlen runs one chunk per CTA: a CTA window spanning several chunks would
    # straddle sequence boundaries, and the main loop below (statically empty at
    # BLOCK_T == BT) is what keeps the prefetch pipeline free of a dynamic bound.
    gl.static_assert(
        not IS_VARLEN or BLOCK_T == BT, "varlen: one chunk per CTA (BLOCK_T == BT)"
    )

    dtype = q.type.element_ty
    blocked_lhs: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((0, 1), (0, 2), (0, 4), (32, 0)),
        lane_bases=((0, 8), (0, 16), (0, 32), (4, 0), (8, 0), (16, 0)),
        warp_bases=((1, 0), (2, 0)),
        block_bases=[],
        shape=[64, 64],
    )
    shared_lhs: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs=[[512, 16]],
        offset_bases=[
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [4, 0],
            [8, 0],
            [16, 0],
            [1, 0],
            [2, 0],
            [32, 0],
        ],
        cga_layout=[],
        shape=[64, 64],
    )

    # BUFFERED buffers: 2 → double-buffered pipeline; 1 → single buffer (small-H path)
    bufs_q1 = gl.allocate_shared_memory(
        dtype, shape=[BUFFERED, BT, BK], layout=shared_lhs
    )
    bufs_q2 = gl.allocate_shared_memory(
        dtype, shape=[BUFFERED, BT, BK], layout=shared_lhs
    )

    # h, v layout
    blocked_rhs: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((0, 1), (0, 2), (0, 4), (8, 0), (4, 0)),
        lane_bases=((0, 8), (0, 16), (0, 32), (0, 64), (16, 0), (32, 0)),
        warp_bases=((1, 0), (2, 0)),
        block_bases=[],
        shape=[64, 128],
    )
    shared_rhs: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs=[[512, 16]],
        offset_bases=[
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [16, 0],
            [32, 0],
            [1, 0],
            [2, 0],
            [8, 0],
            [4, 0],
        ],
        cga_layout=[],
        shape=[64, 128],
    )
    # BUFFERED buffers: 2 → double-buffered pipeline; 1 → single buffer (small-H path)
    bufs_h1 = gl.allocate_shared_memory(
        dtype, shape=[BUFFERED, BT, BV], layout=shared_rhs
    )
    bufs_h2 = gl.allocate_shared_memory(
        dtype, shape=[BUFFERED, BT, BV], layout=shared_rhs
    )
    bufs_v = gl.allocate_shared_memory(
        dtype, shape=[BUFFERED, BT, BV], layout=shared_rhs
    )

    # layout for mfma
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[4, 1],
    )
    mfma_layout_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=8
    )
    mfma_layout_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=8
    )

    blocked_o: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((0, 1), (0, 2), (0, 4), (4, 0), (8, 0)),
        lane_bases=((0, 8), (0, 16), (0, 32), (0, 64), (1, 0), (2, 0)),
        warp_bases=((16, 0), (32, 0)),
        block_bases=[],
        shape=[64, 128],
    )

    i_t_outer = gl.program_id(0)
    i_bh = gl.program_id(1)
    i_v = gl.program_id(2)

    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    NT: gl.constexpr = T // BT
    NT_PER_BLOCK: gl.constexpr = BLOCK_T // BT

    # Row origin of this CTA's sequence (`bos`), its first chunk (`i_t_base`, counted
    # within the sequence) and where that chunk sits in the flat `h` buffer. Varlen
    # takes program_id(0) as a flat chunk id and looks all three up; fixed length
    # derives them from the rectangular batch layout.
    if IS_VARLEN:
        i_n = gl.load(chunk_indices + i_t_outer * 2).to(gl.int32)
        i_t_base = gl.load(chunk_indices + i_t_outer * 2 + 1).to(gl.int32)
        bos = gl.load(cu_seqlens + i_n).to(gl.int32)
        T_local = gl.load(cu_seqlens + i_n + 1).to(gl.int32) - bos
        h_chunk = gl.load(chunk_offsets + i_n).to(gl.int32) + i_t_base
    else:
        i_t_base = i_t_outer * NT_PER_BLOCK
        bos = i_b * T
        T_local = T
        h_chunk = i_b * NT + i_t_base

    # Valid rows in this chunk: BT everywhere except the tail chunk of a sequence.
    rows_valid = T_local - i_t_base * BT

    base_q = (bos * H + i_h) * K
    base_g = (bos * HV + i_hv) * K
    base_v = (bos * HV + i_hv) * V
    base_o = (bos * HV + i_hv) * V
    base_A_off = (bos * HV + i_hv) * BT
    base_h_init = (h_chunk * HV + i_hv) * K * V

    # q (BT, BK) blocked_lhs, stride (H*K, 1); +i_t*BT*H*K  +(i_k*BK)
    # Rows past a sequence's end re-read the last valid row rather than being
    # predicated off: `buffer_load_to_shared` skips masked lanes without honouring
    # `other`, leaving uninitialized LDS whose inf/nan would reach *valid* output rows
    # through the A·v product (A is zero there, and 0 * inf = nan). Re-reading a real
    # row keeps the arithmetic finite; A and g still cancel those rows out.
    range_q_bt_raw = gl.arange(0, BT, layout=gl.SliceLayout(1, blocked_lhs))
    range_q_bk = gl.arange(0, BK, layout=gl.SliceLayout(0, blocked_lhs))
    if IS_VARLEN:
        range_q_bt = gl.minimum(range_q_bt_raw, rows_valid - 1)
    else:
        range_q_bt = range_q_bt_raw
    base_offs_q = range_q_bt[:, None] * (H * K) + range_q_bk[None, :]

    # g (BT, BK) directly in mfma_layout_a (consumed without convert), stride (HV*K, 1)
    range_g_bt = gl.arange(0, BT, layout=gl.SliceLayout(1, mfma_layout_a))
    range_g_bk = gl.arange(0, BK, layout=gl.SliceLayout(0, mfma_layout_a))
    base_offs_g = range_g_bt[:, None] * (HV * K) + range_g_bk[None, :]

    # h (BK, BV) blocked_rhs, stride (V, 1); base_h advances per-iter
    range_h_bk = gl.arange(0, BK, layout=gl.SliceLayout(1, blocked_rhs))
    range_h_bv = gl.arange(0, BV, layout=gl.SliceLayout(0, blocked_rhs))
    base_offs_h = range_h_bk[:, None] * V + (i_v * BV) + range_h_bv[None, :]

    # v (BT, BV) blocked_rhs, stride (HV*V, 1); +i_t*BT*HV*V  (row-clamped, see q)
    range_v_bt_raw = gl.arange(0, BT, layout=gl.SliceLayout(1, blocked_rhs))
    range_v_bv = gl.arange(0, BV, layout=gl.SliceLayout(0, blocked_rhs))
    if IS_VARLEN:
        range_v_bt = gl.minimum(range_v_bt_raw, rows_valid - 1)
    else:
        range_v_bt = range_v_bt_raw
    base_offs_v_arr = range_v_bt[:, None] * (HV * V) + (i_v * BV) + range_v_bv[None, :]

    # A (BT, BT) — was blocked_lhs (LDS path); now mfma_layout_a (sync buffer_load to reg)
    range_a_bt0 = gl.arange(0, BT, layout=gl.SliceLayout(1, mfma_layout_a))
    range_a_bt1 = gl.arange(0, BT, layout=gl.SliceLayout(0, mfma_layout_a))
    base_offs_a = range_a_bt0[:, None] * (HV * BT) + range_a_bt1[None, :]

    # o store offsets — accumulator (blocked_o) layout, stride (HV*V, 1); +i_t*BT*HV*V
    range_o_bt = gl.arange(0, BT, layout=gl.SliceLayout(1, blocked_o))
    range_o_bv = gl.arange(0, BV, layout=gl.SliceLayout(0, blocked_o))
    base_offs_o = range_o_bt[:, None] * (HV * V) + (i_v * BV) + range_o_bv[None, :]

    range_ms_m = gl.arange(0, BT, layout=gl.SliceLayout(1, mfma_layout_a))
    range_ms_n = gl.arange(0, BT, layout=gl.SliceLayout(0, mfma_layout_a))
    m_s = range_ms_m[:, None] >= range_ms_n[None, :]

    # Ragged-tail masks, consumed only when IS_VARLEN. Zeroing A (and, via g, the
    # gate) is what makes the row-clamped tail contribute nothing; the store mask is
    # what keeps this CTA out of the next sequence's rows. The column term carries no
    # condition of its own, it broadcasts the row predicate to the offset shape.
    mask_g = (range_g_bt[:, None] < rows_valid) & (range_g_bk[None, :] < BK)
    mask_a = (range_a_bt0[:, None] < rows_valid) & (range_a_bt1[None, :] < rows_valid)
    mask_o = (range_o_bt[:, None] < rows_valid) & (range_o_bv[None, :] < BV)

    i_t0 = i_t_base  # i_t for i_t_local=0, counted within the sequence

    offs_q1_0 = i_t0 * BT * H * K + base_offs_q
    offs_q2_0 = offs_q1_0 + BK
    offs_h1_0 = base_offs_h
    offs_h2_0 = offs_h1_0 + BK * V
    offs_a_0 = i_t0 * BT * HV * BT + base_offs_a
    offs_v_0 = i_t0 * BT * HV * V + base_offs_v_arr

    # None of these are predicated: q/v are row-clamped above, and h is indexed by
    # chunk over a full (K, V) tile.
    async_load(bufs_q1.index(0), q, base_q + offs_q1_0, None, False, LOAD_CACHE)
    async_load(bufs_q2.index(0), q, base_q + offs_q2_0, None, False, LOAD_CACHE)
    async_load(bufs_h1.index(0), h, base_h_init + offs_h1_0, None, False, LOAD_CACHE)
    async_load(bufs_h2.index(0), h, base_h_init + offs_h2_0, None, False, LOAD_CACHE)
    async_load(bufs_v.index(0), v, base_v + offs_v_0, None, False, LOAD_CACHE)

    offs_g1_0 = i_t0 * BT * HV * K + base_offs_g
    offs_g2_0 = offs_g1_0 + BK
    g_curr1 = reg_load(g, base_g + offs_g1_0, mask_g, IS_VARLEN, LOAD_CACHE)
    g_curr2 = reg_load(g, base_g + offs_g2_0, mask_g, IS_VARLEN, LOAD_CACHE)
    a_curr = reg_load(A, base_A_off + offs_a_0, mask_a, IS_VARLEN, LOAD_CACHE)

    buf_idx = 0

    for i_t_local in range(NT_PER_BLOCK - 1):
        gl.amd.cdna4.async_copy.wait_group(0)

        async_idx = (buf_idx + 1) % BUFFERED
        i_t_next_local = i_t_local + 1
        i_t_next = i_t_base + i_t_next_local
        base_h_next = base_h_init + i_t_next_local * HV * K * V

        # === Stage 1: prefetch iter i_t_next ===
        offs_q1_n = i_t_next * BT * H * K + base_offs_q
        offs_q2_n = offs_q1_n + BK
        offs_h1_n = base_offs_h
        offs_h2_n = offs_h1_n + BK * V
        offs_a_n = i_t_next * BT * HV * BT + base_offs_a
        offs_v_n = i_t_next * BT * HV * V + base_offs_v_arr

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            bufs_q1.index(async_idx), q, base_q + offs_q1_n, cache_modifier=LOAD_CACHE
        )
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            bufs_q2.index(async_idx), q, base_q + offs_q2_n, cache_modifier=LOAD_CACHE
        )
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            bufs_h1.index(async_idx),
            h,
            base_h_next + offs_h1_n,
            cache_modifier=LOAD_CACHE,
        )
        gl.amd.cdna4.async_copy.commit_group()

        # === Stage 2: consume iter i_t_local from buf_idx ===
        # outstanding before stage 2: 9 (5 from current iter — A removed, 4 from next iter just issued)
        gl.amd.cdna4.async_copy.wait_group(3)  # drain current iter's 5 commits

        # i_k = 0 leg
        b_q1 = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_q1.index(buf_idx), mfma_layout_a
        )
        b_qg1 = (b_q1 * exp2(g_curr1)).to(dtype)
        b_h1 = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_h1.index(buf_idx), mfma_layout_b
        )
        b_o = gl.zeros([BT, BV], dtype=gl.float32, layout=mfma_layout)
        b_o = gl.amd.cdna4.mfma(b_qg1, b_h1, b_o)

        # move apart the loads (A is now sync register load below)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            bufs_h2.index(async_idx),
            h,
            base_h_next + offs_h2_n,
            cache_modifier=LOAD_CACHE,
        )
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            bufs_v.index(async_idx), v, base_v + offs_v_n, cache_modifier=LOAD_CACHE
        )
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.wait_group(2)  # drain current iter's 5 commits

        # i_k = 1 leg
        b_q2 = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_q2.index(buf_idx), mfma_layout_a
        )
        b_qg2 = (b_q2 * exp2(g_curr2)).to(dtype)
        b_h2 = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_h2.index(buf_idx), mfma_layout_b
        )
        b_o = gl.amd.cdna4.mfma(b_qg2, b_h2, b_o)

        b_o = b_o * scale

        offs_g1_n = i_t_next * BT * HV * K + base_offs_g
        offs_g2_n = offs_g1_n + BK
        g_next1 = gl.amd.cdna4.buffer_load(g, base_g + offs_g1_n, cache=LOAD_CACHE)
        g_next2 = gl.amd.cdna4.buffer_load(g, base_g + offs_g2_n, cache=LOAD_CACHE)
        a_next = gl.amd.cdna4.buffer_load(A, base_A_off + offs_a_n, cache=LOAD_CACHE)

        # A · v  (a_curr already in mfma_layout_a register; no LDS load)
        b_A_masked = gl.where(m_s, a_curr, 0.0).to(dtype)
        b_v = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_v.index(buf_idx), mfma_layout_b
        )
        b_o = gl.amd.cdna4.mfma(b_A_masked, b_v, b_o)

        b_o_bf16 = b_o.to(dtype)
        i_t_curr = i_t_base + i_t_local
        offs_o_curr = i_t_curr * BT * HV * V + base_offs_o
        b_o = gl.convert_layout(b_o_bf16, blocked_o)
        gl.amd.cdna4.buffer_store(
            b_o, ptr=o, offsets=base_o + offs_o_curr, cache=STORE_CACHE
        )

        g_curr1 = g_next1
        g_curr2 = g_next2
        a_curr = a_next
        buf_idx = async_idx

    i_t_last_local = NT_PER_BLOCK - 1
    i_t_last = i_t_base + i_t_last_local

    gl.amd.cdna4.async_copy.wait_group(0)

    b_q1 = gl.amd.cdna4.async_copy.load_shared_relaxed(
        bufs_q1.index(buf_idx), mfma_layout_a
    )
    b_qg1 = (b_q1 * exp2(g_curr1)).to(dtype)
    b_h1 = gl.amd.cdna4.async_copy.load_shared_relaxed(
        bufs_h1.index(buf_idx), mfma_layout_b
    )
    b_o = gl.zeros([BT, BV], dtype=gl.float32, layout=mfma_layout)
    b_o = gl.amd.cdna4.mfma(b_qg1, b_h1, b_o)

    b_q2 = gl.amd.cdna4.async_copy.load_shared_relaxed(
        bufs_q2.index(buf_idx), mfma_layout_a
    )
    b_qg2 = (b_q2 * exp2(g_curr2)).to(dtype)
    b_h2 = gl.amd.cdna4.async_copy.load_shared_relaxed(
        bufs_h2.index(buf_idx), mfma_layout_b
    )
    b_o = gl.amd.cdna4.mfma(b_qg2, b_h2, b_o)

    b_o = b_o * scale

    b_A_masked = gl.where(m_s, a_curr, 0.0).to(dtype)
    b_v = gl.amd.cdna4.async_copy.load_shared_relaxed(
        bufs_v.index(buf_idx), mfma_layout_b
    )
    b_o = gl.amd.cdna4.mfma(b_A_masked, b_v, b_o)

    b_o_bf16 = b_o.to(dtype)
    offs_o_last = i_t_last * BT * HV * V + base_offs_o
    b_o = gl.convert_layout(b_o_bf16, blocked_o)
    reg_store(b_o, o, base_o + offs_o_last, mask_o, IS_VARLEN, STORE_CACHE)
