# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils.device_info import get_num_xcds


# ---------------------------------------------------------------------------
# Prefill kernel
# ---------------------------------------------------------------------------


@gluon.jit
def _sparse_attn_prefill_kernel(
    q_ptr,              # [T, H, D]
    kv_ptr,             # [num_KV, D]
    kv_indices_ptr,     # [num_None_Zero]
    kv_indptr_ptr,      # [T + 1]
    attn_sink_ptr,      # [H]
    out_ptr,            # [T, H, D]
    q_stride_t,
    q_stride_h,
    q_stride_d,
    kv_stride_n,
    kv_stride_d,
    out_stride_t,
    out_stride_h,
    out_stride_d,
    num_queries,
    num_heads,
    head_dim,
    num_kv,
    scale,
    HAS_ATTN_SINK: gl.constexpr,
    BLOCK_H: gl.constexpr,
    BLOCK_D: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    num_warps: gl.constexpr,
):
    # ---- layouts -------------------------------------------------------
    # Single MFMA layout for both QK and PV: PV contracts over BLOCK_K, which the
    # [16, 16, 32] mfma tiles directly, so the PV mfma reuses mma_qk / qk_a / qk_b
    # (acc is laid out in mma_qk). This keeps the softmax stats in one MFMA parent,
    # avoiding the cross-parent layout conversions a separate PV layout would need.
    mma_qk: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True,
        warps_per_cta=[num_warps, 1],
    )
    qk_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma_qk, k_width=8)
    qk_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma_qk, k_width=8)

    # q [BLOCK_H, BLOCK_D] load layout (D = dim1 contiguous).
    blk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[1, 64],
        warps_per_cta=[num_warps, 1], order=[1, 0],
    )
    blk_kv: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[8, BLOCK_K // num_warps], threads_per_warp=[64, 1],
        warps_per_cta=[1, num_warps], order=[0, 1],
    )
    # KV-slot id async-copy offsets: a clean 64-wide (one lane per slot, warps
    # redundant) linear layout. A 1-D async gather narrower than the 64-lane warp
    # can't tile without lane aliasing (which fails to lower), so slots are always
    # gathered 64 at a time and the [0:BLOCK_K] slice is read back per tile
    # (BLOCK_K=32 reads half; BLOCK_K=64 reads the full SLOT_W tile).
    gl.static_assert(num_warps == 4, "slot async-copy layout assumes num_warps == 4")
    SLOT_W: gl.constexpr = 64
    dll_slot: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[], lane_bases=[[1], [2], [4], [8], [16], [32]],
        warp_bases=[[0], [0]], block_bases=[], shape=[SLOT_W],
    )

    gl.static_assert(BLOCK_D == 512, "shared_q/shared_kv offset_bases assume BLOCK_D == 512")
    gl.static_assert(BLOCK_H == 64, "shared_q is hard-coded to [64, 512]")
    gl.static_assert(BLOCK_K == 32 or BLOCK_K == 64, "shared_kv is hard-coded to [512, 32] or [512, 64]")

    # Q staging layout (async global->LDS): D (dim1) contiguous + BLOCK_H row
    # bits, [[512,16]] padding.
    shared_q: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs=[[512, 16]],
        offset_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256],
                      [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]],
        cga_layout=[],
        shape=[64, 512],
    )

    if BLOCK_K == 32:
        shared_kv: gl.constexpr = gl.PaddedSharedLayout(
            interval_padding_pairs=[[512, 16]],
            offset_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0],
                          [0, 1], [0, 2], [0, 4], [0, 8], [0, 16]],
            cga_layout=[],
            shape=[512, 32],
        )
    else:
        # BLOCK == 64, BLOCK_K choices are guarded by above static_assert
        shared_kv: gl.constexpr = gl.PaddedSharedLayout(
            interval_padding_pairs=[[512, 16]],
            offset_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0],
                          [0, 1], [0, 2], [0, 8], [0, 4], [0, 16], [0, 32]],
            cga_layout=[],
            shape=[512, 64],
        )

    # 1-D LDS staging for the KV-slot ids (global->LDS async-copy, ds_read back).
    shared_slot: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[0],
    )

    # slice layouts for the per-row / per-col index vectors
    sl_h_blk: gl.constexpr = gl.SliceLayout(1, blk)      # heads (axis0 of blk, q)
    sl_d_blk: gl.constexpr = gl.SliceLayout(0, blk)      # dim (axis1 of blk, q)
    sl_d_kv: gl.constexpr = gl.SliceLayout(1, blk_kv)    # dim (axis0 of blk_kv, KV)
    sl_k_kv: gl.constexpr = gl.SliceLayout(0, blk_kv)    # kpos (axis1 of blk_kv, KV)
    sl_h_qk: gl.constexpr = gl.SliceLayout(1, mma_qk)    # heads (scores/acc rows)
    sl_k_qk: gl.constexpr = gl.SliceLayout(0, mma_qk)    # cols: scores kpos AND acc/out dim

    # ---- one program per (query, head-block) tile on the 3-D XCD-pinned grid ----
    query_idx = gl.program_id(axis=0) + gl.program_id(axis=2) * NUM_XCDS
    pid_h = gl.program_id(axis=1)

    # The 3-D grid rounds num_queries up to a multiple of NUM_XCDS; guard the tail.
    if query_idx >= num_queries:
        return

    # ---- Global Load q (async global -> LDS) ---
    head_off = pid_h * BLOCK_H + gl.arange(0, BLOCK_H, layout=sl_h_blk)
    dim_off = gl.arange(0, BLOCK_D, layout=sl_d_blk)
    head_mask = head_off < num_heads
    q_off = head_off[:, None] * q_stride_h + dim_off[None, :] * q_stride_d
    buf_q = gl.allocate_shared_memory(q_ptr.dtype.element_ty, [BLOCK_H, BLOCK_D], shared_q)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        buf_q, q_ptr + query_idx * q_stride_t, q_off, mask=head_mask[:, None]
    )
    gl.amd.cdna4.async_copy.commit_group()

    # ---- running softmax state ----------------------------------------
    LOG2E: gl.constexpr = 1.4426950408889634
    qk_scale = scale * LOG2E
    m_i = gl.full([BLOCK_H], float("-inf"), dtype=gl.float32, layout=sl_h_qk)
    l_i = gl.zeros([BLOCK_H], dtype=gl.float32, layout=sl_h_qk)
    acc = gl.zeros([BLOCK_H, BLOCK_D], dtype=gl.float32, layout=mma_qk)

    k_off_kv = gl.arange(0, BLOCK_K, layout=sl_k_kv)     # kpos (cols of [D,K] tile)
    slot_off = gl.arange(0, SLOT_W, layout=dll_slot)     # 64-wide slot gather lanes
    dim_off_kv = gl.arange(0, BLOCK_D, layout=sl_d_kv)   # dim (rows of [D,K] tile)

    # KV-slot ids staged through LDS too, double-buffered like the KV tiles.
    # Slots are gathered 64 at a time (SLOT_W); each tile reads back its
    # [0:BLOCK_K] slice.
    NUM_BUFFER: gl.constexpr = 2
    smem_slot = gl.allocate_shared_memory(
        kv_indices_ptr.dtype.element_ty, [NUM_BUFFER, SLOT_W], shared_slot
    )

    # ---- Stage-3 software pipeline (slot -> KV -> compute), all async-copy:
    #   * slot ids run 2 tiles ahead (GL slot[i+2]),
    #   * the gathered KV [D,K] tile runs 1 tile ahead (GL KV[i+1], addressed
    #     by the just-arrived slot[i+1]),
    #   * the mfma consumes KV[i] ds_read'd inside the iteration.
    kv_start = gl.load(kv_indptr_ptr + query_idx)
    kv_end = gl.load(kv_indptr_ptr + query_idx + 1)
    kv_len = kv_end - kv_start

    num_iters = (kv_len + BLOCK_K - 1) // BLOCK_K
    # Clamp the pipeline structure to >= 2 tiles so the 2-tile-peeled epilogue
    # always addresses two valid buffers. 
    eff_iters = gl.maximum(num_iters, 2)

    # Prologue: GL slot[0];
    #           GL slot[1];
    #           LL Q;
    #           LL slot[0] + GL KV[0].
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_slot.index(0), kv_indices_ptr + kv_start, slot_off, mask=slot_off < kv_len
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_slot.index(1), kv_indices_ptr + kv_start, BLOCK_K + slot_off,
        mask=(BLOCK_K + slot_off) < kv_len,
    )
    gl.amd.cdna4.async_copy.commit_group()

    gl.amd.cdna4.async_copy.wait_group(2)
    q_dot = gl.amd.cdna4.async_copy.load_shared_relaxed(buf_q, qk_a)

    # KV LDS allocation should be after Q Local Load to reuse Q LDS
    smem_kv = gl.allocate_shared_memory(
        kv_ptr.dtype.element_ty, [NUM_BUFFER, BLOCK_D, BLOCK_K], shared_kv
    )

    gl.amd.cdna4.async_copy.wait_group(1)
    # Tile 0 is full whenever real num_iters >= 2; it can be partial only when
    # num_iters <= 1, so clamp its tail-lane gather (a no-op for the full case)
    # and mask its scores in the epilogue below.
    slot0 = gl.amd.cdna4.async_copy.load_shared_relaxed(smem_slot.index(0).slice(0, BLOCK_K, 0), sl_k_kv)
    tail0 = k_off_kv < kv_len
    kv_off0 = dim_off_kv[:, None] * kv_stride_d + gl.where(tail0, slot0, 0)[None, :] * kv_stride_n
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smem_kv.index(0), kv_ptr, kv_off0)
    gl.amd.cdna4.async_copy.commit_group()

    # Main Loop: GL slot[i+2]
    #            LL KV[i]^T + MFMA QK
    #            LL slot[i+1] + GL KV[i+1];
    #            LL KV[i]   + MFMA PV
    # Entry invariant (iter i): in flight [slot[i+1], KV[i]]. Runs only for real
    # num_iters >= 3, where every tile it computes (0..num_iters-3) is full.
    for i in tl.range(0, eff_iters - 2):

        # GL slot[i+2] (2 tiles ahead) into the slot buffer freed by tile i.
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smem_slot.index(i % 2),
            kv_indices_ptr + kv_start,
            (i + 2) * BLOCK_K + slot_off,
            mask=((i + 2) * BLOCK_K + slot_off) < kv_len,
        )
        gl.amd.cdna4.async_copy.commit_group()

        # LL KV[i] (K^T view) -> QK mfma -> base-2 online softmax for tile i.
        gl.amd.cdna4.async_copy.wait_group(1)
        cbuf = i % 2
        kvT_dot = gl.amd.cdna4.async_copy.load_shared_relaxed(smem_kv.index(cbuf), qk_b)
        acc_qk = gl.zeros([BLOCK_H, BLOCK_K], dtype=gl.float32, layout=mma_qk)
        scores = gl.amd.cdna4.mfma(q_dot, kvT_dot, acc_qk)

        # LL slot[i+1] -> derive KV[i+1] offsets -> GL KV[i+1] (1 tile ahead).
        gl.amd.cdna4.async_copy.wait_group(2)
        sbuf = (i + 1) % 2
        slot_n = gl.amd.cdna4.async_copy.load_shared_relaxed(smem_slot.index(sbuf).slice(0, BLOCK_K, 0), sl_k_kv)
        kv_off_n = dim_off_kv[:, None] * kv_stride_d + slot_n[None, :] * kv_stride_n
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smem_kv.index(sbuf), kv_ptr, kv_off_n)
        gl.amd.cdna4.async_copy.commit_group()

        scores = scores * qk_scale
        m_block = gl.max(scores, axis=1)
        m_new = gl.maximum(m_i, m_block)
        alpha = gl.exp2(m_i - m_new)
        p = gl.exp2(scores - m_new[:, None])
        l_i = l_i * alpha + gl.sum(p, axis=1)
        m_i = m_new
        # LL KV[i] (V view) -> PV mfma.
        v_dot = gl.amd.cdna4.async_copy.load_shared_relaxed(smem_kv.index(cbuf).permute([1, 0]), qk_b)
        p_dot = gl.convert_layout(p.to(kv_ptr.dtype.element_ty), qk_a)
        alpha_pv = gl.convert_layout(alpha, sl_h_qk)
        acc = acc * alpha_pv[:, None]
        acc = gl.amd.cdna4.mfma(p_dot, v_dot, acc)

    # Epilogue: LL slot[last] -> GL KV[last] (not yet prefetched), then compute
    # the two trailing tiles eff_iters-2 and eff_iters-1.
    gl.amd.cdna4.async_copy.wait_group(1)
    nlast = (eff_iters - 1) % 2
    slot_l = gl.amd.cdna4.async_copy.load_shared_relaxed(smem_slot.index(nlast).slice(0, BLOCK_K, 0), sl_k_kv)
    # Tail lanes (kpos >= kv_len) are clamped to slot 0 (a safe in-bounds row)
    # for the gather; the -inf score mask below drops their softmax contribution.
    kpos_l = (eff_iters - 1) * BLOCK_K + k_off_kv
    tail = kpos_l < kv_len
    kv_off_l = dim_off_kv[:, None] * kv_stride_d + gl.where(tail, slot_l, 0)[None, :] * kv_stride_n
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smem_kv.index(nlast), kv_ptr, kv_off_l)
    gl.amd.cdna4.async_copy.commit_group()

    # Epilogue tile eff_iters-2 (full when real num_iters >= 2 -> mask is a
    # no-op; partial when num_iters <= 1 -> masked to the real kv_len).
    gl.amd.cdna4.async_copy.wait_group(1)
    cbuf = (eff_iters - 2) % 2
    tail_qk0 = ((eff_iters - 2) * BLOCK_K + gl.arange(0, BLOCK_K, layout=sl_k_qk)) < kv_len
    kvT_dot = gl.amd.cdna4.async_copy.load_shared_relaxed(smem_kv.index(cbuf), qk_b)
    acc_qk = gl.zeros([BLOCK_H, BLOCK_K], dtype=gl.float32, layout=mma_qk)
    scores = gl.amd.cdna4.mfma(q_dot, kvT_dot, acc_qk)
    scores = scores * qk_scale
    scores = gl.where(tail_qk0[None, :], scores, float("-inf"))
    m_block = gl.max(scores, axis=1)
    m_new = gl.maximum(m_i, m_block)
    alpha = gl.exp2(m_i - m_new)
    p = gl.exp2(scores - m_new[:, None])
    p = gl.where(tail_qk0[None, :], p, 0.0)
    l_i = l_i * alpha + gl.sum(p, axis=1)
    m_i = m_new
    v_dot = gl.amd.cdna4.async_copy.load_shared_relaxed(smem_kv.index(cbuf).permute([1, 0]), qk_b)
    p_dot = gl.convert_layout(p.to(kv_ptr.dtype.element_ty), qk_a)
    alpha_qk = gl.convert_layout(alpha, sl_h_qk)
    acc = acc * alpha_qk[:, None]
    acc = gl.amd.cdna4.mfma(p_dot, v_dot, acc)

    # Epilogue tile eff_iters-1 (always-possibly-partial tile -> mask tail).
    tail_qk = ((eff_iters - 1) * BLOCK_K + gl.arange(0, BLOCK_K, layout=sl_k_qk)) < kv_len
    gl.amd.cdna4.async_copy.wait_group(0)
    kvT_dot = gl.amd.cdna4.async_copy.load_shared_relaxed(smem_kv.index(nlast), qk_b)
    acc_qk = gl.zeros([BLOCK_H, BLOCK_K], dtype=gl.float32, layout=mma_qk)
    scores = gl.amd.cdna4.mfma(q_dot, kvT_dot, acc_qk)
    scores = scores * qk_scale
    scores = gl.where(tail_qk[None, :], scores, float("-inf"))
    m_block = gl.max(scores, axis=1)
    m_new = gl.maximum(m_i, m_block)
    alpha = gl.exp2(m_i - m_new)
    p = gl.exp2(scores - m_new[:, None])
    p = gl.where(tail_qk[None, :], p, 0.0)
    l_i = l_i * alpha + gl.sum(p, axis=1)
    m_i = m_new
    v_dot = gl.amd.cdna4.async_copy.load_shared_relaxed(smem_kv.index(nlast).permute([1, 0]), qk_b)
    p_dot = gl.convert_layout(p.to(kv_ptr.dtype.element_ty), qk_a)
    alpha_pv = gl.convert_layout(alpha, sl_h_qk)
    acc = acc * alpha_pv[:, None]
    acc = gl.amd.cdna4.mfma(p_dot, v_dot, acc)

    if HAS_ATTN_SINK:
        sink = gl.load(
            attn_sink_ptr + head_off, mask=head_mask, other=float("-inf")
        ).to(gl.float32)
        sink = gl.convert_layout(sink, sl_h_qk) * LOG2E
        m_final = gl.maximum(m_i, sink)
        alpha = gl.exp2(m_i - m_final)
        l_final = l_i * alpha + gl.exp2(sink - m_final)
        denom = gl.maximum(l_final, 1.0e-30)
        scale_row = alpha / denom
        guard = l_final > 0.0
    else:
        denom = gl.maximum(l_i, 1.0e-30)
        scale_row = 1.0 / denom
        guard = l_i > 0.0

    scale_qk = gl.convert_layout(scale_row, sl_h_qk)
    guard_qk = gl.convert_layout(guard, sl_h_qk)
    out = gl.where(guard_qk[:, None], acc * scale_qk[:, None], 0.0)

    out_head = pid_h * BLOCK_H + gl.arange(0, BLOCK_H, layout=sl_h_qk)
    out_dim = gl.arange(0, BLOCK_D, layout=sl_k_qk)
    out_off = out_head[:, None] * out_stride_h + out_dim[None, :] * out_stride_d
    if (num_heads % BLOCK_H == 0):
        gl.amd.cdna4.buffer_store(
            out.to(out_ptr.dtype.element_ty), 
            ptr=out_ptr + query_idx * out_stride_t, 
            offsets=out_off
        )
    else:
        gl.amd.cdna4.buffer_store(
            out.to(out_ptr.dtype.element_ty),
            ptr=out_ptr + query_idx * out_stride_t,
            offsets=out_off,
            mask=(out_head < num_heads)[:, None],
        )


def sparse_attn_prefill_gluon(
    q,                 # [T, H, D] bf16 queries
    kv,                # [num_kv, D] bf16 KV pool
    indices,           # [nnz] int32 ragged KV slot ids
    indptr,            # [T + 1] int32 ragged offsets
    out,               # [T, H, D] output (filled in place)
    scale,             # softmax scale
    attn_sink=None,    # optional [H] fp32 per-head sink bias
):
    """Host launcher for the DSV4 CSA(Compressed Sparse Attention) prefill kernel.
    """
    assert q.ndim == 3, f"expected q=[T,H,D], got {tuple(q.shape)}"
    assert kv.ndim == 2, f"expected kv=[num_kv,D], got {tuple(kv.shape)}"
    assert q.is_cuda and kv.is_cuda, "q/kv must be on the GPU"

    num_queries, num_heads, head_dim = q.shape
    BLOCK_D = triton.next_power_of_2(head_dim)
    BLOCK_H = 64
    BLOCK_K = 64
    NUM_XCDS = get_num_xcds()

    has_attn_sink = attn_sink is not None
    if attn_sink is None:
        attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)

    num_pid_h = triton.cdiv(num_heads, BLOCK_H)
    grid = (NUM_XCDS, num_pid_h, triton.cdiv(num_queries, NUM_XCDS))
    _sparse_attn_prefill_kernel[grid](
        q,
        kv,
        indices,
        indptr,
        attn_sink,
        out,
        q.stride(0), q.stride(1), q.stride(2),
        kv.stride(0), kv.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        num_queries,
        num_heads,
        head_dim,
        kv.shape[0],
        float(scale),
        HAS_ATTN_SINK=has_attn_sink,
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,
        BLOCK_K=BLOCK_K,
        NUM_XCDS=NUM_XCDS,
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# Decode kernel (fp8_ds_mla paged cache, dual ragged passes)
# ---------------------------------------------------------------------------


@gluon.jit
def _decode_core_attn(
    cache_ptr,
    indices_ptr,
    seg_start,
    seg_len,
    cache_stride0,
    block_size,
    num_rows,
    q_nope_dot,
    q_rope_dot,
    scale,
    m_i,
    l_i,
    acc_nope,
    acc_rope,
    smem_knT,
    smem_krT,
    NOPE_DIM: gl.constexpr,
    NOPE_BLOCK: gl.constexpr,
    ROPE_DIM: gl.constexpr,
    BLOCK_H: gl.constexpr,
    BLOCK_K: gl.constexpr,
    IS_FNUZ: gl.constexpr,
):
    nw: gl.constexpr = gl.num_warps()
    mma_qk: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True,
        warps_per_cta=[nw, 1],
    )
    mma_pv: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16], transposed=True,
        warps_per_cta=[1, nw],
    )
    qk_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma_qk, k_width=8)
    pv_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma_pv, k_width=4)
    pv_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma_pv, k_width=4)

    blk_n: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16], threads_per_warp=[2, 32],
        warps_per_cta=[nw, 1], order=[1, 0],
    )
    blk_r: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 2], threads_per_warp=[2, 32],
        warps_per_cta=[nw, 1], order=[1, 0],
    )
    sl_k_n: gl.constexpr = gl.SliceLayout(1, blk_n)   # kpos rows (nope tile)
    sl_d_n: gl.constexpr = gl.SliceLayout(0, blk_n)   # nope dim cols
    sl_k_r: gl.constexpr = gl.SliceLayout(1, blk_r)   # kpos rows (rope tile)
    sl_d_r: gl.constexpr = gl.SliceLayout(0, blk_r)   # rope dim cols
    sl_k_qk: gl.constexpr = gl.SliceLayout(0, mma_qk)
    sl_h_pv: gl.constexpr = gl.SliceLayout(1, mma_pv)

    nope_off = gl.arange(0, NOPE_BLOCK, layout=sl_d_n)
    nope_mask = nope_off < NOPE_DIM
    nope_grp = nope_off // 64
    rope_off = gl.arange(0, ROPE_DIM, layout=sl_d_r)
    k_off_n = gl.arange(0, BLOCK_K, layout=sl_k_n)
    k_off_r = gl.arange(0, BLOCK_K, layout=sl_k_r)

    for k_start in tl.range(0, seg_len, BLOCK_K):
        # --- nope tile (fp8 dequant) ---
        kpos = k_start + k_off_n
        slot = gl.load(indices_ptr + seg_start + kpos, mask=kpos < seg_len, other=-1)
        valid = (kpos < seg_len) & (slot >= 0) & (slot < num_rows)
        safe = gl.where(valid, slot, 0)
        block_off = (safe // block_size).to(gl.int64) * cache_stride0
        pos_in_block = safe % block_size
        token_ptr = cache_ptr + block_off + pos_in_block * 576
        scale_ptr = cache_ptr + block_off + block_size * 576 + pos_in_block * 8

        x_u8 = gl.load(
            token_ptr[:, None] + nope_off[None, :],
            mask=valid[:, None] & nope_mask[None, :],
            other=0,
        )
        if IS_FNUZ:
            x_fp8 = x_u8.to(gl.float8e4b15, bitcast=True)
        else:
            x_fp8 = x_u8.to(gl.float8e4nv, bitcast=True)
        enc = gl.load(
            scale_ptr[:, None] + nope_grp[None, :],
            mask=valid[:, None] & nope_mask[None, :],
            other=127,
        )
        scales = gl.exp2(enc.to(gl.float32) - 127.0)
        k_nope = x_fp8.to(gl.bfloat16) * scales.to(gl.bfloat16)
        zero_n = gl.zeros([BLOCK_K, NOPE_BLOCK], dtype=gl.bfloat16, layout=blk_n)
        k_nope = gl.where(valid[:, None] & nope_mask[None, :], k_nope, zero_n)
        k_nope = gl.where(k_nope == k_nope, k_nope, zero_n)

        # --- rope tile (bf16) ---
        kpos_r = k_start + k_off_r
        slot_r = gl.load(
            indices_ptr + seg_start + kpos_r, mask=kpos_r < seg_len, other=-1
        )
        valid_r = (kpos_r < seg_len) & (slot_r >= 0) & (slot_r < num_rows)
        safe_r = gl.where(valid_r, slot_r, 0)
        block_off_r = (safe_r // block_size).to(gl.int64) * cache_stride0
        token_off_r = block_off_r + (safe_r % block_size) * 576 + NOPE_DIM
        rope_base = (cache_ptr + token_off_r).to(gl.pointer_type(gl.bfloat16))
        k_rope = gl.load(
            rope_base[:, None] + rope_off[None, :],
            mask=valid_r[:, None],
            other=0.0,
        )
        zero_r = gl.zeros([BLOCK_K, ROPE_DIM], dtype=gl.bfloat16, layout=blk_r)
        k_rope = gl.where(valid_r[:, None], k_rope, zero_r)
        k_rope = gl.where(k_rope == k_rope, k_rope, zero_r)

        # --- stage K tiles transposed [D, K] in shared memory (single buffer
        # per tensor): QK reads K^T directly, PV reads the permuted view ---
        smem_knT.store(gl.permute(k_nope, [1, 0]))
        smem_krT.store(gl.permute(k_rope, [1, 0]))

        valid_qk = gl.convert_layout(valid, sl_k_qk)

        # --- QK: scores = q_nope @ k_nope^T + q_rope @ k_rope^T ---
        scores = gl.zeros([BLOCK_H, BLOCK_K], dtype=gl.float32, layout=mma_qk)
        scores = gl.amd.cdna4.mfma(q_nope_dot, smem_knT.load(layout=qk_b), scores)
        scores = gl.amd.cdna4.mfma(q_rope_dot, smem_krT.load(layout=qk_b), scores)
        scores = scores * scale
        scores = gl.where(valid_qk[None, :], scores, float("-inf"))

        # --- online softmax ---
        m_block = gl.max(scores, axis=1)
        m_new = gl.maximum(m_i, m_block)
        alpha = gl.exp(m_i - m_new)
        p = gl.exp(scores - m_new[:, None])
        p = gl.where(valid_qk[None, :], p, 0.0)
        l_i = l_i * alpha + gl.sum(p, axis=1)
        m_i = m_new

        # --- PV (V == K; permute [D, K] -> [K, D] for the PV operand) ---
        p_dot = gl.convert_layout(p.to(gl.bfloat16), pv_a)
        alpha_pv = gl.convert_layout(alpha, sl_h_pv)
        acc_nope = acc_nope * alpha_pv[:, None]
        acc_nope = gl.amd.cdna4.mfma(p_dot, smem_knT.permute([1, 0]).load(layout=pv_b), acc_nope)
        acc_rope = acc_rope * alpha_pv[:, None]
        acc_rope = gl.amd.cdna4.mfma(p_dot, smem_krT.permute([1, 0]).load(layout=pv_b), acc_rope)

    return m_i, l_i, acc_nope, acc_rope


@gluon.jit
def _sparse_attn_decode_kernel(
    q_ptr,              # [T, H, D]  (D = NOPE_DIM + ROPE_DIM)
    main_cache_ptr,     # [num_blocks, block_size, 584]  packed bytes
    main_indices_ptr,   # [num_None_Zeros]
    main_indptr_ptr,    # [T + 1]
    extra_cache_ptr,    # [num_blocks, block_size, 584]  (used if HAS_EXTRA)
    extra_indices_ptr,  # [num_None_Zeros]
    extra_indptr_ptr,   # [T + 1]
    attn_sink_ptr,      # [H]
    out_ptr,            # [T, H, D]
    q_stride0,
    q_stride1,
    out_stride0,
    out_stride1,
    main_cache_stride0,
    extra_cache_stride0,
    main_num_rows,
    extra_num_rows,
    main_block_size,
    extra_block_size,
    scale,
    num_queries,
    num_heads,
    HAS_ATTN_SINK: gl.constexpr,
    HAS_EXTRA: gl.constexpr,
    NOPE_DIM: gl.constexpr,
    NOPE_BLOCK: gl.constexpr,
    ROPE_DIM: gl.constexpr,
    IS_FNUZ: gl.constexpr,
    BLOCK_H: gl.constexpr,
    BLOCK_K: gl.constexpr,
):
    # Persistent launch: a fixed pool of programs walks the (query, head-block)
    # tile space in a grid-stride loop instead of one program per tile.
    pid = gl.program_id(axis=0)
    num_programs = gl.num_programs(axis=0)

    nw: gl.constexpr = gl.num_warps()
    mma_qk: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True,
        warps_per_cta=[nw, 1],
    )
    mma_pv: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16], transposed=True,
        warps_per_cta=[1, nw],
    )
    qk_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma_qk, k_width=8)

    blk_n: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16], threads_per_warp=[2, 32],
        warps_per_cta=[nw, 1], order=[1, 0],
    )
    blk_r: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 2], threads_per_warp=[2, 32],
        warps_per_cta=[nw, 1], order=[1, 0],
    )
    # Transposed single-buffer KV with explicit conflict-free padded layouts
    # ([D, K], D contiguous + BLOCK_K bits with the 4<->8 swap, [[512,16]] pad):
    # QK reads K^T directly and PV reads the permuted view, so the duplicate
    # normal-orientation buffers (smem_kn/smem_kr) are dropped.
    if BLOCK_K == 32:
        sh_knT: gl.constexpr = gl.PaddedSharedLayout(interval_padding_pairs=[[512, 16]], offset_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], cga_layout=[], shape=[512, 32])
        sh_krT: gl.constexpr = gl.PaddedSharedLayout(interval_padding_pairs=[[512, 16]], offset_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 1], [0, 2], [0, 8], [0, 4], [0, 16]], cga_layout=[], shape=[64, 32])
    else:
        sh_knT: gl.constexpr = gl.PaddedSharedLayout(interval_padding_pairs=[[512, 16]], offset_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0], [0, 1], [0, 2], [0, 8], [0, 4]], cga_layout=[], shape=[512, 16])
        sh_krT: gl.constexpr = gl.PaddedSharedLayout(interval_padding_pairs=[[512, 16]], offset_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 1], [0, 2], [0, 8], [0, 4]], cga_layout=[], shape=[64, 16])

    sl_h_n: gl.constexpr = gl.SliceLayout(1, blk_n)
    sl_d_n: gl.constexpr = gl.SliceLayout(0, blk_n)
    sl_h_r: gl.constexpr = gl.SliceLayout(1, blk_r)
    sl_d_r: gl.constexpr = gl.SliceLayout(0, blk_r)
    sl_h_qk: gl.constexpr = gl.SliceLayout(1, mma_qk)
    sl_h_pv: gl.constexpr = gl.SliceLayout(1, mma_pv)
    sl_d_pv: gl.constexpr = gl.SliceLayout(0, mma_pv)

    # --- q async-copy (global -> LDS) layouts (mirror the prefill `shared_q`) ---
    # 16-byte (8xbf16) async-copy granule; D (dim1) contiguous, BLOCK_H rows.
    blk_qn: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[1, 64],
        warps_per_cta=[nw, 1], order=[1, 0],
    )
    blk_qr: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1], threads_per_warp=[1, 64],
        warps_per_cta=[nw, 1], order=[1, 0],
    )
    sl_h_qn: gl.constexpr = gl.SliceLayout(1, blk_qn)
    sl_d_qn: gl.constexpr = gl.SliceLayout(0, blk_qn)
    sl_h_qr: gl.constexpr = gl.SliceLayout(1, blk_qr)
    sl_d_qr: gl.constexpr = gl.SliceLayout(0, blk_qr)
    sh_qn: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs=[[512, 16]],
        offset_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256],
                      [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]],
        cga_layout=[],
        shape=[64, 512],
    )

    num_pid_h = (num_heads + BLOCK_H - 1) // BLOCK_H
    total_tiles = num_queries * num_pid_h

    # ---- persistent grid-stride loop over (query, head-block) tiles -----
    # Flattened tile id -> (query_idx, pid_h); BLOCK_H=64 -> num_pid_h = 2 for
    # the 128-head DSV4 shape.
    for tile_id in tl.range(pid, total_tiles, num_programs):
        query_idx = tile_id // num_pid_h
        pid_h = tile_id % num_pid_h

        # --- load q (nope + rope) via async global->LDS copy, then ds_read to
        # the qk_a dot operand (mirrors the prefill q staging). q's head row is
        # [nope(448) | rope(64)] contiguous, so q_nope reads the full 512-wide
        # row -- cols 448:512 hold q's rope values but multiply k_nope==0 there. ---
        head_n = pid_h * BLOCK_H + gl.arange(0, BLOCK_H, layout=sl_h_qn)
        head_r = pid_h * BLOCK_H + gl.arange(0, BLOCK_H, layout=sl_h_qr)
        nope_off = gl.arange(0, NOPE_BLOCK, layout=sl_d_qn)
        rope_off = gl.arange(0, ROPE_DIM, layout=sl_d_qr)

        q_base = q_ptr + query_idx * q_stride0
        buf_qn = gl.allocate_shared_memory(q_ptr.dtype.element_ty, [BLOCK_H, NOPE_BLOCK], sh_qn)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            buf_qn, q_base,
            head_n[:, None] * q_stride1 + nope_off[None, :],
            mask=(head_n < num_heads)[:, None],
        )
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.wait_group(0)
        q_nope_dot = gl.amd.cdna4.async_copy.load_shared_relaxed(buf_qn, qk_a)
        # q_rope is only 64-wide; a 64-wide LDS->qk_a ds-read does not lower, and
        # the async copy of the tiny tile buys nothing, so keep it a direct load.
        q_rope = gl.amd.cdna4.buffer_load(
            ptr=q_base,
            offsets=head_r[:, None] * q_stride1 + NOPE_DIM + rope_off[None, :],
            mask=(head_r < num_heads)[:, None],
            other=0.0,
        )
        q_rope_dot = gl.convert_layout(q_rope, qk_a)

        # --- running softmax state ---
        m_i = gl.full([BLOCK_H], float("-inf"), dtype=gl.float32, layout=sl_h_qk)
        l_i = gl.zeros([BLOCK_H], dtype=gl.float32, layout=sl_h_qk)
        acc_nope = gl.zeros([BLOCK_H, NOPE_BLOCK], dtype=gl.float32, layout=mma_pv)
        acc_rope = gl.zeros([BLOCK_H, ROPE_DIM], dtype=gl.float32, layout=mma_pv)

        # --- shared staging buffers (transposed single buffer, reused both passes) ---
        smem_knT = gl.allocate_shared_memory(gl.bfloat16, [NOPE_BLOCK, BLOCK_K], sh_knT)
        smem_krT = gl.allocate_shared_memory(gl.bfloat16, [ROPE_DIM, BLOCK_K], sh_krT)

        main_start = gl.load(main_indptr_ptr + query_idx)
        main_end = gl.load(main_indptr_ptr + query_idx + 1)
        m_i, l_i, acc_nope, acc_rope = _decode_core_attn(
            main_cache_ptr, main_indices_ptr, main_start, main_end - main_start,
            main_cache_stride0, main_block_size, main_num_rows,
            q_nope_dot, q_rope_dot, scale,
            m_i, l_i, acc_nope, acc_rope,
            smem_knT, smem_krT,
            NOPE_DIM, NOPE_BLOCK, ROPE_DIM, BLOCK_H, BLOCK_K, IS_FNUZ,
        )

        if HAS_EXTRA:
            extra_start = gl.load(extra_indptr_ptr + query_idx)
            extra_end = gl.load(extra_indptr_ptr + query_idx + 1)
            m_i, l_i, acc_nope, acc_rope = _decode_core_attn(
                extra_cache_ptr, extra_indices_ptr, extra_start, extra_end - extra_start,
                extra_cache_stride0, extra_block_size, extra_num_rows,
                q_nope_dot, q_rope_dot, scale,
                m_i, l_i, acc_nope, acc_rope,
                smem_knT, smem_krT,
                NOPE_DIM, NOPE_BLOCK, ROPE_DIM, BLOCK_H, BLOCK_K, IS_FNUZ,
            )

        # --- finalize ---
        head_q = pid_h * BLOCK_H + gl.arange(0, BLOCK_H, layout=sl_h_qk)
        if HAS_ATTN_SINK:
            sink = gl.load(
                attn_sink_ptr + head_q, mask=head_q < num_heads, other=float("-inf")
            ).to(gl.float32)
            m_final = gl.maximum(m_i, sink)
            alpha = gl.exp(m_i - m_final)
            l_final = l_i * alpha + gl.exp(sink - m_final)
            denom = gl.maximum(l_final, 1.0e-30)
            scale_row = alpha / denom
            guard = l_final > 0.0
        else:
            denom = gl.maximum(l_i, 1.0e-30)
            scale_row = 1.0 / denom
            guard = l_i > 0.0

        scale_pv = gl.convert_layout(scale_row, sl_h_pv)
        guard_pv = gl.convert_layout(guard, sl_h_pv)
        out_nope = gl.where(guard_pv[:, None], acc_nope * scale_pv[:, None], 0.0)
        out_rope = gl.where(guard_pv[:, None], acc_rope * scale_pv[:, None], 0.0)

        out_head = pid_h * BLOCK_H + gl.arange(0, BLOCK_H, layout=sl_h_pv)
        nope_off_pv = gl.arange(0, NOPE_BLOCK, layout=sl_d_pv)
        rope_off_pv = gl.arange(0, ROPE_DIM, layout=sl_d_pv)
        out_base = out_ptr + query_idx * out_stride0
        gl.amd.cdna4.buffer_store(
            out_nope.to(out_ptr.dtype.element_ty),
            ptr=out_base,
            offsets=out_head[:, None] * out_stride1 + nope_off_pv[None, :],
            mask=(out_head < num_heads)[:, None] & (nope_off_pv < NOPE_DIM)[None, :],
        )
        gl.amd.cdna4.buffer_store(
            out_rope.to(out_ptr.dtype.element_ty),
            ptr=out_base,
            offsets=out_head[:, None] * out_stride1 + NOPE_DIM + rope_off_pv[None, :],
            mask=(out_head < num_heads)[:, None],
        )


def sparse_attn_decode_gluon(
    q,                  # [T, H, D] bf16 queries (D = NOPE_DIM + ROPE_DIM)
    main_cache,         # [num_blocks, block_size, row_bytes] uint8 fp8_ds_mla cache
    main_indices,       # [nnz] int32 ragged slot ids
    main_indptr,        # [T + 1] int32 ragged offsets
    out,                # [T, H, D] output (filled in place)
    scale,              # softmax scale
    extra_cache=None,   # optional second ragged pass (top-k) cache
    extra_indices=None,
    extra_indptr=None,
    attn_sink=None,     # optional [H] fp32 per-head sink bias
    nope_dim=448,
    rope_dim=64,
    is_fnuz=False,
    num_sms=None,       # persistent program count (defaults to device CU count)
):
    """Host launcher for the persistent DSV4 sparse-MLA decode kernel.

    Mirrors ``sparse_attn_prefill_gluon``: the kernel is persistent, so a fixed
    pool of ``num_sms`` programs grid-strides over the (query, head-block) tile
    space and the launch grid is 1-D instead of ``(num_queries, num_pid_h)``.
    The autotuned search is removed -- the best config is hard-coded here
    (BLOCK_H=64, BLOCK_K=32, num_warps=8, num_ctas=1).

    The optional ``extra_*`` arguments drive the second ragged pass (top-k);
    when omitted the kernel runs the single (main / SWA) pass only.
    """
    assert q.ndim == 3, f"expected q=[T,H,D], got {tuple(q.shape)}"
    assert main_cache.ndim == 3, (
        f"expected main_cache=[blocks,block,bytes], got {tuple(main_cache.shape)}"
    )
    assert q.is_cuda and main_cache.is_cuda, "q/main_cache must be on the GPU"

    num_queries, num_heads, head_dim = q.shape
    nope_block = triton.next_power_of_2(nope_dim)

    has_attn_sink = attn_sink is not None
    if attn_sink is None:
        attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)

    has_extra = (
        extra_cache is not None
        and extra_indices is not None
        and extra_indptr is not None
    )
    if not has_extra:
        # Pass valid (but unused-by-dead-code) pointers so the HAS_EXTRA
        # constexpr branch in the kernel never materializes.
        extra_cache = main_cache
        extra_indices = torch.empty(0, device=q.device, dtype=torch.int32)
        extra_indptr = torch.zeros(num_queries + 1, device=q.device, dtype=torch.int32)

    if num_sms is None:
        num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count

    BLOCK_H = 64
    BLOCK_K = 32

    # Persistent 1-D grid: cap the program count at the number of tiles so small
    # problems never launch idle workgroups.
    num_pid_h = triton.cdiv(num_heads, BLOCK_H)
    grid = (min(num_sms, num_queries * num_pid_h),)
    _sparse_attn_decode_kernel[grid](
        q,
        main_cache,
        main_indices,
        main_indptr,
        extra_cache,
        extra_indices,
        extra_indptr,
        attn_sink,
        out,
        q.stride(0), q.stride(1),
        out.stride(0), out.stride(1),
        main_cache.stride(0),
        extra_cache.stride(0),
        main_cache.shape[0] * main_cache.shape[1],
        extra_cache.shape[0] * extra_cache.shape[1],
        main_cache.shape[1],
        extra_cache.shape[1],
        float(scale),
        num_queries,
        num_heads,
        HAS_ATTN_SINK=has_attn_sink,
        HAS_EXTRA=has_extra,
        NOPE_DIM=nope_dim,
        NOPE_BLOCK=nope_block,
        ROPE_DIM=rope_dim,
        IS_FNUZ=is_fnuz,
        BLOCK_H=BLOCK_H,
        BLOCK_K=BLOCK_K,
        num_warps=8,
    )
    return out
