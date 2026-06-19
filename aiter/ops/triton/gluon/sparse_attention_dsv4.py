# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import math
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
    smem_k_nope,
    smem_k_pe,
    kv_buf_n,
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

    # rope async-copy ([ROPE_DIM, BLOCK_K], D rows contiguous 8xbf16=16B). This
    # triton requires buffer_load_to_shared offsets in a Blocked/Slice layout
    # (NOT a DistributedLinearLayout), matching the in-file q async-copy pattern.
    blk_kr: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[8, 1], threads_per_warp=[8, 8],
        warps_per_cta=[1, nw], order=[0, 1],
    )
    sl_d_kr: gl.constexpr = gl.SliceLayout(1, blk_kr)   # rope dim rows
    sl_k_kr: gl.constexpr = gl.SliceLayout(0, blk_kr)   # kpos cols (slot)

    gl.static_assert(nw == 4, "decode core async layouts assume num_warps == 4")
    # slot-id async gather (1-D 64-wide DLL; nw=4 -> 2 redundant warp bases): one
    # global gather into shared, read back in both the nope (sl_k_n) and rope
    # (sl_k_kr) views instead of two separate sync index loads.
    dll_slot: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[], lane_bases=[[1], [2], [4], [8], [16], [32]],
        warp_bases=[[0], [0]], block_bases=[], shape=[BLOCK_K],
    )
    sh_slot: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[0],
    )
    # bf16 dequant target for the nope tile (MFMA reads it as qk_b / pv_b).
    sh_knT: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs=[[512, 16]],
        offset_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0],
                      [128, 0], [256, 0], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]],
        cga_layout=[], shape=[NOPE_BLOCK, BLOCK_K],
    )

    nope_off = gl.arange(0, NOPE_BLOCK, layout=sl_d_n)
    nope_mask = nope_off < NOPE_DIM
    nope_grp = nope_off // 64
    k_off_n = gl.arange(0, BLOCK_K, layout=sl_k_n)
    k_off_kr = gl.arange(0, BLOCK_K, layout=sl_k_kr)
    d_kr = gl.arange(0, ROPE_DIM, layout=sl_d_kr)
    g_off = gl.arange(0, BLOCK_K, layout=dll_slot)
    krT_base = cache_ptr.to(gl.pointer_type(gl.bfloat16))
    if IS_FNUZ:
        knT_base = cache_ptr.to(gl.pointer_type(gl.float8e4b15))
    else:
        knT_base = cache_ptr.to(gl.pointer_type(gl.float8e4nv))
    # slot ids double-buffered (WAR-free reuse across tiles); bf16 nope target single.
    smem_slot = gl.allocate_shared_memory(indices_ptr.dtype.element_ty, [2, BLOCK_K], sh_slot)
    smem_kn = gl.allocate_shared_memory(gl.bfloat16, [NOPE_BLOCK, BLOCK_K], sh_knT)
    zero_n = gl.zeros([BLOCK_K, NOPE_BLOCK], dtype=gl.bfloat16, layout=blk_n)

    num_iter = (seg_len + BLOCK_K - 1) // BLOCK_K
    # Clamp to >= 2 so the loop is non-empty and async copies stay on a single
    # straight-line path (branching on num_iter trips the SIInsertWaitcnts merge
    # crash). Tiles past num_iter mask to all-invalid and contribute nothing.
    # eff_iter = gl.maximum(num_iter, 2)

    # Main Loop: GL slot[i+2]
    #            LL KV_nope[i] + Load Enc 
    #            K_nope Dequant + MFMA QK_nope
    #            LL KV_rope[i] + MFMA QK_rope
    #            LS K_nope_bf16
    #
    #            LL slot[i+1]_nope + GL KV[i+1]_nope
    #            LL slot[i+1]_rope + GL KV[i+1]_rope
    #            LL K_nope_bf16[i] +  MFMA PV_nope
    #            LL K_rope[i] +  MFMA PV_rope

    # for t in tl.range(0, eff_iter):
    for t in tl.range(0, num_iter):
        kv_buf = (kv_buf_n + t) % 2

        # --- async slot[t] gather: one gather -> shared, read back in both views ---
        gpos = t * BLOCK_K + g_off
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smem_slot.index(kv_buf), indices_ptr + seg_start, gpos, mask=gpos < seg_len,
        )
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.wait_group(0)
        slot_n = gl.amd.cdna4.async_copy.load_shared_relaxed(smem_slot.index(kv_buf), sl_k_n)
        slot_kr = gl.amd.cdna4.async_copy.load_shared_relaxed(smem_slot.index(kv_buf), sl_k_kr)

        # --- nope view: valid + issue this tile's async fp8 nope gather ---
        kpos = t * BLOCK_K + k_off_n
        valid = (kpos < seg_len) & (slot_n >= 0) & (slot_n < num_rows)
        safe = gl.where(valid, slot_n, 0)
        tok_n = ((safe // block_size).to(gl.int64) * cache_stride0 + (safe % block_size) * 576).to(gl.int32)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smem_k_nope.index(kv_buf), knT_base, tok_n[:, None] + nope_off[None, :],
            mask=valid[:, None] & nope_mask[None, :],
        )
        gl.amd.cdna4.async_copy.commit_group()

        # --- rope view: sync k_pe gather (overlaps the fp8 dma) ---
        kpos_kr = t * BLOCK_K + k_off_kr
        valid_kr = (kpos_kr < seg_len) & (slot_kr >= 0) & (slot_kr < num_rows)
        safe_kr = gl.where(valid_kr, slot_kr, 0)
        rope_n = ((safe_kr // block_size).to(gl.int64) * cache_stride0 + (safe_kr % block_size) * 576 + NOPE_DIM) // 2
        k_pe = gl.amd.cdna4.buffer_load(
            ptr=krT_base, offsets=rope_n[None, :].to(gl.int32) + d_kr[:, None],
        )
        smem_k_pe.index(kv_buf).store(k_pe)

        # --- per-group dequant scales (overlaps the fp8 dma) ---
        block_off = (safe // block_size).to(gl.int64) * cache_stride0
        scale_ptr = cache_ptr + block_off + block_size * 576 + (safe % block_size) * 8
        enc = gl.load(
            scale_ptr[:, None] + nope_grp[None, :],
            mask=valid[:, None] & nope_mask[None, :], other=127,
        )
        scales = gl.exp2(enc.to(gl.float32) - 127.0)

        # --- wait the fp8 dma, dequant into the bf16 nope target ---
        gl.amd.cdna4.async_copy.wait_group(0)
        k_nope_fp8 = smem_k_nope.index(kv_buf).load(layout=blk_n)
        k_nope = k_nope_fp8.to(gl.bfloat16) * scales.to(gl.bfloat16)
        # invalid cols -> 0 (else PV's 0*NaN=NaN) and pad dims [NOPE_DIM,NOPE_BLOCK) -> 0.
        k_nope = gl.where(valid[:, None] & nope_mask[None, :], k_nope, zero_n)
        smem_kn.store(gl.permute(k_nope, [1, 0]))

        # --- QK ---
        valid_qk = gl.convert_layout(valid, sl_k_qk)
        scores = gl.zeros([BLOCK_H, BLOCK_K], dtype=gl.float32, layout=mma_qk)
        scores = gl.amd.cdna4.mfma(q_nope_dot, smem_kn.load(layout=qk_b), scores)
        scores = gl.amd.cdna4.mfma(q_rope_dot, smem_k_pe.index(kv_buf).load(layout=qk_b), scores)
        scores = scores * scale
        scores = gl.where(valid_qk[None, :], scores, float("-inf"))

        # --- softmax + PV ---
        m_block = gl.max(scores, axis=1)
        m_new = gl.maximum(m_i, m_block)
        alpha = gl.exp(m_i - m_new)
        p = gl.exp(scores - m_new[:, None])
        p = gl.where(valid_qk[None, :], p, 0.0)
        l_i = l_i * alpha + gl.sum(p, axis=1)
        m_i = m_new
        p_dot = gl.convert_layout(p.to(gl.bfloat16), pv_a)
        alpha_pv = gl.convert_layout(alpha, sl_h_pv)
        acc_nope = acc_nope * alpha_pv[:, None]
        acc_nope = gl.amd.cdna4.mfma(p_dot, smem_kn.permute([1, 0]).load(layout=pv_b), acc_nope)
        acc_rope = acc_rope * alpha_pv[:, None]
        acc_rope = gl.amd.cdna4.mfma(p_dot, smem_k_pe.index(kv_buf).permute([1, 0]).load(layout=pv_b), acc_rope)

    return m_i, l_i, acc_nope, acc_rope

# ---------------------------------------------------------------------------
# Decode kernel — flash-decode (split-K) variant
# ---------------------------------------------------------------------------
# The single-pass decode kernel above launches only num_queries * heads_blocks
# workgroups, under-filling the device at low batch (the latency-bound regime).
# The partial kernel adds a split axis: each (query, split, head-block) program
# walks a contiguous slice of the per-query main / extra segments and writes raw
# (m, l, acc) partials; the reduce kernel combines them (sink + normalize).
# The heavy per-tile work (fp8 dequant, QK/PV MFMA, online softmax) is shared
# with the single-pass kernel via _decode_core_attn.


@gluon.jit
def _sparse_attn_decode_partial_kernel(
    q_ptr,              # [T, H, D]  (D = NOPE_DIM + ROPE_DIM)
    main_cache_ptr,     # [num_blocks, block_size, 584]  packed bytes
    main_indices_ptr,   # [main_nnz]
    main_indptr_ptr,    # [T + 1]
    extra_cache_ptr,    # [num_blocks, block_size, 584]  (used if HAS_EXTRA)
    extra_indices_ptr,  # [extra_nnz]
    extra_indptr_ptr,   # [T + 1]
    part_m_ptr,         # [T, NUM_SPLITS, H]            fp32
    part_l_ptr,         # [T, NUM_SPLITS, H]            fp32
    part_acc_ptr,       # [T, NUM_SPLITS, H, COMB_DIM]  fp32
    q_stride0,
    q_stride1,
    main_cache_stride0,
    extra_cache_stride0,
    pm_stride0,
    pm_stride_s,
    pa_stride0,
    pa_stride_s,
    pa_stride_h,
    main_num_rows,
    extra_num_rows,
    main_block_size,
    extra_block_size,
    scale,
    num_heads,
    HAS_EXTRA: gl.constexpr,
    NOPE_DIM: gl.constexpr,
    NOPE_BLOCK: gl.constexpr,
    ROPE_DIM: gl.constexpr,
    IS_FNUZ: gl.constexpr,
    BLOCK_H: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_SPLITS: gl.constexpr,
):
    # Non-persistent 3-D grid: (query, split, head-block).
    query_idx = gl.program_id(axis=0)
    split_id = gl.program_id(axis=1)
    pid_h = gl.program_id(axis=2)

    nw: gl.constexpr = gl.num_warps()
    gl.static_assert(nw == 4, "decode core slot async gather assumes num_warps == 4")

    mma_qk: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True,
        warps_per_cta=[nw, 1],
    )
    mma_pv: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16], transposed=True,
        warps_per_cta=[1, nw],
    )
    qk_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma_qk, k_width=8)

    # Transposed single-buffer deq-KV staging layouts ([D, BLOCK_K], D contiguous,
    # [[512,16]] pad). BLOCK_H=64, BLOCK_K=64 only: QK reads K^T directly and PV
    # reads the permuted view, so no normal-orientation duplicate is needed.
    gl.static_assert(BLOCK_K == 64, "decode partial kernel is BLOCK_K=64 only")
    sh_slot: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
    # sh_knT: gl.constexpr = gl.PaddedSharedLayout(interval_padding_pairs=[[512, 16]], offset_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], cga_layout=[], shape=[512, 64])
    sh_kn_fp8: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=1, max_phase=1, order=[1, 0],
    )
    sh_krT: gl.constexpr = gl.PaddedSharedLayout(interval_padding_pairs=[[512, 16]], offset_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], cga_layout=[], shape=[64, 64])

    sl_h_qk: gl.constexpr = gl.SliceLayout(1, mma_qk)
    sl_h_pv: gl.constexpr = gl.SliceLayout(1, mma_pv)
    sl_d_pv: gl.constexpr = gl.SliceLayout(0, mma_pv)

    # --- q async-copy (global -> LDS) layouts (mirror the single-pass kernel) ---
    blk_qn: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[1, 64],
        warps_per_cta=[nw, 1], order=[1, 0],
    )
    # q_pe (rope) async-copy: 16-byte granule [1,8] over [BLOCK_H, ROPE_DIM=64].
    blk_qr: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[8, 8],
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
    sh_qpe: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs=[[512, 16]],
        offset_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32],
                      [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]],
        cga_layout=[],
        shape=[64, 64],
    )

    # --- Global Load q (nope + rope) ---
    head_n = pid_h * BLOCK_H + gl.arange(0, BLOCK_H, layout=sl_h_qn)
    head_r = pid_h * BLOCK_H + gl.arange(0, BLOCK_H, layout=sl_h_qr)
    nope_off = gl.arange(0, NOPE_BLOCK, layout=sl_d_qn)
    rope_off = gl.arange(0, ROPE_DIM, layout=sl_d_qr)

    q_base = q_ptr + query_idx * q_stride0
    buf_qn = gl.allocate_shared_memory(q_ptr.dtype.element_ty, [BLOCK_H, NOPE_BLOCK], sh_qn)
    buf_qpe = gl.allocate_shared_memory(q_ptr.dtype.element_ty, [BLOCK_H, ROPE_DIM], sh_qpe)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        buf_qn, q_base,
        head_n[:, None] * q_stride1 + nope_off[None, :],
        mask=(head_n < num_heads)[:, None],
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        buf_qpe, q_base,
        head_r[:, None] * q_stride1 + NOPE_DIM + rope_off[None, :],
        mask=(head_r < num_heads)[:, None],
    )
    gl.amd.cdna4.async_copy.commit_group()

    # --- running softmax state ---
    m_i = gl.full([BLOCK_H], float("-inf"), dtype=gl.float32, layout=sl_h_qk)
    l_i = gl.zeros([BLOCK_H], dtype=gl.float32, layout=sl_h_qk)
    acc_nope = gl.zeros([BLOCK_H, NOPE_BLOCK], dtype=gl.float32, layout=mma_pv)
    acc_rope = gl.zeros([BLOCK_H, ROPE_DIM], dtype=gl.float32, layout=mma_pv)

    # --- main (SWA) pass over this split's contiguous slice ---
    main_start = gl.load(main_indptr_ptr + query_idx)
    main_end = gl.load(main_indptr_ptr + query_idx + 1)
    main_len = main_end - main_start
    main_chunk = (main_len + NUM_SPLITS - 1) // NUM_SPLITS
    main_lo = split_id * main_chunk
    main_hi = gl.minimum(main_lo + main_chunk, main_len)

    gl.amd.cdna4.async_copy.wait_group(0)
    q_nope_dot = gl.amd.cdna4.async_copy.load_shared_relaxed(buf_qn, qk_a)
    q_rope_dot = gl.amd.cdna4.async_copy.load_shared_relaxed(buf_qpe, qk_a)

    # smem_k_nope: raw fp8 nope staging (async target), double-buffered. smem_k_pe:
    # rope raw bf16, double-buffered. Slot ids + the bf16 dequant target are
    # allocated inside the core.
    k_nope_ty = gl.float8e4b15 if IS_FNUZ else gl.float8e4nv
    smem_k_nope = gl.allocate_shared_memory(k_nope_ty, [2, BLOCK_K, NOPE_BLOCK], sh_kn_fp8)
    smem_k_pe = gl.allocate_shared_memory(gl.bfloat16, [2, ROPE_DIM, BLOCK_K], sh_krT)

    # Prologue: GL slot[0]
    #           GL slot[1]
    #           LL Q_nope + LL Q_rope
    #           LL slot[0]_nope + GL KV_nope
    #           LL slot[0]_rope + GL KV_rope


    m_i, l_i, acc_nope, acc_rope = _decode_core_attn(
        main_cache_ptr, main_indices_ptr, main_start + main_lo, main_hi - main_lo,
        main_cache_stride0, main_block_size, main_num_rows,
        q_nope_dot, q_rope_dot, scale,
        m_i, l_i, acc_nope, acc_rope,
        smem_k_nope, smem_k_pe, 0,
        NOPE_DIM, NOPE_BLOCK, ROPE_DIM, BLOCK_H, BLOCK_K, IS_FNUZ,
    )

    # --- Extra (TopK) pass share the same core attention with main pass ---
    if HAS_EXTRA:
        extra_start = gl.load(extra_indptr_ptr + query_idx)
        extra_end = gl.load(extra_indptr_ptr + query_idx + 1)
        extra_len = extra_end - extra_start
        extra_chunk = (extra_len + NUM_SPLITS - 1) // NUM_SPLITS
        extra_lo = split_id * extra_chunk
        extra_hi = gl.minimum(extra_lo + extra_chunk, extra_len)
        # Continue the smem_k_pe slot parity from where the main pass left off.
        # TODO: No need since main cache is already finished?
        extra_kr_buf0 = ((main_hi - main_lo + BLOCK_K - 1) // BLOCK_K) % 2
        m_i, l_i, acc_nope, acc_rope = _decode_core_attn(
            extra_cache_ptr, extra_indices_ptr, extra_start + extra_lo, extra_hi - extra_lo,
            extra_cache_stride0, extra_block_size, extra_num_rows,
            q_nope_dot, q_rope_dot, scale,
            m_i, l_i, acc_nope, acc_rope,
            smem_k_nope, smem_k_pe, extra_kr_buf0,
            NOPE_DIM, NOPE_BLOCK, ROPE_DIM, BLOCK_H, BLOCK_K, IS_FNUZ,
        )

    # --- store raw partials (no sink, no normalize) ---
    head_q = pid_h * BLOCK_H + gl.arange(0, BLOCK_H, layout=sl_h_qk)
    pm_off = query_idx * pm_stride0 + split_id * pm_stride_s + head_q
    gl.amd.cdna4.buffer_store(m_i, ptr=part_m_ptr, offsets=pm_off, mask=head_q < num_heads)
    gl.amd.cdna4.buffer_store(l_i, ptr=part_l_ptr, offsets=pm_off, mask=head_q < num_heads)

    head_pv = pid_h * BLOCK_H + gl.arange(0, BLOCK_H, layout=sl_h_pv)
    nope_off_pv = gl.arange(0, NOPE_BLOCK, layout=sl_d_pv)
    rope_off_pv = gl.arange(0, ROPE_DIM, layout=sl_d_pv)
    acc_base = part_acc_ptr + query_idx * pa_stride0 + split_id * pa_stride_s
    gl.amd.cdna4.buffer_store(
        acc_nope,
        ptr=acc_base,
        offsets=head_pv[:, None] * pa_stride_h + nope_off_pv[None, :],
        mask=(head_pv < num_heads)[:, None] & (nope_off_pv < NOPE_DIM)[None, :],
    )
    gl.amd.cdna4.buffer_store(
        acc_rope,
        ptr=acc_base,
        offsets=head_pv[:, None] * pa_stride_h + NOPE_DIM + rope_off_pv[None, :],
        mask=(head_pv < num_heads)[:, None],
    )


@gluon.jit
def _sparse_attn_decode_reduce_kernel(
    part_m_ptr,         # [T, NUM_SPLITS, H]            fp32
    part_l_ptr,         # [T, NUM_SPLITS, H]            fp32
    part_acc_ptr,       # [T, NUM_SPLITS, H, COMB_DIM]  fp32
    attn_sink_ptr,      # [H]                           fp32
    out_ptr,            # [T, H, COMB_DIM]              bf16
    out_stride0,
    out_stride1,
    pm_stride0,
    pm_stride_s,
    pa_stride0,
    pa_stride_s,
    pa_stride_h,
    num_heads,
    HAS_ATTN_SINK: gl.constexpr,
    COMB_DIM: gl.constexpr,
    BLOCK_H: gl.constexpr,
    NUM_SPLITS: gl.constexpr,
):
    # Grid: (query, head-block). Combines the per-split partials for BLOCK_H
    # heads into the final output, applying the optional attn-sink and softmax
    # normalization. NUM_SPLITS is small (<= 16), so the split loop is unrolled.
    query_idx = gl.program_id(axis=0)
    pid_h = gl.program_id(axis=1)

    # Elementwise/reduction blocked layout for the [BLOCK_H, COMB_DIM] acc.
    blk_acc: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[BLOCK_H // gl.num_warps(), 8],
        threads_per_warp=[1, 64],
        warps_per_cta=[gl.num_warps(), 1],
        order=[1, 0],
    )
    sl_h: gl.constexpr = gl.SliceLayout(1, blk_acc)   # rows (heads)
    sl_c: gl.constexpr = gl.SliceLayout(0, blk_acc)   # cols (comb dim)

    head = pid_h * BLOCK_H + gl.arange(0, BLOCK_H, layout=sl_h)
    head_mask = head < num_heads
    comb = gl.arange(0, COMB_DIM, layout=sl_c)
    neg_large = -3.4028234663852886e38

    # Phase 1: global max over splits (+ optional sink).
    m_final = gl.full([BLOCK_H], neg_large, dtype=gl.float32, layout=sl_h)
    for s in tl.static_range(NUM_SPLITS):
        m_s = gl.amd.cdna4.buffer_load(
            ptr=part_m_ptr + query_idx * pm_stride0 + s * pm_stride_s,
            offsets=head, mask=head_mask, other=neg_large,
        )
        m_final = gl.maximum(m_final, m_s)
    if HAS_ATTN_SINK:
        sink = gl.amd.cdna4.buffer_load(
            ptr=attn_sink_ptr, offsets=head, mask=head_mask, other=neg_large,
        )
        m_final = gl.maximum(m_final, sink)

    # Phase 2: weighted sum of per-split (l, acc) into the m_final frame.
    l_final = gl.zeros([BLOCK_H], dtype=gl.float32, layout=sl_h)
    acc = gl.zeros([BLOCK_H, COMB_DIM], dtype=gl.float32, layout=blk_acc)
    for s in tl.static_range(NUM_SPLITS):
        m_s = gl.amd.cdna4.buffer_load(
            ptr=part_m_ptr + query_idx * pm_stride0 + s * pm_stride_s,
            offsets=head, mask=head_mask, other=neg_large,
        )
        l_s = gl.amd.cdna4.buffer_load(
            ptr=part_l_ptr + query_idx * pm_stride0 + s * pm_stride_s,
            offsets=head, mask=head_mask, other=0.0,
        )
        w_s = gl.exp(m_s - m_final)
        l_final = l_final + w_s * l_s
        acc_s = gl.amd.cdna4.buffer_load(
            ptr=part_acc_ptr + query_idx * pa_stride0 + s * pa_stride_s,
            offsets=head[:, None] * pa_stride_h + comb[None, :],
            mask=head_mask[:, None], other=0.0,
        )
        acc = acc + w_s[:, None] * acc_s

    if HAS_ATTN_SINK:
        l_final = l_final + gl.exp(sink - m_final)
    denom = gl.maximum(l_final, 1.0e-30)
    out = gl.where(l_final[:, None] > 0.0, acc / denom[:, None], 0.0)

    gl.amd.cdna4.buffer_store(
        out.to(out_ptr.dtype.element_ty),
        ptr=out_ptr + query_idx * out_stride0,
        offsets=head[:, None] * out_stride1 + comb[None, :],
        mask=head_mask[:, None],
    )


def _decode_partial_iters(
    avg_main_len: float, avg_extra_len: float, splits: int, block_k: int
) -> int:
    """BLOCK_K iterations one partial workgroup walks for ``splits`` splits.

    Each split processes ``ceil(seg_len / splits)`` tokens of a segment, walked
    ``BLOCK_K`` at a time, and the main/extra segments are handled separately.
    """
    main_iters = (
        math.ceil(math.ceil(avg_main_len / splits) / block_k) if avg_main_len > 0 else 0
    )
    extra_iters = (
        math.ceil(math.ceil(avg_extra_len / splits) / block_k)
        if avg_extra_len > 0
        else 0
    )
    return main_iters + extra_iters


def _decode_num_splits(
    num_queries: int,
    heads_blocks: int,
    main_indices: torch.Tensor,
    extra_indices: torch.Tensor,
    has_extra: bool,
    block_k: int = 32,
    max_splits: int = 8,
    num_cus: int = 256,
) -> int:
    """Pick a flash-decode split count for the gather-latency-bound decode.

    Decode launches only ``num_queries * heads_blocks`` workgroups, which
    under-fills the device in the low-batch regime that dominates latency.
    Splitting the per-query KV segment across workgroups adds memory-level
    parallelism (more concurrent gathers), which is the right lever for this
    HBM-gather-latency-bound kernel.

    But each split also writes a full fp32 partial accumulator
    (``BLOCK_H x COMB_DIM`` per workgroup) to HBM that the reduce kernel reads
    back, so the overhead grows with the split count. The cost model works in
    ``BLOCK_K``-iteration units::

        cost(s) = waves(s) * (iters_per_split(s) + MU) + NU * (s - 1)

    where ``waves = ceil(base * s / CU)``, ``iters_per_split`` is the real
    per-workgroup BLOCK_K iteration count (``_decode_partial_iters``), ``MU``
    charges per-wave launch/tail latency, and ``NU`` charges the per-split
    reduce + partial-accumulator HBM traffic. Splitting is chosen only when the
    iteration drop outweighs the extra-wave and reduce overhead; ties favour
    fewer splits. This avoids splitting short segments (where the gather is too
    cheap to offset the reduce overhead) and over-splitting tiny batches (where
    the partial-acc HBM traffic dominates) — both measured net losses on gfx950.
    Returns 1 (use the single-pass kernel, no reduce) when no split helps.

    Tuned on gfx950 against the split-vs-single-pass decode sweep.
    """
    inv_q = 1.0 / max(1, num_queries)
    avg_main_len = main_indices.numel() * inv_q
    avg_extra_len = (extra_indices.numel() * inv_q) if has_extra else 0.0
    if avg_main_len <= 0.0 and avg_extra_len <= 0.0:
        return 1
    # Minimum per-query work to split at all. Short segments are too cheap to
    # gather for the split to offset the second kernel launch + partial-acc HBM
    # round-trip: measured net loss on gfx950 once the single-pass per-query
    # iteration count drops below ~16 BLOCK_K tiles (e.g. topk<=256).
    MIN_ITERS_TO_SPLIT = 16
    if _decode_partial_iters(avg_main_len, avg_extra_len, 1, block_k) < MIN_ITERS_TO_SPLIT:
        return 1
    base = max(1, num_queries * heads_blocks)
    cu = max(1, num_cus)
    MU = 1.0  # per-wave launch/tail latency penalty
    NU = 4.0  # per-extra-split reduce + partial-accumulator HBM penalty
    best_splits = 1
    best_cost = None
    for splits in range(1, max_splits + 1):
        waves = (base * splits + cu - 1) // cu
        iters = _decode_partial_iters(avg_main_len, avg_extra_len, splits, block_k)
        cost = waves * (iters + MU) + NU * (splits - 1)
        if best_cost is None or cost < best_cost - 1e-9:
            best_splits = splits
            best_cost = cost
    return best_splits


def sparse_attn_decode_gluon(
    q,                  # [T, H, D] bf16 queries (D = NOPE_DIM + ROPE_DIM)
    main_cache,         # [num_blocks, block_size, row_bytes] uint8 fp8_ds_mla cache
    main_indices,       # [nnz] int32 ragged slot ids
    main_indptr,        # [T + 1] int32 ragged offsets
    out,                # [T, H, D] output (filled in place)
    scale,              # softmax scale
    num_splits=None,    # flash-decode split count (chosen by the caller)
    extra_cache=None,   # optional second ragged pass (top-k) cache
    extra_indices=None,
    extra_indptr=None,
    attn_sink=None,     # optional [H] fp32 per-head sink bias
    nope_dim=448,
    rope_dim=64,
    is_fnuz=False,
    num_warps=4,        # partial-kernel warps (8 -> 2 waves/SIMD on gfx950)
):
    """Host launcher for the flash-decode (split-K) DSV4 sparse-MLA decode.

    Stage 1 (_sparse_attn_decode_partial_kernel): a (query, split, head-block)
    grid writes per-split raw (m, l, acc) partials. Stage 2
    (_sparse_attn_decode_reduce_kernel) combines them (sink + normalize) into
    ``out``. Mirrors mla_decode_gluon's split + reduce structure.
    """
    assert q.ndim == 3, f"expected q=[T,H,D], got {tuple(q.shape)}"
    assert main_cache.ndim == 3, (
        f"expected main_cache=[blocks,block,bytes], got {tuple(main_cache.shape)}"
    )
    assert nope_dim == 448, f"DSv4 sparse_attn_decode_gluon requires nope_dim=448, got {nope_dim=}"
    assert rope_dim == 64, f"DSv4 sparse_attn_decode_gluon requires rope_dim=448, got {rope_dim=}"

    num_queries, num_heads, head_dim = q.shape
    assert head_dim == 512, f"DSv4 sparse_attn_decode_gluon requires Q head_dim=512, got {head_dim=}"

    nope_block = triton.next_power_of_2(nope_dim)
    comb_dim = nope_dim + rope_dim

    has_attn_sink = attn_sink is not None
    if attn_sink is None:
        attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)

    has_extra = (
        extra_cache is not None
        and extra_indices is not None
        and extra_indptr is not None
    )
    if has_extra:
        assert extra_cache is not None
        assert extra_indices is not None
        assert extra_indptr is not None
        assert extra_indices.ndim == 1, (
            f"expected extra_indices=[nnz], got {extra_indices.shape}"
        )
        assert extra_indptr.ndim == 1, (
            f"expected extra_indptr=[b+1], got {extra_indptr.shape}"
        )
        assert extra_indptr.numel() == num_queries + 1, (
            f"expected extra_indptr shape [{num_queries + 1}], got {extra_indptr.shape}"
        )
    else:
        extra_cache = main_cache
        extra_indices = torch.empty(0, device=q.device, dtype=torch.int32)
        extra_indptr = torch.zeros(num_queries + 1, device=q.device, dtype=torch.int32)

    BLOCK_H = 64
    BLOCK_K = 64
    # NUM_XCDS = get_num_xcds()
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    heads_blocks = triton.cdiv(num_heads, BLOCK_H)
    num_splits = _decode_num_splits(
        num_queries,
        heads_blocks,
        main_indices,
        extra_indices,
        has_extra,
        block_k = BLOCK_K,
        num_cus = NUM_SMS,
        )

    part_m = torch.empty(
        (num_queries, num_splits, num_heads), dtype=torch.float32, device=q.device
    )
    part_l = torch.empty_like(part_m)
    part_acc = torch.empty(
        (num_queries, num_splits, num_heads, comb_dim),
        dtype=torch.float32,
        device=q.device,
    )

    _sparse_attn_decode_partial_kernel[(num_queries, num_splits, heads_blocks)](
        q,
        main_cache,
        main_indices,
        main_indptr,
        extra_cache,
        extra_indices,
        extra_indptr,
        part_m,
        part_l,
        part_acc,
        q.stride(0), q.stride(1),
        main_cache.stride(0),
        extra_cache.stride(0),
        part_m.stride(0), part_m.stride(1),
        part_acc.stride(0), part_acc.stride(1), part_acc.stride(2),
        main_cache.shape[0] * main_cache.shape[1],
        extra_cache.shape[0] * extra_cache.shape[1],
        main_cache.shape[1],
        extra_cache.shape[1],
        float(scale),
        num_heads,
        HAS_EXTRA=has_extra,
        NOPE_DIM=nope_dim,
        NOPE_BLOCK=nope_block,
        ROPE_DIM=rope_dim,
        IS_FNUZ=is_fnuz,
        BLOCK_H=BLOCK_H,
        BLOCK_K=BLOCK_K,
        NUM_SPLITS=num_splits,
        num_warps=num_warps,
    )

    _sparse_attn_decode_reduce_kernel[(num_queries, heads_blocks)](
        part_m,
        part_l,
        part_acc,
        attn_sink,
        out,
        out.stride(0), out.stride(1),
        part_m.stride(0), part_m.stride(1),
        part_acc.stride(0), part_acc.stride(1), part_acc.stride(2),
        num_heads,
        HAS_ATTN_SINK=has_attn_sink,
        COMB_DIM=comb_dim,
        BLOCK_H=BLOCK_H,
        NUM_SPLITS=num_splits,
        num_warps=4,
    )
    return out
