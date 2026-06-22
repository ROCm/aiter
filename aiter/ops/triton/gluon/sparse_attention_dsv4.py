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
    q_ptr,  # [T, H, D]
    kv_ptr,  # [num_KV, D]
    kv_indices_ptr,  # [num_None_Zero]
    kv_indptr_ptr,  # [T + 1]
    attn_sink_ptr,  # [H]
    out_ptr,  # [T, H, D]
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
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[num_warps, 1],
    )
    qk_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma_qk, k_width=8)
    qk_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma_qk, k_width=8)

    # q [BLOCK_H, BLOCK_D] load layout (D = dim1 contiguous).
    blk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[1, 64],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )
    blk_kv: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[8, BLOCK_K // num_warps],
        threads_per_warp=[64, 1],
        warps_per_cta=[1, num_warps],
        order=[0, 1],
    )
    # KV-slot id async-copy offsets: a clean 64-wide (one lane per slot, warps
    # redundant) linear layout. A 1-D async gather narrower than the 64-lane warp
    # can't tile without lane aliasing (which fails to lower), so slots are always
    # gathered 64 at a time and the [0:BLOCK_K] slice is read back per tile
    # (BLOCK_K=32 reads half; BLOCK_K=64 reads the full SLOT_W tile).
    gl.static_assert(num_warps == 4, "slot async-copy layout assumes num_warps == 4")
    SLOT_W: gl.constexpr = 64
    dll_slot: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[],
        lane_bases=[[1], [2], [4], [8], [16], [32]],
        warp_bases=[[0], [0]],
        block_bases=[],
        shape=[SLOT_W],
    )

    gl.static_assert(
        BLOCK_D == 512, "shared_q/shared_kv offset_bases assume BLOCK_D == 512"
    )
    gl.static_assert(BLOCK_H == 64, "shared_q is hard-coded to [64, 512]")
    gl.static_assert(
        BLOCK_K == 32 or BLOCK_K == 64,
        "shared_kv is hard-coded to [512, 32] or [512, 64]",
    )

    # Q staging layout (async global->LDS): D (dim1) contiguous + BLOCK_H row
    # bits, [[512,16]] padding.
    shared_q: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs=[[512, 16]],
        offset_bases=[
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [0, 128],
            [0, 256],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [16, 0],
            [32, 0],
        ],
        cga_layout=[],
        shape=[64, 512],
    )

    if BLOCK_K == 32:
        shared_kv: gl.constexpr = gl.PaddedSharedLayout(
            interval_padding_pairs=[[512, 16]],
            offset_bases=[
                [1, 0],
                [2, 0],
                [4, 0],
                [8, 0],
                [16, 0],
                [32, 0],
                [64, 0],
                [128, 0],
                [256, 0],
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
                [0, 16],
            ],
            cga_layout=[],
            shape=[512, 32],
        )
    else:
        # BLOCK == 64, BLOCK_K choices are guarded by above static_assert
        shared_kv: gl.constexpr = gl.PaddedSharedLayout(
            interval_padding_pairs=[[512, 16]],
            offset_bases=[
                [1, 0],
                [2, 0],
                [4, 0],
                [8, 0],
                [16, 0],
                [32, 0],
                [64, 0],
                [128, 0],
                [256, 0],
                [0, 1],
                [0, 2],
                [0, 8],
                [0, 4],
                [0, 16],
                [0, 32],
            ],
            cga_layout=[],
            shape=[512, 64],
        )

    # 1-D LDS staging for the KV-slot ids (global->LDS async-copy, ds_read back).
    shared_slot: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1,
        per_phase=1,
        max_phase=1,
        order=[0],
    )

    # slice layouts for the per-row / per-col index vectors
    sl_h_blk: gl.constexpr = gl.SliceLayout(1, blk)  # heads (axis0 of blk, q)
    sl_d_blk: gl.constexpr = gl.SliceLayout(0, blk)  # dim (axis1 of blk, q)
    sl_d_kv: gl.constexpr = gl.SliceLayout(1, blk_kv)  # dim (axis0 of blk_kv, KV)
    sl_k_kv: gl.constexpr = gl.SliceLayout(0, blk_kv)  # kpos (axis1 of blk_kv, KV)
    sl_h_qk: gl.constexpr = gl.SliceLayout(1, mma_qk)  # heads (scores/acc rows)
    sl_k_qk: gl.constexpr = gl.SliceLayout(
        0, mma_qk
    )  # cols: scores kpos AND acc/out dim

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
    buf_q = gl.allocate_shared_memory(
        q_ptr.dtype.element_ty, [BLOCK_H, BLOCK_D], shared_q
    )
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

    k_off_kv = gl.arange(0, BLOCK_K, layout=sl_k_kv)  # kpos (cols of [D,K] tile)
    slot_off = gl.arange(0, SLOT_W, layout=dll_slot)  # 64-wide slot gather lanes
    dim_off_kv = gl.arange(0, BLOCK_D, layout=sl_d_kv)  # dim (rows of [D,K] tile)

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
        smem_slot.index(1),
        kv_indices_ptr + kv_start,
        BLOCK_K + slot_off,
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
    slot0 = gl.amd.cdna4.async_copy.load_shared_relaxed(
        smem_slot.index(0).slice(0, BLOCK_K, 0), sl_k_kv
    )
    tail0 = k_off_kv < kv_len
    kv_off0 = (
        dim_off_kv[:, None] * kv_stride_d
        + gl.where(tail0, slot0, 0)[None, :] * kv_stride_n
    )
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
        slot_n = gl.amd.cdna4.async_copy.load_shared_relaxed(
            smem_slot.index(sbuf).slice(0, BLOCK_K, 0), sl_k_kv
        )
        kv_off_n = dim_off_kv[:, None] * kv_stride_d + slot_n[None, :] * kv_stride_n
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smem_kv.index(sbuf), kv_ptr, kv_off_n
        )
        gl.amd.cdna4.async_copy.commit_group()

        scores = scores * qk_scale
        m_block = gl.max(scores, axis=1)
        m_new = gl.maximum(m_i, m_block)
        alpha = gl.exp2(m_i - m_new)
        p = gl.exp2(scores - m_new[:, None])
        l_i = l_i * alpha + gl.sum(p, axis=1)
        m_i = m_new
        # LL KV[i] (V view) -> PV mfma.
        v_dot = gl.amd.cdna4.async_copy.load_shared_relaxed(
            smem_kv.index(cbuf).permute([1, 0]), qk_b
        )
        p_dot = gl.convert_layout(p.to(kv_ptr.dtype.element_ty), qk_a)
        alpha_pv = gl.convert_layout(alpha, sl_h_qk)
        acc = acc * alpha_pv[:, None]
        acc = gl.amd.cdna4.mfma(p_dot, v_dot, acc)

    # Epilogue: LL slot[last] -> GL KV[last] (not yet prefetched), then compute
    # the two trailing tiles eff_iters-2 and eff_iters-1.
    gl.amd.cdna4.async_copy.wait_group(1)
    nlast = (eff_iters - 1) % 2
    slot_l = gl.amd.cdna4.async_copy.load_shared_relaxed(
        smem_slot.index(nlast).slice(0, BLOCK_K, 0), sl_k_kv
    )
    # Tail lanes (kpos >= kv_len) are clamped to slot 0 (a safe in-bounds row)
    # for the gather; the -inf score mask below drops their softmax contribution.
    kpos_l = (eff_iters - 1) * BLOCK_K + k_off_kv
    tail = kpos_l < kv_len
    kv_off_l = (
        dim_off_kv[:, None] * kv_stride_d
        + gl.where(tail, slot_l, 0)[None, :] * kv_stride_n
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smem_kv.index(nlast), kv_ptr, kv_off_l
    )
    gl.amd.cdna4.async_copy.commit_group()

    # Epilogue tile eff_iters-2 (full when real num_iters >= 2 -> mask is a
    # no-op; partial when num_iters <= 1 -> masked to the real kv_len).
    gl.amd.cdna4.async_copy.wait_group(1)
    cbuf = (eff_iters - 2) % 2
    tail_qk0 = (
        (eff_iters - 2) * BLOCK_K + gl.arange(0, BLOCK_K, layout=sl_k_qk)
    ) < kv_len
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
    v_dot = gl.amd.cdna4.async_copy.load_shared_relaxed(
        smem_kv.index(cbuf).permute([1, 0]), qk_b
    )
    p_dot = gl.convert_layout(p.to(kv_ptr.dtype.element_ty), qk_a)
    alpha_qk = gl.convert_layout(alpha, sl_h_qk)
    acc = acc * alpha_qk[:, None]
    acc = gl.amd.cdna4.mfma(p_dot, v_dot, acc)

    # Epilogue tile eff_iters-1 (always-possibly-partial tile -> mask tail).
    tail_qk = (
        (eff_iters - 1) * BLOCK_K + gl.arange(0, BLOCK_K, layout=sl_k_qk)
    ) < kv_len
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
    v_dot = gl.amd.cdna4.async_copy.load_shared_relaxed(
        smem_kv.index(nlast).permute([1, 0]), qk_b
    )
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
    if num_heads % BLOCK_H == 0:
        gl.amd.cdna4.buffer_store(
            out.to(out_ptr.dtype.element_ty),
            ptr=out_ptr + query_idx * out_stride_t,
            offsets=out_off,
        )
    else:
        gl.amd.cdna4.buffer_store(
            out.to(out_ptr.dtype.element_ty),
            ptr=out_ptr + query_idx * out_stride_t,
            offsets=out_off,
            mask=(out_head < num_heads)[:, None],
        )


def sparse_attn_prefill_gluon(
    q,  # [T, H, D] bf16 queries
    kv,  # [num_kv, D] bf16 KV pool
    indices,  # [nnz] nnz: num non-zeros, int32 ragged KV slot ids
    indptr,  # [T + 1] int32 ragged offsets
    out,  # [T, H, D] output (filled in place)
    scale,  # softmax scale
    attn_sink=None,  # optional [H] fp32 per-head sink bias
):
    """Host launcher for the DSV4 CSA(Compressed Sparse Attention) prefill kernel."""
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
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
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
