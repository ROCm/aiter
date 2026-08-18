# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa: EXE005  the `#!=----` separators below are decorative
# comments, not shebangs; ruff's heuristic flags all 66 of them.

import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

try:
    from triton.experimental.gluon.language.amd.cdna3 import (
        s_set_prio as _amd_s_set_prio,
    )
    from triton.experimental.gluon.language.amd.cdna3 import (
        sched_barrier as _amd_iglp_sched_barrier,
    )
    from triton.experimental.gluon.language.amd.cdna3 import (
        sched_group_barrier as _amd_iglp_sched_group_barrier,
    )
except ImportError:
    # ignore iglp hint
    @gluon.jit
    def _amd_iglp_sched_barrier(inst_mask):
        pass

    @gluon.jit
    def _amd_iglp_sched_group_barrier(inst_mask, cnt, _):
        pass

    @gluon.jit
    def _amd_s_set_prio(prio):
        pass


@triton.jit
def _sum_combine(a, b):
    return a + b


@gluon.jit
def _gluon_deepgemm_fp8_paged_mqa_logits(
    batch_size,
    next_n,
    heads_num,
    Q_buffer,
    stride_q_batch,
    stride_q_next_n,
    stride_q_heads,
    KV_buffer,
    stride_k_seq,
    scale_buffer,
    stride_scale_seq,
    context_len_ptr,
    kv_indices,
    weights,
    stride_w_batch,
    OutLogits_buffer,
    stride_out_batch,
    max_model_len,
    max_block_len,
    num_block,
    SplitKV,
    ChunkQ: tl.constexpr,
    ChunkK: tl.constexpr,
    HiddenDim: tl.constexpr,
    KVBlockSize: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    num_block_q_head = tl.cdiv(heads_num, ChunkQ)

    pid_q_head, remain_pid = pid % num_block_q_head, pid // num_block_q_head
    pid_next_n, remain_pid = remain_pid % next_n, remain_pid // next_n
    pid_batch, pid_split_kv = remain_pid % batch_size, remain_pid // batch_size

    context_length = gl.load(context_len_ptr + pid_batch)

    context_chunk_num = tl.cdiv(context_length, ChunkK)
    split_context_chunk_num = tl.cdiv(context_chunk_num, SplitKV)

    split_context_start = (pid_split_kv * split_context_chunk_num) * ChunkK
    split_context_length = min(
        context_length - split_context_start, split_context_chunk_num * ChunkK
    )

    if split_context_length <= 0:
        return

    residual_context = (ChunkK - split_context_length % ChunkK) % ChunkK

    NumWarps: gl.constexpr = 4
    ThreadsPerWarp: gl.constexpr = 32

    # ===---------------------------------------------------
    # Gluon Layout
    # ===---------------------------------------------------
    ValQMPerThread: gl.constexpr = ChunkQ // (
        NumWarps * ThreadsPerWarp // (HiddenDim // 16)
    )
    layout_q: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[ValQMPerThread, 16],  # q type is fp8 (E4M3)
        threads_per_warp=[ThreadsPerWarp // (HiddenDim // 16), HiddenDim // 16],
        warps_per_cta=[NumWarps, 1],
        order=[1, 0],
    )

    ValKNPerThread: gl.constexpr = ChunkK // (
        NumWarps * ThreadsPerWarp // (HiddenDim // 16)
    )
    layout_kv: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[ValKNPerThread, 16],  # k type is fp8 (E4M3)
        threads_per_warp=[ThreadsPerWarp // (HiddenDim // 16), HiddenDim // 16],
        warps_per_cta=[NumWarps, 1],
        order=[1, 0],
    )

    FP8_K_DIM: gl.constexpr = 128 if HiddenDim > 64 else 64
    if NumWarps == 1:
        warp_bases: gl.constexpr = []
    elif NumWarps == 2:
        warp_bases: gl.constexpr = [[0, 1]]
    elif NumWarps == 4:
        warp_bases: gl.constexpr = [[0, 1], [0, 2]]
    else:
        warp_bases: gl.constexpr = [[0, 1], [0, 2], [0, 4]]
    mfma_layout: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=False,
        instr_shape=[16, 16, FP8_K_DIM],
        warp_bases=warp_bases,
    )
    mfma_layout_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    mfma_layout_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )

    layout_scale: gl.constexpr = gl.SliceLayout(1, mfma_layout)

    # ===---------------------------------------------------
    # Pipeline Start
    # ===---------------------------------------------------
    q = gl.amd.cdna3.buffer_load(
        ptr=Q_buffer,
        offsets=pid_batch * stride_q_batch
        + pid_next_n * stride_q_next_n
        + (
            (
                pid_q_head * ChunkQ
                + gl.arange(0, ChunkQ, layout=gl.SliceLayout(1, layout_q))
            )
            * stride_q_heads
        )[:, None]
        + gl.arange(0, HiddenDim, layout=gl.SliceLayout(0, layout_q))[None, :],
    )
    scale_weight = gl.amd.cdna3.buffer_load(
        ptr=weights,
        offsets=(pid_batch * next_n + pid_next_n) * stride_w_batch
        + pid_q_head * ChunkQ
        + gl.arange(0, ChunkQ, layout=layout_scale),
    )

    mask_kv_next = (
        split_context_start
        - residual_context
        + gl.arange(0, ChunkK, layout=gl.SliceLayout(1, layout_kv))
        >= 0
    )
    mask_kv_scale_next = (
        split_context_start
        - residual_context
        + gl.arange(0, ChunkK, layout=gl.SliceLayout(0, mfma_layout))
        >= 0
    )
    context_kv_idx_next = gl.amd.cdna3.buffer_load(
        ptr=kv_indices,
        offsets=pid_batch * max_block_len
        + split_context_start
        - residual_context
        + gl.arange(0, ChunkK, layout=gl.SliceLayout(1, layout_kv)),
        mask=mask_kv_next,
    )
    context_kv_scale_idx_next = gl.amd.cdna3.buffer_load(
        ptr=kv_indices,
        offsets=pid_batch * max_block_len
        + split_context_start
        - residual_context
        + gl.arange(0, ChunkK, layout=gl.SliceLayout(0, mfma_layout)),
        mask=mask_kv_scale_next,
    )

    mfma_q = gl.convert_layout(q, mfma_layout_a)

    context_kv_idx_next = tl.where(mask_kv_next, context_kv_idx_next, 0)
    k_next = gl.amd.cdna3.buffer_load(
        ptr=KV_buffer,
        offsets=context_kv_idx_next[:, None] * stride_k_seq
        + gl.arange(0, HiddenDim, layout=gl.SliceLayout(0, layout_kv))[None, :],
    )
    context_kv_scale_idx_next = tl.where(
        mask_kv_scale_next, context_kv_scale_idx_next, 0
    )
    k_scale_f_next = gl.amd.cdna3.buffer_load(
        ptr=scale_buffer, offsets=context_kv_scale_idx_next * stride_scale_seq
    )

    zero = gl.zeros((ChunkQ, ChunkK), dtype=tl.float32, layout=mfma_layout)
    for context_idx in range(
        split_context_start - residual_context,
        split_context_start + split_context_length - ChunkK,
        ChunkK,
    ):
        k = k_next
        k_scale_f = k_scale_f_next

        context_kv_idx_next = gl.amd.cdna3.buffer_load(
            ptr=kv_indices,
            offsets=pid_batch * max_block_len
            + context_idx
            + ChunkK
            + gl.arange(0, ChunkK, layout=gl.SliceLayout(1, layout_kv)),
        )
        context_kv_scale_idx_next = gl.amd.cdna3.buffer_load(
            ptr=kv_indices,
            offsets=pid_batch * max_block_len
            + context_idx
            + ChunkK
            + gl.arange(0, ChunkK, layout=gl.SliceLayout(0, mfma_layout)),
        )

        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------
        mfma_k = gl.convert_layout(k.T, mfma_layout_b)

        o = gl.amd.gfx1250.wmma(mfma_q, mfma_k, zero)
        o = o * k_scale_f[None, :]

        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------
        k_next = gl.amd.cdna3.buffer_load(
            ptr=KV_buffer,
            offsets=context_kv_idx_next[:, None] * stride_k_seq
            + gl.arange(0, HiddenDim, layout=gl.SliceLayout(0, layout_kv))[None, :],
        )
        o = gl.maximum(o, 0.0)
        o = o * scale_weight[:, None]

        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------
        k_scale_f_next = gl.amd.cdna3.buffer_load(
            ptr=scale_buffer, offsets=context_kv_scale_idx_next * stride_scale_seq
        )

        mask = (
            context_idx + gl.arange(0, ChunkK, layout=gl.SliceLayout(0, mfma_layout))
            <= context_length - next_n + pid_next_n
        )
        o = tl.where(mask[None, :], o, float("-inf"))

        logits = gl.reduce(o, axis=0, combine_fn=_sum_combine)
        gl.amd.cdna3.buffer_store(
            logits,
            ptr=OutLogits_buffer
            + (pid_batch * next_n + pid_next_n).to(tl.int64) * stride_out_batch,
            offsets=(
                context_idx
                + gl.arange(0, ChunkK, layout=gl.SliceLayout(0, mfma_layout))
            ),
            mask=context_idx
            + gl.arange(0, ChunkK, layout=gl.SliceLayout(0, mfma_layout))
            >= 0,
        )

    context_idx = split_context_start + split_context_length - ChunkK
    k = k_next
    k_scale_f = k_scale_f_next

    mfma_k = gl.convert_layout(k.T, mfma_layout_b)
    o = gl.amd.gfx1250.wmma(mfma_q, mfma_k, zero)

    o = o * k_scale_f[None, :]
    o = gl.maximum(o, 0.0)
    o = o * scale_weight[:, None]

    mask = (
        context_idx + gl.arange(0, ChunkK, layout=gl.SliceLayout(0, mfma_layout))
        <= context_length - next_n + pid_next_n
    )
    o = tl.where(mask[None, :], o, float("-inf"))

    logits = gl.reduce(o, axis=0, combine_fn=_sum_combine)
    gl.amd.cdna3.buffer_store(
        logits,
        ptr=OutLogits_buffer
        + (pid_batch * next_n + pid_next_n).to(tl.int64) * stride_out_batch,
        offsets=(
            context_idx + gl.arange(0, ChunkK, layout=gl.SliceLayout(0, mfma_layout))
        ),
        mask=context_idx + gl.arange(0, ChunkK, layout=gl.SliceLayout(0, mfma_layout))
        >= 0,
    )


@gluon.jit
def _gluon_deepgemm_fp8_paged_mqa_logits_preshuffle(
    batch_size,
    next_n,
    heads_num,
    Q_buffer,
    stride_q_batch,
    stride_q_next_n,
    stride_q_heads,
    KV_buffer,
    stride_k_seq,
    scale_buffer,
    stride_scale_seq,
    context_len_ptr,
    kv_indices,
    weights,
    stride_w_batch,
    OutLogits_buffer,
    stride_out_batch,
    max_model_len,
    max_block_len,
    num_block,
    SplitKV: gl.constexpr,
    ChunkQ: gl.constexpr,
    ChunkK: gl.constexpr,
    HiddenDim: gl.constexpr,
    KVBlockSize: gl.constexpr = 16,
):
    # ===---------------------------------------------------
    # Gluon Layout
    # ===---------------------------------------------------
    NumWarps: gl.constexpr = 1
    ThreadsPerWarp: gl.constexpr = 32

    ValQMPerThread: gl.constexpr = ChunkQ // (
        NumWarps * ThreadsPerWarp // (HiddenDim // 16)
    )
    layout_q: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[ValQMPerThread, 16],  # q type is fp8 (E4M3)
        threads_per_warp=[ThreadsPerWarp // (HiddenDim // 16), HiddenDim // 16],
        warps_per_cta=[NumWarps, 1],
        order=[1, 0],
    )

    FP8_K_DIM: gl.constexpr = 128 if HiddenDim > 64 else 64
    if NumWarps == 1:
        warp_bases: gl.constexpr = []
    elif NumWarps == 2:
        warp_bases: gl.constexpr = [[0, 1]]
    elif NumWarps == 4:
        warp_bases: gl.constexpr = [[0, 1], [0, 2]]
    else:
        warp_bases: gl.constexpr = [[0, 1], [0, 2], [0, 4]]
    mfma_layout: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=False,
        instr_shape=[16, 16, FP8_K_DIM],
        warp_bases=warp_bases,
    )

    mfma_layout_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    mfma_layout_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )

    layout_scale: gl.constexpr = gl.SliceLayout(1, mfma_layout)

    NUM_BUFFERS: gl.constexpr = 2
    KV_SHARED: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[1, 0]
    )
    kv_shared = gl.allocate_shared_memory(
        KV_buffer.type.element_ty,
        [NUM_BUFFERS, 1, KVBlockSize * HiddenDim],
        KV_SHARED,
    )
    kv_scale_shared = gl.allocate_shared_memory(
        scale_buffer.type.element_ty,
        [NUM_BUFFERS, 1, KVBlockSize * HiddenDim // 128],
        KV_SHARED,
    )

    pid = tl.program_id(0)
    pid_batch, remain_pid = pid % batch_size, pid // batch_size
    pid_next_n, pid_split_kv = remain_pid % next_n, remain_pid // next_n
    context_length = gl.load(context_len_ptr + pid_batch)

    context_chunk_num = tl.cdiv(context_length, ChunkK)
    split_context_chunk_num = context_chunk_num // SplitKV
    residual_context_chunks = context_chunk_num % SplitKV
    split_context_start = (
        pid_split_kv * split_context_chunk_num * ChunkK
        + min(pid_split_kv, residual_context_chunks) * ChunkK
    )
    split_context_length = min(
        context_length - split_context_start,
        split_context_chunk_num * ChunkK
        + (ChunkK if pid_split_kv < residual_context_chunks else 0),
    )
    if split_context_length <= 0:
        return

    q = gl.amd.cdna3.buffer_load(
        ptr=Q_buffer,
        offsets=pid_batch * stride_q_batch
        + pid_next_n * stride_q_next_n
        + (gl.arange(0, ChunkQ, layout=gl.SliceLayout(1, layout_q)) * stride_q_heads)[
            :, None
        ]
        + gl.arange(0, HiddenDim, layout=gl.SliceLayout(0, layout_q))[None, :],
    )
    scale_weight = gl.amd.cdna3.buffer_load(
        ptr=weights,
        offsets=(pid_batch * next_n + pid_next_n) * stride_w_batch
        + gl.arange(0, ChunkQ, layout=layout_scale),
    )

    mfma_q = gl.convert_layout(q, mfma_layout_a)
    zero_acc = gl.zeros((ChunkQ, KVBlockSize), dtype=tl.float32, layout=mfma_layout)
    col = gl.arange(0, KVBlockSize, layout=gl.SliceLayout(0, mfma_layout))

    n_blocks = tl.cdiv(split_context_length, KVBlockSize)
    cblk_base = split_context_start // KVBlockSize
    kv_idx_base = kv_indices + pid_batch * max_block_len + cblk_base

    # Prologue: kick off the async_load for block 0 into buffer 0.
    save_blk_idx: gl.int32 = n_blocks - 1
    blk_cur = gl.load(kv_idx_base)
    blk_next = gl.load(kv_idx_base + gl.minimum(1, save_blk_idx))
    desc_cur = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=KV_buffer,
        shape=(num_block, KVBlockSize * HiddenDim),
        strides=(stride_k_seq, 1),
        block_shape=(1, KVBlockSize * HiddenDim),
        layout=KV_SHARED,
    )
    BLOCKSCALE_SIZE: gl.constexpr = 128
    assert HiddenDim // BLOCKSCALE_SIZE == 1
    desc_scale_cur = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=scale_buffer,
        shape=(num_block, KVBlockSize * HiddenDim // BLOCKSCALE_SIZE),
        strides=(stride_scale_seq, 1),
        block_shape=(1, KVBlockSize * HiddenDim // BLOCKSCALE_SIZE),
        layout=KV_SHARED,
    )
    gl.amd.gfx1250.tdm.async_load(desc_cur, [blk_cur, 0], kv_shared.index(0))
    gl.amd.gfx1250.tdm.async_load(
        desc_scale_cur, [blk_cur, 0], kv_scale_shared.index(0)
    )

    for j in range(n_blocks):
        buf = j % NUM_BUFFERS
        context_idx = split_context_start + j * KVBlockSize
        # blk = blk_cur
        # k_scale_f = gl.amd.cdna3.buffer_load(
        #     ptr=scale_buffer,
        #     offsets=blk * stride_scale_seq + col,
        # )

        # Prefetch block j+1 into the other buffer before waiting on block j.
        if j + 1 < n_blocks:
            blk_cur = blk_next
            blk_next = gl.load(kv_idx_base + gl.minimum(j + 2, save_blk_idx))
            gl.amd.gfx1250.tdm.async_load(
                desc_cur, [blk_cur, 0], kv_shared.index((j + 1) % NUM_BUFFERS)
            )
            gl.amd.gfx1250.tdm.async_load(
                desc_scale_cur,
                [blk_cur, 0],
                kv_scale_shared.index((j + 1) % NUM_BUFFERS),
            )
            # Leave exactly the just-issued load (block j+1) in flight.
            gl.amd.gfx1250.tdm.async_wait(3)
        else:
            gl.amd.gfx1250.tdm.async_wait(1)

        K_WIDTH: gl.constexpr = 16
        mfma_k = (
            kv_shared.index(buf)
            .reshape(
                (
                    KVBlockSize // 16,
                    HiddenDim // (2 * K_WIDTH),
                    2,
                    16,
                    K_WIDTH,
                )
            )
            .permute((0, 3, 1, 2, 4))
            .reshape((KVBlockSize, HiddenDim))
            .permute([1, 0])
            .load(layout=mfma_layout_b)
        )
        o = gl.amd.gfx1250.wmma(mfma_q, mfma_k, zero_acc)

        if j + 1 < n_blocks:
            gl.amd.gfx1250.tdm.async_wait(2)
        else:
            gl.amd.gfx1250.tdm.async_wait(0)

        k_scale_f = (
            kv_scale_shared.index(buf)
            .reshape((1, KVBlockSize))
            .load(layout=mfma_layout)
        )
        o = o * k_scale_f
        o = gl.maximum(o, 0.0)
        o = o * scale_weight[:, None]

        valid = (context_idx + col) <= (context_length - next_n + pid_next_n)
        o = tl.where(valid[None, :], o, float("-inf"))
        logits = gl.reduce(o, axis=0, combine_fn=_sum_combine)

        store_off = context_idx + col
        gl.amd.cdna3.buffer_store(
            logits,
            ptr=OutLogits_buffer
            + (pid_batch * next_n + pid_next_n).to(tl.int64) * stride_out_batch,
            offsets=store_off,
            mask=(store_off >= 0) & (store_off < max_model_len),
        )
    return
