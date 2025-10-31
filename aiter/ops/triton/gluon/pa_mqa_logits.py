# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import math
import triton
import triton.language as tl

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


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
    ThreadsPerWarp: gl.constexpr = 64

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

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3,
        instr_shape=[16, 16],
        transposed=False,
        warps_per_cta=[1, NumWarps],
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
        gl.amd.cdna3.sched_barrier(0x0)
        #!=----------------------------
        mfma_k = gl.convert_layout(k.T, mfma_layout_b)

        o = gl.amd.cdna3.mfma(mfma_q, mfma_k, zero)
        o = o * k_scale_f[None, :]

        #!=----------------------------
        gl.amd.cdna3.sched_barrier(0x0)
        #!=----------------------------
        k_next = gl.amd.cdna3.buffer_load(
            ptr=KV_buffer,
            offsets=context_kv_idx_next[:, None] * stride_k_seq
            + gl.arange(0, HiddenDim, layout=gl.SliceLayout(0, layout_kv))[None, :],
        )
        o = gl.maximum(o, 0.0)
        o = o * scale_weight[:, None]

        #!=----------------------------
        gl.amd.cdna3.sched_barrier(0x0)
        #!=----------------------------
        k_scale_f_next = gl.amd.cdna3.buffer_load(
            ptr=scale_buffer, offsets=context_kv_scale_idx_next * stride_scale_seq
        )

        mask = (
            context_idx + gl.arange(0, ChunkK, layout=gl.SliceLayout(0, mfma_layout))
            <= context_length - pid_next_n
        )
        o = tl.where(mask[None, :], o, float("-inf"))

        logits = gl.reduce(o, axis=0, combine_fn=_sum_combine)
        gl.amd.cdna3.buffer_store(
            logits,
            ptr=OutLogits_buffer,
            offsets=(pid_batch * next_n + pid_next_n) * stride_out_batch
            + (
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
    o = gl.amd.cdna3.mfma(mfma_q, mfma_k, zero)

    o = o * k_scale_f[None, :]
    o = gl.maximum(o, 0.0)
    o = o * scale_weight[:, None]

    mask = (
        context_idx + gl.arange(0, ChunkK, layout=gl.SliceLayout(0, mfma_layout))
        <= context_length - pid_next_n
    )
    o = tl.where(mask[None, :], o, float("-inf"))

    logits = gl.reduce(o, axis=0, combine_fn=_sum_combine)
    gl.amd.cdna3.buffer_store(
        logits,
        ptr=OutLogits_buffer,
        offsets=(pid_batch * next_n + pid_next_n) * stride_out_batch
        + (context_idx + gl.arange(0, ChunkK, layout=gl.SliceLayout(0, mfma_layout))),
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
    SplitKV,
    ChunkQ: tl.constexpr,
    ChunkK: tl.constexpr,
    HiddenDim: tl.constexpr,
    KVBlockSize: tl.constexpr = 16,
):
    # ===---------------------------------------------------
    # Gluon Layout
    # ===---------------------------------------------------
    NumWarps: gl.constexpr = 4
    ThreadsPerWarp: gl.constexpr = 64

    ValQMPerThread: gl.constexpr = ChunkQ // (
        NumWarps * ThreadsPerWarp // (HiddenDim // 16)
    )
    layout_q: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[ValQMPerThread, 16],  # q type is fp8 (E4M3)
        threads_per_warp=[ThreadsPerWarp // (HiddenDim // 16), HiddenDim // 16],
        warps_per_cta=[NumWarps, 1],
        order=[1, 0],
    )

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3,
        instr_shape=[16, 16],
        transposed=False,
        warps_per_cta=[1, NumWarps],
    )
    mfma_layout_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    mfma_layout_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )

    layout_scale: gl.constexpr = gl.SliceLayout(1, mfma_layout)

    ContextBlockPerChunkK: gl.constexpr = ChunkK // KVBlockSize

    DS_WRITE: gl.constexpr = 0x200
    DS_READ: gl.constexpr = 0x100
    BUFFER_LOAD: gl.constexpr = 0x020
    MFMA: gl.constexpr = 0x008
    VALU: gl.constexpr = 0x002

    # ===---------------------------------------------------
    # Mapping WorkTile
    # ===---------------------------------------------------
    pid = tl.program_id(0)
    num_block_q_head = tl.cdiv(heads_num, ChunkQ)

    pid_batch, remain_pid = pid % batch_size, pid // batch_size
    pid_q_head, remain_pid = (
        remain_pid % num_block_q_head,
        remain_pid // num_block_q_head,
    )
    pid_next_n, pid_split_kv = remain_pid % next_n, remain_pid // next_n

    context_length = gl.load(context_len_ptr + pid_batch)

    context_chunk_num = tl.cdiv(context_length, ChunkK)
    split_context_chunk_num = tl.cdiv(context_chunk_num, SplitKV)

    split_context_start = (pid_split_kv * split_context_chunk_num) * ChunkK
    split_context_length = min(
        context_length - split_context_start, split_context_chunk_num * ChunkK
    )

    if split_context_length <= 0:
        return

    split_context_block = tl.cdiv(split_context_length, KVBlockSize)
    split_context_length = split_context_block * KVBlockSize

    residual_context_blocks = (
        ContextBlockPerChunkK - split_context_block % ContextBlockPerChunkK
    ) % ContextBlockPerChunkK
    residual_context = residual_context_blocks * KVBlockSize

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
        (split_context_start - residual_context) // KVBlockSize
        + gl.arange(0, ChunkK, layout=gl.SliceLayout(0, mfma_layout_b)) // KVBlockSize
    ) >= 0
    context_kv_idx_next = gl.amd.cdna3.buffer_load(
        ptr=kv_indices,
        offsets=pid_batch * max_block_len
        + (split_context_start - residual_context) // KVBlockSize
        + gl.arange(0, ChunkK, layout=gl.SliceLayout(0, mfma_layout_b)) // KVBlockSize,
        mask=mask_kv_next,
    )

    offset_k_fixed = (
        gl.arange(0, HiddenDim, layout=gl.SliceLayout(1, mfma_layout_b)) % 16
        + gl.arange(0, HiddenDim, layout=gl.SliceLayout(1, mfma_layout_b)) // 16 * 256
    )[:, None] + (
        gl.arange(0, ChunkK, layout=gl.SliceLayout(0, mfma_layout_b)) % 16 * 16
    )[
        None, :
    ]

    # #!=----------------------------
    # gl.amd.cdna3.sched_barrier(0x0)
    # #!=----------------------------

    context_kv_idx_next = tl.where(mask_kv_next, context_kv_idx_next, 0)
    k_next = gl.amd.cdna3.buffer_load(
        ptr=KV_buffer,
        offsets=offset_k_fixed + context_kv_idx_next[None, :] * stride_k_seq,
    )
    k_scale_f_next = gl.amd.cdna3.buffer_load(
        ptr=scale_buffer,
        offsets=context_kv_idx_next * stride_scale_seq
        + gl.arange(0, ChunkK, layout=gl.SliceLayout(0, mfma_layout_b)) % KVBlockSize,
    )
    mfma_q = gl.convert_layout(q, mfma_layout_a)

    zero = gl.zeros((ChunkQ, ChunkK), dtype=tl.float32, layout=mfma_layout)
    for context_idx in range(
        split_context_start - residual_context,
        split_context_start + split_context_length - ChunkK,
        ChunkK,
    ):
        k = k_next
        k_scale_f = k_scale_f_next

        # #!=----------------------------
        gl.amd.cdna3.sched_barrier(0x0)
        # #!=----------------------------

        context_kv_idx_next = gl.amd.cdna3.buffer_load(
            ptr=kv_indices,
            offsets=pid_batch * max_block_len
            + (context_idx + ChunkK) // KVBlockSize
            + gl.arange(0, ChunkK, layout=gl.SliceLayout(0, mfma_layout_b))
            // KVBlockSize,
        )

        #!=----------------------------
        gl.amd.cdna3.sched_barrier(0x0)
        #!=----------------------------

        mfma_k = gl.convert_layout(k, mfma_layout_b)
        o = gl.amd.cdna3.mfma(mfma_q, mfma_k, zero)

        #!=----------------------------
        gl.amd.cdna3.sched_barrier(0x0)
        #!=----------------------------
        k_next = gl.amd.cdna3.buffer_load(
            ptr=KV_buffer,
            offsets=offset_k_fixed + context_kv_idx_next[None, :] * stride_k_seq,
        )

        k_scale_f = gl.convert_layout(k_scale_f, gl.SliceLayout(0, mfma_layout))
        o = o * k_scale_f[None, :]
        o = gl.maximum(o, 0.0)

        #!=----------------------------
        gl.amd.cdna3.sched_barrier(0x0)
        #!=----------------------------
        k_scale_f_next = gl.amd.cdna3.buffer_load(
            ptr=scale_buffer,
            offsets=context_kv_idx_next * stride_scale_seq
            + gl.arange(0, ChunkK, layout=gl.SliceLayout(0, mfma_layout_b))
            % KVBlockSize,
        )
        o = o * scale_weight[:, None]

        mask = (
            context_idx + gl.arange(0, ChunkK, layout=gl.SliceLayout(0, mfma_layout))
            <= context_length - pid_next_n
        )
        o = tl.where(mask[None, :], o, float("-inf"))

        logits = gl.reduce(o, axis=0, combine_fn=_sum_combine)
        gl.amd.cdna3.buffer_store(
            logits,
            ptr=OutLogits_buffer,
            offsets=(pid_batch * next_n + pid_next_n) * stride_out_batch
            + (
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

    mfma_k = gl.convert_layout(k, mfma_layout_b)
    o = gl.amd.cdna3.mfma(mfma_q, mfma_k, zero)

    k_scale_f = gl.convert_layout(k_scale_f, gl.SliceLayout(0, mfma_layout))
    o = o * k_scale_f[None, :]
    o = gl.maximum(o, 0.0)
    o = o * scale_weight[:, None]

    mask = (
        context_idx + gl.arange(0, ChunkK, layout=gl.SliceLayout(0, mfma_layout))
        <= context_length - pid_next_n
    )
    o = tl.where(mask[None, :], o, float("-inf"))

    logits = gl.reduce(o, axis=0, combine_fn=_sum_combine)
    gl.amd.cdna3.buffer_store(
        logits,
        ptr=OutLogits_buffer,
        offsets=(pid_batch * next_n + pid_next_n) * stride_out_batch
        + (context_idx + gl.arange(0, ChunkK, layout=gl.SliceLayout(0, mfma_layout))),
        mask=context_idx + gl.arange(0, ChunkK, layout=gl.SliceLayout(0, mfma_layout))
        >= 0,
    )
