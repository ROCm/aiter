# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

try:
    from triton.experimental.gluon.language.amd.cdna3 import (
        sched_barrier as _amd_iglp_sched_barrier,
        sched_group_barrier as _amd_iglp_sched_group_barrier,
        s_set_prio as _amd_s_set_prio,
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


# for some newer triton>=3.5 version, a 3D instr_shape is required.
try:
    _: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16],
        transposed=False,
        warps_per_cta=[1, 1],
        tiles_per_warp=[1, 1],
    )
    _Use_2d_instr_shape_mfma_layout = tl.constexpr(True)
except Exception:
    _Use_2d_instr_shape_mfma_layout = tl.constexpr(False)


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
        version=4,
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
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------
        mfma_k = gl.convert_layout(k.T, mfma_layout_b)

        o = gl.amd.cdna3.mfma(mfma_q, mfma_k, zero)
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
        <= context_length - next_n + pid_next_n
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

    ChunkKPerStage: gl.constexpr = ChunkK // 2
    MFMAPerWarp: gl.constexpr = ChunkKPerStage // 16 // NumWarps

    if _Use_2d_instr_shape_mfma_layout:
        mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[16, 16],
            transposed=False,
            warps_per_cta=[1, NumWarps],
            tiles_per_warp=[1, MFMAPerWarp],
        )
    else:
        mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[16, 16, 32],
            transposed=False,
            warps_per_cta=[1, NumWarps],
            tiles_per_warp=[1, MFMAPerWarp],
        )

    mfma_layout_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    mfma_layout_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )

    layout_scale: gl.constexpr = gl.SliceLayout(1, mfma_layout)

    ContextBlockPerChunkK: gl.constexpr = ChunkK // KVBlockSize
    ChunkKStagePerContextBlock: gl.constexpr = KVBlockSize // ChunkKPerStage

    LoadBlockIndiceForEachStage: gl.constexpr = ChunkKPerStage % KVBlockSize == 0

    # DS_WRITE: gl.constexpr = 0x200
    DS_READ: gl.constexpr = 0x100
    BUFFER_LOAD: gl.constexpr = 0x020
    MFMA: gl.constexpr = 0x008
    # VALU: gl.constexpr = 0x002

    # ===---------------------------------------------------
    # Mapping WorkTile
    # ===---------------------------------------------------
    pid = tl.program_id(0)

    # ===---------------------------------------------------
    pid_batch, remain_pid = pid % batch_size, pid // batch_size
    pid_next_n, pid_split_kv = remain_pid % next_n, remain_pid // next_n
    # ===---------------------------------------------------
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

    if LoadBlockIndiceForEachStage:
        split_context_block = tl.cdiv(split_context_length, KVBlockSize)
        split_context_length = split_context_block * KVBlockSize

        residual_context_blocks = (
            ContextBlockPerChunkK - split_context_block % ContextBlockPerChunkK
        ) % ContextBlockPerChunkK
        residual_context = residual_context_blocks * KVBlockSize

        # ===---------------------------------------------------
        # Pipeline Start
        _amd_iglp_sched_barrier(0x0)
        # ===---------------------------------------------------
        q = gl.amd.cdna3.buffer_load(
            ptr=Q_buffer,
            offsets=pid_batch * stride_q_batch
            + pid_next_n * stride_q_next_n
            + (
                gl.arange(0, ChunkQ, layout=gl.SliceLayout(1, layout_q))
                * stride_q_heads
            )[:, None]
            + gl.arange(0, HiddenDim, layout=gl.SliceLayout(0, layout_q))[None, :],
        )

        context_idx = split_context_start - residual_context

        mask_kv_next_0 = (
            context_idx // KVBlockSize
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            // KVBlockSize
        ) >= split_context_start // KVBlockSize
        context_kv_idx_next_0 = gl.amd.cdna3.buffer_load(
            ptr=kv_indices,
            offsets=pid_batch * max_block_len
            + context_idx // KVBlockSize
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            // KVBlockSize,
            mask=mask_kv_next_0,
        )

        mask_kv_next_1 = (
            (context_idx + ChunkKPerStage) // KVBlockSize
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            // KVBlockSize
        ) >= split_context_start // KVBlockSize
        context_kv_idx_next_1 = gl.amd.cdna3.buffer_load(
            ptr=kv_indices,
            offsets=pid_batch * max_block_len
            + (context_idx + ChunkKPerStage) // KVBlockSize
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            // KVBlockSize,
            mask=mask_kv_next_1,
        )

        scale_weight = gl.amd.cdna3.buffer_load(
            ptr=weights,
            offsets=(pid_batch * next_n + pid_next_n) * stride_w_batch
            + gl.arange(0, ChunkQ, layout=layout_scale),
        )

        offset_k_fixed = (
            gl.arange(0, HiddenDim, layout=gl.SliceLayout(1, mfma_layout_b)) % 16
            + gl.arange(0, HiddenDim, layout=gl.SliceLayout(1, mfma_layout_b))
            // 16
            * 256
        )[:, None] + (
            gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            % 16
            * 16
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            % KVBlockSize
            // 16
            * 16
            * 128
        )[
            None, :
        ]

        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------
        mfma_q = gl.convert_layout(q, mfma_layout_a)

        context_kv_idx_next_0 = tl.where(mask_kv_next_0, context_kv_idx_next_0, 0)
        k_next_0 = gl.amd.cdna3.buffer_load(
            ptr=KV_buffer,
            offsets=offset_k_fixed + context_kv_idx_next_0[None, :] * stride_k_seq,
        )
        k_scale_f_next_0 = gl.amd.cdna3.buffer_load(
            ptr=scale_buffer,
            offsets=context_kv_idx_next_0 * stride_scale_seq
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            % KVBlockSize,
        )

        _amd_iglp_sched_group_barrier(DS_READ, 4, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 4, 0)
        _amd_iglp_sched_group_barrier(DS_READ, 2, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        _amd_iglp_sched_group_barrier(DS_READ, 2, 0)

        if context_idx + ChunkK < split_context_start + split_context_length:
            context_kv_idx_next_0 = gl.amd.cdna3.buffer_load(
                ptr=kv_indices,
                offsets=pid_batch * max_block_len
                + (context_idx + ChunkK) // KVBlockSize
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
                // KVBlockSize,
            )
        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------

        # ===---------------------------------------------------
        # Precompute First Iteration
        # ===---------------------------------------------------
        zero = gl.zeros((ChunkQ, ChunkKPerStage), dtype=tl.float32, layout=mfma_layout)

        k = k_next_0
        k_scale_f = k_scale_f_next_0

        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------

        context_kv_idx_next_1 = tl.where(mask_kv_next_1, context_kv_idx_next_1, 0)
        k_next_1 = gl.amd.cdna3.buffer_load(
            ptr=KV_buffer,
            offsets=offset_k_fixed + context_kv_idx_next_1[None, :] * stride_k_seq,
        )
        k_scale_f_next_1 = gl.amd.cdna3.buffer_load(
            ptr=scale_buffer,
            offsets=context_kv_idx_next_1 * stride_scale_seq
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            % KVBlockSize,
        )

        _amd_s_set_prio(3)
        mfma_k = gl.convert_layout(k, mfma_layout_b)
        o = gl.amd.cdna3.mfma(mfma_q, mfma_k, zero)

        _amd_iglp_sched_group_barrier(MFMA, 8, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        _amd_iglp_sched_group_barrier(MFMA, 8, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        _amd_iglp_sched_group_barrier(MFMA, 8, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        _amd_iglp_sched_group_barrier(MFMA, 8, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------

        k_scale_f = gl.convert_layout(k_scale_f, gl.SliceLayout(0, mfma_layout))

        o = o * k_scale_f[None, :]
        o = gl.maximum(o, 0.0)
        o = o * scale_weight[:, None]
        _amd_s_set_prio(1)

        logits = gl.reduce(o, axis=0, combine_fn=_sum_combine)
        gl.amd.cdna3.buffer_store(
            logits,
            ptr=OutLogits_buffer,
            offsets=(pid_batch * next_n + pid_next_n) * stride_out_batch
            + (
                context_idx
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
            ),
            mask=context_idx
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
            >= split_context_start,
        )

        for context_idx in range(
            split_context_start - residual_context,
            split_context_start + split_context_length - ChunkK,
            ChunkK,
        ):
            k = k_next_1
            k_scale_f = k_scale_f_next_1

            #!=----------------------------
            _amd_iglp_sched_barrier(0x0)
            #!=----------------------------

            context_kv_idx_next_1 = gl.amd.cdna3.buffer_load(
                ptr=kv_indices,
                offsets=pid_batch * max_block_len
                + (context_idx + ChunkK + ChunkKPerStage) // KVBlockSize
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
                // KVBlockSize,
            )
            k_next_0 = gl.amd.cdna3.buffer_load(
                ptr=KV_buffer,
                offsets=offset_k_fixed + context_kv_idx_next_0[None, :] * stride_k_seq,
            )
            k_scale_f_next_0 = gl.amd.cdna3.buffer_load(
                ptr=scale_buffer,
                offsets=context_kv_idx_next_0 * stride_scale_seq
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
                % KVBlockSize,
            )

            _amd_s_set_prio(3)
            mfma_k = gl.convert_layout(k, mfma_layout_b)
            o = gl.amd.cdna3.mfma(mfma_q, mfma_k, zero)

            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            #!=----------------------------
            _amd_iglp_sched_barrier(0x0)
            #!=----------------------------
            k_scale_f = gl.convert_layout(k_scale_f, gl.SliceLayout(0, mfma_layout))
            o = o * k_scale_f[None, :]
            o = gl.maximum(o, 0.0)
            o = o * scale_weight[:, None]
            _amd_s_set_prio(1)

            logits = gl.reduce(o, axis=0, combine_fn=_sum_combine)
            gl.amd.cdna3.buffer_store(
                logits,
                ptr=OutLogits_buffer,
                offsets=(pid_batch * next_n + pid_next_n) * stride_out_batch
                + (
                    context_idx
                    + ChunkKPerStage
                    + gl.arange(
                        0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout)
                    )
                ),
                mask=context_idx
                + ChunkKPerStage
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
                >= split_context_start,
            )

            # =======================================================================================

            k = k_next_0
            k_scale_f = k_scale_f_next_0

            # #!=----------------------------
            _amd_iglp_sched_barrier(0x0)
            # #!=----------------------------
            if (
                context_idx + ChunkK + ChunkK
                < split_context_start + split_context_length
            ):
                context_kv_idx_next_0 = gl.amd.cdna3.buffer_load(
                    ptr=kv_indices,
                    offsets=pid_batch * max_block_len
                    + (context_idx + ChunkK + ChunkK) // KVBlockSize
                    + gl.arange(
                        0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b)
                    )
                    // KVBlockSize,
                )
            k_next_1 = gl.amd.cdna3.buffer_load(
                ptr=KV_buffer,
                offsets=offset_k_fixed + context_kv_idx_next_1[None, :] * stride_k_seq,
            )
            k_scale_f_next_1 = gl.amd.cdna3.buffer_load(
                ptr=scale_buffer,
                offsets=context_kv_idx_next_1 * stride_scale_seq
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
                % KVBlockSize,
            )
            _amd_s_set_prio(2)
            mfma_k = gl.convert_layout(k, mfma_layout_b)
            o = gl.amd.cdna3.mfma(mfma_q, mfma_k, zero)

            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            #!=----------------------------
            _amd_iglp_sched_barrier(0x0)
            #!=----------------------------

            k_scale_f = gl.convert_layout(k_scale_f, gl.SliceLayout(0, mfma_layout))

            o = o * k_scale_f[None, :]
            o = gl.maximum(o, 0.0)
            o = o * scale_weight[:, None]
            _amd_s_set_prio(0)

            logits = gl.reduce(o, axis=0, combine_fn=_sum_combine)
            gl.amd.cdna3.buffer_store(
                logits,
                ptr=OutLogits_buffer,
                offsets=(pid_batch * next_n + pid_next_n) * stride_out_batch
                + (
                    context_idx
                    + ChunkK
                    + gl.arange(
                        0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout)
                    )
                ),
            )

        context_idx = split_context_start + split_context_length - ChunkK

        k = k_next_1
        k_scale_f = k_scale_f_next_1

        _amd_s_set_prio(1)
        mfma_k = gl.convert_layout(k, mfma_layout_b)
        o = gl.amd.cdna3.mfma(mfma_q, mfma_k, zero)
        k_scale_f = gl.convert_layout(k_scale_f, gl.SliceLayout(0, mfma_layout))
        o = o * k_scale_f[None, :]
        o = gl.maximum(o, 0.0)
        o = o * scale_weight[:, None]
        _amd_s_set_prio(0)

        mask = (
            context_idx
            + ChunkKPerStage
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
            <= context_length - next_n + pid_next_n
        )

        logits = gl.reduce(o, axis=0, combine_fn=_sum_combine)
        logits = tl.where(mask, logits, float("-inf"))
        gl.amd.cdna3.buffer_store(
            logits,
            ptr=OutLogits_buffer,
            offsets=(pid_batch * next_n + pid_next_n) * stride_out_batch
            + (
                context_idx
                + ChunkKPerStage
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
            ),
            mask=context_idx
            + ChunkKPerStage
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
            >= split_context_start,
        )
    else:
        context_idx = split_context_start
        current_chunk_rank = context_idx // ChunkKPerStage % ChunkKStagePerContextBlock
        block_idx = context_idx // KVBlockSize
        batch_blocks = tl.cdiv(context_length, KVBlockSize)

        q = gl.amd.cdna3.buffer_load(
            ptr=Q_buffer,
            offsets=pid_batch * stride_q_batch
            + pid_next_n * stride_q_next_n
            + (
                gl.arange(0, ChunkQ, layout=gl.SliceLayout(1, layout_q))
                * stride_q_heads
            )[:, None]
            + gl.arange(0, HiddenDim, layout=gl.SliceLayout(0, layout_q))[None, :],
        )

        context_kv_idx_next_0 = gl.load(
            kv_indices + pid_batch * max_block_len + block_idx
        )
        block_idx += 1

        context_kv_idx_next_1 = gl.load(
            kv_indices + pid_batch * max_block_len + block_idx,
            mask=block_idx < batch_blocks,
        )
        block_idx += 1

        scale_weight = gl.amd.cdna3.buffer_load(
            ptr=weights,
            offsets=(pid_batch * next_n + pid_next_n) * stride_w_batch
            + gl.arange(0, ChunkQ, layout=layout_scale),
        )

        offset_k_fixed = (
            gl.arange(0, HiddenDim, layout=gl.SliceLayout(1, mfma_layout_b)) % 16
            + gl.arange(0, HiddenDim, layout=gl.SliceLayout(1, mfma_layout_b))
            // 16
            * 256
        )[:, None] + (
            gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            % 16
            * 16
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            % KVBlockSize
            // 16
            * 16
            * 128
        )[
            None, :
        ]

        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------
        mfma_q = gl.convert_layout(q, mfma_layout_a)

        k_next_0 = gl.amd.cdna3.buffer_load(
            ptr=KV_buffer,
            offsets=offset_k_fixed
            + context_kv_idx_next_0 * stride_k_seq
            + context_idx % KVBlockSize * HiddenDim,
        )
        k_scale_f_next_0 = gl.amd.cdna3.buffer_load(
            ptr=scale_buffer,
            offsets=context_kv_idx_next_0 * stride_scale_seq
            + (
                context_idx
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            )
            % KVBlockSize,
        )

        _amd_iglp_sched_group_barrier(DS_READ, 4, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 4, 0)
        _amd_iglp_sched_group_barrier(DS_READ, 2, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        _amd_iglp_sched_group_barrier(DS_READ, 2, 0)

        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------

        # ===---------------------------------------------------
        # Precompute First Iteration
        # ===---------------------------------------------------
        zero = gl.zeros((ChunkQ, ChunkKPerStage), dtype=tl.float32, layout=mfma_layout)

        k = k_next_0
        k_scale_f = k_scale_f_next_0

        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------
        k_next_1 = gl.amd.cdna3.buffer_load(
            ptr=KV_buffer,
            offsets=offset_k_fixed
            + context_kv_idx_next_0 * stride_k_seq
            + (context_idx + ChunkKPerStage) % KVBlockSize * HiddenDim,
        )
        k_scale_f_next_1 = gl.amd.cdna3.buffer_load(
            ptr=scale_buffer,
            offsets=context_kv_idx_next_0 * stride_scale_seq
            + (
                context_idx
                + ChunkKPerStage
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            )
            % KVBlockSize,
        )
        current_chunk_rank += 2

        _amd_s_set_prio(3)
        mfma_k = gl.convert_layout(k, mfma_layout_b)
        o = gl.amd.cdna3.mfma(mfma_q, mfma_k, zero)

        _amd_iglp_sched_group_barrier(MFMA, 8, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        _amd_iglp_sched_group_barrier(MFMA, 8, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        _amd_iglp_sched_group_barrier(MFMA, 8, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        _amd_iglp_sched_group_barrier(MFMA, 8, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------

        k_scale_f = gl.convert_layout(k_scale_f, gl.SliceLayout(0, mfma_layout))

        o = o * k_scale_f[None, :]
        o = gl.maximum(o, 0.0)
        o = o * scale_weight[:, None]
        _amd_s_set_prio(1)

        mask = (
            context_idx
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
            <= context_length - next_n + pid_next_n
        )

        logits = gl.reduce(o, axis=0, combine_fn=_sum_combine)
        logits = tl.where(mask, logits, float("-inf"))
        gl.amd.cdna3.buffer_store(
            logits,
            ptr=OutLogits_buffer,
            offsets=(pid_batch * next_n + pid_next_n) * stride_out_batch
            + (
                context_idx
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
            ),
        )

        for context_idx_ in range(
            split_context_start,
            split_context_start + split_context_length - ChunkK,
            ChunkK,
        ):
            k = k_next_1
            k_scale_f = k_scale_f_next_1

            #!=----------------------------
            _amd_iglp_sched_barrier(0x0)
            #!=----------------------------

            if current_chunk_rank == ChunkKStagePerContextBlock:
                current_chunk_rank = 0
                context_kv_idx_next_0 = context_kv_idx_next_1
                context_kv_idx_next_1 = gl.load(
                    kv_indices + pid_batch * max_block_len + block_idx,
                    mask=block_idx < batch_blocks,
                )
                block_idx += 1

            k_next_0 = gl.amd.cdna3.buffer_load(
                ptr=KV_buffer,
                offsets=offset_k_fixed
                + context_kv_idx_next_0 * stride_k_seq
                + (context_idx_ + ChunkK) % KVBlockSize * HiddenDim,
            )
            k_scale_f_next_0 = gl.amd.cdna3.buffer_load(
                ptr=scale_buffer,
                offsets=context_kv_idx_next_0 * stride_scale_seq
                + (
                    context_idx_
                    + ChunkK
                    + gl.arange(
                        0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b)
                    )
                )
                % KVBlockSize,
            )

            _amd_s_set_prio(3)
            mfma_k = gl.convert_layout(k, mfma_layout_b)
            o = gl.amd.cdna3.mfma(mfma_q, mfma_k, zero)

            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            #!=----------------------------
            _amd_iglp_sched_barrier(0x0)
            #!=----------------------------
            k_scale_f = gl.convert_layout(k_scale_f, gl.SliceLayout(0, mfma_layout))
            o = o * k_scale_f[None, :]
            o = gl.maximum(o, 0.0)
            o = o * scale_weight[:, None]
            _amd_s_set_prio(1)

            logits = gl.reduce(o, axis=0, combine_fn=_sum_combine)
            gl.amd.cdna3.buffer_store(
                logits,
                ptr=OutLogits_buffer,
                offsets=(pid_batch * next_n + pid_next_n) * stride_out_batch
                + (
                    context_idx_
                    + ChunkKPerStage
                    + gl.arange(
                        0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout)
                    )
                ),
            )

            # =======================================================================================

            k = k_next_0
            k_scale_f = k_scale_f_next_0

            #!=----------------------------
            _amd_iglp_sched_barrier(0x0)
            #!=----------------------------
            k_next_1 = gl.amd.cdna3.buffer_load(
                ptr=KV_buffer,
                offsets=offset_k_fixed
                + context_kv_idx_next_0 * stride_k_seq
                + (context_idx_ + ChunkK + ChunkKPerStage) % KVBlockSize * HiddenDim,
            )
            k_scale_f_next_1 = gl.amd.cdna3.buffer_load(
                ptr=scale_buffer,
                offsets=context_kv_idx_next_0 * stride_scale_seq
                + (
                    context_idx_
                    + ChunkK
                    + ChunkKPerStage
                    + gl.arange(
                        0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b)
                    )
                )
                % KVBlockSize,
            )
            current_chunk_rank += 2

            _amd_s_set_prio(2)
            mfma_k = gl.convert_layout(k, mfma_layout_b)
            o = gl.amd.cdna3.mfma(mfma_q, mfma_k, zero)

            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            #!=----------------------------
            _amd_iglp_sched_barrier(0x0)
            #!=----------------------------

            k_scale_f = gl.convert_layout(k_scale_f, gl.SliceLayout(0, mfma_layout))

            o = o * k_scale_f[None, :]
            o = gl.maximum(o, 0.0)
            o = o * scale_weight[:, None]
            _amd_s_set_prio(0)

            mask = (
                context_idx_
                + ChunkK
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
                <= context_length - next_n + pid_next_n
            )

            logits = gl.reduce(o, axis=0, combine_fn=_sum_combine)
            logits = tl.where(mask, logits, float("-inf"))
            gl.amd.cdna3.buffer_store(
                logits,
                ptr=OutLogits_buffer,
                offsets=(pid_batch * next_n + pid_next_n) * stride_out_batch
                + (
                    context_idx_
                    + ChunkK
                    + gl.arange(
                        0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout)
                    )
                ),
            )
            context_idx = context_idx_ + ChunkK

        k = k_next_1
        k_scale_f = k_scale_f_next_1

        _amd_s_set_prio(1)
        mfma_k = gl.convert_layout(k, mfma_layout_b)
        o = gl.amd.cdna3.mfma(mfma_q, mfma_k, zero)
        k_scale_f = gl.convert_layout(k_scale_f, gl.SliceLayout(0, mfma_layout))
        o = o * k_scale_f[None, :]
        o = gl.maximum(o, 0.0)
        o = o * scale_weight[:, None]
        _amd_s_set_prio(0)

        mask = (
            context_idx
            + ChunkKPerStage
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
            <= context_length - next_n + pid_next_n
        )

        logits = gl.reduce(o, axis=0, combine_fn=_sum_combine)
        logits = tl.where(mask, logits, float("-inf"))
        gl.amd.cdna3.buffer_store(
            logits,
            ptr=OutLogits_buffer,
            offsets=(pid_batch * next_n + pid_next_n) * stride_out_batch
            + (
                context_idx
                + ChunkKPerStage
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
            ),
        )


@gluon.jit
def _gluon_deepgemm_fp8_paged_mqa_logits_preshuffle_varctx(
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
    safe_chunks_per_cta_ptr,
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

    ChunkKPerStage: gl.constexpr = ChunkK // 2
    MFMAPerWarp: gl.constexpr = ChunkKPerStage // 16 // NumWarps

    if _Use_2d_instr_shape_mfma_layout:
        mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[16, 16],
            transposed=False,
            warps_per_cta=[1, NumWarps],
            tiles_per_warp=[1, MFMAPerWarp],
        )
    else:
        mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[16, 16, 32],
            transposed=False,
            warps_per_cta=[1, NumWarps],
            tiles_per_warp=[1, MFMAPerWarp],
        )

    mfma_layout_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    mfma_layout_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )

    layout_scale: gl.constexpr = gl.SliceLayout(1, mfma_layout)

    ContextBlockPerChunkK: gl.constexpr = ChunkK // KVBlockSize
    ChunkKStagePerContextBlock: gl.constexpr = KVBlockSize // ChunkKPerStage

    LoadBlockIndiceForEachStage: gl.constexpr = ChunkKPerStage % KVBlockSize == 0

    # DS_WRITE: gl.constexpr = 0x200
    DS_READ: gl.constexpr = 0x100
    BUFFER_LOAD: gl.constexpr = 0x020
    MFMA: gl.constexpr = 0x008
    # VALU: gl.constexpr = 0x002

    # ===---------------------------------------------------
    # Mapping WorkTile
    # ===---------------------------------------------------
    pid = tl.program_id(0)

    pid_split_kv = pid
    safe_chunks_per_cta = gl.load(safe_chunks_per_cta_ptr)

    pid_batch = 0
    context_length = gl.load(context_len_ptr + pid_batch)

    cur_batch_chunk_num = tl.cdiv(context_length, ChunkK)
    cur_batch_cta_count = tl.cdiv(cur_batch_chunk_num, safe_chunks_per_cta)

    while pid_split_kv >= cur_batch_cta_count * next_n and cur_batch_cta_count > 0:
        pid_split_kv -= cur_batch_cta_count * next_n
        pid_batch += 1
        context_length = gl.load(
            context_len_ptr + pid_batch, mask=pid_batch < batch_size, other=0
        )
        cur_batch_chunk_num = tl.cdiv(context_length, ChunkK)
        cur_batch_cta_count = tl.cdiv(cur_batch_chunk_num, safe_chunks_per_cta)

    if context_length == 0:
        return

    pid_next_n = pid_split_kv % next_n
    pid_split_kv = pid_split_kv // next_n

    split_context_chunk_num = cur_batch_chunk_num // cur_batch_cta_count
    residual_context_chunks = cur_batch_chunk_num % cur_batch_cta_count
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

    if LoadBlockIndiceForEachStage:
        split_context_block = tl.cdiv(split_context_length, KVBlockSize)
        split_context_length = split_context_block * KVBlockSize

        residual_context_blocks = (
            ContextBlockPerChunkK - split_context_block % ContextBlockPerChunkK
        ) % ContextBlockPerChunkK
        residual_context = residual_context_blocks * KVBlockSize

        # ===---------------------------------------------------
        # Pipeline Start
        _amd_iglp_sched_barrier(0x0)
        # ===---------------------------------------------------
        q = gl.amd.cdna3.buffer_load(
            ptr=Q_buffer,
            offsets=pid_batch * stride_q_batch
            + pid_next_n * stride_q_next_n
            + (
                gl.arange(0, ChunkQ, layout=gl.SliceLayout(1, layout_q))
                * stride_q_heads
            )[:, None]
            + gl.arange(0, HiddenDim, layout=gl.SliceLayout(0, layout_q))[None, :],
        )

        context_idx = split_context_start - residual_context

        mask_kv_next_0 = (
            context_idx // KVBlockSize
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            // KVBlockSize
        ) >= split_context_start // KVBlockSize
        context_kv_idx_next_0 = gl.amd.cdna3.buffer_load(
            ptr=kv_indices,
            offsets=pid_batch * max_block_len
            + context_idx // KVBlockSize
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            // KVBlockSize,
            mask=mask_kv_next_0,
        )

        mask_kv_next_1 = (
            (context_idx + ChunkKPerStage) // KVBlockSize
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            // KVBlockSize
        ) >= split_context_start // KVBlockSize
        context_kv_idx_next_1 = gl.amd.cdna3.buffer_load(
            ptr=kv_indices,
            offsets=pid_batch * max_block_len
            + (context_idx + ChunkKPerStage) // KVBlockSize
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            // KVBlockSize,
            mask=mask_kv_next_1,
        )

        scale_weight = gl.amd.cdna3.buffer_load(
            ptr=weights,
            offsets=(pid_batch * next_n + pid_next_n) * stride_w_batch
            + gl.arange(0, ChunkQ, layout=layout_scale),
        )

        offset_k_fixed = (
            gl.arange(0, HiddenDim, layout=gl.SliceLayout(1, mfma_layout_b)) % 16
            + gl.arange(0, HiddenDim, layout=gl.SliceLayout(1, mfma_layout_b))
            // 16
            * 256
        )[:, None] + (
            gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            % 16
            * 16
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            % KVBlockSize
            // 16
            * 16
            * 128
        )[
            None, :
        ]

        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------
        mfma_q = gl.convert_layout(q, mfma_layout_a)

        context_kv_idx_next_0 = tl.where(mask_kv_next_0, context_kv_idx_next_0, 0)
        k_next_0 = gl.amd.cdna3.buffer_load(
            ptr=KV_buffer,
            offsets=offset_k_fixed + context_kv_idx_next_0[None, :] * stride_k_seq,
        )
        k_scale_f_next_0 = gl.amd.cdna3.buffer_load(
            ptr=scale_buffer,
            offsets=context_kv_idx_next_0 * stride_scale_seq
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            % KVBlockSize,
        )

        _amd_iglp_sched_group_barrier(DS_READ, 4, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 4, 0)
        _amd_iglp_sched_group_barrier(DS_READ, 2, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        _amd_iglp_sched_group_barrier(DS_READ, 2, 0)

        if context_idx + ChunkK < split_context_start + split_context_length:
            context_kv_idx_next_0 = gl.amd.cdna3.buffer_load(
                ptr=kv_indices,
                offsets=pid_batch * max_block_len
                + (context_idx + ChunkK) // KVBlockSize
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
                // KVBlockSize,
            )
        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------

        # ===---------------------------------------------------
        # Precompute First Iteration
        # ===---------------------------------------------------
        zero = gl.zeros((ChunkQ, ChunkKPerStage), dtype=tl.float32, layout=mfma_layout)

        k = k_next_0
        k_scale_f = k_scale_f_next_0

        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------

        context_kv_idx_next_1 = tl.where(mask_kv_next_1, context_kv_idx_next_1, 0)
        k_next_1 = gl.amd.cdna3.buffer_load(
            ptr=KV_buffer,
            offsets=offset_k_fixed + context_kv_idx_next_1[None, :] * stride_k_seq,
        )
        k_scale_f_next_1 = gl.amd.cdna3.buffer_load(
            ptr=scale_buffer,
            offsets=context_kv_idx_next_1 * stride_scale_seq
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            % KVBlockSize,
        )

        _amd_s_set_prio(3)
        mfma_k = gl.convert_layout(k, mfma_layout_b)
        o = gl.amd.cdna3.mfma(mfma_q, mfma_k, zero)

        _amd_iglp_sched_group_barrier(MFMA, 8, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        _amd_iglp_sched_group_barrier(MFMA, 8, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        _amd_iglp_sched_group_barrier(MFMA, 8, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        _amd_iglp_sched_group_barrier(MFMA, 8, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------

        k_scale_f = gl.convert_layout(k_scale_f, gl.SliceLayout(0, mfma_layout))

        o = o * k_scale_f[None, :]
        o = gl.maximum(o, 0.0)
        o = o * scale_weight[:, None]
        _amd_s_set_prio(1)

        logits = gl.reduce(o, axis=0, combine_fn=_sum_combine)
        gl.amd.cdna3.buffer_store(
            logits,
            ptr=OutLogits_buffer,
            offsets=(pid_batch * next_n + pid_next_n) * stride_out_batch
            + (
                context_idx
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
            ),
            mask=context_idx
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
            >= split_context_start,
        )

        for context_idx in range(
            split_context_start - residual_context,
            split_context_start + split_context_length - ChunkK,
            ChunkK,
        ):
            k = k_next_1
            k_scale_f = k_scale_f_next_1

            #!=----------------------------
            _amd_iglp_sched_barrier(0x0)
            #!=----------------------------

            context_kv_idx_next_1 = gl.amd.cdna3.buffer_load(
                ptr=kv_indices,
                offsets=pid_batch * max_block_len
                + (context_idx + ChunkK + ChunkKPerStage) // KVBlockSize
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
                // KVBlockSize,
            )
            k_next_0 = gl.amd.cdna3.buffer_load(
                ptr=KV_buffer,
                offsets=offset_k_fixed + context_kv_idx_next_0[None, :] * stride_k_seq,
            )
            k_scale_f_next_0 = gl.amd.cdna3.buffer_load(
                ptr=scale_buffer,
                offsets=context_kv_idx_next_0 * stride_scale_seq
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
                % KVBlockSize,
            )

            _amd_s_set_prio(3)
            mfma_k = gl.convert_layout(k, mfma_layout_b)
            o = gl.amd.cdna3.mfma(mfma_q, mfma_k, zero)

            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            #!=----------------------------
            _amd_iglp_sched_barrier(0x0)
            #!=----------------------------
            k_scale_f = gl.convert_layout(k_scale_f, gl.SliceLayout(0, mfma_layout))
            o = o * k_scale_f[None, :]
            o = gl.maximum(o, 0.0)
            o = o * scale_weight[:, None]
            _amd_s_set_prio(1)

            logits = gl.reduce(o, axis=0, combine_fn=_sum_combine)
            gl.amd.cdna3.buffer_store(
                logits,
                ptr=OutLogits_buffer,
                offsets=(pid_batch * next_n + pid_next_n) * stride_out_batch
                + (
                    context_idx
                    + ChunkKPerStage
                    + gl.arange(
                        0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout)
                    )
                ),
                mask=context_idx
                + ChunkKPerStage
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
                >= split_context_start,
            )

            # =======================================================================================

            k = k_next_0
            k_scale_f = k_scale_f_next_0

            # #!=----------------------------
            _amd_iglp_sched_barrier(0x0)
            # #!=----------------------------
            if (
                context_idx + ChunkK + ChunkK
                < split_context_start + split_context_length
            ):
                context_kv_idx_next_0 = gl.amd.cdna3.buffer_load(
                    ptr=kv_indices,
                    offsets=pid_batch * max_block_len
                    + (context_idx + ChunkK + ChunkK) // KVBlockSize
                    + gl.arange(
                        0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b)
                    )
                    // KVBlockSize,
                )
            k_next_1 = gl.amd.cdna3.buffer_load(
                ptr=KV_buffer,
                offsets=offset_k_fixed + context_kv_idx_next_1[None, :] * stride_k_seq,
            )
            k_scale_f_next_1 = gl.amd.cdna3.buffer_load(
                ptr=scale_buffer,
                offsets=context_kv_idx_next_1 * stride_scale_seq
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
                % KVBlockSize,
            )
            _amd_s_set_prio(2)
            mfma_k = gl.convert_layout(k, mfma_layout_b)
            o = gl.amd.cdna3.mfma(mfma_q, mfma_k, zero)

            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            #!=----------------------------
            _amd_iglp_sched_barrier(0x0)
            #!=----------------------------

            k_scale_f = gl.convert_layout(k_scale_f, gl.SliceLayout(0, mfma_layout))

            o = o * k_scale_f[None, :]
            o = gl.maximum(o, 0.0)
            o = o * scale_weight[:, None]
            _amd_s_set_prio(0)

            logits = gl.reduce(o, axis=0, combine_fn=_sum_combine)
            gl.amd.cdna3.buffer_store(
                logits,
                ptr=OutLogits_buffer,
                offsets=(pid_batch * next_n + pid_next_n) * stride_out_batch
                + (
                    context_idx
                    + ChunkK
                    + gl.arange(
                        0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout)
                    )
                ),
            )

        context_idx = split_context_start + split_context_length - ChunkK

        k = k_next_1
        k_scale_f = k_scale_f_next_1

        _amd_s_set_prio(1)
        mfma_k = gl.convert_layout(k, mfma_layout_b)
        o = gl.amd.cdna3.mfma(mfma_q, mfma_k, zero)
        k_scale_f = gl.convert_layout(k_scale_f, gl.SliceLayout(0, mfma_layout))
        o = o * k_scale_f[None, :]
        o = gl.maximum(o, 0.0)
        o = o * scale_weight[:, None]
        _amd_s_set_prio(0)

        mask = (
            context_idx
            + ChunkKPerStage
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
            <= context_length - next_n + pid_next_n
        )

        logits = gl.reduce(o, axis=0, combine_fn=_sum_combine)
        logits = tl.where(mask, logits, float("-inf"))
        gl.amd.cdna3.buffer_store(
            logits,
            ptr=OutLogits_buffer,
            offsets=(pid_batch * next_n + pid_next_n) * stride_out_batch
            + (
                context_idx
                + ChunkKPerStage
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
            ),
            mask=context_idx
            + ChunkKPerStage
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
            >= split_context_start,
        )
    else:
        context_idx = split_context_start
        current_chunk_rank = context_idx // ChunkKPerStage % ChunkKStagePerContextBlock
        block_idx = context_idx // KVBlockSize
        batch_blocks = tl.cdiv(context_length, KVBlockSize)

        q = gl.amd.cdna3.buffer_load(
            ptr=Q_buffer,
            offsets=pid_batch * stride_q_batch
            + pid_next_n * stride_q_next_n
            + (
                gl.arange(0, ChunkQ, layout=gl.SliceLayout(1, layout_q))
                * stride_q_heads
            )[:, None]
            + gl.arange(0, HiddenDim, layout=gl.SliceLayout(0, layout_q))[None, :],
        )

        context_kv_idx_next_0 = gl.load(
            kv_indices + pid_batch * max_block_len + block_idx
        )
        block_idx += 1

        context_kv_idx_next_1 = gl.load(
            kv_indices + pid_batch * max_block_len + block_idx,
            mask=block_idx < batch_blocks,
        )
        block_idx += 1

        scale_weight = gl.amd.cdna3.buffer_load(
            ptr=weights,
            offsets=(pid_batch * next_n + pid_next_n) * stride_w_batch
            + gl.arange(0, ChunkQ, layout=layout_scale),
        )

        offset_k_fixed = (
            gl.arange(0, HiddenDim, layout=gl.SliceLayout(1, mfma_layout_b)) % 16
            + gl.arange(0, HiddenDim, layout=gl.SliceLayout(1, mfma_layout_b))
            // 16
            * 256
        )[:, None] + (
            gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            % 16
            * 16
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            % KVBlockSize
            // 16
            * 16
            * 128
        )[
            None, :
        ]

        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------
        mfma_q = gl.convert_layout(q, mfma_layout_a)

        k_next_0 = gl.amd.cdna3.buffer_load(
            ptr=KV_buffer,
            offsets=offset_k_fixed
            + context_kv_idx_next_0 * stride_k_seq
            + context_idx % KVBlockSize * HiddenDim,
        )
        k_scale_f_next_0 = gl.amd.cdna3.buffer_load(
            ptr=scale_buffer,
            offsets=context_kv_idx_next_0 * stride_scale_seq
            + (
                context_idx
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            )
            % KVBlockSize,
        )

        _amd_iglp_sched_group_barrier(DS_READ, 4, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 4, 0)
        _amd_iglp_sched_group_barrier(DS_READ, 2, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        _amd_iglp_sched_group_barrier(DS_READ, 2, 0)

        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------

        # ===---------------------------------------------------
        # Precompute First Iteration
        # ===---------------------------------------------------
        zero = gl.zeros((ChunkQ, ChunkKPerStage), dtype=tl.float32, layout=mfma_layout)

        k = k_next_0
        k_scale_f = k_scale_f_next_0

        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------
        k_next_1 = gl.amd.cdna3.buffer_load(
            ptr=KV_buffer,
            offsets=offset_k_fixed
            + context_kv_idx_next_0 * stride_k_seq
            + (context_idx + ChunkKPerStage) % KVBlockSize * HiddenDim,
        )
        k_scale_f_next_1 = gl.amd.cdna3.buffer_load(
            ptr=scale_buffer,
            offsets=context_kv_idx_next_0 * stride_scale_seq
            + (
                context_idx
                + ChunkKPerStage
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            )
            % KVBlockSize,
        )
        current_chunk_rank += 2

        _amd_s_set_prio(3)
        mfma_k = gl.convert_layout(k, mfma_layout_b)
        o = gl.amd.cdna3.mfma(mfma_q, mfma_k, zero)

        _amd_iglp_sched_group_barrier(MFMA, 8, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        _amd_iglp_sched_group_barrier(MFMA, 8, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        _amd_iglp_sched_group_barrier(MFMA, 8, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        _amd_iglp_sched_group_barrier(MFMA, 8, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------

        k_scale_f = gl.convert_layout(k_scale_f, gl.SliceLayout(0, mfma_layout))

        o = o * k_scale_f[None, :]
        o = gl.maximum(o, 0.0)
        o = o * scale_weight[:, None]
        _amd_s_set_prio(1)

        mask = (
            context_idx
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
            <= context_length - next_n + pid_next_n
        )

        logits = gl.reduce(o, axis=0, combine_fn=_sum_combine)
        logits = tl.where(mask, logits, float("-inf"))
        gl.amd.cdna3.buffer_store(
            logits,
            ptr=OutLogits_buffer,
            offsets=(pid_batch * next_n + pid_next_n) * stride_out_batch
            + (
                context_idx
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
            ),
        )

        for context_idx_ in range(
            split_context_start,
            split_context_start + split_context_length - ChunkK,
            ChunkK,
        ):
            k = k_next_1
            k_scale_f = k_scale_f_next_1

            #!=----------------------------
            _amd_iglp_sched_barrier(0x0)
            #!=----------------------------

            if current_chunk_rank == ChunkKStagePerContextBlock:
                current_chunk_rank = 0
                context_kv_idx_next_0 = context_kv_idx_next_1
                context_kv_idx_next_1 = gl.load(
                    kv_indices + pid_batch * max_block_len + block_idx,
                    mask=block_idx < batch_blocks,
                )
                block_idx += 1

            k_next_0 = gl.amd.cdna3.buffer_load(
                ptr=KV_buffer,
                offsets=offset_k_fixed
                + context_kv_idx_next_0 * stride_k_seq
                + (context_idx_ + ChunkK) % KVBlockSize * HiddenDim,
            )
            k_scale_f_next_0 = gl.amd.cdna3.buffer_load(
                ptr=scale_buffer,
                offsets=context_kv_idx_next_0 * stride_scale_seq
                + (
                    context_idx_
                    + ChunkK
                    + gl.arange(
                        0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b)
                    )
                )
                % KVBlockSize,
            )

            _amd_s_set_prio(3)
            mfma_k = gl.convert_layout(k, mfma_layout_b)
            o = gl.amd.cdna3.mfma(mfma_q, mfma_k, zero)

            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            #!=----------------------------
            _amd_iglp_sched_barrier(0x0)
            #!=----------------------------
            k_scale_f = gl.convert_layout(k_scale_f, gl.SliceLayout(0, mfma_layout))
            o = o * k_scale_f[None, :]
            o = gl.maximum(o, 0.0)
            o = o * scale_weight[:, None]
            _amd_s_set_prio(1)

            logits = gl.reduce(o, axis=0, combine_fn=_sum_combine)
            gl.amd.cdna3.buffer_store(
                logits,
                ptr=OutLogits_buffer,
                offsets=(pid_batch * next_n + pid_next_n) * stride_out_batch
                + (
                    context_idx_
                    + ChunkKPerStage
                    + gl.arange(
                        0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout)
                    )
                ),
            )

            # =======================================================================================

            k = k_next_0
            k_scale_f = k_scale_f_next_0

            #!=----------------------------
            _amd_iglp_sched_barrier(0x0)
            #!=----------------------------
            k_next_1 = gl.amd.cdna3.buffer_load(
                ptr=KV_buffer,
                offsets=offset_k_fixed
                + context_kv_idx_next_0 * stride_k_seq
                + (context_idx_ + ChunkK + ChunkKPerStage) % KVBlockSize * HiddenDim,
            )
            k_scale_f_next_1 = gl.amd.cdna3.buffer_load(
                ptr=scale_buffer,
                offsets=context_kv_idx_next_0 * stride_scale_seq
                + (
                    context_idx_
                    + ChunkK
                    + ChunkKPerStage
                    + gl.arange(
                        0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b)
                    )
                )
                % KVBlockSize,
            )
            current_chunk_rank += 2

            _amd_s_set_prio(2)
            mfma_k = gl.convert_layout(k, mfma_layout_b)
            o = gl.amd.cdna3.mfma(mfma_q, mfma_k, zero)

            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            #!=----------------------------
            _amd_iglp_sched_barrier(0x0)
            #!=----------------------------

            k_scale_f = gl.convert_layout(k_scale_f, gl.SliceLayout(0, mfma_layout))

            o = o * k_scale_f[None, :]
            o = gl.maximum(o, 0.0)
            o = o * scale_weight[:, None]
            _amd_s_set_prio(0)

            mask = (
                context_idx_
                + ChunkK
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
                <= context_length - next_n + pid_next_n
            )

            logits = gl.reduce(o, axis=0, combine_fn=_sum_combine)
            logits = tl.where(mask, logits, float("-inf"))
            gl.amd.cdna3.buffer_store(
                logits,
                ptr=OutLogits_buffer,
                offsets=(pid_batch * next_n + pid_next_n) * stride_out_batch
                + (
                    context_idx_
                    + ChunkK
                    + gl.arange(
                        0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout)
                    )
                ),
            )
            context_idx = context_idx_ + ChunkK

        k = k_next_1
        k_scale_f = k_scale_f_next_1

        _amd_s_set_prio(1)
        mfma_k = gl.convert_layout(k, mfma_layout_b)
        o = gl.amd.cdna3.mfma(mfma_q, mfma_k, zero)
        k_scale_f = gl.convert_layout(k_scale_f, gl.SliceLayout(0, mfma_layout))
        o = o * k_scale_f[None, :]
        o = gl.maximum(o, 0.0)
        o = o * scale_weight[:, None]
        _amd_s_set_prio(0)

        mask = (
            context_idx
            + ChunkKPerStage
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
            <= context_length - next_n + pid_next_n
        )

        logits = gl.reduce(o, axis=0, combine_fn=_sum_combine)
        logits = tl.where(mask, logits, float("-inf"))
        gl.amd.cdna3.buffer_store(
            logits,
            ptr=OutLogits_buffer,
            offsets=(pid_batch * next_n + pid_next_n) * stride_out_batch
            + (
                context_idx
                + ChunkKPerStage
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
            ),
        )
