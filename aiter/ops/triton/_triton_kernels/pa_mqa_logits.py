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


@triton.jit
def _deepgemm_fp8_paged_mqa_logits_stage1_ragged_k(
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
    prefix_sum_context_lens,
    kv_indices,
    weights,
    stride_w_batch,
    Out_buffer,
    stride_out_heads,
    stride_out_batch,
    max_model_len,
    ChunkQ: tl.constexpr,
    ChunkK: tl.constexpr,
    HiddenDim: tl.constexpr,
    SplitKV: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    num_block_q_head = tl.cdiv(heads_num, ChunkQ)

    pid_q_head, remain_pid = pid % num_block_q_head, pid // num_block_q_head
    pid_next_n, remain_pid = remain_pid % next_n, remain_pid // next_n
    pid_batch, pid_split_kv = remain_pid % batch_size, remain_pid // batch_size

    context_start = tl.load(prefix_sum_context_lens + pid_batch)
    context_end = tl.load(prefix_sum_context_lens + pid_batch + 1)

    context_length = context_end - context_start
    context_chunk_num = tl.cdiv(context_length, ChunkK)
    split_context_chunk_num = tl.cdiv(context_chunk_num, SplitKV)

    split_context_start = (pid_split_kv * split_context_chunk_num) * ChunkK
    split_context_length = min(context_length - split_context_start, split_context_chunk_num * ChunkK)

    q = tl.load(
        Q_buffer
        + pid_batch * stride_q_batch
        + pid_next_n * stride_q_next_n
        + ((pid_q_head * ChunkQ + tl.arange(0, ChunkQ)) * stride_q_heads)[:, None]
        + tl.arange(0, HiddenDim)[None, :],
    )
    scale_weight = tl.load(weights + (pid_batch * next_n + pid_next_n) * stride_w_batch + pid_q_head * ChunkQ + tl.arange(0, ChunkQ))

    for context_idx in range(split_context_start, split_context_start + split_context_length, ChunkK):
        mask_kv = context_idx + tl.arange(0, ChunkK) < context_length
        context_kv_idx = tl.load(
            kv_indices + context_start + context_idx + tl.arange(0, ChunkK),
            mask=mask_kv,
            other=0,
        )

        k = tl.load(
            KV_buffer + context_kv_idx[:, None] * stride_k_seq + tl.arange(0, HiddenDim)[None, :],
            mask=mask_kv[:, None],
            other=0.0,
        )
        k_scale_f = tl.load(scale_buffer + context_kv_idx[:, None] * stride_scale_seq)

        o = tl.dot(q, k.T)
        o = o * k_scale_f.T
        o = tl.maximum(o, 0.0)
        o = o * scale_weight[None, :].T

        mask = context_idx + tl.arange(0, ChunkK) <= context_length - pid_next_n
        o = tl.where(mask[None, :], o, float("-inf"))

        tl.store(
            Out_buffer
            + (pid_batch * next_n + pid_next_n) * stride_out_batch
            + (pid_q_head * ChunkQ + tl.arange(0, ChunkQ)[:, None, None]) * stride_out_heads
            + (context_idx + tl.arange(0, ChunkK)[None, None, :]),
            o[:, None, :],
        )


@triton.jit
def _deepgemm_fp8_paged_mqa_logits_ragged_k(
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
    prefix_sum_context_lens,
    kv_indices,
    weights,
    stride_w_batch,
    OutLogits_buffer,
    stride_out_batch,
    max_model_len,
    ChunkQ: tl.constexpr,
    ChunkK: tl.constexpr,
    HiddenDim: tl.constexpr,
    SplitKV: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    num_block_q_head = tl.cdiv(heads_num, ChunkQ)

    pid_q_head, remain_pid = pid % num_block_q_head, pid // num_block_q_head
    pid_next_n, remain_pid = remain_pid % next_n, remain_pid // next_n
    pid_batch, pid_split_kv = remain_pid % batch_size, remain_pid // batch_size

    context_start = tl.load(prefix_sum_context_lens + pid_batch)
    context_end = tl.load(prefix_sum_context_lens + pid_batch + 1)

    context_length = context_end - context_start
    context_chunk_num = tl.cdiv(context_length, ChunkK)
    split_context_chunk_num = tl.cdiv(context_chunk_num, SplitKV)

    split_context_start = (pid_split_kv * split_context_chunk_num) * ChunkK
    split_context_length = min(context_length - split_context_start, split_context_chunk_num * ChunkK)

    q = tl.load(
        Q_buffer
        + pid_batch * stride_q_batch
        + pid_next_n * stride_q_next_n
        + ((pid_q_head * ChunkQ + tl.arange(0, ChunkQ)) * stride_q_heads)[:, None]
        + tl.arange(0, HiddenDim)[None, :],
    )
    scale_weight = tl.load(weights + (pid_batch * next_n + pid_next_n) * stride_w_batch + pid_q_head * ChunkQ + tl.arange(0, ChunkQ))

    for context_idx in range(split_context_start, split_context_start + split_context_length, ChunkK):
        mask_kv = context_idx + tl.arange(0, ChunkK) < context_length
        context_kv_idx = tl.load(
            kv_indices + context_start + context_idx + tl.arange(0, ChunkK),
            mask=mask_kv,
            other=0,
        )

        k = tl.load(
            KV_buffer + context_kv_idx[:, None] * stride_k_seq + tl.arange(0, HiddenDim)[None, :],
            mask=mask_kv[:, None],
            other=0.0,
        )
        k_scale_f = tl.load(scale_buffer + context_kv_idx[:, None] * stride_scale_seq)

        o = tl.dot(q, k.T)
        o = o * k_scale_f.T
        o = tl.maximum(o, 0.0)
        o = o * scale_weight[None, :].T

        mask = context_idx + tl.arange(0, ChunkK) <= context_length - pid_next_n
        o = tl.where(mask[None, :], o, float("-inf"))

        logits = tl.reduce(o, axis=0, combine_fn=_sum_combine)
        tl.store(
            OutLogits_buffer + (pid_batch * next_n + pid_next_n) * stride_out_batch + (context_idx + tl.arange(0, ChunkK)),
            logits,
        )


@triton.jit
def _deepgemm_fp8_paged_mqa_logits_stage1(
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
    Out_buffer,
    stride_out_heads,
    stride_out_batch,
    max_model_len,
    max_blk_len,
    ChunkQ: tl.constexpr,
    ChunkK: tl.constexpr,
    HiddenDim: tl.constexpr,
    SplitKV: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    num_block_q_head = tl.cdiv(heads_num, ChunkQ)

    pid_q_head, remain_pid = pid % num_block_q_head, pid // num_block_q_head
    pid_next_n, remain_pid = remain_pid % next_n, remain_pid // next_n
    pid_batch, pid_split_kv = remain_pid % batch_size, remain_pid // batch_size

    context_length = tl.load(context_len_ptr + pid_batch)

    context_chunk_num = tl.cdiv(context_length, ChunkK)
    split_context_chunk_num = tl.cdiv(context_chunk_num, SplitKV)

    split_context_start = (pid_split_kv * split_context_chunk_num) * ChunkK
    split_context_length = min(context_length - split_context_start, split_context_chunk_num * ChunkK)

    q = tl.load(
        Q_buffer
        + pid_batch * stride_q_batch
        + pid_next_n * stride_q_next_n
        + ((pid_q_head * ChunkQ + tl.arange(0, ChunkQ)) * stride_q_heads)[:, None]
        + tl.arange(0, HiddenDim)[None, :],
    )
    scale_weight = tl.load(weights + (pid_batch * next_n + pid_next_n) * stride_w_batch + pid_q_head * ChunkQ + tl.arange(0, ChunkQ))

    for context_idx in range(split_context_start, split_context_start + split_context_length, ChunkK):
        mask_kv = context_idx + tl.arange(0, ChunkK) < context_length
        context_kv_idx = tl.load(
            kv_indices + pid_batch * max_blk_len + context_idx + tl.arange(0, ChunkK),
            mask=mask_kv,
            other=0,
        )

        k = tl.load(
            KV_buffer + context_kv_idx[:, None] * stride_k_seq + tl.arange(0, HiddenDim)[None, :],
            mask=mask_kv[:, None],
            other=0.0,
        )
        k_scale_f = tl.load(scale_buffer + context_kv_idx[:, None] * stride_scale_seq)

        o = tl.dot(q, k.T)
        o = o * k_scale_f.T
        o = tl.maximum(o, 0.0)
        o = o * scale_weight[None, :].T

        mask = context_idx + tl.arange(0, ChunkK) <= context_length - pid_next_n
        o = tl.where(mask[None, :], o, float("-inf"))

        tl.store(
            Out_buffer
            + (pid_batch * next_n + pid_next_n) * stride_out_batch
            + (pid_q_head * ChunkQ + tl.arange(0, ChunkQ)[:, None, None]) * stride_out_heads
            + (context_idx + tl.arange(0, ChunkK)[None, None, :]),
            o[:, None, :],
        )


@triton.jit
def _deepgemm_fp8_paged_mqa_logits(
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
    max_blk_len,
    ChunkQ: tl.constexpr,
    ChunkK: tl.constexpr,
    HiddenDim: tl.constexpr,
    SplitKV: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    num_block_q_head = tl.cdiv(heads_num, ChunkQ)

    pid_q_head, remain_pid = pid % num_block_q_head, pid // num_block_q_head
    pid_next_n, remain_pid = remain_pid % next_n, remain_pid // next_n
    pid_batch, pid_split_kv = remain_pid % batch_size, remain_pid // batch_size

    context_length = tl.load(context_len_ptr + pid_batch)

    context_chunk_num = tl.cdiv(context_length, ChunkK)
    split_context_chunk_num = tl.cdiv(context_chunk_num, SplitKV)

    split_context_start = (pid_split_kv * split_context_chunk_num) * ChunkK
    split_context_length = min(context_length - split_context_start, split_context_chunk_num * ChunkK)

    q = tl.load(
        Q_buffer
        + pid_batch * stride_q_batch
        + pid_next_n * stride_q_next_n
        + ((pid_q_head * ChunkQ + tl.arange(0, ChunkQ)) * stride_q_heads)[:, None]
        + tl.arange(0, HiddenDim)[None, :],
    )
    scale_weight = tl.load(weights + (pid_batch * next_n + pid_next_n) * stride_w_batch + pid_q_head * ChunkQ + tl.arange(0, ChunkQ))

    for context_idx in range(split_context_start, split_context_start + split_context_length, ChunkK):
        mask_kv = context_idx + tl.arange(0, ChunkK) < context_length
        context_kv_idx = tl.load(
            kv_indices + pid_batch * max_blk_len + context_idx + tl.arange(0, ChunkK),
            mask=mask_kv,
            other=0,
        )

        k = tl.load(
            KV_buffer + context_kv_idx[:, None] * stride_k_seq + tl.arange(0, HiddenDim)[None, :],
            mask=mask_kv[:, None],
            other=0.0,
        )
        k_scale_f = tl.load(scale_buffer + context_kv_idx[:, None] * stride_scale_seq)

        o = tl.dot(q, k.T)
        o = o * k_scale_f.T
        o = tl.maximum(o, 0.0)
        o = o * scale_weight[None, :].T

        mask = context_idx + tl.arange(0, ChunkK) <= context_length - pid_next_n
        o = tl.where(mask[None, :], o, float("-inf"))

        logits = tl.reduce(o, axis=0, combine_fn=_sum_combine)
        tl.store(
            OutLogits_buffer + (pid_batch * next_n + pid_next_n) * stride_out_batch + (context_idx + tl.arange(0, ChunkK)),
            logits,
        )


@gluon.jit
def _gluon_deepgemm_fp8_paged_mqa_logits_ragged_k(
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
    prefix_sum_context_lens,
    kv_indices,
    weights,
    stride_w_batch,
    OutLogits_buffer,
    stride_out_batch,
    max_model_len,
    ChunkQ: tl.constexpr,
    ChunkK: tl.constexpr,
    HiddenDim: tl.constexpr,
    SplitKV: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    num_block_q_head = tl.cdiv(heads_num, ChunkQ)

    pid_q_head, remain_pid = pid % num_block_q_head, pid // num_block_q_head
    pid_next_n, remain_pid = remain_pid % next_n, remain_pid // next_n
    pid_batch, pid_split_kv = remain_pid % batch_size, remain_pid // batch_size

    context_start = gl.load(prefix_sum_context_lens + pid_batch)
    context_end = gl.load(prefix_sum_context_lens + pid_batch + 1)

    context_length = context_end - context_start
    context_chunk_num = tl.cdiv(context_length, ChunkK)
    split_context_chunk_num = tl.cdiv(context_chunk_num, SplitKV)

    split_context_start = (pid_split_kv * split_context_chunk_num) * ChunkK
    split_context_length = min(context_length - split_context_start, split_context_chunk_num * ChunkK)

    if split_context_length <= 0:
        return

    residual_context = (ChunkK - split_context_length % ChunkK) % ChunkK

    NumWarps: gl.constexpr = 4
    ThreadsPerWarp: gl.constexpr = 64

    # ===---------------------------------------------------
    # Gluon Layout
    # ===---------------------------------------------------
    ValQMPerThread: gl.constexpr = ChunkQ // (NumWarps * ThreadsPerWarp // (HiddenDim // 16))
    layout_q: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[ValQMPerThread, 16],  # q type is fp8 (E4M3)
        threads_per_warp=[ThreadsPerWarp // (HiddenDim // 16), HiddenDim // 16],
        warps_per_cta=[NumWarps, 1],
        order=[1, 0],
    )

    ValKNPerThread: gl.constexpr = ChunkK // (NumWarps * ThreadsPerWarp // (HiddenDim // 16))
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
    mfma_layout_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=16)
    mfma_layout_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=16)

    mfma_layout_a_linear: gl.constexpr = gl.DistributedLinearLayout( # 64x128
        reg_bases=((0,1), (0,2), (0,4), (0,8), (16,0), (32,0), (0,64)),
        lane_bases=((1,0), (2,0), (4,0), (8,0), (0,16), (0,32)),
        warp_bases=((0,0), (0,0)),
        block_bases=[],
        shape=[64, 128],
    )
    mfma_layout_b_linear: gl.constexpr = gl.DistributedLinearLayout( # 128x256
        reg_bases=((1,0), (2,0), (4,0), (8,0), (0,64), (0,128), (64,0)),
        lane_bases=((0,1), (0,2), (0,4), (0,8), (16,0), (32,0)),
        warp_bases=((0,16), (0,32)),
        block_bases=[],
        shape=[128, 256],
    )

    layout_scale: gl.constexpr = gl.SliceLayout(1, mfma_layout)

    # ===---------------------------------------------------
    # Pipeline Start
    # ===---------------------------------------------------
    q = gl.amd.cdna3.buffer_load(
        ptr=Q_buffer,
        offsets=pid_batch * stride_q_batch
        + pid_next_n * stride_q_next_n
        + ((pid_q_head * ChunkQ + gl.arange(0, ChunkQ, layout=gl.SliceLayout(1, layout_q))) * stride_q_heads)[:, None]
        + gl.arange(0, HiddenDim, layout=gl.SliceLayout(0, layout_q))[None, :],
    )
    scale_weight = gl.amd.cdna3.buffer_load(
        ptr=weights,
        offsets=(pid_batch * next_n + pid_next_n) * stride_w_batch + pid_q_head * ChunkQ + gl.arange(0, ChunkQ, layout=layout_scale),
    )

    mask_kv_next = split_context_start - residual_context + gl.arange(0, ChunkK, layout=gl.SliceLayout(1, layout_kv)) >= 0
    mask_kv_scale_next = split_context_start - residual_context + gl.arange(0, ChunkK, layout=gl.SliceLayout(0, mfma_layout)) >= 0
    context_kv_idx_next = gl.amd.cdna3.buffer_load(
        ptr=kv_indices,
        offsets=context_start + split_context_start - residual_context + gl.arange(0, ChunkK, layout=gl.SliceLayout(1, layout_kv)),
        mask=mask_kv_next,
    )
    context_kv_scale_idx_next = gl.amd.cdna3.buffer_load(
        ptr=kv_indices,
        offsets=context_start + split_context_start - residual_context + gl.arange(0, ChunkK, layout=gl.SliceLayout(0, mfma_layout)),
        mask=mask_kv_scale_next,
    )

    linear_mfma_q = gl.convert_layout(q, mfma_layout_a_linear)
    linear_mfma_q = gl.reshape(linear_mfma_q, (2, 2, 16, 2, 64))
    linear_mfma_q = gl.permute(linear_mfma_q, (2, 4, 1, 0, 3))
    # split n-0 axis origin axis=3  --> 64 x 64
    linear_mfma_q0, linear_mfma_q1 = gl.split(linear_mfma_q)
    # split m-0 axis origin axis=0  --> 32 x 64
    linear_mfma_q0_0, linear_mfma_q1_0 = gl.split(linear_mfma_q0)
    linear_mfma_q0_1, linear_mfma_q1_1 = gl.split(linear_mfma_q1)
    # split n-1 axis origin axis=1  --> 16 x 64
    linear_mfma_q00_0, linear_mfma_q01_0 = gl.split(linear_mfma_q0_0)
    linear_mfma_q10_0, linear_mfma_q11_0 = gl.split(linear_mfma_q1_0)
    linear_mfma_q00_1, linear_mfma_q01_1 = gl.split(linear_mfma_q0_1)
    linear_mfma_q10_1, linear_mfma_q11_1 = gl.split(linear_mfma_q1_1)

    # 
    mfma_q00_0 = gl.convert_layout(linear_mfma_q00_0, mfma_layout_a)
    mfma_q01_0 = gl.convert_layout(linear_mfma_q01_0, mfma_layout_a)
    mfma_q10_0 = gl.convert_layout(linear_mfma_q10_0, mfma_layout_a)
    mfma_q11_0 = gl.convert_layout(linear_mfma_q11_0, mfma_layout_a)
    mfma_q00_1 = gl.convert_layout(linear_mfma_q00_1, mfma_layout_a)
    mfma_q01_1 = gl.convert_layout(linear_mfma_q01_1, mfma_layout_a)
    mfma_q10_1 = gl.convert_layout(linear_mfma_q10_1, mfma_layout_a)
    mfma_q11_1 = gl.convert_layout(linear_mfma_q11_1, mfma_layout_a)

    context_kv_idx_next = tl.where(mask_kv_next, context_kv_idx_next, 0)
    k_next = gl.amd.cdna3.buffer_load(
        ptr=KV_buffer,
        offsets=context_kv_idx_next[:, None] * stride_k_seq + gl.arange(0, HiddenDim, layout=gl.SliceLayout(0, layout_kv))[None, :],
    )
    context_kv_scale_idx_next = tl.where(mask_kv_scale_next, context_kv_scale_idx_next, 0)
    k_scale_f_next = gl.amd.cdna3.buffer_load(ptr=scale_buffer, offsets=context_kv_scale_idx_next * stride_scale_seq)

    zero = gl.zeros((16, 64), dtype=tl.float32, layout=mfma_layout)
    for context_idx in range(split_context_start - residual_context, split_context_start + split_context_length - ChunkK, ChunkK):
        k = k_next
        k_scale_f = k_scale_f_next


        context_kv_idx_next = gl.amd.cdna3.buffer_load(
            ptr=kv_indices,
            offsets=context_start + context_idx + ChunkK + gl.arange(0, ChunkK, layout=gl.SliceLayout(1, layout_kv)),
        )
        context_kv_scale_idx_next = gl.amd.cdna3.buffer_load(
            ptr=kv_indices,
            offsets=context_start + context_idx + ChunkK + gl.arange(0, ChunkK, layout=gl.SliceLayout(0, mfma_layout)),
        )

        #!=----------------------------
        gl.amd.cdna3.sched_barrier(0x0)
        #!=----------------------------
        kT = k.T
        mfma_k = gl.convert_layout(kT, mfma_layout_b_linear) 

        mfma_k = gl.reshape(mfma_k, (2, 64, 2, 2, 64))
        mfma_k = gl.permute(mfma_k, (1, 4, 3, 2, 0))
        mfma_k0, mfma_k1 = gl.split(mfma_k) # 64 x 256
        mfma_k0_0, mfma_k0_1 = gl.split(mfma_k0) # 64 x 128
        mfma_k1_0, mfma_k1_1 = gl.split(mfma_k1)
        mfma_k0_00, mfma_k0_01 = gl.split(mfma_k0_0) # 64 x 64
        mfma_k0_10, mfma_k0_11 = gl.split(mfma_k0_1)
        mfma_k1_00, mfma_k1_01 = gl.split(mfma_k1_0)
        mfma_k1_10, mfma_k1_11 = gl.split(mfma_k1_1)

        mfma_k0_00 = gl.convert_layout(mfma_k0_00, mfma_layout_b)
        mfma_k0_01 = gl.convert_layout(mfma_k0_01, mfma_layout_b)
        mfma_k0_10 = gl.convert_layout(mfma_k0_10, mfma_layout_b)
        mfma_k0_11 = gl.convert_layout(mfma_k0_11, mfma_layout_b)
        mfma_k1_00 = gl.convert_layout(mfma_k1_00, mfma_layout_b)
        mfma_k1_01 = gl.convert_layout(mfma_k1_01, mfma_layout_b)
        mfma_k1_10 = gl.convert_layout(mfma_k1_10, mfma_layout_b)
        mfma_k1_11 = gl.convert_layout(mfma_k1_11, mfma_layout_b)

        # k=0, n=0
        o00_00 = gl.amd.cdna3.mfma(mfma_q00_0, mfma_k0_00, zero)
        o01_00 = gl.amd.cdna3.mfma(mfma_q01_0, mfma_k0_00, zero)
        o10_00 = gl.amd.cdna3.mfma(mfma_q10_0, mfma_k0_00, zero)
        o11_00 = gl.amd.cdna3.mfma(mfma_q11_0, mfma_k0_00, zero)
        gl.amd.cdna3.sched_barrier(0x0)
        # k=0, n=1
        o00_01 = gl.amd.cdna3.mfma(mfma_q00_0, mfma_k0_01, zero)
        o01_01 = gl.amd.cdna3.mfma(mfma_q01_0, mfma_k0_01, zero)
        o10_01 = gl.amd.cdna3.mfma(mfma_q10_0, mfma_k0_01, zero)
        o11_01 = gl.amd.cdna3.mfma(mfma_q11_0, mfma_k0_01, zero)
        gl.amd.cdna3.sched_barrier(0x0)
        # k=0, n=2
        o00_10 = gl.amd.cdna3.mfma(mfma_q00_0, mfma_k0_10, zero)
        o01_10 = gl.amd.cdna3.mfma(mfma_q01_0, mfma_k0_10, zero)
        o10_10 = gl.amd.cdna3.mfma(mfma_q10_0, mfma_k0_10, zero)
        o11_10 = gl.amd.cdna3.mfma(mfma_q11_0, mfma_k0_10, zero)
        gl.amd.cdna3.sched_barrier(0x0)
        # k=0, n=3
        o00_11 = gl.amd.cdna3.mfma(mfma_q00_0, mfma_k0_11, zero)
        o01_11 = gl.amd.cdna3.mfma(mfma_q01_0, mfma_k0_11, zero)
        o10_11 = gl.amd.cdna3.mfma(mfma_q10_0, mfma_k0_11, zero)
        o11_11 = gl.amd.cdna3.mfma(mfma_q11_0, mfma_k0_11, zero)
        gl.amd.cdna3.sched_barrier(0x0)
        # k=64, n=0
        o00_00 = gl.amd.cdna3.mfma(mfma_q00_1, mfma_k1_00, o00_00)
        o01_00 = gl.amd.cdna3.mfma(mfma_q01_1, mfma_k1_00, o01_00)
        o10_00 = gl.amd.cdna3.mfma(mfma_q10_1, mfma_k1_00, o10_00)
        o11_00 = gl.amd.cdna3.mfma(mfma_q11_1, mfma_k1_00, o11_00)
        gl.amd.cdna3.sched_barrier(0x0)
        # k=64, n=1
        o00_01 = gl.amd.cdna3.mfma(mfma_q00_1, mfma_k1_01, o00_01)
        o01_01 = gl.amd.cdna3.mfma(mfma_q01_1, mfma_k1_01, o01_01)
        o10_01 = gl.amd.cdna3.mfma(mfma_q10_1, mfma_k1_01, o10_01)
        o11_01 = gl.amd.cdna3.mfma(mfma_q11_1, mfma_k1_01, o11_01)
        gl.amd.cdna3.sched_barrier(0x0)
        # k=64, n=2
        o00_10 = gl.amd.cdna3.mfma(mfma_q00_1, mfma_k1_10, o00_10)
        o01_10 = gl.amd.cdna3.mfma(mfma_q01_1, mfma_k1_10, o01_10)
        o10_10 = gl.amd.cdna3.mfma(mfma_q10_1, mfma_k1_10, o10_10)
        o11_10 = gl.amd.cdna3.mfma(mfma_q11_1, mfma_k1_10, o11_10)
        gl.amd.cdna3.sched_barrier(0x0)
        # k=64, n=3
        o00_11 = gl.amd.cdna3.mfma(mfma_q00_1, mfma_k1_11, o00_11)
        o01_11 = gl.amd.cdna3.mfma(mfma_q01_1, mfma_k1_11, o01_11)
        o10_11 = gl.amd.cdna3.mfma(mfma_q10_1, mfma_k1_11, o10_11)
        o11_11 = gl.amd.cdna3.mfma(mfma_q11_1, mfma_k1_11, o11_11)
        gl.amd.cdna3.sched_barrier(0x0)

        o00_0 = gl.join(o00_00, o00_01) # (16, 64, 2)
        o00_1 = gl.join(o00_10, o00_11)
        o01_0 = gl.join(o01_00, o01_01)
        o01_1 = gl.join(o01_10, o01_11)
        o10_0 = gl.join(o10_00, o10_01)
        o10_1 = gl.join(o10_10, o10_11)
        o11_0 = gl.join(o11_00, o11_01)
        o11_1 = gl.join(o11_10, o11_11)

        o00 = gl.join(o00_0, o00_1) # (16, 64, 2, 2)
        o01 = gl.join(o01_0, o01_1)
        o10 = gl.join(o10_0, o10_1)
        o11 = gl.join(o11_0, o11_1)

        o0 = gl.join(o00, o01) # (16, 64, 2, 2, 2)
        o1 = gl.join(o10, o11)
        o = gl.join(o0, o1) # [16, 64, 2, 2, 2, 2]
        o = gl.permute(o, (5, 4, 0, 3, 2, 1))
        o = gl.reshape(o, (64, 256))
        o = gl.convert_layout(o, mfma_layout)

        # o = gl.amd.cdna3.mfma(mfma_q, mfma_k, zero)
        o = o * k_scale_f[None, :]

        #!=----------------------------
        gl.amd.cdna3.sched_barrier(0x0)
        #!=----------------------------
        k_next = gl.amd.cdna3.buffer_load(
            ptr=KV_buffer,
            offsets=context_kv_idx_next[:, None] * stride_k_seq + gl.arange(0, HiddenDim, layout=gl.SliceLayout(0, layout_kv))[None, :],
        )
        o = gl.maximum(o, 0.0)
        o = o * scale_weight[:, None]

        #!=----------------------------
        gl.amd.cdna3.sched_barrier(0x0)
        #!=----------------------------
        k_scale_f_next = gl.amd.cdna3.buffer_load(ptr=scale_buffer, offsets=context_kv_scale_idx_next * stride_scale_seq)

        mask = context_idx + gl.arange(0, ChunkK, layout=gl.SliceLayout(0, mfma_layout)) <= context_length - pid_next_n
        o = tl.where(mask[None, :], o, float("-inf"))

        logits = gl.reduce(o, axis=0, combine_fn=_sum_combine)
        gl.amd.cdna3.buffer_store(
            logits,
            ptr=OutLogits_buffer,
            offsets=(pid_batch * next_n + pid_next_n) * stride_out_batch
            + (context_idx + gl.arange(0, ChunkK, layout=gl.SliceLayout(0, mfma_layout))),
            mask=context_idx + gl.arange(0, ChunkK, layout=gl.SliceLayout(0, mfma_layout)) >= 0,
        )

    context_idx = split_context_start + split_context_length - ChunkK
    k = k_next
    k_scale_f = k_scale_f_next
    kT = k.T
    mfma_k = gl.convert_layout(kT, mfma_layout_b_linear) 

    mfma_k = gl.reshape(mfma_k, (2, 64, 2, 2, 64))
    mfma_k = gl.permute(mfma_k, (1, 4, 3, 2, 0))
    mfma_k0, mfma_k1 = gl.split(mfma_k) # 64 x 256
    mfma_k0_0, mfma_k0_1 = gl.split(mfma_k0) # 64 x 128
    mfma_k1_0, mfma_k1_1 = gl.split(mfma_k1)
    mfma_k0_00, mfma_k0_01 = gl.split(mfma_k0_0) # 64 x 64
    mfma_k0_10, mfma_k0_11 = gl.split(mfma_k0_1)
    mfma_k1_00, mfma_k1_01 = gl.split(mfma_k1_0)
    mfma_k1_10, mfma_k1_11 = gl.split(mfma_k1_1)

    mfma_k0_00 = gl.convert_layout(mfma_k0_00, mfma_layout_b)
    mfma_k0_01 = gl.convert_layout(mfma_k0_01, mfma_layout_b)
    mfma_k0_10 = gl.convert_layout(mfma_k0_10, mfma_layout_b)
    mfma_k0_11 = gl.convert_layout(mfma_k0_11, mfma_layout_b)
    mfma_k1_00 = gl.convert_layout(mfma_k1_00, mfma_layout_b)
    mfma_k1_01 = gl.convert_layout(mfma_k1_01, mfma_layout_b)
    mfma_k1_10 = gl.convert_layout(mfma_k1_10, mfma_layout_b)
    mfma_k1_11 = gl.convert_layout(mfma_k1_11, mfma_layout_b)

    # k=0, n=0
    o00_00 = gl.amd.cdna3.mfma(mfma_q00_0, mfma_k0_00, zero)
    o01_00 = gl.amd.cdna3.mfma(mfma_q01_0, mfma_k0_00, zero)
    o10_00 = gl.amd.cdna3.mfma(mfma_q10_0, mfma_k0_00, zero)
    o11_00 = gl.amd.cdna3.mfma(mfma_q11_0, mfma_k0_00, zero)
    gl.amd.cdna3.sched_barrier(0x0)
    # k=0, n=1
    o00_01 = gl.amd.cdna3.mfma(mfma_q00_0, mfma_k0_01, zero)
    o01_01 = gl.amd.cdna3.mfma(mfma_q01_0, mfma_k0_01, zero)
    o10_01 = gl.amd.cdna3.mfma(mfma_q10_0, mfma_k0_01, zero)
    o11_01 = gl.amd.cdna3.mfma(mfma_q11_0, mfma_k0_01, zero)
    gl.amd.cdna3.sched_barrier(0x0)
    # k=0, n=2
    o00_10 = gl.amd.cdna3.mfma(mfma_q00_0, mfma_k0_10, zero)
    o01_10 = gl.amd.cdna3.mfma(mfma_q01_0, mfma_k0_10, zero)
    o10_10 = gl.amd.cdna3.mfma(mfma_q10_0, mfma_k0_10, zero)
    o11_10 = gl.amd.cdna3.mfma(mfma_q11_0, mfma_k0_10, zero)
    gl.amd.cdna3.sched_barrier(0x0)
    # k=0, n=3
    o00_11 = gl.amd.cdna3.mfma(mfma_q00_0, mfma_k0_11, zero)
    o01_11 = gl.amd.cdna3.mfma(mfma_q01_0, mfma_k0_11, zero)
    o10_11 = gl.amd.cdna3.mfma(mfma_q10_0, mfma_k0_11, zero)
    o11_11 = gl.amd.cdna3.mfma(mfma_q11_0, mfma_k0_11, zero)
    gl.amd.cdna3.sched_barrier(0x0)
    # k=64, n=0
    o00_00 = gl.amd.cdna3.mfma(mfma_q00_1, mfma_k1_00, o00_00)
    o01_00 = gl.amd.cdna3.mfma(mfma_q01_1, mfma_k1_00, o01_00)
    o10_00 = gl.amd.cdna3.mfma(mfma_q10_1, mfma_k1_00, o10_00)
    o11_00 = gl.amd.cdna3.mfma(mfma_q11_1, mfma_k1_00, o11_00)
    gl.amd.cdna3.sched_barrier(0x0)
    # k=64, n=1
    o00_01 = gl.amd.cdna3.mfma(mfma_q00_1, mfma_k1_01, o00_01)
    o01_01 = gl.amd.cdna3.mfma(mfma_q01_1, mfma_k1_01, o01_01)
    o10_01 = gl.amd.cdna3.mfma(mfma_q10_1, mfma_k1_01, o10_01)
    o11_01 = gl.amd.cdna3.mfma(mfma_q11_1, mfma_k1_01, o11_01)
    gl.amd.cdna3.sched_barrier(0x0)
    # k=64, n=2
    o00_10 = gl.amd.cdna3.mfma(mfma_q00_1, mfma_k1_10, o00_10)
    o01_10 = gl.amd.cdna3.mfma(mfma_q01_1, mfma_k1_10, o01_10)
    o10_10 = gl.amd.cdna3.mfma(mfma_q10_1, mfma_k1_10, o10_10)
    o11_10 = gl.amd.cdna3.mfma(mfma_q11_1, mfma_k1_10, o11_10)
    gl.amd.cdna3.sched_barrier(0x0)
    # k=64, n=3
    o00_11 = gl.amd.cdna3.mfma(mfma_q00_1, mfma_k1_11, o00_11)
    o01_11 = gl.amd.cdna3.mfma(mfma_q01_1, mfma_k1_11, o01_11)
    o10_11 = gl.amd.cdna3.mfma(mfma_q10_1, mfma_k1_11, o10_11)
    o11_11 = gl.amd.cdna3.mfma(mfma_q11_1, mfma_k1_11, o11_11)
    gl.amd.cdna3.sched_barrier(0x0)

    o00_0 = gl.join(o00_00, o00_01) # (16, 64, 2)
    o00_1 = gl.join(o00_10, o00_11)
    o01_0 = gl.join(o01_00, o01_01)
    o01_1 = gl.join(o01_10, o01_11)
    o10_0 = gl.join(o10_00, o10_01)
    o10_1 = gl.join(o10_10, o10_11)
    o11_0 = gl.join(o11_00, o11_01)
    o11_1 = gl.join(o11_10, o11_11)

    o00 = gl.join(o00_0, o00_1) # (16, 64, 2, 2)
    o01 = gl.join(o01_0, o01_1)
    o10 = gl.join(o10_0, o10_1)
    o11 = gl.join(o11_0, o11_1)

    o0 = gl.join(o00, o01) # (16, 64, 2, 2, 2)
    o1 = gl.join(o10, o11)
    o = gl.join(o0, o1) # [16, 64, 2, 2, 2, 2]
    o = gl.permute(o, (5, 4, 0, 3, 2, 1))
    o = gl.reshape(o, (64, 256))
    o = gl.convert_layout(o, mfma_layout)
    # mfma_k = gl.convert_layout(k.T, mfma_layout_b)
    # o = gl.amd.cdna3.mfma(mfma_q, mfma_k, zero)

    o = o * k_scale_f[None, :]
    o = gl.maximum(o, 0.0)
    o = o * scale_weight[:, None]

    mask = context_idx + gl.arange(0, ChunkK, layout=gl.SliceLayout(0, mfma_layout)) <= context_length - pid_next_n
    o = tl.where(mask[None, :], o, float("-inf"))

    logits = gl.reduce(o, axis=0, combine_fn=_sum_combine)
    gl.amd.cdna3.buffer_store(
        logits,
        ptr=OutLogits_buffer,
        offsets=(pid_batch * next_n + pid_next_n) * stride_out_batch + (context_idx + gl.arange(0, ChunkK, layout=gl.SliceLayout(0, mfma_layout))),
        mask=context_idx + gl.arange(0, ChunkK, layout=gl.SliceLayout(0, mfma_layout)) >= 0,
    )
