# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import math
import triton
import triton.language as tl


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
    split_context_length = min(
        context_length - split_context_start, split_context_chunk_num * ChunkK
    )

    q = tl.load(
        Q_buffer
        + pid_batch * stride_q_batch
        + pid_next_n * stride_q_next_n
        + ((pid_q_head * ChunkQ + tl.arange(0, ChunkQ)) * stride_q_heads)[:, None]
        + tl.arange(0, HiddenDim)[None, :],
    )
    scale_weight = tl.load(
        weights
        + (pid_batch * next_n + pid_next_n) * stride_w_batch
        + pid_q_head * ChunkQ
        + tl.arange(0, ChunkQ)
    )

    for context_idx in range(
        split_context_start, split_context_start + split_context_length, ChunkK
    ):
        mask_kv = context_idx + tl.arange(0, ChunkK) < context_length
        context_kv_idx = tl.load(
            kv_indices + context_start + context_idx + tl.arange(0, ChunkK),
            mask=mask_kv,
            other=0,
        )

        k = tl.load(
            KV_buffer
            + context_kv_idx[:, None] * stride_k_seq
            + tl.arange(0, HiddenDim)[None, :],
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
            + (pid_q_head * ChunkQ + tl.arange(0, ChunkQ)[:, None, None])
            * stride_out_heads
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
    split_context_length = min(
        context_length - split_context_start, split_context_chunk_num * ChunkK
    )

    q = tl.load(
        Q_buffer
        + pid_batch * stride_q_batch
        + pid_next_n * stride_q_next_n
        + ((pid_q_head * ChunkQ + tl.arange(0, ChunkQ)) * stride_q_heads)[:, None]
        + tl.arange(0, HiddenDim)[None, :],
    )
    scale_weight = tl.load(
        weights
        + (pid_batch * next_n + pid_next_n) * stride_w_batch
        + pid_q_head * ChunkQ
        + tl.arange(0, ChunkQ)
    )

    for context_idx in range(
        split_context_start, split_context_start + split_context_length, ChunkK
    ):
        mask_kv = context_idx + tl.arange(0, ChunkK) < context_length
        context_kv_idx = tl.load(
            kv_indices + context_start + context_idx + tl.arange(0, ChunkK),
            mask=mask_kv,
            other=0,
        )

        k = tl.load(
            KV_buffer
            + context_kv_idx[:, None] * stride_k_seq
            + tl.arange(0, HiddenDim)[None, :],
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
            OutLogits_buffer
            + (pid_batch * next_n + pid_next_n) * stride_out_batch
            + (context_idx + tl.arange(0, ChunkK)),
            logits,
            mask=(context_idx + tl.arange(0, ChunkK)) < max_model_len,
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
    split_context_length = min(
        context_length - split_context_start, split_context_chunk_num * ChunkK
    )

    q = tl.load(
        Q_buffer
        + pid_batch * stride_q_batch
        + pid_next_n * stride_q_next_n
        + ((pid_q_head * ChunkQ + tl.arange(0, ChunkQ)) * stride_q_heads)[:, None]
        + tl.arange(0, HiddenDim)[None, :],
    )
    scale_weight = tl.load(
        weights
        + (pid_batch * next_n + pid_next_n) * stride_w_batch
        + pid_q_head * ChunkQ
        + tl.arange(0, ChunkQ)
    )

    for context_idx in range(
        split_context_start, split_context_start + split_context_length, ChunkK
    ):
        mask_kv = context_idx + tl.arange(0, ChunkK) < context_length
        context_kv_idx = tl.load(
            kv_indices + pid_batch * max_blk_len + context_idx + tl.arange(0, ChunkK),
            mask=mask_kv,
            other=0,
        )

        k = tl.load(
            KV_buffer
            + context_kv_idx[:, None] * stride_k_seq
            + tl.arange(0, HiddenDim)[None, :],
            mask=mask_kv[:, None],
            other=0.0,
        )
        k_scale_f = tl.load(scale_buffer + context_kv_idx[:, None] * stride_scale_seq)

        o = tl.dot(q, k.T)
        o = o * k_scale_f.T
        o = tl.maximum(o, 0.0)
        o = o * scale_weight[None, :].T

        mask = (
            context_idx + tl.arange(0, ChunkK) <= context_length - next_n + pid_next_n
        )
        o = tl.where(mask[None, :], o, float("-inf"))

        tl.store(
            Out_buffer
            + (pid_batch * next_n + pid_next_n) * stride_out_batch
            + (pid_q_head * ChunkQ + tl.arange(0, ChunkQ)[:, None, None])
            * stride_out_heads
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
    split_context_length = min(
        context_length - split_context_start, split_context_chunk_num * ChunkK
    )

    q = tl.load(
        Q_buffer
        + pid_batch * stride_q_batch
        + pid_next_n * stride_q_next_n
        + ((pid_q_head * ChunkQ + tl.arange(0, ChunkQ)) * stride_q_heads)[:, None]
        + tl.arange(0, HiddenDim)[None, :],
    )
    scale_weight = tl.load(
        weights
        + (pid_batch * next_n + pid_next_n) * stride_w_batch
        + pid_q_head * ChunkQ
        + tl.arange(0, ChunkQ)
    )

    for context_idx in range(
        split_context_start, split_context_start + split_context_length, ChunkK
    ):
        mask_kv = context_idx + tl.arange(0, ChunkK) < context_length
        context_kv_idx = tl.load(
            kv_indices + pid_batch * max_blk_len + context_idx + tl.arange(0, ChunkK),
            mask=mask_kv,
            other=0,
        )

        k = tl.load(
            KV_buffer
            + context_kv_idx[:, None] * stride_k_seq
            + tl.arange(0, HiddenDim)[None, :],
            mask=mask_kv[:, None],
            other=0.0,
        )
        k_scale_f = tl.load(scale_buffer + context_kv_idx[:, None] * stride_scale_seq)

        o = tl.dot(q, k.T)
        o = o * k_scale_f.T
        o = tl.maximum(o, 0.0)
        o = o * scale_weight[None, :].T

        mask = (
            context_idx + tl.arange(0, ChunkK) <= context_length - next_n + pid_next_n
        )
        o = tl.where(mask[None, :], o, float("-inf"))

        logits = tl.reduce(o, axis=0, combine_fn=_sum_combine)
        tl.store(
            OutLogits_buffer
            + (pid_batch * next_n + pid_next_n) * stride_out_batch
            + (context_idx + tl.arange(0, ChunkK)),
            logits,
            mask=(context_idx + tl.arange(0, ChunkK)) < max_model_len,
        )


@triton.jit
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
    dummyPointerArg,  # dummy pointer for compatibility with triton3.5 on lower version
    ChunkQ: tl.constexpr,
    ChunkK: tl.constexpr,
    HiddenDim: tl.constexpr,
    KVBlockSize: tl.constexpr = 1,
):
    # for AOT load use, only need kernel have the same signature as implementation side
    pass


@triton.jit
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
    dummyPointerArg,  # dummy pointer for compatibility with triton3.5 on lower version
    ChunkQ: tl.constexpr,
    ChunkK: tl.constexpr,
    HiddenDim: tl.constexpr,
    KVBlockSize: tl.constexpr = 16,
):
    # for AOT load use, only need kernel have the same signature as implementation side
    pass
