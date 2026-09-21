# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@triton.jit
def _sum_combine(a, b):
    return a + b


_deepgemm_fp8_paged_mqa_logits_stage1_ragged_k_repr = make_kernel_repr(
    "_deepgemm_fp8_paged_mqa_logits_stage1_ragged_k",
    [
        "ChunkQ",
        "ChunkK",
        "HiddenDim",
        "SplitKV",
    ],
)


@triton.jit(repr=_deepgemm_fp8_paged_mqa_logits_stage1_ragged_k_repr)
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
    stride_out_batch: tl.int64,
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


_deepgemm_fp8_paged_mqa_logits_ragged_k_repr = make_kernel_repr(
    "_deepgemm_fp8_paged_mqa_logits_ragged_k",
    [
        "ChunkQ",
        "ChunkK",
        "HiddenDim",
        "SplitKV",
    ],
)


@triton.jit(repr=_deepgemm_fp8_paged_mqa_logits_ragged_k_repr)
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
    stride_out_batch: tl.int64,
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


_deepgemm_fp8_paged_mqa_logits_stage1_repr = make_kernel_repr(
    "_deepgemm_fp8_paged_mqa_logits_stage1",
    [
        "ChunkQ",
        "ChunkK",
        "HiddenDim",
        "KVBlockSize",
        "SplitKV",
    ],
)


@triton.jit(repr=_deepgemm_fp8_paged_mqa_logits_stage1_repr)
def _deepgemm_fp8_paged_mqa_logits_stage1(
    batch_size,
    next_n,
    heads_num,
    Q_buffer,
    stride_q_batch: tl.int64,
    stride_q_next_n: tl.int64,
    stride_q_heads: tl.int64,
    KV_buffer,
    stride_k_block: tl.int64,
    stride_k_token: tl.int64,
    scale_buffer,
    stride_scale_block: tl.int64,
    stride_scale_token: tl.int64,
    context_len_ptr,
    kv_indices,
    weights,
    stride_w_batch: tl.int64,
    Out_buffer,
    stride_out_heads: tl.int64,
    stride_out_batch: tl.int64,
    max_model_len,
    max_blk_len,
    ChunkQ: tl.constexpr,
    ChunkK: tl.constexpr,
    HiddenDim: tl.constexpr,
    KVBlockSize: tl.constexpr,
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
        logical_kv_idx = context_idx + tl.arange(0, ChunkK)
        logical_block_idx = logical_kv_idx // KVBlockSize
        mask_kv = (logical_kv_idx < context_length) & (logical_block_idx < max_blk_len)
        physical_block_idx = tl.load(
            kv_indices + pid_batch * max_blk_len + logical_block_idx,
            mask=mask_kv,
            other=0,
        )
        block_offset = logical_kv_idx % KVBlockSize

        k = tl.load(
            KV_buffer
            + physical_block_idx[:, None] * stride_k_block
            + block_offset[:, None] * stride_k_token
            + tl.arange(0, HiddenDim)[None, :],
            mask=mask_kv[:, None],
            other=0.0,
        )
        k_scale_f = tl.load(
            scale_buffer
            + physical_block_idx[:, None] * stride_scale_block
            + block_offset[:, None] * stride_scale_token,
            mask=mask_kv[:, None],
            other=0.0,
        )

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


_deepgemm_fp8_paged_mqa_logits_varctx_schedule_repr = make_kernel_repr(
    "_deepgemm_fp8_paged_mqa_logits_varctx_schedule",
    [
        "ChunkK",
        "AlignedBatchSize",
        "TryCount",
    ],
)


@triton.jit
def _deepgemm_fp8_paged_mqa_logits_persistent_schedule(
    context_len_ptr,
    cta_info_ptr,
    batch_size: tl.constexpr,
    slots: tl.constexpr,
    max_model_len,
    ChunkK: tl.constexpr,
    NEXT_N: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Build [packed query row, first chunk, chunk count, context length].

    Like the FlyDSL FP4 schedule, find the smallest chunk budget whose CTA
    count fits the persistent grid. Empty sequences consume no slots. Balance
    the chunks within each sequence so its last CTA is not a short straggler.
    """
    b = tl.arange(0, BLOCK_B)
    ctx = tl.load(context_len_ptr + b, mask=b < batch_size, other=0)
    ctx = tl.minimum(tl.maximum(ctx, 0), max_model_len)
    chunks = tl.cdiv(ctx, ChunkK)
    slot = tl.program_id(0) * BLOCK_S + tl.arange(0, BLOCK_S)
    max_chunks = tl.max(chunks, 0)
    min_chunks = tl.min(tl.where(b < batch_size, chunks, max_chunks), 0)
    if (min_chunks == max_chunks) & (max_chunks >= slots // batch_size):
        # Enough uniform chunks to fill the grid: all divisors are compile-time
        # constants. Contexts can still differ within their final chunk.
        per_seq: tl.constexpr = slots // batch_size
        batch = slot // per_seq
        split = slot % per_seq
        valid = (slot < per_seq * batch_size) & (slot < slots)
        ctx_slot = tl.load(context_len_ptr + batch, mask=valid, other=0)
        ctx_slot = tl.minimum(tl.maximum(ctx_slot, 0), max_model_len)
        uniform_per_cta = max_chunks // per_seq
        uniform_extra = max_chunks % per_seq
        start = split * uniform_per_cta + tl.minimum(split, uniform_extra)
        count = uniform_per_cta + (split < uniform_extra).to(tl.int32)
    else:
        total_chunks = tl.sum(chunks, 0)
        nonempty = tl.sum((chunks > 0).to(tl.int32), 0)
        lo = tl.maximum(tl.cdiv(total_chunks, slots), 1)
        # sum(ceil(chunks / s)) < total_chunks / s + nonempty. With this
        # upper bound it is < slots + 1, hence <= slots for integer CTA counts.
        hi = tl.maximum(
            tl.minimum(max_chunks, tl.cdiv(total_chunks, slots - nonempty + 1)), 1
        )
        while lo < hi:
            mid = (lo + hi) // 2
            fits = tl.sum(tl.cdiv(chunks, mid), 0) <= slots
            hi = tl.where(fits, mid, hi)
            lo = tl.where(fits, lo, mid + 1)

        ctas = tl.cdiv(chunks, lo)
        end = tl.cumsum(ctas, 0)
        begin = end - ctas
        batch = tl.sum(
            ((end[None, :] <= slot[:, None]) & (b[None, :] < batch_size)).to(tl.int32),
            1,
        )
        valid = (slot < tl.sum(ctas, 0)) & (slot < slots)
        selected = (batch[:, None] == b[None, :]) & valid[:, None]
        seq_chunks = tl.sum(tl.where(selected, chunks[None, :], 0), 1)
        seq_ctas = tl.maximum(tl.sum(tl.where(selected, ctas[None, :], 0), 1), 1)
        split = slot - tl.sum(tl.where(selected, begin[None, :], 0), 1)
        ctx_slot = tl.sum(tl.where(selected, ctx[None, :], 0), 1)
        per_cta = seq_chunks // seq_ctas
        extra = seq_chunks % seq_ctas
        start = split * per_cta + tl.minimum(split, extra)
        count = per_cta + (split < extra).to(tl.int32)

    field = tl.arange(0, 4)
    for n in tl.static_range(NEXT_N):
        row = slot * NEXT_N + n
        info = tl.where(
            field[None, :] == 0,
            (batch * NEXT_N + n)[:, None],
            tl.where(
                field[None, :] == 1,
                start[:, None],
                tl.where(field[None, :] == 2, count[:, None], ctx_slot[:, None]),
            ),
        )
        tl.store(
            cta_info_ptr + row[:, None] * 4 + field[None, :],
            tl.where(valid[:, None], info, 0),
            (slot < slots)[:, None],
        )


@triton.jit(repr=_deepgemm_fp8_paged_mqa_logits_varctx_schedule_repr)
def _deepgemm_fp8_paged_mqa_logits_varctx_schedule(
    batch_size,
    context_len_ptr,
    safe_chunks_per_cta_ptr,
    parallel_unit_num,
    ChunkK: tl.constexpr,
    AlignedBatchSize: tl.constexpr,
    TryCount: tl.constexpr,
):
    pid = tl.program_id(0)

    ctx_lens = tl.load(
        context_len_ptr + tl.arange(0, AlignedBatchSize),
        mask=tl.arange(0, AlignedBatchSize) < batch_size,
        other=0,
    )
    ctx_blks = tl.cdiv(ctx_lens, ChunkK)

    has_successed = False
    safe_seg_lens = 0
    for t in range(TryCount):
        try_seg_per_pu = 1 + pid * TryCount + TryCount - t
        ctx_segs = tl.cdiv(ctx_blks, try_seg_per_pu)
        total_segs = tl.sum(ctx_segs)

        if total_segs <= parallel_unit_num:
            has_successed = True
        elif has_successed:
            safe_seg_lens = try_seg_per_pu + 1
            has_successed = False

    try_seg_per_pu = 1 + pid * TryCount
    ctx_segs = tl.cdiv(ctx_blks, try_seg_per_pu)
    total_segs = tl.sum(ctx_segs)

    if has_successed:
        if total_segs > parallel_unit_num:
            safe_seg_lens = try_seg_per_pu + 1
        elif try_seg_per_pu == 1:
            safe_seg_lens = 1

    if safe_seg_lens != 0:
        tl.store(safe_chunks_per_cta_ptr, safe_seg_lens)


_deepgemm_fp8_paged_mqa_logits_repr = make_kernel_repr(
    "_deepgemm_fp8_paged_mqa_logits",
    [
        "ChunkQ",
        "ChunkK",
        "HiddenDim",
        "KVBlockSize",
        "SplitKV",
    ],
)


@triton.jit(repr=_deepgemm_fp8_paged_mqa_logits_repr)
def _deepgemm_fp8_paged_mqa_logits(
    batch_size,
    next_n,
    heads_num,
    Q_buffer,
    stride_q_batch,
    stride_q_next_n,
    stride_q_heads,
    KV_buffer,
    stride_k_block,
    stride_k_token,
    scale_buffer,
    stride_scale_block,
    stride_scale_token,
    context_len_ptr,
    kv_indices,
    weights,
    stride_w_batch,
    OutLogits_buffer,
    stride_out_batch: tl.int64,
    max_model_len,
    max_blk_len,
    ChunkQ: tl.constexpr,
    ChunkK: tl.constexpr,
    HiddenDim: tl.constexpr,
    KVBlockSize: tl.constexpr,
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
        logical_kv_idx = context_idx + tl.arange(0, ChunkK)
        logical_block_idx = logical_kv_idx // KVBlockSize
        mask_kv = (logical_kv_idx < context_length) & (logical_block_idx < max_blk_len)
        physical_block_idx = tl.load(
            kv_indices + pid_batch * max_blk_len + logical_block_idx,
            mask=mask_kv,
            other=0,
        )
        block_offset = logical_kv_idx % KVBlockSize

        k = tl.load(
            KV_buffer
            + physical_block_idx[:, None] * stride_k_block
            + block_offset[:, None] * stride_k_token
            + tl.arange(0, HiddenDim)[None, :],
            mask=mask_kv[:, None],
            other=0.0,
        )
        k_scale_f = tl.load(
            scale_buffer
            + physical_block_idx[:, None] * stride_scale_block
            + block_offset[:, None] * stride_scale_token,
            mask=mask_kv[:, None],
            other=0.0,
        )

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
    stride_out_batch: tl.int64,
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
    stride_out_batch: tl.int64,
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


@triton.jit
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
    stride_out_batch: tl.int64,
    max_model_len,
    max_block_len,
    safe_chunks_per_cta_ptr,
    dummyPointerArg,  # dummy pointer for compatibility with triton3.5 on lower version
    ChunkQ: tl.constexpr,
    ChunkK: tl.constexpr,
    HiddenDim: tl.constexpr,
    KVBlockSize: tl.constexpr = 16,
):
    # for AOT load use, only need kernel have the same signature as implementation side
    pass
