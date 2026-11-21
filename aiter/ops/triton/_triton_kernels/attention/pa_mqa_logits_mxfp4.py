# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Triton kernel for the paged MXFP4 MQA logits, used where the gfx950 Gluon
# kernel is not available. It reads the cache token-major (preshuffle=0).

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_pa_mqa_logits_mxfp4_kernel_repr = make_kernel_repr(
    "_pa_mqa_logits_mxfp4_kernel",
    [
        "NUM_HEADS",
        "HEAD_SIZE",
        "PAGE_SIZE",
        "BLOCK_KV",
        "HAS_ROW_ENDS",
    ],
)


@triton.jit(repr=_pa_mqa_logits_mxfp4_kernel_repr)
def _pa_mqa_logits_mxfp4_kernel(
    q_ptr,
    q_scales_ptr,
    kv_ptr,
    kv_scales_ptr,
    weights_ptr,
    context_lens_ptr,
    row_ends_ptr,
    block_table_ptr,
    logits_ptr,
    next_n,
    max_model_len,
    stride_q_b,
    stride_q_n,
    stride_qs_b,
    stride_qs_n,
    stride_w_s,
    stride_logits_s,
    stride_blk_b,
    kv_page_stride,
    kvs_page_stride,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HAS_ROW_ENDS: tl.constexpr,
):
    # One program per (query row, KV tile), all heads at once. The row is
    # int64 so the batch, token and row offsets below cannot wrap.
    row = tl.program_id(0).to(tl.int64)
    pid_kv = tl.program_id(1)
    pid_b = row // next_n
    pid_n = row % next_n

    context_length = tl.load(context_lens_ptr + pid_b)
    # A logits row has max_model_len columns. A longer context must not
    # read the block table or write past the row.
    n_cols = tl.minimum(context_length, max_model_len)
    # A tile past the context writes nothing, so it skips the loads and dot.
    if pid_kv * BLOCK_KV >= n_cols:
        return
    if HAS_ROW_ENDS:
        row_end = tl.minimum(tl.load(row_ends_ptr + row), context_length)
    else:
        row_end = context_length - next_n + pid_n + 1

    offs_h = tl.arange(0, NUM_HEADS)
    offs_d = tl.arange(0, HEAD_SIZE // 2)
    offs_s = tl.arange(0, HEAD_SIZE // 32)
    q = tl.load(
        q_ptr
        + pid_b * stride_q_b
        + pid_n * stride_q_n
        + offs_h[:, None] * (HEAD_SIZE // 2)
        + offs_d[None, :]
    )
    q_scales = tl.load(
        q_scales_ptr
        + pid_b * stride_qs_b
        + pid_n * stride_qs_n
        + offs_h[:, None] * (HEAD_SIZE // 32)
        + offs_s[None, :]
    )
    scale_weight = tl.load(weights_ptr + row * stride_w_s + offs_h)

    offs_kv = pid_kv * BLOCK_KV + tl.arange(0, BLOCK_KV)
    mask_kv = offs_kv < n_cols
    page = tl.load(
        block_table_ptr + pid_b * stride_blk_b + offs_kv // PAGE_SIZE,
        mask=mask_kv,
        other=0,
    ).to(tl.int64)
    slot = offs_kv % PAGE_SIZE
    k = tl.load(
        kv_ptr
        + page[:, None] * kv_page_stride
        + slot[:, None] * (HEAD_SIZE // 2)
        + offs_d[None, :],
        mask=mask_kv[:, None],
        other=0,
    )
    # 127 is 2^0. A masked lane must not read as 255, the e8m0 NaN, or the
    # whole dot turns NaN even though its values are zero.
    k_scales = tl.load(
        kv_scales_ptr
        + page[:, None] * kvs_page_stride
        + slot[:, None] * (HEAD_SIZE // 32)
        + offs_s[None, :],
        mask=mask_kv[:, None],
        other=127,
    )

    o = tl.dot_scaled(q, q_scales, "e2m1", tl.trans(k), k_scales, "e2m1")
    o = tl.maximum(o, 0.0)
    o = o * scale_weight[:, None]
    logits = tl.sum(o, axis=0)
    logits = tl.where(offs_kv < row_end, logits, float("-inf"))
    tl.store(
        logits_ptr + row * stride_logits_s + offs_kv,
        logits,
        mask=mask_kv,
    )
