# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Triton kernel pooling a BSHD tensor along the sequence axis into one row per block of tokens, for
Sol-Attn's pooled Q/K/V: the block mean, plus optionally the block's population variance or its
mean of squares, from a single read of the input.
"""

import triton
import triton.language as tl
from triton.language.extra import libdevice

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

POOL_AUX_NONE = 0
POOL_AUX_VARIANCE = 1
POOL_AUX_SQUARE_MEAN = 2

_sol_attn_pool_kernel_repr = make_kernel_repr(
    "_sol_attn_pool_kernel",
    ["BLOCK", "ROWS", "HEAD_DIM", "HEADS_PER_PROGRAM", "AUX", "ROUND"],
)


@triton.jit(repr=_sol_attn_pool_kernel_repr)
def _sol_attn_pool_kernel(
    x_ptr,
    mean_ptr,
    aux_ptr,
    seqlen,
    nheads,
    num_blocks,
    stride_x_b,
    stride_x_s,
    stride_x_h,
    stride_x_d,
    aux_multiplier,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
    AUX: tl.constexpr,
    ROUND: tl.constexpr,
):
    """
    Each program pools one block of BLOCK tokens for HEADS_PER_PROGRAM adjacent heads, reading it
    ROWS tokens at a time. Head groups are the fastest-varying program index, so neighbouring
    programs read neighbouring bytes of the same token rows.

    Outputs are contiguous (batch, num_blocks, nheads, HEAD_DIM). A short last block is divided by
    its real token count.
    """
    COLS: tl.constexpr = HEADS_PER_PROGRAM * HEAD_DIM
    pid = tl.program_id(0)
    num_head_groups = tl.cdiv(nheads, HEADS_PER_PROGRAM)
    head_group = pid % num_head_groups
    block = (pid // num_head_groups) % num_blocks
    b = pid // (num_head_groups * num_blocks)

    cols = tl.arange(0, COLS)
    head = head_group * HEADS_PER_PROGRAM + cols // HEAD_DIM
    dim = cols % HEAD_DIM
    col_ok = head < nheads
    token0 = block * BLOCK
    count = tl.minimum(seqlen - token0, BLOCK).to(tl.float32)
    x_base = (
        x_ptr
        + b.to(tl.int64) * stride_x_b
        + head[None, :].to(tl.int64) * stride_x_h
        + dim[None, :] * stride_x_d
    )

    # Accumulated elementwise and reduced across rows once at the end: a cross-thread reduction
    # per chunk costs more than the chunk's load.
    acc = tl.zeros([ROWS, COLS], dtype=tl.float32)
    acc_sq = tl.zeros([ROWS, COLS], dtype=tl.float32)
    for r0 in tl.static_range(0, BLOCK, ROWS):
        tokens = token0 + r0 + tl.arange(0, ROWS)
        ok = (tokens < seqlen)[:, None] & col_ok[None, :]
        ptrs = x_base + tokens.to(tl.int64)[:, None] * stride_x_s
        x = tl.where(ok, tl.load(ptrs, mask=ok).to(tl.float32), 0.0)
        acc += x
        if AUX == 2:
            acc_sq += x * x
    mean = tl.sum(acc, axis=0) / count

    out_off = ((b.to(tl.int64) * num_blocks + block) * nheads + head) * HEAD_DIM + dim
    stored_mean = mean
    if ROUND:
        stored_mean = libdevice.nearbyint(stored_mean)
    tl.store(mean_ptr + out_off, stored_mean.to(mean_ptr.dtype.element_ty), mask=col_ok)

    if AUX == 1:
        # Taken about the mean rather than as E[x^2] - mean^2, which cancels badly for a channel
        # whose offset dwarfs its spread. The block was just read, so the second pass hits cache.
        for r0 in tl.static_range(0, BLOCK, ROWS):
            tokens = token0 + r0 + tl.arange(0, ROWS)
            ok = (tokens < seqlen)[:, None] & col_ok[None, :]
            ptrs = x_base + tokens.to(tl.int64)[:, None] * stride_x_s
            centered = tl.where(
                ok, tl.load(ptrs, mask=ok).to(tl.float32) - mean[None, :], 0.0
            )
            acc_sq += centered * centered
    if AUX != 0:
        # Under torch.compile aux_multiplier arrives as fp64, and gfx942 has no fp64 -> FP8
        # conversion.
        aux_value = (tl.sum(acc_sq, axis=0) / count * aux_multiplier).to(tl.float32)
        tl.store(aux_ptr + out_off, aux_value.to(aux_ptr.dtype.element_ty), mask=col_ok)
