# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Triton kernel merging attention partials computed over disjoint key ranges by their
log-sum-exp.
"""

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_lse_merge_kernel_repr = make_kernel_repr(
    "_lse_merge_kernel",
    [
        "SPLITS",
        "BLOCK_M",
        "HEAD_DIM",
        "WRITE_LSE",
    ],
)


@triton.jit(repr=_lse_merge_kernel_repr)
def _lse_merge_kernel(
    partial_out_ptr,
    partial_lse_ptr,
    out_ptr,
    lse_ptr,
    sequence,
    stride_po_b,
    stride_po_s,
    stride_po_h,
    stride_pl_b,
    stride_pl_h,
    stride_o_b,
    stride_o_s,
    stride_o_h,
    stride_l_b,
    stride_l_h,
    SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    """
    Each program merges BLOCK_M query rows of one (batch, head). Partial batch
    batch * SPLITS + i holds key range i, as BSHD output and [batch, head, Sq] LSE.
    """
    pid_m = tl.program_id(0)
    # Offsets are 64-bit: one partial, or the splits ahead of the last, can pass 2^31 elements.
    # Each split is reached by stepping the pointer rather than by i * stride, which would be
    # computed in the stride's 32 bits.
    head = tl.program_id(1).to(tl.int64)
    batch = tl.program_id(2).to(tl.int64)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, HEAD_DIM)
    valid = rows < sequence
    rows = rows.to(tl.int64)

    lse_base = (
        partial_lse_ptr + batch * SPLITS * stride_pl_b + head * stride_pl_h + rows
    )
    lse_max = tl.full([BLOCK_M], float("-inf"), tl.float32)
    lse_split = lse_base
    for _ in tl.static_range(SPLITS):
        partial = tl.load(lse_split, mask=valid, other=float("-inf"))
        lse_max = tl.maximum(lse_max, partial)
        lse_split += stride_pl_b
    # A row no key range reached keeps every weight at zero rather than exp(-inf - -inf).
    lse_max = tl.where(lse_max == float("-inf"), 0.0, lse_max)

    out_split = (
        partial_out_ptr
        + batch * SPLITS * stride_po_b
        + rows[:, None] * stride_po_s
        + head * stride_po_h
        + cols[None, :]
    )
    lse_split = lse_base
    total = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    for _ in tl.static_range(SPLITS):
        partial = tl.load(lse_split, mask=valid, other=float("-inf"))
        weight = tl.exp(partial - lse_max)
        total += weight
        # An empty range weighs 0, but 0 * NaN or 0 * Inf is NaN, so its value must not be read.
        reached = valid & (partial != float("-inf"))
        value = tl.load(out_split, mask=reached[:, None], other=0.0)
        acc += weight[:, None] * value.to(tl.float32)
        lse_split += stride_pl_b
        out_split += stride_po_b
    acc = acc * tl.where(total > 0, 1.0 / total, 0.0)[:, None]

    out_offsets = (
        batch * stride_o_b
        + rows[:, None] * stride_o_s
        + head * stride_o_h
        + cols[None, :]
    )
    tl.store(
        out_ptr + out_offsets,
        acc.to(out_ptr.dtype.element_ty),
        mask=valid[:, None],
    )
    if WRITE_LSE:
        tl.store(
            lse_ptr + batch * stride_l_b + head * stride_l_h + rows,
            lse_max + tl.log(total),
            mask=valid,
        )
