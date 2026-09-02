# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from TransformerEngine's Triton cross-entropy kernels.
# Original copyright: NVIDIA CORPORATION & AFFILIATES.

"""Triton JIT kernels for vocab-parallel cross-entropy loss.

Three kernels are provided:

1. ``_ce_local_softmax_stats_kernel`` — per-rank online softmax to compute local
   ``max``, ``denominator``, and ``X_y`` (the logit at the target index).
2. ``_ce_fused_loss_grad_kernel`` — merges per-rank softmax stats (after
   ``all_gather``), computes the loss, and writes the gradient into the
   logits tensor in-place.
3. ``_ce_grad_scale_kernel`` — element-wise multiply for the backward pass
   (scales the stored gradient by ``grad_output``).
"""

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

# repr keys are the integer/bool constexprs that select the compiled variant;
# label_smoothing (a float) is intentionally excluded — a fractional value would
# put a "." in the trace name.
_ce_local_softmax_stats_kernel_repr = make_kernel_repr(
    "_ce_local_softmax_stats_kernel", ["BLOCK_SIZE"]
)
_ce_fused_loss_grad_kernel_repr = make_kernel_repr(
    "_ce_fused_loss_grad_kernel", ["BLOCK_SIZE", "reduce_loss"]
)
_ce_grad_scale_kernel_repr = make_kernel_repr("_ce_grad_scale_kernel", ["BLOCK_SIZE"])


@triton.jit(repr=_ce_local_softmax_stats_kernel_repr)
def _ce_local_softmax_stats_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    m_d_Xy_ptr,
    m_d_Xy_stride,
    rank,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute per-rank softmax statistics for one row.

    For each row (program), computes:
      - ``m``   — row-wise max of logits on this TP rank
      - ``d``   — sum of exp(x - m) for the local logit shard
      - ``X_y`` — the logit value at the target token index (if it falls on
                   this rank, otherwise ``-inf``)

    The three scalars are written contiguously so that a subsequent
    ``all_gather`` can collect them from every rank.
    """
    pid = tl.program_id(0).to(tl.int64)
    X_ptr += pid * X_stride
    Y_ptr += pid * Y_stride

    y = tl.load(Y_ptr)
    vocab_start = rank * n_cols
    vocab_end = (rank + 1) * n_cols
    X_y: tl.float32 = float("-inf")
    if y >= vocab_start and y < vocab_end:
        X_y = tl.load(X_ptr + y - vocab_start).to(tl.float32)

    base = pid * m_d_Xy_stride * 3
    m: tl.float32 = float("-inf")
    d: tl.float32 = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        blk = tl.load(X_ptr + offs, mask=offs < n_cols, other=float("-inf")).to(
            tl.float32
        )
        blk_max = tl.max(blk)
        m_new = tl.maximum(m, blk_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(blk - m_new))
        m = m_new

    tl.store(m_d_Xy_ptr + base, m)
    tl.store(m_d_Xy_ptr + base + m_d_Xy_stride, d)
    tl.store(m_d_Xy_ptr + base + 2 * m_d_Xy_stride, X_y)


@triton.jit(repr=_ce_fused_loss_grad_kernel_repr)
def _ce_fused_loss_grad_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    loss_ptr,
    loss_stride,
    m_d_Xy_ptr,
    m_d_Xy_stride,
    rank,
    world_size,
    ignore_idx,
    n_cols,
    n_rows,
    n_valid_ptr,
    reduce_loss: tl.constexpr,
    label_smoothing: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused cross-entropy loss + gradient computation.

    Merges the per-rank online-softmax statistics (``m``, ``d``, ``X_y``)
    gathered from all TP ranks into globally correct values, then:

    - Writes the softmax-based gradient into *X* in-place.
    - Stores the scalar loss for the row.

    ``reduce_loss=True`` normalizes the *gradient* by ``n_valid`` (the number
    of non-ignored rows, read from ``n_valid_ptr``) so that it matches a
    mean-over-non-ignored reduction; ``n_rows`` is retained only for the
    all-gather stride offset. ``label_smoothing`` (``0`` = standard CE) is
    only correct for ``world_size == 1``: ``scaled_x_sum`` is collected from
    the local vocab shard alone, so smoothing under tensor parallelism is
    unsupported.
    """
    pid = tl.program_id(0).to(tl.int64)
    X_ptr += pid * X_stride
    Y_ptr += pid * Y_stride
    y = tl.load(Y_ptr)

    if y == ignore_idx:
        for i in range(0, n_cols, BLOCK_SIZE):
            offs = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + offs, 0.0, mask=offs < n_cols)
        return

    loss_ptr += pid * loss_stride
    base = pid * 3 * m_d_Xy_stride
    m = tl.load(m_d_Xy_ptr + base)
    d = tl.load(m_d_Xy_ptr + base + m_d_Xy_stride)
    ori_Xy = tl.load(m_d_Xy_ptr + base + 2 * m_d_Xy_stride)

    for i in range(1, world_size):
        off = i * 3 * n_rows * m_d_Xy_stride
        ptr = m_d_Xy_ptr + base + off
        m_new = tl.load(ptr)
        d_new = tl.load(ptr + m_d_Xy_stride)
        Xy_new = tl.load(ptr + 2 * m_d_Xy_stride)
        d = d * tl.exp(m - tl.maximum(m, m_new)) + d_new * tl.exp(
            m_new - tl.maximum(m, m_new)
        )
        m = tl.maximum(m, m_new)
        # Recover the target logit by max across ranks: only the owning rank
        # loaded the real X_y; every other rank wrote -inf (see the stats
        # kernel), so the max picks the true value. A genuinely -inf target
        # logit (a masked vocab entry) is indistinguishable from "not my
        # shard" — an accepted limitation of the sentinel.
        ori_Xy = tl.maximum(ori_Xy, Xy_new)

    n_valid: tl.float32 = 1.0
    if reduce_loss:
        n_valid = tl.load(n_valid_ptr).to(tl.float32)

    eps = label_smoothing / (n_cols * world_size)
    scaled_x_sum: tl.float32 = 0.0

    for i in range(0, n_cols, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        blk = tl.load(X_ptr + offs, mask=offs < n_cols, other=float("-inf"))
        grad_dtype = blk.dtype
        blk = blk.to(tl.float32)
        if label_smoothing > 0:
            scaled_x_sum += tl.sum(tl.where(offs < n_cols, -eps * blk, 0.0))
        if reduce_loss:
            blk = (tl.exp(blk - m) / d - eps) / n_valid
        else:
            blk = tl.exp(blk - m) / d - eps
        tl.store(X_ptr + offs, blk.to(grad_dtype), mask=offs < n_cols)

    tl.debug_barrier()

    loss = -(ori_Xy - m - tl.log(d))
    if label_smoothing > 0:
        smooth = scaled_x_sum + label_smoothing * (m + tl.log(d))
        loss = loss * (1 - label_smoothing) + smooth

    vocab_start = rank * n_cols
    vocab_end = (rank + 1) * n_cols
    if y >= vocab_start and y < vocab_end:
        Xy_grad = tl.load(X_ptr + y - vocab_start)
        if reduce_loss:
            Xy_grad += -(1 - label_smoothing) / n_valid
        else:
            Xy_grad += -(1 - label_smoothing)
        tl.store(X_ptr + y - vocab_start, Xy_grad)

    tl.store(loss_ptr, loss)


@triton.jit(repr=_ce_grad_scale_kernel_repr)
def _ce_grad_scale_kernel(
    X_ptr,
    X_stride,
    grad_ptr,
    grad_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Element-wise multiply for backward: ``X[row] *= grad[row]``."""
    pid = tl.program_id(0).to(tl.int64)
    X_ptr += pid * X_stride
    grad_ptr += pid * grad_stride
    g = tl.load(grad_ptr)
    for i in range(0, n_cols, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        blk = tl.load(X_ptr + offs, mask=offs < n_cols)
        tl.store(X_ptr + offs, blk * g, mask=offs < n_cols)
