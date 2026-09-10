# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Vocab-parallel cross-entropy: Python API wrapping Triton kernels.

This module provides ``cross_entropy_forward`` and ``cross_entropy_backward``
that orchestrate kernel launches and ``all_gather`` communication so that
higher-level frameworks (e.g. Lumen, Megatron) only need a single function
call.
"""

import torch
import torch.distributed as dist
import triton

from aiter.ops.triton._triton_kernels.cross_entropy import (
    _ce_fused_loss_grad_kernel,
    _ce_grad_scale_kernel,
    _ce_local_softmax_stats_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

__all__ = [
    "cross_entropy_backward",
    "cross_entropy_forward",
    "cross_entropy_forward_chunked",
]

_LOGGER = AiterTritonLogger()
_WAVES_PER_EU = 2
_NUM_STAGES = 2


def _block_size_and_warps(element_size: int, n_cols: int) -> tuple[int, int]:
    """Compute BLOCK_SIZE and num_warps for a given vocab width."""
    block_size = min(65536 // element_size, triton.next_power_of_2(n_cols))
    num_warps = min(16, max(1, block_size // 64))
    return block_size, num_warps


def cross_entropy_forward(
    _input: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float,
    reduce_loss: bool,
    dist_group: dist.ProcessGroup | None,
    ignore_idx: int,
    validate_targets: bool = True,
):
    """Compute vocab-parallel cross-entropy loss (forward).

    Args:
        _input:  Logits shard for this TP rank — ``[B, SQ, V_local]``.
        target:  Label indices — ``[B, SQ]``  (global vocab ids).
        label_smoothing:  Label-smoothing factor (0 = standard CE). Must be 0
            when ``world_size > 1`` — the smoothing term only sums the local
            vocab shard, so it is asserted against under tensor parallelism.
        reduce_loss:  If ``True``, return a scalar loss and gradient averaged
            over the *non-ignored* rows (matching ``F.cross_entropy`` mean),
            not over all ``B*SQ`` rows.  When all rows are ignored the loss is
            0.0 (not ``nan`` as ``F.cross_entropy`` would return).
        dist_group:  TP process group (``None`` for single-GPU).
        ignore_idx:  Target value to ignore (required; callers typically
            pass ``-100``).
        validate_targets:  If ``True`` (default), check that every target is
            either ``ignore_idx`` or in ``[0, V*world_size)`` before launching
            kernels.  Set to ``False`` to skip the host sync after the first
            few warm-up steps.

    Returns:
        ``(loss, grad_input)`` where *grad_input* has the same shape as
        ``_input`` and already contains the gradient (to be scaled by
        ``grad_output`` in the backward pass).
    """
    _LOGGER.info(f"CROSS_ENTROPY_FORWARD: input={tuple(_input.shape)}")
    B, SQ, V = _input.shape
    n_rows = B * SQ
    if target.numel() != n_rows:
        raise ValueError(f"target.numel() ({target.numel()}) != B*SQ ({n_rows})")
    BLOCK_SIZE, num_warps = _block_size_and_warps(_input.element_size(), V)

    loss_1d = torch.zeros(n_rows, dtype=torch.float32, device=_input.device)
    m_d_Xy = torch.zeros(n_rows * 3, dtype=torch.float32, device=_input.device)

    # Full contiguity, not just stride(-1) == 1: both kernels flatten B*SQ with
    # a single row stride, so a tensor sliced along SQ (inner stride still 1 but
    # batch stride != SQ*stride(-2)) would make later programs read wrong rows.
    if not _input.is_contiguous():
        _input = _input.contiguous()
    if not target.is_contiguous():
        target = target.contiguous()

    rank = 0 if dist_group is None else dist.get_rank(dist_group)
    world_size = 1 if dist_group is None else dist.get_world_size(dist_group)

    # The distributed label-smoothing term only sums this rank's vocab shard,
    # so it is incorrect under TP. Guard the known-wrong combination until a
    # multi-process TP test covers it.
    assert not (label_smoothing > 0 and world_size > 1), (
        "label_smoothing > 0 is not supported with world_size > 1: the "
        "smoothing term only covers the local vocab shard"
    )

    # Validate targets: an out-of-range label leaves X_y = -inf on every rank,
    # which silently yields +inf loss. This is one reduction over [B*SQ] ints
    # (a host sync), paid once per forward to avoid silent corruption.
    # Set validate_targets=False to skip after the first few warm-up steps.
    if validate_targets:
        valid_target = (target == ignore_idx) | (
            (target >= 0) & (target < V * world_size)
        )
        if not valid_target.all():
            raise ValueError(
                f"target out of range: expected in [0, {V * world_size}) or == "
                f"ignore_idx ({ignore_idx})"
            )

    # For reduce_loss, both the loss and the in-kernel gradient normalization
    # divide by the count of non-ignored rows (matching F.cross_entropy mean),
    # not by n_rows. Kept on-device as a 0-d tensor to avoid a CPU sync.
    # clamp_min(1): an all-ignored batch would divide by 0 -> inf grad / nan
    # loss; clamping keeps the gradient finite (0.0). Note: F.cross_entropy
    # returns nan for an all-ignored batch; this op returns 0.0 instead.
    if reduce_loss:
        n_valid = (target != ignore_idx).sum().clamp_min(1)
    else:
        n_valid = torch.ones((), dtype=torch.int64, device=_input.device)

    _ce_local_softmax_stats_kernel[(n_rows,)](
        _input,
        _input.stride(-2),
        target,
        target.stride(-1),
        m_d_Xy,
        m_d_Xy.stride(-1),
        rank,
        V,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        waves_per_eu=_WAVES_PER_EU,
        num_stages=_NUM_STAGES,
    )

    if world_size > 1:
        gathered = torch.zeros(
            n_rows * 3 * world_size, dtype=torch.float32, device=_input.device
        )
        dist.all_gather_into_tensor(gathered, m_d_Xy, group=dist_group)
    else:
        gathered = m_d_Xy

    _ce_fused_loss_grad_kernel[(n_rows,)](
        _input,
        _input.stride(-2),
        target,
        target.stride(-1),
        loss_1d,
        loss_1d.stride(-1),
        gathered,
        gathered.stride(-1),
        rank,
        world_size,
        ignore_idx,
        V,
        n_rows,
        n_valid,
        reduce_loss=reduce_loss,
        label_smoothing=label_smoothing,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        waves_per_eu=_WAVES_PER_EU,
        num_stages=_NUM_STAGES,
    )

    loss = loss_1d.reshape(B, SQ) if not reduce_loss else (loss_1d.sum() / n_valid)
    return loss, _input


def cross_entropy_forward_chunked(
    _input: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float,
    reduce_loss: bool,
    dist_group: dist.ProcessGroup | None,
    ignore_idx: int,
    chunk_rows: int,
    validate_targets: bool = True,
):
    """Chunked vocab-parallel cross-entropy forward.

    Splits the row dimension (B*SQ) into chunks of ``chunk_rows`` and
    processes each independently: online_softmax → allgather → ce_kernel.

    This does *not* shrink the ``B*SQ*V_local`` logits activation (the caller
    still materializes it in full and it is updated in-place). What it bounds
    is the per-step online-softmax scratch and, under tensor parallelism, the
    all-gather/communication buffer, which drop from ``B*SQ * 3`` to
    ``chunk_rows * 3`` floats per rank — at the cost of repeated launches and
    collectives.

    The gradient is written in-place into ``_input`` exactly as in the
    non-chunked path.

    Args:
        _input:    Logit shard ``[B, SQ, V_local]`` — modified in-place.
        target:    Label indices ``[B, SQ]`` (global vocab ids).
        chunk_rows: Number of rows per chunk.  Must be >= 1.
        validate_targets:  Same as in :func:`cross_entropy_forward`.
        (other args: same semantics as :func:`cross_entropy_forward`)

    Returns:
        ``(loss, _input)`` where *loss* is ``[B, SQ]`` or a scalar.
    """
    _LOGGER.info(f"CROSS_ENTROPY_FORWARD_CHUNKED: input={tuple(_input.shape)}")
    if chunk_rows < 1:
        raise ValueError(f"chunk_rows must be >= 1, got {chunk_rows}")

    # Enforce the contiguity contract before flattening: reshape() on a
    # non-contiguous tensor would return a *copy*, the kernels would write the
    # gradient into that copy, and we would return the untouched original.
    if not _input.is_contiguous():
        _input = _input.contiguous()
    if not target.is_contiguous():
        target = target.contiguous()

    B, SQ, V = _input.shape
    n_rows = B * SQ

    # Flatten to 2-D views so we can slice by row without copying.
    input_2d = _input.view(n_rows, V)  # view, no alloc
    target_1d = target.view(n_rows)  # view, no alloc
    # zeros (not empty): ignored rows never store a loss, so they must default
    # to 0 to match the non-chunked path and F.cross_entropy(reduction="none").
    loss_1d = torch.zeros(n_rows, dtype=torch.float32, device=_input.device)

    BLOCK_SIZE, num_warps = _block_size_and_warps(_input.element_size(), V)
    world_size = 1 if dist_group is None else dist.get_world_size(dist_group)
    rank = 0 if dist_group is None else dist.get_rank(dist_group)

    # See cross_entropy_forward: label smoothing is only correct single-shard.
    assert not (label_smoothing > 0 and world_size > 1), (
        "label_smoothing > 0 is not supported with world_size > 1: the "
        "smoothing term only covers the local vocab shard"
    )

    # See cross_entropy_forward: reject out-of-range labels (silent +inf loss).
    if validate_targets:
        valid_target = (target_1d == ignore_idx) | (
            (target_1d >= 0) & (target_1d < V * world_size)
        )
        if not valid_target.all():
            raise ValueError(
                f"target out of range: expected in [0, {V * world_size}) or == "
                f"ignore_idx ({ignore_idx})"
            )

    # Global non-ignored row count — every chunk normalizes its gradient by the
    # same denominator so the result matches the non-chunked reduce path.
    # clamp_min(1): avoid divide-by-zero for an all-ignored batch (see forward).
    if reduce_loss:
        n_valid = (target_1d != ignore_idx).sum().clamp_min(1)
    else:
        n_valid = torch.ones((), dtype=torch.int64, device=_input.device)

    # Allocate scratch buffers once at max chunk size to avoid per-chunk alloc.
    m_d_Xy = torch.empty(chunk_rows * 3, dtype=torch.float32, device=_input.device)
    gathered_buf = (
        torch.empty(
            chunk_rows * 3 * world_size, dtype=torch.float32, device=_input.device
        )
        if world_size > 1
        else None
    )

    row = 0
    while row < n_rows:
        rows_this = min(chunk_rows, n_rows - row)
        chunk_x = input_2d[row : row + rows_this]  # view [rows_this, V]
        chunk_y = target_1d[row : row + rows_this]  # view [rows_this]
        chunk_loss = loss_1d[row : row + rows_this]  # view [rows_this]
        m_d_Xy_chunk = m_d_Xy[: rows_this * 3]

        _ce_local_softmax_stats_kernel[(rows_this,)](
            chunk_x,
            chunk_x.stride(0),
            chunk_y,
            chunk_y.stride(0),
            m_d_Xy_chunk,
            1,  # stride=1: (m,d,Xy) packed per row
            rank,
            V,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            waves_per_eu=_WAVES_PER_EU,
            num_stages=_NUM_STAGES,
        )

        if world_size > 1:
            gathered = gathered_buf[: rows_this * 3 * world_size]
            dist.all_gather_into_tensor(gathered, m_d_Xy_chunk, group=dist_group)
        else:
            gathered = m_d_Xy_chunk

        _ce_fused_loss_grad_kernel[(rows_this,)](
            chunk_x,
            chunk_x.stride(0),
            chunk_y,
            chunk_y.stride(0),
            chunk_loss,
            chunk_loss.stride(0),
            gathered,
            1,  # stride=1: same packed layout
            rank,
            world_size,
            ignore_idx,
            V,
            rows_this,
            n_valid,  # global denominator, shared across chunks
            reduce_loss=reduce_loss,
            label_smoothing=label_smoothing,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            waves_per_eu=_WAVES_PER_EU,
            num_stages=_NUM_STAGES,
        )

        row += rows_this

    if reduce_loss:
        loss = loss_1d.sum() / n_valid
    else:
        loss = loss_1d.reshape(B, SQ)
    return loss, _input


def cross_entropy_backward(
    _input: torch.Tensor,
    grad_output: torch.Tensor,
    is_cg_capturable: bool = False,
):
    """Backward pass: scale the pre-computed gradient by ``grad_output``.

    Always launches the scale kernel. A ``grad_output == 1.0`` fast path is
    deliberately avoided: detecting it needs ``torch.equal`` (a host sync on
    every backward, and a dtype-sensitive miss under bf16) which costs more
    than the single elementwise multiply it would skip.

    Args:
        _input:  Gradient tensor stored during forward (``[B, SQ, V_local]``).
        grad_output:  Upstream gradient.
        is_cg_capturable:  Retained for call-site compatibility; the kernel
            launch is CUDA-graph safe regardless, so it no longer gates a path.

    Returns:
        Gradient w.r.t. the logits, same shape as ``_input``.
    """
    _LOGGER.info(f"CROSS_ENTROPY_BACKWARD: input={tuple(_input.shape)}")
    B, SQ, V = _input.shape
    n_rows = B * SQ

    # The scale kernel reads grad_output as either a scalar or one value per
    # row with unit stride. Reject any other size (would read OOB) and make it
    # contiguous so a transposed per-row tensor can't scale rows with wrong
    # values.
    if grad_output.numel() not in (1, n_rows):
        raise ValueError(
            f"grad_output must be scalar or one value per row (n_rows={n_rows}), "
            f"got numel={grad_output.numel()}"
        )
    grad_output = grad_output.contiguous()

    BLOCK_SIZE, num_warps = _block_size_and_warps(_input.element_size(), V)
    _ce_grad_scale_kernel[(n_rows,)](
        _input,
        _input.stride(-2),
        grad_output,
        1 if grad_output.numel() > 1 else 0,
        V,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        waves_per_eu=_WAVES_PER_EU,
        num_stages=_NUM_STAGES,
    )
    return _input
