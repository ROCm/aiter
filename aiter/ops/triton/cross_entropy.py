###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""Vocab-parallel cross-entropy: Python API wrapping Triton kernels.

This module provides ``cross_entropy_forward`` and ``cross_entropy_backward``
that orchestrate kernel launches and ``all_gather`` communication so that
higher-level frameworks (e.g. Lumen, Megatron) only need a single function
call.
"""

from functools import reduce
from operator import mul

import torch
import torch.distributed as dist
import triton

from aiter.ops.triton._triton_kernels.cross_entropy import (
    cross_entropy_kernel,
    element_mul_kernel,
    online_softmax_kernel,
)

__all__ = [
    "cross_entropy_backward",
    "cross_entropy_forward",
    "cross_entropy_forward_chunked",
]

MAX_FUSED_SIZE = 65536 // 2
NUM_WARPS = 16


def cross_entropy_forward(
    _input: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float,
    reduce_loss: bool,
    dist_group: dist.ProcessGroup | None,
    ignore_idx: int,
):
    """Compute vocab-parallel cross-entropy loss (forward).

    Args:
        _input:  Logits shard for this TP rank — ``[B, SQ, V_local]``.
        target:  Label indices — ``[B, SQ]``  (global vocab ids).
        label_smoothing:  Label-smoothing factor (0 = standard CE).
        reduce_loss:  If ``True``, return scalar loss averaged over rows.
        dist_group:  TP process group (``None`` for single-GPU).
        ignore_idx:  Target value to ignore (default ``-100``).

    Returns:
        ``(loss, grad_input)`` where *grad_input* has the same shape as
        ``_input`` and already contains the gradient (to be scaled by
        ``grad_output`` in the backward pass).
    """
    B, SQ, V = _input.shape
    n_rows = B * SQ
    assert reduce(mul, list(target.size())) == n_rows
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    loss_1d = torch.zeros(n_rows, dtype=torch.float32, device=_input.device)
    m_d_Xy = torch.zeros(n_rows * 3, dtype=torch.float32, device=_input.device)

    if _input.stride(-1) != 1:
        _input = _input.contiguous()
    if target.stride(-1) != 1:
        target = target.contiguous()

    rank = 0 if dist_group is None else dist.get_rank(dist_group)

    online_softmax_kernel[(n_rows,)](
        _input,
        _input.stride(-2),
        target,
        target.stride(-1),
        m_d_Xy,
        m_d_Xy.stride(-1),
        rank,
        V,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )

    world_size = 1 if dist_group is None else dist.get_world_size(dist_group)
    if world_size > 1:
        gathered = torch.zeros(
            n_rows * 3 * world_size, dtype=torch.float32, device=_input.device
        )
        dist.all_gather_into_tensor(gathered, m_d_Xy, group=dist_group)
    else:
        gathered = m_d_Xy

    cross_entropy_kernel[(n_rows,)](
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
        reduce_loss=reduce_loss,
        label_smoothing=label_smoothing,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )

    loss = loss_1d.reshape(B, SQ) if not reduce_loss else (loss_1d.sum() / n_rows)
    return loss, _input


def cross_entropy_forward_chunked(
    _input: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float,
    reduce_loss: bool,
    dist_group: dist.ProcessGroup | None,
    ignore_idx: int,
    chunk_rows: int,
):
    """Chunked vocab-parallel cross-entropy forward.

    Splits the row dimension (B*SQ) into chunks of ``chunk_rows`` and
    processes each independently: online_softmax → allgather → ce_kernel.
    Peak activation memory is proportional to ``chunk_rows * V_local``
    rather than ``B*SQ * V_local``.

    The gradient is written in-place into ``_input`` exactly as in the
    non-chunked path.  The caller must ensure ``_input`` is contiguous.

    Args:
        _input:    Logit shard ``[B, SQ, V_local]`` — modified in-place.
        target:    Label indices ``[B, SQ]`` (global vocab ids).
        chunk_rows: Number of rows per chunk.  Must be >= 1.
        (other args: same semantics as :func:`cross_entropy_forward`)

    Returns:
        ``(loss, _input)`` where *loss* is ``[B, SQ]`` or a scalar.
    """
    B, SQ, V = _input.shape
    n_rows = B * SQ

    # Flatten to 2-D views so we can slice by row without copying.
    input_2d = _input.reshape(n_rows, V)  # view, no alloc
    target_1d = target.reshape(n_rows)  # view, no alloc
    loss_1d = torch.empty(n_rows, dtype=torch.float32, device=_input.device)

    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    world_size = 1 if dist_group is None else dist.get_world_size(dist_group)
    rank = 0 if dist_group is None else dist.get_rank(dist_group)

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

        online_softmax_kernel[(rows_this,)](
            chunk_x,
            chunk_x.stride(0),
            chunk_y,
            chunk_y.stride(0),
            m_d_Xy_chunk,
            1,  # stride=1: (m,d,Xy) packed per row
            rank,
            V,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=NUM_WARPS,
        )

        if world_size > 1:
            gathered = gathered_buf[: rows_this * 3 * world_size]
            dist.all_gather_into_tensor(gathered, m_d_Xy_chunk, group=dist_group)
        else:
            gathered = m_d_Xy_chunk

        cross_entropy_kernel[(rows_this,)](
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
            reduce_loss=False,  # accumulate manually after loop
            label_smoothing=label_smoothing,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=NUM_WARPS,
        )

        row += rows_this

    if reduce_loss:
        loss = loss_1d.sum() / n_rows
    else:
        loss = loss_1d.reshape(B, SQ)
    return loss, _input


def cross_entropy_backward(
    _input: torch.Tensor,
    grad_output: torch.Tensor,
    is_cg_capturable: bool = False,
):
    """Backward pass: scale pre-computed gradient by ``grad_output``.

    If ``grad_output`` is a scalar 1.0 (and not CUDA-graph capturable),
    the multiplication is skipped as an optimisation.

    Args:
        _input:  Gradient tensor stored during forward (``[B, SQ, V_local]``).
        grad_output:  Upstream gradient.
        is_cg_capturable:  Whether the operation must be CUDA-graph safe.

    Returns:
        Gradient w.r.t. the logits, same shape as ``_input``.
    """
    if not is_cg_capturable and torch.equal(
        grad_output, torch.tensor(1.0, device=grad_output.device)
    ):
        return _input

    B, SQ, V = _input.shape
    n_rows = B * SQ
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    element_mul_kernel[(n_rows,)](
        _input,
        _input.stride(-2),
        grad_output,
        1 if grad_output.numel() > 1 else 0,
        V,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    return _input
