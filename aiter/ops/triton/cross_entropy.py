###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""Vocab-parallel cross-entropy: Python API wrapping Triton kernels.

This module provides ``cross_entropy_forward`` and ``cross_entropy_backward``
that orchestrate kernel launches and ``all_gather`` communication so that
higher-level frameworks (e.g. Lumen, Megatron) only need a single function
call.
"""

from typing import Union
from functools import reduce
from operator import mul

import torch
import torch.distributed as dist
import triton

from aiter.ops.triton._triton_kernels.cross_entropy import (
    online_softmax_kernel,
    cross_entropy_kernel,
    element_mul_kernel,
)

__all__ = [
    "cross_entropy_forward",
    "cross_entropy_backward",
]

MAX_FUSED_SIZE = 65536 // 2
NUM_WARPS = 16


def cross_entropy_forward(
    _input: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float,
    reduce_loss: bool,
    dist_group: Union[dist.ProcessGroup, None],
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
        _input, _input.stride(-2),
        target, target.stride(-1),
        m_d_Xy, m_d_Xy.stride(-1),
        rank, V,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS,
    )

    world_size = 1 if dist_group is None else dist.get_world_size(dist_group)
    if world_size > 1:
        gathered = torch.zeros(n_rows * 3 * world_size, dtype=torch.float32, device=_input.device)
        dist.all_gather_into_tensor(gathered, m_d_Xy, group=dist_group)
    else:
        gathered = m_d_Xy

    cross_entropy_kernel[(n_rows,)](
        _input, _input.stride(-2),
        target, target.stride(-1),
        loss_1d, loss_1d.stride(-1),
        gathered, gathered.stride(-1),
        rank, world_size, ignore_idx, V, n_rows,
        reduce_loss=reduce_loss,
        label_smoothing=label_smoothing,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS,
    )

    loss = loss_1d.reshape(B, SQ) if not reduce_loss else (loss_1d.sum() / n_rows)
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
        _input, _input.stride(-2),
        grad_output, 1 if grad_output.numel() > 1 else 0,
        V,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS,
    )
    return _input
