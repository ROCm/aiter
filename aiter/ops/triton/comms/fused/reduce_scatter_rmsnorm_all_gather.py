# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import ctypes
from typing import NamedTuple

import torch
import torch.distributed as dist
import triton

from aiter.ops.triton._gluon_kernels.gfx1250.comms.fused.reduce_scatter_rmsnorm_all_gather import (
    HW_WORDS,
    _reduce_scatter_rmsnorm_all_gather_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()
_HIDDEN = 2048
_POOLS = {}  # the IPC pool of each state, by the address of its box


class RsRmsnormAgState(NamedTuple):
    """The IPC boxes of the 4 ranks and the device counters of one group."""

    rank: int
    boxes: tuple
    ctrl: torch.Tensor  # [call counter, programs done], advanced on the device
    err: torch.Tensor  # polls that timed out
    nrows: torch.Tensor  # the row count of the call
    max_rows: int
    programs: int
    rows_per_quarter: int
    num_warps: int
    group_rows: int
    max_polls: int
    rs_flags: int  # word offsets in a box
    ag_flags: int
    inbox: int
    rows_out: int


def reduce_scatter_rmsnorm_all_gather_init(
    group, max_rows=16384, programs=512, rows_per_quarter=8, num_warps=4
):
    """
    Allocates the IPC box of this rank and maps the boxes of the 4 ranks of
    ``group`` (gfx1250). All ranks must call it. The box holds the flags, two
    banks of reduce-scatter inboxes and the all-gather rows.

    Args:
        group: A process group of 4 ranks, for example the TP group's cpu_group.
        max_rows (int): The largest row count of a call (hidden 2048).
        programs (int): The persistent programs of a call. Every call of a forward
            pass must use the same value, because it sets the rows a rank owns.
        rows_per_quarter (int): The rows a program can own.

    Returns:
        RsRmsnormAgState: The state of ``reduce_scatter_rmsnorm_all_gather``.
    """
    from aiter.dist.device_communicators.custom_all_reduce import (
        IPCBufferPool,
        _loaded_hip_runtime,
    )

    assert dist.get_world_size(group) == 4
    assert max_rows <= programs * 4 * rows_per_quarter
    rs_flags = 0
    ag_flags = rs_flags + 8 * programs
    inbox = triton.cdiv(ag_flags + 8 * programs, 1024) * 1024
    rows_out = inbox + 8 * programs * rows_per_quarter * HW_WORDS
    words = rows_out + max_rows * HW_WORDS
    device = torch.device("cuda", torch.cuda.current_device())
    pool = IPCBufferPool(device, group)
    box = pool.create("rs_rmsnorm_ag_box", words * 4, raw_cached=True)
    err = _loaded_hip_runtime().hipMemset(
        ctypes.c_void_p(box.data_ptr), 0, ctypes.c_size_t(words * 4)
    )
    assert err == 0, f"hipMemset failed: {err}"  # 0 is never a call number
    torch.cuda.synchronize()
    boxes = tuple(int(a) for a in pool.open_peer_buffers("rs_rmsnorm_ag_box"))
    rank = dist.get_rank(group)
    _POOLS[boxes[rank]] = pool
    dist.barrier(group=group)
    return RsRmsnormAgState(
        rank=rank,
        boxes=boxes,
        ctrl=torch.zeros(4, dtype=torch.int32, device=device),
        err=torch.zeros(1, dtype=torch.int32, device=device),
        nrows=torch.zeros(1, dtype=torch.int32, device=device),
        max_rows=max_rows,
        programs=programs,
        rows_per_quarter=rows_per_quarter,
        num_warps=num_warps,
        group_rows=4,
        max_polls=2_000_000,
        rs_flags=rs_flags,
        ag_flags=ag_flags,
        inbox=inbox,
        rows_out=rows_out,
    )


def reduce_scatter_rmsnorm_all_gather_free(group, state):
    """Unmaps the peer boxes and frees the box of this rank. All ranks must call it."""
    pool = _POOLS.pop(state.boxes[state.rank])
    torch.cuda.synchronize()
    dist.barrier(group=group)
    pool.close_peer_buffers()
    dist.barrier(group=group)  # no peer uses the box of this rank now
    pool.close()


def reduce_scatter_rmsnorm_all_gather_supports(x, residual, weight, state):
    """Whether ``reduce_scatter_rmsnorm_all_gather`` takes these tensors."""
    return (
        x.dim() == 2
        and x.shape[1] == _HIDDEN
        and x.shape[0] <= state.max_rows
        and x.dtype == residual.dtype == weight.dtype == torch.bfloat16
        and x.is_contiguous()
        and residual.is_contiguous()
        and weight.is_contiguous()
        and residual.shape == x.shape
        and weight.shape == (_HIDDEN,)
    )


def reduce_scatter_rmsnorm_all_gather(x, residual, weight, eps, gemma, state):
    """
    Returns out = RMSNorm(all-reduce(x) + residual) on every rank, and
    residual_out = all-reduce(x) + residual for the rows that this rank owns, over
    the 4 ranks of the group of ``state`` on gfx1250, with one kernel launch. The
    sum is in fp32 in rank order. The other rows of residual_out are not written:
    the residual stays row-sharded. This is correct when every call of a forward
    pass has the same row count, because the rows a rank owns depend only on the
    row count and ``programs``.

    Args:
        x (torch.Tensor): bf16 (tokens, 2048), the partial sums of this rank.
        residual (torch.Tensor): bf16 (tokens, 2048). Only the rows that this rank
            owns are read.
        weight (torch.Tensor): bf16 (2048,). With gemma, the norm uses 1 + weight.
        state (RsRmsnormAgState): The result of
            ``reduce_scatter_rmsnorm_all_gather_init``.

    Returns:
        tuple: (out, residual_out), bf16 (tokens, 2048).
    """
    assert reduce_scatter_rmsnorm_all_gather_supports(x, residual, weight, state)
    _LOGGER.info("REDUCE_SCATTER_RMSNORM_ALL_GATHER [gluon]: x=%s", x.shape)
    out = torch.empty_like(x)
    residual_out = torch.empty_like(residual)
    state.nrows.fill_(x.shape[0])
    _reduce_scatter_rmsnorm_all_gather_kernel[(state.programs,)](
        x.view(torch.int32),
        residual.view(torch.int32),
        residual_out.view(torch.int32),
        out.view(torch.int32),
        weight.view(torch.int32),
        state.nrows,
        state.ctrl,
        state.err,
        *state.boxes,
        RANK=state.rank,
        G=state.programs,
        QRM=state.rows_per_quarter,
        NW=state.num_warps,
        RG=state.group_rows,
        MAXIT=state.max_polls,
        EPS=eps,
        GEMMA=gemma,
        RSF=state.rs_flags,
        ZF=state.ag_flags,
        INB=state.inbox,
        ZO=state.rows_out,
        num_warps=state.num_warps,
    )
    return out, residual_out
