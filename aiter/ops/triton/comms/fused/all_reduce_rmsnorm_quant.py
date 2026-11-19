# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.distributed as dist
import triton

from aiter.ops.triton._gluon_kernels.gfx1250.comms.fused.all_reduce_rmsnorm_quant import (
    _all_reduce_rmsnorm_kernel,
)
from aiter.ops.triton.comms.all_reduce import _LINES, _MAXIT
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()
_HIDDEN = {}  # the hidden size of each state, by the address of its calls tensor
_ROW_PROGRAMS = 512  # receivers: more polling receivers can fill the GPU


def all_reduce_rmsnorm_quant(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    gemma: bool,
    quant: bool,
    group,
    inboxes,
    calls,
    per_token: bool = False,
    store_out: bool = True,
):
    """
    Returns out = RMSNorm(residual_out) and residual_out = all-reduce(x) + residual
    over the 4 ranks of ``group`` on gfx1250, with one kernel launch. All ranks get
    the same bits.

    Args:
        x, residual (torch.Tensor): bf16 (tokens, hidden). hidden is a multiple of
            128 and at least 256. residual is the same on all ranks.
        weight (torch.Tensor): bf16 (hidden,). With gemma, the norm uses 1 + weight.
        quant (bool): Also return q, out as fp8, and scale, one fp32 value for
            each 128 values.
        per_token (bool): The quant has one scale for each row: scale is
            (tokens, 1).
        store_out (bool): Store out. Without it, out is None. Needs quant.
        inboxes, calls: The result of ``all_reduce_init``, for this op and one
            hidden size only, with max_bytes not less than the bytes of x.

    Returns:
        tuple: (out, residual_out), and with quant (out, residual_out, q, scale).
    """
    tokens, hidden = x.shape
    assert x.dtype == residual.dtype == weight.dtype == torch.bfloat16
    assert residual.shape == x.shape and weight.shape == (hidden,)
    assert x.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()
    assert hidden >= 256 and hidden % 128 == 0
    assert quant or store_out, "store_out=False needs quant"
    # The inbox lines and the call counters of a row depend on hidden: rows of
    # another hidden size would meet old call numbers.
    first = _HIDDEN.setdefault(calls.data_ptr(), hidden)
    assert first == hidden, f"these inboxes serve hidden {first}, not {hidden}"
    slot = 4 * _LINES * ((calls.numel() - 1) // 2)  # as in all_reduce_init
    chunks = triton.cdiv(hidden, 6 * _LINES)  # chunks of 384 values in a row
    # A chunk is 4 * _LINES int32 words of inbox.
    assert 4 * _LINES * chunks * tokens <= slot, "x is larger than max_bytes"

    _LOGGER.info("ALL_REDUCE_RMSNORM_QUANT [gluon]: x=%s quant=%s", x.shape, quant)

    out = torch.empty_like(x) if store_out else None
    residual_out = torch.empty_like(x)
    q, scale = residual_out, residual_out  # not used without quant
    group_size = hidden if per_token else 128
    if quant:
        q = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
        scale = torch.empty(
            (tokens, hidden // group_size), dtype=torch.float32, device=x.device
        )
    programs = min(tokens, _ROW_PROGRAMS)
    _all_reduce_rmsnorm_kernel[(tokens + programs,)](
        x,
        residual,
        weight,
        residual_out if out is None else out,  # not used without store_out
        residual_out,
        q,
        scale,
        inboxes,
        calls,
        calls[-1:],
        tokens,
        programs,
        eps,
        RANK=dist.get_rank(group),
        HIDDEN=hidden,
        SLOT=slot,
        MAXIT=_MAXIT,
        GEMMA=gemma,
        QUANT=quant,
        GROUP=group_size,
        STORE_OUT=store_out,
        # A wave for each chunk, up to 16. The chunks of a row go up to a power of 2.
        num_warps=min(16, triton.next_power_of_2(chunks)),
    )
    if quant:
        return out, residual_out, q, scale
    return out, residual_out
