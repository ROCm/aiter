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
    shared: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
):
    """
    Returns out = RMSNorm(residual_out) and residual_out = all-reduce(x) + residual
    over the 4 ranks of ``group`` on gfx1250, with one kernel launch. All ranks get
    the same bits.

    With ``shared`` and ``gate``, x is ``x + sigmoid(gate) * shared`` (the MoE output
    of Qwen3-Next / Qwen3.5: routed experts + gated shared expert), rounded as the
    unfused ops: the sigmoid and the product to bf16, then the add in fp32 to bf16.
    That sum is never stored; the result is bitwise the one of the unfused ops.

    Args:
        x, residual (torch.Tensor): bf16 (tokens, hidden). hidden is a power of 2
            and at least 256. residual is the same on all ranks.
        weight (torch.Tensor): bf16 (hidden,). With gemma, the norm uses 1 + weight.
        quant (bool): Also return q, out as fp8, and scale, one fp32 value for
            each 128 values.
        inboxes, calls: The result of ``all_reduce_init``, for this op and one
            hidden size only, with max_bytes of 2 rows or more.
        shared (torch.Tensor): bf16 (tokens, hidden), contiguous, or None.
        gate (torch.Tensor): bf16 (tokens, 1), any row stride (the last column of
            the router GEMM's output), or None.
    """
    tokens, hidden = x.shape
    assert x.dtype == residual.dtype == weight.dtype == torch.bfloat16
    assert residual.shape == x.shape and weight.shape == (hidden,)
    assert x.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()
    assert hidden >= 256 and hidden & (hidden - 1) == 0
    # A row is one chunk, so rows of another hidden size would meet old call numbers.
    first = _HIDDEN.setdefault(calls.data_ptr(), hidden)
    assert first == hidden, f"these inboxes serve hidden {first}, not {hidden}"
    slot = 4 * _LINES * ((calls.numel() - 1) // 2)  # as in all_reduce_init
    triples = triton.cdiv(tokens, 3)  # a program sends or receives 3 rows
    # A triple is 2 * hidden int32 words of inbox.
    assert 2 * hidden * triples <= slot, "x is larger than max_bytes"

    gated = shared is not None
    assert gated == (gate is not None), "shared and gate come together"
    if gated:
        assert shared.shape == x.shape and shared.dtype == torch.bfloat16
        assert shared.is_contiguous()
        assert gate.shape == (tokens, 1) and gate.dtype == torch.bfloat16
    else:
        shared, gate = x, x  # not read

    _LOGGER.info(
        "ALL_REDUCE_RMSNORM_QUANT [gluon]: x=%s quant=%s gated=%s",
        x.shape,
        quant,
        gated,
    )

    out = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    q, scale = out, out  # not used without quant
    if quant:
        q = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
        scale = torch.empty(
            (tokens, hidden // 128), dtype=torch.float32, device=x.device
        )
    programs = min(triples, _ROW_PROGRAMS)
    _all_reduce_rmsnorm_kernel[(triples + programs,)](
        x,
        residual,
        weight,
        out,
        residual_out,
        q,
        scale,
        inboxes,
        calls,
        calls[-1:],
        tokens,
        programs,
        eps,
        shared,
        gate,
        gate.stride(0),
        RANK=dist.get_rank(group),
        HIDDEN=hidden,
        SLOT=slot,
        MAXIT=_MAXIT,
        GEMMA=gemma,
        QUANT=quant,
        GATED=gated,
        num_warps=min(16, hidden // 128),  # one group of 128 values a wave
    )
    if quant:
        return out, residual_out, q, scale
    return out, residual_out
