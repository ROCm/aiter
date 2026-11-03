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
_QUANT_NONE = 0
_QUANT_GROUP = 1
_QUANT_TOKEN = 2


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
    *,
    per_token_quant: bool = False,
    return_norm: bool = True,
):
    """
    Returns out = RMSNorm(residual_out) and residual_out = all-reduce(x) + residual
    over the 4 ranks of ``group`` on gfx1250, with one kernel launch. All ranks get
    the same bits.

    Args:
        x, residual (torch.Tensor): bf16 (tokens, hidden). hidden is a power of 2
            and at least 256, or 6144. residual is the same on all ranks.
        weight (torch.Tensor): bf16 (hidden,). With gemma, the norm uses 1 + weight.
        quant (bool): Also return q as fp8 and its fp32 scale. The existing default
            is one scale per 128 values; ``per_token_quant`` selects one per row.
        return_norm (bool): Return/store the bf16 normalized output. This must be
            true without quantization.
        inboxes, calls: The result of ``all_reduce_init``, for this op and one
            hidden size only. Its slot must hold 2 * next_power_of_2(hidden)
            int32 words per triple.
    """
    tokens, hidden = x.shape
    assert x.dtype == residual.dtype == weight.dtype == torch.bfloat16
    assert residual.shape == x.shape and weight.shape == (hidden,)
    assert x.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()
    assert hidden >= 256 and (hidden & (hidden - 1) == 0 or hidden == 6144)
    assert not per_token_quant or quant
    assert return_norm or quant
    quant_mode = (
        _QUANT_TOKEN
        if per_token_quant
        else (_QUANT_GROUP if quant else _QUANT_NONE)
    )
    # A row is one chunk, so rows of another hidden size would meet old call numbers.
    first = _HIDDEN.setdefault(calls.data_ptr(), hidden)
    assert first == hidden, f"these inboxes serve hidden {first}, not {hidden}"
    slot = 4 * _LINES * ((calls.numel() - 1) // 2)  # as in all_reduce_init
    triples = triton.cdiv(tokens, 3)  # a program sends or receives 3 rows
    hidden_pad = triton.next_power_of_2(hidden)
    # A triple is 2 * hidden_pad int32 words of inbox.
    assert 2 * hidden_pad * triples <= slot, "x is larger than max_bytes"

    _LOGGER.info(
        "ALL_REDUCE_RMSNORM_QUANT [gluon]: x=%s quant_mode=%d return_norm=%s",
        x.shape,
        quant_mode,
        return_norm,
    )

    residual_out = torch.empty_like(x)
    # The kernel argument remains present for one compiled signature. It is dead
    # when RETURN_NORM is false, so aliasing it avoids an unused allocation.
    out = torch.empty_like(x) if return_norm else residual_out
    q, scale = out, out  # not used without quant
    if quant:
        q = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
        scale_shape = (tokens, 1) if per_token_quant else (tokens, hidden // 128)
        scale = torch.empty(scale_shape, dtype=torch.float32, device=x.device)
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
        RANK=dist.get_rank(group),
        HIDDEN=hidden,
        SLOT=slot,
        MAXIT=_MAXIT,
        GEMMA=gemma,
        QUANT_MODE=quant_mode,
        RETURN_NORM=return_norm,
        num_warps=min(16, hidden // 128),  # one group of 128 values a wave
    )
    if quant:
        if not return_norm:
            return q, residual_out, scale
        return out, residual_out, q, scale
    return out, residual_out
