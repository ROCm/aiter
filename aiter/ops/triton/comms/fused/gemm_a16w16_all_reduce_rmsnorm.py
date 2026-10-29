# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.distributed as dist
import triton

from aiter.ops.triton._gluon_kernels.gfx1250.comms.fused.gemm_a16w16_all_reduce_rmsnorm import (
    _gemm_a16w16_all_reduce_rmsnorm_kernel,
)
from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16 import (
    create_shared_layouts,
    create_wmma_layouts,
)
from aiter.ops.triton.comms.all_reduce import _LINES, _MAXIT
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()
_HIDDEN = {}  # the hidden size of each state, by the address of its calls tensor
_ROW_PROGRAMS = 512  # receivers: more polling receivers can fill the GPU


def gemm_a16w16_all_reduce_rmsnorm(
    x: torch.Tensor,
    w: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    gemma: bool,
    group,
    inboxes,
    calls,
):
    """
    Returns out = RMSNorm(residual_out) and residual_out = all-reduce(x @ w.T) +
    residual over the 4 ranks of ``group`` on gfx1250, with one kernel launch. x @ w.T
    is the bf16 GEMM of ``gemm_a16w16``, and the all-reduce + RMSNorm is the one of
    ``all_reduce_rmsnorm_quant``. All ranks get the same bits.

    Args:
        x (torch.Tensor): bf16 (tokens, K), the input of this rank. K is a multiple
            of the BLOCK_K of the tile.
        w (torch.Tensor): bf16 (hidden, K), the weight of this rank. hidden is a
            power of 2 and at least 256.
        residual (torch.Tensor): bf16 (tokens, hidden), the same on all ranks.
        weight (torch.Tensor): bf16 (hidden,). With gemma, the norm uses 1 + weight.
        inboxes, calls: The result of ``all_reduce_init``, for this op and one hidden
            size only.
    """
    tokens, k = x.shape
    hidden = w.shape[0]
    assert x.dtype == w.dtype == residual.dtype == weight.dtype == torch.bfloat16
    assert w.shape == (hidden, k) and residual.shape == (tokens, hidden)
    assert weight.shape == (hidden,)
    assert x.is_contiguous() and w.is_contiguous() and residual.is_contiguous()
    assert weight.is_contiguous()
    assert hidden >= 256 and hidden & (hidden - 1) == 0
    config, _ = get_gemm_config(
        "GEMM-A16W16-ALL-REDUCE-RMSNORM", tokens, hidden, k, backend="gluon"
    )
    rows, block_n, block_k = config["ROWS"], config["BLOCK_N"], config["BLOCK_K"]
    num_buffers, num_warps = config["NUM_BUFFERS"], config["num_warps"]
    assert k % block_k == 0 and k // block_k >= num_buffers - 1
    # A triple of rows is one place in the inbox, so a state serves one hidden size.
    first = _HIDDEN.setdefault(calls.data_ptr(), hidden)
    assert first == hidden, f"these inboxes serve hidden {first}, not {hidden}"
    slot = 4 * _LINES * ((calls.numel() - 1) // 2)  # as in all_reduce_init
    blocks = triton.cdiv(tokens, 3 * rows)
    triples = blocks * rows
    tiles = blocks * (hidden // block_n)
    # A triple is 2 * hidden int32 words of inbox and has a counter.
    assert 2 * hidden * triples <= slot, "x is larger than max_bytes"
    assert triples <= calls.numel() - 1

    _LOGGER.info("GEMM_A16W16_ALL_REDUCE_RMSNORM [gluon]: x=%s w=%s", x.shape, w.shape)

    out = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=x.device)
    residual_out = torch.empty_like(out)
    wmma, operand_a, operand_b = create_wmma_layouts(num_warps)
    shared_a, shared_b = create_shared_layouts(4 * rows, block_n, block_k, "TN")
    programs = min(triples, _ROW_PROGRAMS)
    _gemm_a16w16_all_reduce_rmsnorm_kernel[(tiles + programs,)](
        x,
        w,
        residual,
        weight,
        out,
        residual_out,
        inboxes,
        calls,
        calls[-1:],
        tokens,
        programs,
        eps,
        RANK=dist.get_rank(group),
        HIDDEN=hidden,
        K=k,
        ROWS=rows,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        NUM_BUFFERS=num_buffers,
        N_FIRST=config["N_FIRST"],
        SLOT=slot,
        MAXIT=_MAXIT,
        GEMMA=gemma,
        SHARED_A=shared_a,
        SHARED_B=shared_b,
        WMMA=wmma,
        OPERAND_A=operand_a,
        OPERAND_B=operand_b,
        num_warps=num_warps,
    )
    return out, residual_out
