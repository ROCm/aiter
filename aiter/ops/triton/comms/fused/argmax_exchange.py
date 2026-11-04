# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.distributed as dist
import triton

from aiter.ops.triton._gluon_kernels.gfx1250.comms.fused.argmax_exchange import (
    _argmax_exchange_pick_kernel,
)
from aiter.ops.triton.comms.all_reduce import _LINES, _MAXIT
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()
_ROWS = 16  # rows of a program


def argmax_exchange_max_bytes(max_rows: int) -> int:
    """The ``max_bytes`` of ``all_reduce_init`` that serves up to max_rows rows."""
    return max(16 * max_rows, 8 * _LINES * triton.cdiv(max_rows, _ROWS))


def argmax_exchange_pick(
    keys: torch.Tensor,
    vocab_start: int,
    out: torch.Tensor,
    group,
    inboxes,
    calls,
) -> torch.Tensor:
    """
    Writes into ``out`` the token of the largest logit of each row over the vocab
    shards of the 4 ranks of ``group`` (gfx1250), in one kernel launch: the greedy
    pick of a TP LM head. All ranks get the same tokens. All ranks must make the
    same calls in the same order. HIP graphs are supported.

    The token is torch.argmax of the gathered logits: of equal maxima the lower
    rank, and in a rank the lower column (``gemm_a16w16_argmax_keys``).

    Args:
        keys (torch.Tensor): int64 (M, tiles), ``gemm_a16w16_argmax_keys`` of this
            rank's shard.
        vocab_start (int): The token of column 0 of this rank's shard.
        out (torch.Tensor): Contiguous int32 (M,).
        inboxes, calls: ``all_reduce_init(group, argmax_exchange_max_bytes(M'))``
            for this op only, M' >= M. If a poll times out, the tokens are wrong
            and calls[-1] is more than 0.
    """
    M, num_tiles = keys.shape
    assert keys.dtype == torch.int64 and keys.stride(1) == 1
    assert out.dtype == torch.int32 and out.is_contiguous() and out.numel() == M
    chunks = (calls.numel() - 1) // 2
    slot = 4 * _LINES * chunks  # int32 words of a sender's slot, as in all_reduce
    blocks = triton.cdiv(M, _ROWS)
    assert 4 * M <= slot and blocks <= chunks, "more rows than the inboxes serve"
    _LOGGER.info("ARGMAX_EXCHANGE_PICK [gluon]: keys=%s", keys.shape)
    _argmax_exchange_pick_kernel[(2 * blocks,)](
        keys,
        out,
        inboxes,
        calls,
        calls[-1:],
        M,
        num_tiles,
        keys.stride(0),
        vocab_start,
        RANK=dist.get_rank(group),
        ROWS=_ROWS,
        BLOCK_T=64,
        SLOT=slot,
        MAXIT=_MAXIT,
        num_warps=4,
    )
    return out
