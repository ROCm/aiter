# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 decode index block scores over the NVFP4 index cache (gfx1250).

Bitwise drop-in for ATOM's Triton ``_decode_index_score_transposed_kernel`` on
the FP4 index cache; see ``kernels.index_score_decode_nvfp4_gfx1250``.
"""

import flydsl.expr as fx
import torch

from aiter.jit.utils.chip_info import get_cu_num

from .kernels.index_score_decode_nvfp4_gfx1250 import (
    MAX_BLOCKS,
    MAX_ROWS,
    compile_index_score_decode_nvfp4,
)
from .index_score_prefill import nvfp4_index_inputs_supported
from .kernels.index_score_prefill_nvfp4_gfx1250 import HEAD_DIM, sm_scale_log2e
from .kernels.tensor_shim import _run_compiled, ptr_arg

# Workgroups a compute unit the launch aims for. The kernel fits 6 a CU (VGPRs);
# 4 left the most bandwidth in the context-parallel scorer's sweeps on MI455X
# (batch 16-32, 64K-256K contexts).
WORKGROUPS_PER_CU = 4


def index_score_decode_nvfp4_supported(idx_q, payload, scale, max_query_len) -> bool:
    """Whether :func:`index_score_decode_nvfp4` serves this call: the scorers'
    shared layout, at most 32 (index head, query row) pairs a request, and
    index planes its async loads can address (16 B pieces of the payload, 8 B
    of the scales: bases and page strides aligned to that)."""
    return (
        nvfp4_index_inputs_supported(idx_q, payload, scale)
        and 1 <= idx_q.shape[1] * max_query_len <= MAX_ROWS
        and scale.stride(0) % 8 == 0
        and payload.data_ptr() % 16 == 0
        and scale.data_ptr() % 8 == 0
    )


def decode_score_programs(batch: int, blocks: int, workgroups: int) -> int:
    """Workgroups a request: the batch's share of ``workgroups``, and enough
    that a workgroup's chunk (``ceil(blocks / P)``) fits its 64-block window."""
    return max(1, workgroups // max(1, batch), -(-blocks // MAX_BLOCKS))


def index_score_decode_nvfp4(
    idx_q: torch.Tensor,  # [batch * max_query_len, heads, 128] bf16
    payload: torch.Tensor,  # [pages, 128, 64] uint8
    scale: torch.Tensor,  # [pages, 128, 8] uint8 (E4M3 bits)
    score: torch.Tensor,  # [heads, >= batch * max_query_len, max_block] fp32
    block_table: torch.Tensor,  # [batch, blocks] int32
    seq_lens: torch.Tensor,  # [batch] int32
    sm_scale: float,
    max_query_len: int,
    *,
    workgroups: int | None = None,
    ring: int = 4,
) -> None:
    """``score[h, b * max_query_len + t, blk]`` for the first
    ``ceil(seq_lens[b] / 128)`` blocks of every request (at most what ``score``
    and ``block_table`` hold), exactly as ATOM's Triton decode scorer (same
    bits); later blocks are not written.
    Row t of a request sees the keys before ``seq_lens[b] - max_query_len + t +
    1``.

    Everything read here is fixed once the buffers exist (shapes, strides,
    devices): safe to capture in a graph. ``workgroups``: the launch's
    workgroup budget, ``WORKGROUPS_PER_CU`` a compute unit by default; ``ring``:
    LDS slots a wave. Neither changes a score.

    Raises ``ValueError`` on a layout :func:`index_score_decode_nvfp4_supported`
    declines and on metadata the kernel cannot address; what only the device
    can see (page ids inside the cache) is the caller's, as with the Triton
    scorer.
    """
    if not index_score_decode_nvfp4_supported(idx_q, payload, scale, max_query_len):
        raise ValueError(
            "[FlyDSL index_score_decode_nvfp4] needs gfx1250, a bf16 query "
            f"[rows, heads, {HEAD_DIM}] of at most {MAX_ROWS} (head, row) pairs a "
            "request and byte index planes [pages, 128, 64] / [pages, 128, 8] "
            "with contiguous rows, 16 B / 8 B aligned"
        )
    total_q, heads, _ = idx_q.shape
    batch = seq_lens.shape[0]
    # Blocks a request can have: what both the scores and its table row hold.
    max_block = min(score.shape[2], block_table.shape[-1])
    dev = idx_q.device
    if any(
        t.device != dev or not t.is_cuda
        for t in (payload, scale, score, block_table, seq_lens)
    ):
        raise ValueError("every tensor must be on the query's GPU")
    if (
        block_table.dtype != torch.int32
        or seq_lens.dtype != torch.int32
        or seq_lens.dim() != 1
        or block_table.dim() != 2
        or (batch > 1 and seq_lens.stride(0) != 1)
        or (block_table.shape[1] > 1 and block_table.stride(1) != 1)
        or block_table.shape[0] < batch
    ):
        raise ValueError(
            "seq_lens must be contiguous int32 [batch] and block_table int32 "
            f"[>= {batch}, blocks] with unit inner stride"
        )
    if (
        score.dtype != torch.float32
        or score.dim() != 3
        or score.shape[0] != heads
        or score.shape[1] < batch * max_query_len
        or (score.shape[2] > 1 and score.stride(2) != 1)
    ):
        raise ValueError(
            f"score must be fp32 [{heads}, >= {batch * max_query_len}, blocks] "
            "with unit inner stride"
        )
    if total_q < batch * max_query_len:
        raise ValueError(
            f"idx_q holds {total_q} rows, the batch needs {batch * max_query_len}"
        )
    strides = (
        idx_q.stride(0),
        idx_q.stride(1),
        payload.stride(0),
        scale.stride(0),
        block_table.stride(0),
        score.stride(0),
        score.stride(1),
    )
    if max(strides) >= 2**31 or batch * block_table.stride(0) + max_block >= 2**31:
        raise ValueError("strides must fit int32")
    if batch == 0 or max_block == 0:
        return
    if workgroups is None:
        workgroups = WORKGROUPS_PER_CU * get_cu_num()
    compiled = compile_index_score_decode_nvfp4(
        heads=heads, query_len=max_query_len, ring=ring
    )
    _run_compiled(
        compiled["launch"],
        ptr_arg(idx_q, fx.BFloat16),
        ptr_arg(payload, fx.Uint8),
        ptr_arg(scale, fx.Uint8),
        ptr_arg(block_table, fx.Int32),
        ptr_arg(seq_lens, fx.Int32),
        ptr_arg(score, fx.Float32),
        *strides,
        max_block,
        decode_score_programs(batch, max_block, workgroups),
        sm_scale_log2e(sm_scale),
        batch,
        torch.cuda.current_stream(dev),
        specialization_key=(dev.index,),
    )
