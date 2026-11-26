# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 decode index block scores over the NVFP4 index cache (gfx1250),
and the decode top-k over them.

:func:`index_score_decode_nvfp4` is a bitwise drop-in for ATOM's Triton
``_decode_index_score_transposed_kernel`` on the FP4 index cache (see
``kernels.index_score_decode_nvfp4_gfx1250``); :func:`index_select_decode` for
its ``_topk_index_packed_kernel`` in decode mode, sparse-table emit included
(``kernels.index_select_decode_gfx1250``); :func:`index_topk_decode_nvfp4` is
both, the scorer on its flat plan.
"""

import flydsl.expr as fx
import torch

from aiter.jit.utils.chip_info import (
    get_cu_num,
    get_gfx_runtime,
    get_lds_capacity_bytes,
)

from .index_score_prefill import nvfp4_index_inputs_supported
from .kernels.index_score_decode_nvfp4_gfx1250 import (
    FLAT_MAX_BLOCKS,
    MAX_BLOCKS,
    MAX_ROWS,
    compile_index_score_decode_nvfp4,
    flat_lds_bytes,
)
from .kernels.index_score_prefill_nvfp4_gfx1250 import HEAD_DIM, sm_scale_log2e
from .kernels.index_select_decode_gfx1250 import (
    WAVE,
    compile_index_select_decode,
    lds_bytes,
    plane_columns,
)
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


# The flat scorer's ring a wave (LDS slots; blocks in flight + 1).
FLAT_RING = 4


def flat_workgroups_per_cu(ring: int) -> int:
    """The flat scorer's resident workgroups a CU: ``WORKGROUPS_PER_CU``, or
    fewer when a build's LDS (its ring) admits fewer -- a budget past
    residency would leave a second round on an emptied machine."""
    lds = flat_lds_bytes(ring)
    return max(1, min(WORKGROUPS_PER_CU, get_lds_capacity_bytes("gfx1250") // lds))


def flat_programs(batch: int, blocks: int, workgroups: int) -> int:
    """The flat plan's grid: the budget, and enough that a split never needs
    more than the window (``FLAT_MAX_BLOCKS``) -- see the kernel builder."""
    return max(workgroups, batch * (-(-blocks // FLAT_MAX_BLOCKS) + 1))


def index_topk_decode_score_plane(heads: int, rows: int, max_block: int, device):
    """An fp32 score plane [heads, rows, max_block] the selector can read: a
    view of rows padded to whole 128-block steps, 16 B aligned."""
    full = torch.empty(heads, rows, plane_columns(max_block), device=device)
    return full[..., :max_block]


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
    flat: bool = False,
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
    LDS slots a wave; ``flat``: the length-proportional plan (a ragged batch's
    long requests get proportionally more workgroups). None changes a score.

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
        per_cu = flat_workgroups_per_cu(ring) if flat else WORKGROUPS_PER_CU
        workgroups = per_cu * get_cu_num()
    programs = (flat_programs if flat else decode_score_programs)(
        batch, max_block, workgroups
    )
    compiled = compile_index_score_decode_nvfp4(
        heads=heads, query_len=max_query_len, ring=ring, flat=flat
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
        programs,
        sm_scale_log2e(sm_scale),
        batch,
        torch.cuda.current_stream(dev),
        specialization_key=(dev.index,),
    )


def index_select_decode_supported(score, topk: int) -> bool:
    """Whether :func:`index_select_decode` serves a score plane ``[heads, rows,
    max_block]`` fp32 at this ``topk``: gfx1250, ``topk`` within a wave, block
    ids within the packed key's 16 bits, a row's LDS (8 B a column at top-16)
    within half a CU's -- two selector workgroups a CU, as D's ~500 rows on
    256 CUs need -- and the plane's rows padded to
    whole 128-block steps and 16 B aligned
    (:func:`index_topk_decode_score_plane`)."""
    if score.dim() != 3 or score.shape[0] * score.shape[1] == 0:
        return False
    cols = plane_columns(score.shape[2])
    last = (score.shape[0] - 1) * score.stride(0) + (score.shape[1] - 1) * score.stride(
        1
    )
    storage = score.untyped_storage()
    return (
        get_gfx_runtime() == "gfx1250"
        and score.dtype == torch.float32
        and score.stride(2) == 1
        and 1 <= topk <= WAVE
        and 1 <= score.shape[2] < 0xFFFF
        and lds_bytes(topk, cols) <= get_lds_capacity_bytes("gfx1250") // 2
        and score.stride(1) % 4 == 0
        and score.stride(0) % 4 == 0
        and score.data_ptr() % 16 == 0
        and score.stride(1) >= cols
        and storage.nbytes()
        >= score.data_ptr() - storage.data_ptr() + 4 * (last + cols)
    )


def index_select_decode(
    score: torch.Tensor,  # [heads, >= batch * max_query_len, max_block] fp32
    seq_lens: torch.Tensor,  # [batch] int32
    topk_idx: torch.Tensor,  # [heads, >= batch * max_query_len, topk] int32
    max_query_len: int,
    init_blocks: int,
    local_blocks: int,
    *,
    block_table: torch.Tensor | None = None,  # [batch, blocks] int32 (emit)
    sparse_bt: torch.Tensor | None = None,  # [rows * heads, topk * pages] int32
    sparse_ctx: torch.Tensor | None = None,  # [rows * heads] int32
    pages_per_block: int = 1,
    block_page_stride: int = 1,
) -> None:
    """ATOM's decode block selection (``_topk_index_packed_kernel`` with
    ``DECODE_MAX_Q = max_query_len``, ``NUM_KV_HEADS = heads``) over ``score``:
    for row ``r = b * max_query_len + t`` and head ``h``, the ``topk`` best of
    the row's ``(seq_lens[b] - max_query_len + t + 128) // 128`` blocks by
    ATOM's ``_pack_score_key`` (the first ``init_blocks`` pinned at 1e30, the
    last ``local_blocks`` at 1e29), in that order, into ``topk_idx[h, r]``, -1
    past their number. With ``sparse_bt`` (and ``block_table``, ``sparse_ctx``)
    also the attention's table row ``r * heads + h`` and its context length
    (``_emit_sparse_block_table_row``; the row is zero-filled to its
    ``topk * pages_per_block`` columns). Only the first
    ``batch * max_query_len`` rows are written; graph-safe."""
    emit = sparse_bt is not None
    heads, rows_cap, max_block = score.shape
    batch = seq_lens.shape[0]
    rows = batch * max_query_len
    topk = topk_idx.shape[-1]
    if not index_select_decode_supported(score, topk):
        raise ValueError(
            "[FlyDSL index_select_decode] needs gfx1250, topk in [1, "
            f"{WAVE}] and an fp32 score plane from index_topk_decode_score_plane"
        )
    if (
        seq_lens.dtype != torch.int32
        or seq_lens.dim() != 1
        or (batch > 1 and seq_lens.stride(0) != 1)
        or topk_idx.dtype != torch.int32
        or topk_idx.dim() != 3
        or topk_idx.shape[0] != heads
        or topk_idx.shape[1] < rows
        or (topk > 1 and topk_idx.stride(2) != 1)
        or rows_cap < rows
        or not 1 <= max_query_len < 128
    ):
        raise ValueError(
            "seq_lens must be contiguous int32 [batch], topk_idx int32 "
            f"[{heads}, >= {rows}, topk] with unit inner stride, the score plane "
            f">= {rows} rows, 1 <= max_query_len < 128"
        )
    if emit and (
        block_table is None
        or sparse_ctx is None
        or block_table.dtype != torch.int32
        or block_table.dim() != 2
        or block_table.shape[0] < batch
        or (block_table.shape[1] > 1 and block_table.stride(1) != 1)
        or sparse_bt.dtype != torch.int32
        or sparse_bt.dim() != 2
        or sparse_bt.shape[0] < rows * heads
        or sparse_bt.shape[1] != topk * pages_per_block
        or sparse_bt.stride(1) != 1
        or sparse_ctx.dtype != torch.int32
        or sparse_ctx.numel() < rows * heads
        or not sparse_ctx.is_contiguous()
    ):
        raise ValueError(
            "the emit needs block_table int32 [>= batch, blocks], sparse_bt "
            f"int32 [>= {rows * heads}, {topk * pages_per_block}] and sparse_ctx "
            f"int32 [>= {rows * heads}], unit inner strides"
        )
    strides = (
        score.stride(0),
        score.stride(1),
        block_table.stride(0) if emit else 0,
        topk_idx.stride(0),
        topk_idx.stride(1),
        sparse_bt.stride(0) if emit else 0,
    )
    if max(strides) >= 2**31:
        raise ValueError("strides must fit int32")
    if rows == 0:
        return
    compiled = compile_index_select_decode(
        heads=heads,
        query_len=max_query_len,
        topk=topk,
        columns=plane_columns(max_block),
        emit=emit,
        pages_per_block=pages_per_block if emit else 1,
        block_page_stride=block_page_stride if emit else 1,
    )
    dev = score.device
    _run_compiled(
        compiled["launch"],
        ptr_arg(score, fx.Float32),
        ptr_arg(seq_lens, fx.Int32),
        ptr_arg(block_table if emit else seq_lens, fx.Int32),
        ptr_arg(topk_idx, fx.Int32),
        ptr_arg(sparse_bt if emit else topk_idx, fx.Int32),
        ptr_arg(sparse_ctx if emit else topk_idx, fx.Int32),
        *strides,
        max_block,
        init_blocks,
        local_blocks,
        rows,
        torch.cuda.current_stream(dev),
        specialization_key=(dev.index,),
    )


def index_topk_decode_nvfp4_supported(
    idx_q, payload, scale, score, max_query_len: int, topk: int
) -> bool:
    """Whether :func:`index_topk_decode_nvfp4` serves the call: what both its
    launches serve (``score`` from :func:`index_topk_decode_score_plane`)."""
    return index_score_decode_nvfp4_supported(
        idx_q, payload, scale, max_query_len
    ) and index_select_decode_supported(score, topk)


def index_topk_decode_nvfp4(
    idx_q: torch.Tensor,
    payload: torch.Tensor,
    scale: torch.Tensor,
    score: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float,
    max_query_len: int,
    topk_idx: torch.Tensor,
    init_blocks: int,
    local_blocks: int,
    *,
    sparse_bt: torch.Tensor | None = None,
    sparse_ctx: torch.Tensor | None = None,
    pages_per_block: int = 1,
    block_page_stride: int = 1,
    workgroups: int | None = None,
    ring: int = FLAT_RING,
) -> None:
    """The decode indexer: :func:`index_score_decode_nvfp4` on its flat plan
    into ``score`` (a plane from :func:`index_topk_decode_score_plane`, made
    for the call: its live blocks are overwritten), then
    :func:`index_select_decode`; arguments as there. Graph-safe."""
    index_score_decode_nvfp4(
        idx_q,
        payload,
        scale,
        score,
        block_table,
        seq_lens,
        sm_scale,
        max_query_len,
        workgroups=workgroups,
        ring=ring,
        flat=True,
    )
    index_select_decode(
        score,
        seq_lens,
        topk_idx,
        max_query_len,
        init_blocks,
        local_blocks,
        block_table=block_table,
        sparse_bt=sparse_bt,
        sparse_ctx=sparse_ctx,
        pages_per_block=pages_per_block,
        block_page_stride=block_page_stride,
    )
