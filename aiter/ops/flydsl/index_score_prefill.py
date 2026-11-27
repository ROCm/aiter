# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Causal prefill block scores of index queries over an NVFP4 key cache (gfx1250),
and the same scorer fused with the top-16 block selection.

See ``kernels.index_score_prefill_nvfp4_gfx1250``.
"""

import flydsl.expr as fx
import torch

from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

from .kernels.index_score_prefill_nvfp4_gfx1250 import (
    CHUNK_MAX,
    INT32_MIN,
    TOPK,
    compile_index_score_prefill_nvfp4,
    sm_scale_log2e,
)
from .kernels.minimax_m3_index_score import HEAD_DIM, PAGE
from .kernels.tensor_shim import _run_compiled, ptr_arg


def _byte_pages(t, row_bytes):
    """``[pages, PAGE, row_bytes]`` bytes, each page contiguous and 16 B aligned:
    a page is one tensor DMA of ``PAGE * row_bytes`` bytes."""
    return (
        t.element_size() == 1
        and t.shape[1:] == (PAGE, row_bytes)
        and t.stride()[1:] == (row_bytes, 1)
        and t.stride(0) % 16 == 0
        and t.data_ptr() % 16 == 0
    )


def _inputs_supported(idx_q, payload, scale, meta, sm_scale) -> bool:
    """What the scorer and the fused op both assume of the query, the cache and
    the batch metadata ``(block_table, cu_seqlens_q, seq_lens, prefix_lens)``."""
    block_table, cu_seqlens_q, seq_lens, prefix_lens = meta
    batch = cu_seqlens_q.shape[0] - 1
    return (
        get_gfx_runtime() == "gfx1250"
        and idx_q.dtype == torch.bfloat16
        and idx_q.shape[2:] == (HEAD_DIM,)
        and idx_q.stride(2) == 1
        and _byte_pages(payload, HEAD_DIM // 2)
        and _byte_pages(scale, HEAD_DIM // 16)
        and all(t.device == idx_q.device for t in (payload, scale) + meta)
        and all(t.dtype == torch.int32 and t.stride(-1) == 1 for t in meta)
        and min(block_table.shape[0], seq_lens.shape[0], prefix_lens.shape[0]) >= batch
        # The kernel takes the block max before the scale.
        and 0.0 < sm_scale_log2e(sm_scale) < float("inf")
    )


def index_score_prefill_nvfp4_supported(
    idx_q,
    payload,
    scale,
    score,
    block_table,
    cu_seqlens_q,
    seq_lens,
    prefix_lens,
    sm_scale,
) -> bool:
    """Whether ``index_score_prefill_nvfp4`` takes these, its first nine arguments.
    The kernel gets bare pointers and int32 strides, so these are what it assumes
    of them; the op raises ``ValueError`` when this is False."""
    meta = (block_table, cu_seqlens_q, seq_lens, prefix_lens)
    return (
        _inputs_supported(idx_q, payload, scale, meta, sm_scale)
        and score.device == idx_q.device
        and score.dtype == torch.float32
        and score.dim() == 3
        and score.shape[0] == idx_q.shape[1]
        and score.shape[1] >= idx_q.shape[0]
        and score.stride(2) == 1
        and max(score.stride()) < 2**31
    )


def index_score_prefill_nvfp4(
    idx_q: torch.Tensor,  # [total_q, heads, 128] bf16
    payload: torch.Tensor,  # [pages, 128, 64] uint8: e2m1 pairs
    scale: torch.Tensor,  # [pages, 128, 8] uint8 / float8_e4m3fn: one per 16 values
    score: torch.Tensor,  # [heads, >= total_q, max_block] fp32
    block_table: torch.Tensor,  # [reqs, max_blocks] int32
    cu_seqlens_q: torch.Tensor,  # [reqs + 1] int32
    seq_lens: torch.Tensor,  # [reqs] int32
    prefix_lens: torch.Tensor,  # [reqs] int32
    sm_scale: float,
    max_query_len: int,
    chunk_blocks: int,
    *,
    block_q: int = 256,
    waves: int = 4,
    ring: int = 4,
    fold_max: bool = False,
    pad_lds: bool = False,
) -> None:
    """For query row ``r`` of request ``b`` (at position ``prefix_lens[b] + r -
    cu_seqlens_q[b]``), index head ``h`` and 128-key block ``blk`` of ``b``'s pages
    (page ``block_table[b, blk]``)::

        score[h, r, blk] = max over the block's keys k at or before r's position of
                           dot(e4m3(clamp(idx_q[r, h], +-448)), dequant(K[k]))
                           * (sm_scale * log2(e))

    ``-inf`` for a block past the row's own. Written for the blocks up to that of
    the last row of r's ``block_q``-row tile; later entries are left as they are.
    ``chunk_blocks`` (blocks a CTA scores) is clamped to ``[1, CHUNK_MAX]``; the
    scores do not depend on it, nor on the tile, ``fold_max`` or ``pad_lds``
    (see ``compile_index_score_prefill_nvfp4``).

    What only the device can see is the caller's to guarantee: ``cu_seqlens_q``
    ends at ``total_q``, ``max_query_len`` covers every request, and
    ``block_table`` and ``score`` hold ``ceil(seq_len / 128)`` blocks of each
    request (pages inside the cache)."""
    meta = (block_table, cu_seqlens_q, seq_lens, prefix_lens)
    if not index_score_prefill_nvfp4_supported(
        idx_q, payload, scale, score, *meta, sm_scale
    ):
        raise ValueError(
            "index_score_prefill_nvfp4: unsupported device, dtype, layout or scale"
        )
    total_q, heads, _ = idx_q.shape
    batch = cu_seqlens_q.shape[0] - 1
    max_block = score.shape[2]
    if total_q == 0 or batch == 0:
        return
    dev = idx_q.device
    chunk_blocks = max(1, min(chunk_blocks, CHUNK_MAX))
    q_tiles = (max_query_len + block_q - 1) // block_q
    chunks = (max_block + chunk_blocks - 1) // chunk_blocks
    compiled = compile_index_score_prefill_nvfp4(
        block_q=block_q,
        waves=waves,
        ring=ring,
        fold_max=bool(fold_max),
        pad_lds=bool(pad_lds),
    )
    _run_compiled(
        compiled["launch"],
        ptr_arg(idx_q, fx.BFloat16),
        ptr_arg(payload, fx.Uint8),
        ptr_arg(scale, fx.Uint8),
        ptr_arg(score, fx.Float32),
        ptr_arg(block_table, fx.Int32),
        ptr_arg(cu_seqlens_q, fx.Int32),
        ptr_arg(seq_lens, fx.Int32),
        ptr_arg(prefix_lens, fx.Int32),
        heads,
        sm_scale_log2e(sm_scale),
        chunk_blocks,
        idx_q.stride(0),
        idx_q.stride(1),
        payload.stride(0),
        scale.stride(0),
        score.stride(0),
        score.stride(1),
        block_table.stride(0),
        q_tiles,
        batch,
        chunks,
        torch.cuda.current_stream(dev),
        specialization_key=(dev.index,),
    )


# Blocks a workgroup of the fused scorer + top-16 owns: a row hands on ``TOPK``
# keys a chunk. A longer chunk loads its query tile and fills its page ring
# fewer times, tries fewer blocks (a list restarts with its chunk) and leaves
# the merge fewer keys; a shorter one gives a short prefill more workgroups.
# The default is the longest power of two in [TOPK_CHUNK_MIN, TOPK_CHUNK_MAX]
# that still gives each CU TOPK_WORKGROUPS_PER_CU workgroups, and never so
# short that a row hands the merge more than TOPK_MERGE_MAX keys. On MI455X the
# derived chunk is level with 512 / 1024 at 384K (within run noise).
TOPK_CHUNK_MIN, TOPK_CHUNK_MAX = 32, 1024
TOPK_WORKGROUPS_PER_CU = 16
TOPK_MERGE_MAX = 1024


def index_topk_prefill_chunk_blocks(
    idx_q, max_query_len, batch, max_block, *, block_q=256, **_launch
) -> int:
    """The default ``chunk_blocks`` of :func:`index_topk_prefill_nvfp4` for this
    grid (the other launch keywords are accepted and do not matter). A power of
    two, so a caller compiling per chunk sees few values."""
    tiles = batch * idx_q.shape[1] * -(-max_query_len // block_q)
    floor = TOPK_CHUNK_MIN
    while floor < TOPK_CHUNK_MAX and -(-max_block // floor) * TOPK > TOPK_MERGE_MAX:
        floor *= 2
    chunk = TOPK_CHUNK_MAX
    while (
        chunk > floor
        and tiles * -(-max_block // chunk) < TOPK_WORKGROUPS_PER_CU * get_cu_num()
    ):
        chunk //= 2
    return chunk


# Where the fused op is the faster indexer (MI455X, 4 index heads, the fused op
# against the scorer at block_q=128, waves=2, ring=2, fold_max, pad_lds + ATOM's
# Triton selector, us):
#     32.7K rows on 384K / 600K / 300K context  2897 / 4362 / 2537 vs 3370 / 5994 / 2889
#     32.7K rows on 150K                        1310 vs 1322
#     32.7K rows on 50K                          587 vs  408
#     2K rows on 384K, 8K rows on 600K           354 vs  269, 1364 vs 1426
# A short context or few rows leave the fused op's per-chunk selection passes
# (its lists restart a chunk) a large share; there the two launches win.
TOPK_PREFERRED_MIN_ROWS = 8192
TOPK_PREFERRED_MIN_BLOCKS = 1024


def index_topk_prefill_nvfp4_preferred(idx_q, max_block) -> bool:
    """Whether the fused op is the faster prefill indexer for this call (rows of
    ``idx_q``, ``max_block`` 128-token blocks); a performance choice, not a
    limit: :func:`index_topk_prefill_nvfp4_supported` says what it serves."""
    return (
        idx_q.shape[0] >= TOPK_PREFERRED_MIN_ROWS
        and max_block >= TOPK_PREFERRED_MIN_BLOCKS
    )


def index_topk_prefill_nvfp4_supported(
    idx_q,
    payload,
    scale,
    block_table,
    cu_seqlens_q,
    seq_lens,
    prefix_lens,
    sm_scale,
    topk,
    max_block,
) -> bool:
    """Whether :func:`index_topk_prefill_nvfp4` serves this call: what the scorer
    assumes of its inputs, 16 blocks a row and block ids that fit the key's 16
    bits."""
    meta = (block_table, cu_seqlens_q, seq_lens, prefix_lens)
    return (
        _inputs_supported(idx_q, payload, scale, meta, sm_scale)
        and topk == TOPK
        and max_block <= 0xFFFF
    )


def index_topk_prefill_nvfp4(
    idx_q: torch.Tensor,  # [total_q, heads, 128] bf16
    payload: torch.Tensor,  # [pages, 128, 64] uint8: e2m1 pairs
    scale: torch.Tensor,  # [pages, 128, 8] uint8 / float8_e4m3fn: one per 16 values
    block_table: torch.Tensor,  # [reqs, max_blocks] int32
    cu_seqlens_q: torch.Tensor,  # [reqs + 1] int32
    seq_lens: torch.Tensor,  # [reqs] int32
    prefix_lens: torch.Tensor,  # [reqs] int32
    sm_scale: float,
    max_query_len: int,
    max_block: int,
    init_blocks: int,
    local_blocks: int,
    *,
    chunk_blocks: int | None = None,
    block_q: int = 256,
    waves: int = 4,
    ring: int = 3,
    fold_max: bool = True,
    pad_lds: bool = True,
    sel32: bool = True,
) -> torch.Tensor:
    """The prefill scorer and ATOM's top-16 block selection in one launch: the
    scores of :func:`index_score_prefill_nvfp4` never reach memory.

    Returns ``[heads, total_q, chunks * 16]`` int64 (not contiguous), chunks =
    ceil(max_block / chunk_blocks): at ``z * 16`` the 16 best blocks of row r
    among the blocks ``[z, z + 1) * chunk_blocks`` it may attend, as the keys of
    ATOM's ``_pack_score_key`` (score, then the larger block; a NaN last; the
    first ``init_blocks`` and the row's last ``local_blocks`` blocks pinned),
    in no order, 0 for no block. The 16 largest keys of a row are its selection,
    in order: every one of its 16 best blocks is in its chunk's 16, though a
    chunk may hold fewer than its own 16 best (blocks another chunk's rule out).
    Chunks from ``ceil(blocks of r / chunk_blocks)`` on hold no key or are not
    written; rows past a request's ``cu_seqlens_q`` are not written.

    ``chunk_blocks`` defaults to :func:`index_topk_prefill_chunk_blocks`; it,
    the tile, ``fold_max``, ``pad_lds`` and ``sel32`` change no selection (see
    ``compile_index_score_prefill_nvfp4``). The defaults are the fastest build
    measured on MI455X. Raises ``ValueError`` where
    :func:`index_topk_prefill_nvfp4_supported` is False."""
    meta = (block_table, cu_seqlens_q, seq_lens, prefix_lens)
    if not index_topk_prefill_nvfp4_supported(
        idx_q, payload, scale, *meta, sm_scale, TOPK, max_block
    ):
        raise ValueError(
            "index_topk_prefill_nvfp4: unsupported device, dtype, layout, scale or "
            "block count"
        )
    total_q, heads, _ = idx_q.shape
    batch = cu_seqlens_q.shape[0] - 1
    if chunk_blocks is None:
        chunk_blocks = index_topk_prefill_chunk_blocks(
            idx_q, max_query_len, batch, max_block, block_q=block_q
        )
    if chunk_blocks < 1:
        raise ValueError(f"chunk_blocks must be positive, got {chunk_blocks}")
    chunks = (max_block + chunk_blocks - 1) // chunk_blocks
    # Column 0: the row's floor, shared by its workgroups.
    keys = torch.empty(
        (heads, total_q, 1 + chunks * TOPK), dtype=torch.int64, device=idx_q.device
    )
    if total_q == 0 or batch == 0 or chunks == 0:
        return keys[:, :, 1:]
    keys[:, :, 0] = INT32_MIN
    compiled = compile_index_score_prefill_nvfp4(
        block_q=block_q,
        waves=waves,
        ring=ring,
        select=(init_blocks, local_blocks),
        fold_max=bool(fold_max),
        pad_lds=bool(pad_lds),
        sel32=bool(sel32),
    )
    dev = idx_q.device
    _run_compiled(
        compiled["launch"],
        ptr_arg(idx_q, fx.BFloat16),
        ptr_arg(payload, fx.Uint8),
        ptr_arg(scale, fx.Uint8),
        ptr_arg(keys, fx.Int64),
        ptr_arg(block_table, fx.Int32),
        ptr_arg(cu_seqlens_q, fx.Int32),
        ptr_arg(seq_lens, fx.Int32),
        ptr_arg(prefix_lens, fx.Int32),
        heads,
        sm_scale_log2e(sm_scale),
        chunk_blocks,
        idx_q.stride(0),
        idx_q.stride(1),
        payload.stride(0),
        scale.stride(0),
        keys.stride(0),
        keys.stride(1),
        block_table.stride(0),
        (max_query_len + block_q - 1) // block_q,
        batch,
        chunks,
        torch.cuda.current_stream(dev),
        specialization_key=(dev.index,),
    )
    return keys[:, :, 1:]
