# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL QSA public surface (SILOTIGER-1047).

``qsa_k1_block_ids`` writes indexer ``block_ids [M, 512]`` from paged
compressed K. Rows up to 512 blocks use the fused emit kernel; longer rows
use tiled BLOCK_N=32 BF16 MFMA scoring plus the stable decode radix below
32768 columns and streaming radix at or above that width. Single-request
prefill scores 16 query rows per workgroup. The indexer head count is 4 or
8, each a separate compile.

``qsa_k2`` writes sparse GQA ``o [M, Hq, D]`` from paged K/V at the selected
token ids. Decode runs a BLOCK_N=16 two-wave tile with split-K and an LSE
merge; prefill runs BLOCK_N=32 two waves and writes output directly once the
grid alone fills the machine. Softmax is online in log2 space and the split
partials are FP32. Expand+tail and the sigmoid gate stay unfused.

``qsa_oracle`` is the fp32 reference. The concrete shapes all of these are
validated against are test fixtures in ``op_tests/qsa_shapes.py``.

``qsa_layer`` is the ``qwen4_exp`` opt-in. It runs K1, the vendored
expand+tail, and K2. A rank-3 query pair missing from the measured
table logs once. A query that is not rank 3 is not a table miss; K1
or K2 reports it. Sigmoid and partial RoPE stay outside the layer.
"""

import functools

import torch

from .kernels.qsa import (
    QsaGqaSpec,
    QsaIndexerSpec,
    QsaOracleResult,
    gather_paged_cache,
    gather_qsa_caches,
    pack_paged_cache,
    qsa_expand_tail,
    qsa_indexer_scores,
    qsa_oracle,
    qsa_sparse_gqa,
    qsa_topk_blocks,
    qsa_visible_blocks,
)
from .kernels.qsa.k1 import qsa_k1_block_ids, qsa_k1_selection_serves
from .kernels.qsa.k2 import qsa_k2

# GQA query ``(n_q_heads, head_dim)`` -> the indexer head counts swept with
# it. An unswept rank-3 pair still runs; the table only feeds the warning.
# The measured thing is the pair.
_MEASURED_QUERIES = {
    (24, 256): (4, 8),  # Flash-Next / qwen4_exp, TP1
    (12, 256): (4, 8),  # Flash-Next TP2: 12 query heads, 1 KV head
    (6, 256): (4, 8),  # Flash-Next TP4: 6 query heads, 1 KV head
    (3, 256): (4, 8),  # Flash-Next TP8: 3 query heads, 1 KV head
    (10, 128): (4, 8),
}

# GPU 6 / gfx950, cold ``--rotate 0``, page_size 16, M in {1, 2, 3, 4, 6, 8,
# 12, 16, 32, 64, 128, 256, 512} crossed with L in {512, 2048, 8192, 32768}.
# All 104 rows won with err=0: 24x256 by 1.03x to 1.67x, 10x128 by
# 1.17x to 2.66x, and 10x128 also won at both indexer widths. The same grid
# with an 8-head indexer on 24x256 also won all 52 rows, err=0, by 1.08x to
# 1.70x. Indexer width does not change the K2 ladder. That reaches
# every launch config the K2 policy can pick, and M past 512 reuses M=512's
# BN32 single-split config with a larger grid, so neither M nor the
# selection width filters this gate. The ``_launch_config`` decode bands are
# now fitted at both head widths. A per-rank shard with a different KV-head
# count is still its own grid, because the K2 launch is
# ``(rows, kv_heads, splits)``. GPU 7 / gfx942, cold ``rotate=0``, page size
# 16, the same M x L grid: q ``[M, 12, 256]`` over 1 KV head and the
# replicated 4-head indexer. All 52 rows won with err=0, by 1.73x
# to 6.06x. The same grid with an 8-head indexer also won all 52 rows,
# FlyDSL err=0, by 1.46x to 4.86x. GPU 6 / gfx950, the same cold grid:
# q ``[M, 6, 256]`` over 1 KV head (Flash-Next TP4, group 6). The 4-head
# indexer won all 52 rows with err=0, by 1.28x to 3.29x. The 8-head indexer
# also won all 52, FlyDSL err=0, by 1.38x to 3.28x. The same gfx950 grid
# with q ``[M, 3, 256]`` over 1 KV head (Flash-Next TP8, group 3) won all
# 52 rows at both indexer widths, FlyDSL err=0: 4-head by 1.21x to 3.29x,
# 8-head by 1.27x to 3.34x.

__all__ = [
    "QsaGqaSpec",
    "QsaIndexerSpec",
    "QsaOracleResult",
    "gather_paged_cache",
    "gather_qsa_caches",
    "pack_paged_cache",
    "qsa_expand_tail",
    "qsa_indexer_scores",
    "qsa_k1_block_ids",
    "qsa_k2",
    "qsa_layer",
    "qsa_oracle",
    "qsa_sparse_gqa",
    "qsa_topk_blocks",
    "qsa_visible_blocks",
]


def _rank3_query_pair(
    q_indexer: torch.Tensor, q_gqa: torch.Tensor
) -> tuple[tuple[int, int], int] | None:
    """``((Hq, D), indexer heads)`` when both queries are rank 3.

    A scalar or any other rank is not a table key. K1 and K2 report that.
    """
    if q_indexer.dim() != 3 or q_gqa.dim() != 3:
        return None
    gqa_key = (int(q_gqa.shape[1]), int(q_gqa.shape[2]))
    return gqa_key, int(q_indexer.shape[1])


def _warn_unmeasured_query(q_indexer: torch.Tensor, q_gqa: torch.Tensor) -> None:
    """Log an unswept rank-3 pair. A bad rank stays quiet."""
    pair = _rank3_query_pair(q_indexer, q_gqa)
    if pair is None:
        return
    gqa_key, index_heads = pair
    allowed = _MEASURED_QUERIES.get(gqa_key)
    if allowed is None or index_heads not in allowed:
        _log_unmeasured_gqa_query(gqa_key, index_heads)


@functools.cache
def _log_unmeasured_gqa_query(gqa_key: tuple[int, int], index_heads: int) -> None:
    """Log one table miss. The same pair does not log again."""
    from aiter import logger

    logger.warning(
        "QSA: query pair (GQA %s, indexer heads %s) is not in the measured table",
        gqa_key,
        index_heads,
    )


def _expand_block_ids(
    block_ids: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    token_to_req: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
    indices: torch.Tensor | None,
) -> torch.Tensor:
    # Phase 5 owns fusing expand into K1 or K2. Until then this is the
    # vendored expand+tail kernel.
    from aiter.ops.triton.attention.qsa_vllm_amd import expand_qsa_block_indices_cuda

    return expand_qsa_block_indices_cuda(
        block_ids,
        query_positions,
        context_lens,
        token_to_req,
        compress_ratio,
        token_topk,
        out=indices,
    )


def qsa_layer(
    q_indexer: torch.Tensor,
    index_k_cache: torch.Tensor,
    index_page_table: torch.Tensor,
    q_gqa: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    indices: torch.Tensor | None = None,
    block_ids: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    token_topk: int = 2048,
    compress_ratio: int = 4,
    score_scale: float | None = None,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Run one QSA layer: indexer select, expand+tail, sparse GQA.

    Returns the bf16 GQA tensor ``[M, Hq, D]``. A caller-provided ``out``
    is that tensor.

    The layer is K1, the vendored expand, and K2. It raises on a token
    budget other than 512 blocks at compress ratio 4. A rank-3 pair
    missing from the measured table logs once and still runs. A query
    that is not rank 3 does not log; K1 or K2 reports it. Indexer scores
    are multiplied by ``score_scale`` and QK by ``softmax_scale``.
    ``None`` is ``head_dim**-0.5`` for that tensor.

    Args:
        q_indexer: bf16 ``[M, H, 128]``. FlyDSL accepts ``H`` of 4 or 8.
        index_k_cache: bf16 ``[pages, page_size, 1, 128]``. The D stride
            is 1. The page, token, and head strides are multiples of 8
            elements. The cache is read in place.
        index_page_table: int32 ``[n_req, n_pages]``.
        q_gqa: bf16 ``[M, Hq, D]``, on the same GPU as the other tensors.
        k_cache, v_cache: bf16 ``[pages, page_size, Hkv, D]``, matching
            each other in shape and stride, with a unit D stride and the
            same 8-element stride rule. ``Hq`` divides across ``Hkv`` in
            a group of at most 16. ``D`` is a multiple of 32 and fills
            the K2 gather, K LDS map, and PV tile.
        kv_page_table: int32 ``[n_req, n_pages]``.
        token_to_req, query_positions: int32 ``[M]``.
        context_lens: 1-D int32, with no more rows than ``index_page_table``.
        indices: optional int32 ``[M, token_topk + compress_ratio - 1]``
            expand buffer. ``None`` allocates it.
        block_ids: optional contiguous int32 ``[M, 512]`` K1 buffer. The
            layer writes it.
        out: optional contiguous bf16 ``[M, Hq, D]`` matching ``q_gqa``.
            ``None`` allocates it.
        token_topk: selected tokens. The default is 2048, which is the
            FlyDSL budget of 512 blocks.
        compress_ratio: block size. The default, and the FlyDSL value, is 4.
        score_scale, softmax_scale: see above.

    A span past 4 GiB on a cache selects the wide gather compile. An empty
    query, cache, or page table writes ``-1`` block ids and a zero ``out``
    and does not launch. K2 allocates its own split partials on a
    multi-split launch. Sigmoid and partial RoPE stay outside this call.

    Returns:
        Sparse GQA output, bf16 ``[M, Hq, D]``.
    """
    reason = qsa_k1_selection_serves(token_topk, compress_ratio)
    if reason is not None:
        raise ValueError(reason)
    _warn_unmeasured_query(q_indexer, q_gqa)
    # A bad rank is not a table key and has no head or D to read here.
    extra = {}
    if q_indexer.dim() == 3:
        if score_scale is None:
            score_scale = float(q_indexer.shape[2]) ** -0.5
        extra["heads"] = (int(q_indexer.shape[1]),)
    if score_scale is not None:
        extra["score_scale"] = score_scale
    block_ids = qsa_k1_block_ids(
        q_indexer,
        index_k_cache,
        index_page_table,
        token_to_req,
        query_positions,
        context_lens,
        out=block_ids,
        **extra,
    )
    indices = _expand_block_ids(
        block_ids,
        query_positions,
        context_lens,
        token_to_req,
        compress_ratio,
        token_topk,
        indices,
    )
    return qsa_k2(
        q_gqa,
        k_cache,
        v_cache,
        indices,
        kv_page_table,
        token_to_req,
        out=out,
        softmax_scale=softmax_scale,
    )
