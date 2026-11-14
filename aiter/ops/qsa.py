# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""QSA top-level dispatch: selects the best backend (triton, FlyDSL, etc.)
for each QSA operator based on the input shape and available backends.

Consumers should import from here rather than from the backend-specific
modules directly.
"""

from collections.abc import Sequence

import torch

from aiter.ops.topk import top_k_per_row_prefill
from aiter.ops.triton.attention.qsa import (
    FP4_GROUP,
    FP8_DTYPE,
    FP8_MAX,
    QSA_CACHE_FORMATS,
    qsa_cache_row_layout,
    qsa_expand_block_indices,
    qsa_sparse_paged_gqa,
    qsa_store_quantized_rows,
)
from aiter.ops.triton.attention.qsa import (
    qsa_paged_mqa_logits as _triton_qsa_paged_mqa_logits,
)

try:
    from aiter.ops.flydsl.kernels.mqa_logits.qsa_flydsl_layout import (
        PAGE_SIZE as _FLYDSL_PAGE_SIZE,
    )
    from aiter.ops.flydsl.kernels.mqa_logits.qsa_flydsl_layout import (
        qsa_paged_mqa_logits_flydsl,
    )

    _HAS_FLYDSL = True
except ImportError:
    _HAS_FLYDSL = False

try:
    from aiter.ops.flydsl.qsa_sparse_prefill import flydsl_qsa_sparse_prefill

    _HAS_FLYDSL_PREFILL = True
except ImportError:
    _HAS_FLYDSL_PREFILL = False

DEFAULT_LOGITS_WORKSPACE_BYTES = 128 * 1024 * 1024


def qsa_paged_mqa_logits(
    q: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    compress_ratio: int = 4,
    score_divisor: float | None = None,
    max_columns: int | None = None,
    row_starts: torch.Tensor | None = None,
    allow_flydsl: bool = False,
    compressed_k_scale: torch.Tensor | None = None,
    stacked_decode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score compressed key groups.

    Dispatches to the FlyDSL MFMA scorer for large prefill batches when
    ``allow_flydsl`` is set and FlyDSL is available, otherwise falls through
    to the triton wrapper.

    FP8 / MXFP4 keys (with ``compressed_k_scale``) and ``stacked_decode`` are
    triton only.
    """
    if (
        allow_flydsl
        and _HAS_FLYDSL
        and compressed_k_scale is None
        and q.shape[0] > 256
        and q.stride(2) == 1
        and compressed_k_cache.stride(3) == 1
        and page_table.stride(1) == 1
        and q.shape[2] >= 16
        and compressed_k_cache.shape[1] == _FLYDSL_PAGE_SIZE
    ):
        divisor = q.shape[2] ** 0.5 if score_divisor is None else score_divisor
        columns = page_table.shape[1] * compressed_k_cache.shape[1]
        if max_columns is not None:
            columns = max(1, min(columns, int(max_columns)))
        logits = torch.empty(
            (q.shape[0], columns), dtype=torch.float32, device=q.device
        )
        visible_groups = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
        qsa_paged_mqa_logits_flydsl(
            q,
            compressed_k_cache,
            page_table,
            token_to_request,
            query_positions,
            context_lens,
            compress_ratio,
            float(divisor),
            logits,
            visible_groups,
            row_starts,
        )
        return logits, visible_groups
    return _triton_qsa_paged_mqa_logits(
        q,
        compressed_k_cache,
        page_table,
        token_to_request,
        query_positions,
        context_lens,
        compress_ratio,
        score_divisor,
        max_columns,
        row_starts,
        compressed_k_scale=compressed_k_scale,
        stacked_decode=stacked_decode,
    )


def _selection_widths(
    compressed_k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    max_seq_len: int | None,
) -> tuple[int, int, int]:
    """(scored columns, selected groups per row, emitted token width).

    Width is `budget + ratio - 1` unless `max_seq_len` -- the longest
    sequence in the batch -- shows the context cannot fill it, in which case
    both the scored width and the emitted width shrink to what the context can
    actually reach. Everything dropped would have been `-1` padding, so the
    result is identical column for column.
    """
    if compress_ratio <= 0 or token_topk <= 0 or token_topk % compress_ratio:
        raise ValueError("token_topk must be positive and divisible by compress_ratio")
    columns = page_table.shape[1] * compressed_k_cache.shape[1]
    if columns < 1:
        raise ValueError("compressed paged-cache capacity must be positive")
    block_topk = min(token_topk // compress_ratio, columns)
    if max_seq_len is not None:
        reachable = (int(max_seq_len) + compress_ratio - 1) // compress_ratio
        # Round up to a power of two: `BLOCK_TOPK`, `OUTPUT_WIDTH` and `TOPK`
        # are Triton constexprs, so an exact width would recompile the expand
        # and GQA kernels every few decoded tokens. A top-k wider than the
        # visible groups just returns -1 for the extras.
        bucket = 1 << (max(reachable, 1) - 1).bit_length()
        columns = max(1, min(columns, bucket))
        block_topk = max(1, min(block_topk, bucket))
    return columns, block_topk, block_topk * compress_ratio + compress_ratio - 1


def qsa_select_paged_groups(
    q: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    token_topk: int,
    compress_ratio: int = 4,
    logits_workspace_bytes: int = DEFAULT_LOGITS_WORKSPACE_BYTES,
    max_seq_len: int | None = None,
    allow_flydsl: bool = False,
    compressed_k_scale: torch.Tensor | None = None,
    stacked_decode: bool = False,
) -> tuple[torch.Tensor, int]:
    """Score and select: `[tokens, block_topk]` int32 group ids, and the token
    width `qsa_expand_block_indices` must emit for them.

    Every row depends only on its own query, request and position, so callers
    may select a row range (for example one tensor-parallel rank's shard) and
    assemble the rows elsewhere; the widths depend only on the batch-wide
    `page_table` and `max_seq_len`.
    """
    columns, block_topk, output_width = _selection_widths(
        compressed_k_cache, page_table, token_topk, compress_ratio, max_seq_len
    )
    rows = q.shape[0]
    groups = torch.empty((rows, block_topk), dtype=torch.int32, device=q.device)
    # Chunk over query rows so the FP32 logits buffer stays within budget.
    rows_per_chunk = max(1, logits_workspace_bytes // max(columns * 4, 1))
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        row_slice = slice(row_start, row_end)
        row_starts = torch.empty(
            row_end - row_start, dtype=torch.int32, device=q.device
        )
        logits, visible_groups = qsa_paged_mqa_logits(
            q[row_slice],
            compressed_k_cache,
            page_table,
            token_to_request[row_slice],
            query_positions[row_slice],
            context_lens,
            compress_ratio,
            max_columns=columns,
            row_starts=row_starts,
            allow_flydsl=allow_flydsl,
            compressed_k_scale=compressed_k_scale,
            stacked_decode=stacked_decode,
        )
        top_k_per_row_prefill(
            logits,
            row_starts,
            visible_groups,
            groups[row_slice],
            None,
            logits.shape[0],
            logits.stride(0),
            logits.stride(1),
            block_topk,
            # Equal index scores must choose the same keys and ordering in
            # single-token decode and multi-token verification.
            stable=True,
        )
    return groups, output_width


def qsa_select_paged_tokens(
    q: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    token_topk: int,
    compress_ratio: int = 4,
    out: torch.Tensor | None = None,
    logits_workspace_bytes: int = DEFAULT_LOGITS_WORKSPACE_BYTES,
    max_seq_len: int | None = None,
    allow_flydsl: bool = False,
    compressed_k_scale: torch.Tensor | None = None,
    stacked_decode: bool = False,
) -> torch.Tensor:
    """Score, select and expand in one call: `[tokens, width]` token ids
    (see `_selection_widths` for the width)."""
    rows = q.shape[0]
    _, _, output_width = _selection_widths(
        compressed_k_cache, page_table, token_topk, compress_ratio, max_seq_len
    )
    if out is None:
        out = torch.empty((rows, output_width), dtype=torch.int32, device=q.device)
    elif out.shape[0] != rows or out.shape[1] < output_width:
        raise ValueError("out has an invalid shape")
    else:
        out = out[:, :output_width]
    if rows == 0:
        return out
    groups, _ = qsa_select_paged_groups(
        q,
        compressed_k_cache,
        page_table,
        token_to_request,
        query_positions,
        context_lens,
        token_topk,
        compress_ratio,
        logits_workspace_bytes=logits_workspace_bytes,
        max_seq_len=max_seq_len,
        allow_flydsl=allow_flydsl,
        compressed_k_scale=compressed_k_scale,
        stacked_decode=stacked_decode,
    )
    return qsa_expand_block_indices(
        groups,
        query_positions,
        context_lens,
        token_to_request,
        compress_ratio,
        token_topk,
        out,
        output_width=output_width,
    )


def qsa_sparse_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    *,
    query_lens: Sequence[int] | None = None,
    prefix_lens: Sequence[int] | None = None,
    softmax_scale: float | None = None,
    out: torch.Tensor | None = None,
    kv_slots: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sparse prefill/extend attention over packed K/V, or a token pool read through
    kv_slots; FlyDSL on gfx942 only.

    See ``aiter.ops.flydsl.qsa_sparse_prefill.flydsl_qsa_sparse_prefill``.
    """
    if not _HAS_FLYDSL_PREFILL:
        raise RuntimeError("QSA sparse prefill requires FlyDSL")
    return flydsl_qsa_sparse_prefill(
        q,
        k,
        v,
        indices,
        query_lens=query_lens,
        prefix_lens=prefix_lens,
        softmax_scale=softmax_scale,
        out=out,
        kv_slots=kv_slots,
    )


__all__ = [
    "DEFAULT_LOGITS_WORKSPACE_BYTES",
    "FP4_GROUP",
    "FP8_DTYPE",
    "FP8_MAX",
    "QSA_CACHE_FORMATS",
    "qsa_cache_row_layout",
    "qsa_expand_block_indices",
    "qsa_paged_mqa_logits",
    "qsa_select_paged_groups",
    "qsa_select_paged_tokens",
    "qsa_sparse_paged_gqa",
    "qsa_sparse_prefill",
    "qsa_store_quantized_rows",
]
