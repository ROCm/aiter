# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""QSA top-level dispatch: selects the best backend (triton, FlyDSL, etc.)
for each QSA operator based on the input shape and available backends.

Consumers should import from here rather than from the backend-specific
modules directly.
"""

from collections.abc import Sequence
from typing import NamedTuple

import torch

from aiter.ops.topk import top_k_per_row_prefill
from aiter.ops.triton.attention.qsa import (
    FP4_GROUP,
    FP8_DTYPE,
    FP8_MAX,
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
    from aiter.ops.flydsl.qsa_sparse_prefill import (
        flydsl_qsa_sparse_prefill,
    )
    from aiter.ops.flydsl.qsa_sparse_prefill import (
        qsa_sparse_prefill_problem as _flydsl_qsa_sparse_prefill_problem,
    )

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
    """Score, select and expand in one call."""
    if compress_ratio <= 0 or token_topk <= 0 or token_topk % compress_ratio:
        raise ValueError("token_topk must be positive and divisible by compress_ratio")
    rows = q.shape[0]

    columns = page_table.shape[1] * compressed_k_cache.shape[1]
    if columns < 1:
        raise ValueError("compressed paged-cache capacity must be positive")
    block_topk = min(token_topk // compress_ratio, columns)
    if max_seq_len is not None:
        reachable = (int(max_seq_len) + compress_ratio - 1) // compress_ratio
        bucket = 1 << (max(reachable, 1) - 1).bit_length()
        columns = max(1, min(columns, bucket))
        block_topk = max(1, min(block_topk, bucket))
    output_width = block_topk * compress_ratio + compress_ratio - 1

    if out is None:
        out = torch.empty((rows, output_width), dtype=torch.int32, device=q.device)
    elif out.shape[0] != rows or out.shape[1] < output_width:
        raise ValueError("out has an invalid shape")
    else:
        out = out[:, :output_width]
    if rows == 0:
        return out

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
        selected_groups = torch.empty(
            (row_end - row_start, block_topk), dtype=torch.int32, device=q.device
        )
        top_k_per_row_prefill(
            logits,
            row_starts,
            visible_groups,
            selected_groups,
            None,
            logits.shape[0],
            logits.stride(0),
            logits.stride(1),
            block_topk,
            stable=True,
        )
        qsa_expand_block_indices(
            selected_groups,
            query_positions[row_slice],
            context_lens,
            token_to_request[row_slice],
            compress_ratio,
            token_topk,
            out[row_slice],
            output_width=output_width,
        )
    return out


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
    kv_slots; FlyDSL on gfx942 and gfx950.

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


class PagedKVPool(NamedTuple):
    """K and V of a paged cache as the token pools ``qsa_sparse_prefill`` reads in place
    through ``kv_slots``: ``k`` and ``v`` are [rows, heads, dim] views of the cache
    memory, and token ``o`` of page ``p`` is row ``p * rows_per_page + o``."""

    k: torch.Tensor
    v: torch.Tensor
    page_size: int
    rows_per_page: int


def qsa_paged_kv_pool(
    k_cache: torch.Tensor, v_cache: torch.Tensor
) -> PagedKVPool | None:
    """K and V of a paged cache [pages, page_size, heads, dim] as token pools, or None
    when ``qsa_sparse_prefill`` cannot read the cache that way.

    K and V share their shape and strides (they may be the two halves of one record per
    token), and strides are arbitrary otherwise: pages may sit further apart than they
    are long. The cache must hold dense head vectors, and K and V must start 16-byte
    aligned. The op reads four-token blocks at four consecutive pool rows starting at a
    multiple of four: a block lies in one page and starts at a multiple of four tokens
    when the page holds a multiple of four tokens, and its row is a multiple of four
    when pages start a multiple of four rows apart. dtype, head dimension, row stride
    and the other operand rules are those of ``qsa_sparse_prefill_problem``, asked of
    the pools.
    """
    if k_cache.ndim != 4 or k_cache.shape != v_cache.shape or 0 in k_cache.shape:
        return None
    if k_cache.stride() != v_cache.stride():
        return None
    pages, page_size, heads, dim = k_cache.shape
    page_stride, token_stride, head_stride, element_stride = k_cache.stride()
    if (
        element_stride != 1
        or (heads > 1 and head_stride != dim)
        or token_stride < heads * dim
    ):
        return None
    rows_per_page, rest = divmod(page_stride, token_stride)
    if (
        rest
        or page_size % 4
        or rows_per_page % 4
        or k_cache.data_ptr() % 16
        or v_cache.data_ptr() % 16
    ):
        return None
    shape = ((pages - 1) * rows_per_page + page_size, heads, dim)
    stride = (token_stride, dim, 1)
    return PagedKVPool(
        k_cache.as_strided(shape, stride),
        v_cache.as_strided(shape, stride),
        page_size,
        rows_per_page,
    )


def qsa_paged_kv_slots(
    query_lens: Sequence[int],
    prefix_lens: Sequence[int],
    block_table: torch.Tensor,
    page_size: int,
    rows_per_page: int,
) -> torch.Tensor:
    """``kv_slots`` for ``qsa_sparse_prefill`` over a ``qsa_paged_kv_pool``: the int32 pool
    row of every key of the requests, in packed order (request r's tokens
    0 .. prefix_r + query_r - 1, requests in order). Token t of request r is row
    ``block_table[r, t // page_size] * rows_per_page + t % page_size``.

    ``block_table`` [requests, pages] holds exactly these requests' rows, on the device
    of the op. The host lengths are copied without a synchronization and the rest is
    device work, so the table can be built per step and shared by every layer. Build it
    outside graph capture: the lengths are staged in host memory and copied when it runs.
    """
    lengths = [q + p for q, p in zip(query_lens, prefix_lens, strict=True)]
    longest = max(lengths, default=0)
    if block_table.ndim != 2 or block_table.shape[0] != len(lengths):
        raise ValueError("block_table must be [requests, pages], one row per request")
    if -(-longest // page_size) > block_table.shape[1]:
        raise ValueError("block_table is too narrow for the longest request")
    device = block_table.device
    counts = torch.tensor(
        lengths, dtype=torch.int64, device="cpu", pin_memory=device.type == "cuda"
    )
    counts = counts.to(device, non_blocking=True)
    total = sum(lengths)
    request = torch.repeat_interleave(
        torch.arange(len(lengths), device=device), counts, output_size=total
    )
    first = torch.cumsum(counts, 0) - counts
    token = torch.arange(total, device=device) - first[request]
    page = block_table[request, token // page_size].long()
    return (page * rows_per_page + token % page_size).to(torch.int32)


def qsa_sparse_prefill_problem(
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
) -> str | None:
    """The first operand rule a ``qsa_sparse_prefill`` call with these arguments breaks,
    as the message it raises, or None when the operands are accepted. States of the
    process (graph capture, one GPU per process) are not operand rules.

    See ``aiter.ops.flydsl.qsa_sparse_prefill.qsa_sparse_prefill_problem``.
    """
    if not _HAS_FLYDSL_PREFILL:
        return "QSA sparse prefill requires FlyDSL"
    return _flydsl_qsa_sparse_prefill_problem(
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


def can_use_flydsl_qsa_sparse_prefill(*args, **kwargs) -> bool:
    """Whether ``qsa_sparse_prefill_problem`` finds no rule broken (same arguments)."""
    return qsa_sparse_prefill_problem(*args, **kwargs) is None


def qsa_sparse_prefill_available() -> bool:
    """Whether the FlyDSL backend of ``qsa_sparse_prefill`` imports in this process: the
    answer a consumer asks once. The device and each call are asked of
    ``qsa_sparse_prefill_problem``."""
    return _HAS_FLYDSL_PREFILL


__all__ = [
    "DEFAULT_LOGITS_WORKSPACE_BYTES",
    "FP4_GROUP",
    "FP8_DTYPE",
    "FP8_MAX",
    "PagedKVPool",
    "can_use_flydsl_qsa_sparse_prefill",
    "qsa_cache_row_layout",
    "qsa_expand_block_indices",
    "qsa_paged_kv_pool",
    "qsa_paged_kv_slots",
    "qsa_paged_mqa_logits",
    "qsa_select_paged_tokens",
    "qsa_sparse_paged_gqa",
    "qsa_sparse_prefill",
    "qsa_sparse_prefill_available",
    "qsa_sparse_prefill_problem",
    "qsa_store_quantized_rows",
]
