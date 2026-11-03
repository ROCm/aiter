# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""QSA top-level dispatch: selects the best backend (triton, FlyDSL, etc.)
for each QSA operator based on the input shape and available backends.

Consumers should import from here rather than from the backend-specific
modules directly.
"""

import torch

from aiter.ops.topk import top_k_per_row_prefill
from aiter.ops.triton.attention.qsa import (
    qsa_expand_block_indices,
    qsa_sparse_paged_gqa,
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score compressed key groups.

    Dispatches to the FlyDSL MFMA scorer for large prefill batches when
    ``allow_flydsl`` is set and FlyDSL is available, otherwise falls through
    to the triton wrapper.
    """
    if (
        allow_flydsl
        and _HAS_FLYDSL
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


__all__ = [
    "DEFAULT_LOGITS_WORKSPACE_BYTES",
    "qsa_expand_block_indices",
    "qsa_paged_mqa_logits",
    "qsa_select_paged_tokens",
    "qsa_sparse_paged_gqa",
]
