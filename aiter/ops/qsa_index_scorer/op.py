# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Public op ``aiter.qsa_paged_mqa_logits`` -- QSA index scorer, op contract v1.

The public signature is exactly ATOM's 12-parameter scorer signature
(``api.ATOM_SCORER_SIGNATURE``); implementation policy (profile, buckets, output
buffers) is not part of it. Backend selection is a dispatch decision:

* fused FlyDSL gfx1250 kernel when the profile named by
  ``$AITER_QSA_INDEX_SCORER_PROFILE`` covers the call;
* otherwise aiter's native pure-torch implementation (``native.torch.v1``).

Default: no profile selected => every call is served natively (reason ``disabled``)
and FlyDSL is not imported. ``aiter.ops.qsa_index_scorer.decide`` reports which path
a call takes and why, without running it. Contract sheet:
``docs/qsa_index_scorer.md``.

Provider readiness: once ``aiter.plugin`` is merged, this function is the routing
point (``@route(op="qsa_paged_mqa_logits", version=1)`` above it); its body is the
native implementation AITER owns. Nothing here imports ``aiter.plugin``.
"""

from __future__ import annotations

import torch

from . import api as _api

#: op contract version (bumped on any breaking semantic change)
OP_CONTRACT_VERSION = 1
OP_NAME = "qsa_paged_mqa_logits"


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score compressed key groups (QSA indexer), op contract v1.

    Args:
        q: ``[rows, heads, head_dim]`` bf16 (fp16/fp32 accepted with an fp8 cache).
        compressed_k_cache: ``[pages, page_size, 1, head_dim]``; bf16, or fp8 E4M3
            with ``compressed_k_scale``. Any page size (fused path: 4 / 8 / 16).
        page_table: ``[requests, width]`` int32/int64; entries outside
            ``[0, pages)`` are invalid pages.
        token_to_request, query_positions: ``[rows]`` int32/int64, contiguous;
            a request id outside ``[0, requests)`` makes the row invisible.
        context_lens: ``[requests]`` int32/int64, contiguous.
        compress_ratio: tokens per compressed group (> 0).
        score_divisor: logit divisor (> 0); ``None`` = ``sqrt(head_dim)``.
        max_columns: ``columns = max(1, min(width * page_size, max_columns))``.
        row_starts: optional ``[rows]`` int32; zero-filled in place.
        allow_flydsl: ATOM's switch for its own page-16 prefill kernel; accepted
            for signature compatibility and ignored.
        compressed_k_scale: fp32 ``[pages, page_size, 1]`` per-row scale of an fp8
            cache; ``None`` for bf16.

    Returns:
        ``(logits, visible_groups)``: fp32 ``[rows, columns]`` (``-inf`` where
        ``c >= visible`` or the page is invalid) and int32 ``[rows]``
        ``visible = clamp(min((pos + 1) // ratio, ctx // ratio), min=0)``.

    Raises:
        ValueError: the call violates the contract (shapes, dtypes, devices,
            MXFP4/MXFP8 caches, non-positive divisor or ratio).
    """
    return _api.qsa_paged_mqa_logits(
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
        allow_flydsl,
        compressed_k_scale,
    )
