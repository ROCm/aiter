# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Paged QSA caches: pack a dense ``[S, H, D]`` tensor into vLLM-style pages.

Layout matches live AMD ``qwen4_exp``: ``cache [n_pages, page_size, H, D]`` and
``block_table [n_req, n_pages]`` mapping logical page -> physical page. Indexer
K is one compressed block per logical slot (``S = n_blocks``); GQA K/V use
uncompressed tokens (``S = seq_len``).
"""

from __future__ import annotations

import torch

from aiter.ops.flydsl.kernels.kernels_common import ceildiv


def pack_paged_cache(
    dense: torch.Tensor,
    page_size: int,
    permute_pages: bool = True,
    generator: torch.Generator | None = None,
    physical: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack ``dense [S, H, D]`` into ``[n_pages, page_size, H, D]`` + table.

    One request. Physical pages are shuffled when ``permute_pages`` is true so
    a gather that ignores ``block_table`` cannot pass. Pass ``physical`` to
    reuse an existing logical->physical map (GQA K and V must share a table).
    """
    if dense.ndim != 3:
        raise ValueError(f"dense cache must be [S, H, D], got {tuple(dense.shape)}")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    n_logical, n_heads, head_dim = dense.shape
    n_pages = ceildiv(n_logical, page_size)
    pad = n_pages * page_size - n_logical
    if pad:
        dense = torch.nn.functional.pad(dense, (0, 0, 0, 0, 0, pad))
    packed = dense.view(n_pages, page_size, n_heads, head_dim).contiguous()
    if n_pages == 0:
        table = torch.empty(1, 0, dtype=torch.int32, device=dense.device)
        return packed, table
    if physical is None:
        if permute_pages and n_pages > 1:
            physical = torch.randperm(
                n_pages, device=dense.device, generator=generator
            ).to(torch.int32)
        else:
            physical = torch.arange(n_pages, dtype=torch.int32, device=dense.device)
    elif physical.numel() != n_pages:
        raise ValueError(f"physical map length {physical.numel()} != n_pages {n_pages}")
    cache = packed.new_empty(packed.shape)
    cache[physical.to(torch.int64)] = packed
    table = physical.reshape(1, n_pages).contiguous()
    return cache, table


def gather_paged_cache(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    n_logical: int,
    req: int = 0,
) -> torch.Tensor:
    """Gather ``n_logical`` rows back to dense ``[S, H, D]`` using the table."""
    if cache.ndim != 4:
        raise ValueError(
            f"paged cache must be [n_pages, page_size, H, D], got {tuple(cache.shape)}"
        )
    page_size = cache.shape[1]
    device = cache.device
    logical = torch.arange(n_logical, device=device)
    logical_page = logical // page_size
    offset = logical % page_size
    physical = block_table[req].index_select(0, logical_page).to(torch.int64)
    return cache[physical, offset]


def gather_qsa_caches(
    index_cache: torch.Tensor,
    index_table: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_table: torch.Tensor,
    n_blocks: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack paged indexer K and GQA K/V to dense tensors for the oracle."""
    k_bar = gather_paged_cache(index_cache, index_table, n_blocks)[:, 0, :]
    k = gather_paged_cache(k_cache, kv_table, seq_len)
    v = gather_paged_cache(v_cache, kv_table, seq_len)
    return k_bar, k, v
