# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shared utilities for group_sizes-based MoE GEMM wrappers."""

import torch

__all__ = ["build_block_mapping"]


def build_block_mapping(
    group_sizes: torch.Tensor,
    block_m: int,
    total_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pre-compute per-M-block expert IDs, token offsets, and valid row ends.

    Performs all computation with GPU tensor operations; the only Python-side
    integer needed is ``total_tokens`` (already available as ``lhs.shape[0]``
    in the caller), so zero ``.item()`` calls are required.

    Args:
        group_sizes:   Expert token counts ``[E]`` (int tensor, any device).
        block_m:       M-tile height used by the kernel.
        total_tokens:  Sum of group_sizes as a Python int (``lhs.shape[0]``).

    Returns:
        ``(block_expert_ids, block_token_offsets, block_token_ends, max_blocks)``

        * ``max_blocks`` is a Python-int upper bound on the number of M-blocks.
          Slots ``[actual_blocks, max_blocks)`` are padded with
          ``block_expert_ids = -1``; the kernel's ``off_expert == -1`` guard
          returns early for those programs.
        * ``block_token_ends[i]`` is the exclusive valid-row end for block ``i``,
          capped at the expert group boundary to prevent cross-expert reads when
          ``group_size < block_m``.
    """
    device = group_sizes.device
    E = group_sizes.shape[0]

    # Upper bound computable from Python ints — zero GPU ops, zero .item().
    max_blocks = (total_tokens + block_m - 1) // block_m + E

    if E == 0 or total_tokens == 0:
        empty = torch.empty(0, dtype=torch.int32, device=device)
        return empty, empty, empty, max_blocks

    gs = group_sizes.long()

    # Prefix sums for token and block boundaries (GPU ops, no sync).
    token_cumsum = torch.zeros(E + 1, dtype=torch.int64, device=device)
    token_cumsum[1:] = gs.cumsum(0)

    n_blocks_per_expert = (gs + block_m - 1) // block_m
    block_cumsum = torch.zeros(E + 1, dtype=torch.int64, device=device)
    block_cumsum[1:] = n_blocks_per_expert.cumsum(0)

    # For each block slot [0, max_blocks), determine its expert via binary
    # search on block_cumsum[1:].  right=True returns the first index where
    # boundary > block_id, which equals the expert index.
    block_ids = torch.arange(max_blocks, device=device, dtype=torch.int64)
    expert_for_block = torch.bucketize(block_ids, block_cumsum[1:], right=True).clamp(
        max=E - 1
    )

    # Local block index within the expert.
    local_idx = block_ids - block_cumsum[expert_for_block]

    # Valid = local_idx is within the expert's allocated blocks.
    valid = local_idx < n_blocks_per_expert[expert_for_block]

    # Token offsets and exclusive ends.
    expert_token_start = token_cumsum[expert_for_block]
    offsets = (expert_token_start + local_idx * block_m).to(torch.int32)
    ends = torch.minimum(
        expert_token_start + local_idx * block_m + block_m,
        token_cumsum[expert_for_block + 1],
    ).to(torch.int32)

    neg_one = torch.tensor(-1, dtype=torch.int32, device=device)
    zero = torch.zeros(1, dtype=torch.int32, device=device)

    block_expert_ids = torch.where(valid, expert_for_block.to(torch.int32), neg_one)
    block_token_offsets = torch.where(valid, offsets, zero.expand_as(offsets))
    block_token_ends = torch.where(valid, ends, zero.expand_as(ends))

    return block_expert_ids, block_token_offsets, block_token_ends, max_blocks
