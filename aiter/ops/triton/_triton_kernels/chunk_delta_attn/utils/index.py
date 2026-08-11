# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""Index preparation utilities for variable-length sequence processing."""

import torch
import triton

from aiter.ops.triton._triton_kernels.gated_delta_rule.gated_delta_rule_utils import (
    tensor_cache,
)


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    """
    Prepare chunk indices for variable-length sequences.

    The ``.tolist()`` below is a D2H sync, so this is cached on the identity of
    ``cu_seqlens``: callers must pass the ORIGINAL metadata tensor rather than a
    freshly sliced view, otherwise every call misses. One forward runs this once
    per layer with the same tensor, which is what the cache is for.

    Args:
        cu_seqlens: Cumulative sequence lengths [N+1] (original, unsliced)
        chunk_size: Size of each chunk

    Returns:
        Tensor of shape [num_chunks, 2] where each row is
        [sequence_id, chunk_idx_in_seq]
    """
    lens = torch.diff(cu_seqlens)
    indices = torch.cat(
        [torch.arange(n) for n in triton.cdiv(lens, chunk_size).tolist()]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)
