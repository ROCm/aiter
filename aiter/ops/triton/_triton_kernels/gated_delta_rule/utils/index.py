# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""
Index preparation utilities for variable-length sequence processing.

This module provides functions for preparing various indices needed for
chunk-based and variable-length sequence operations.
"""

import torch
import torch.nn.functional as F
import triton

from ..gated_delta_rule_utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    """Compute sequence lengths from cumulative sequence lengths."""
    return torch.diff(cu_seqlens)


@tensor_cache
def prepare_lens_from_mask(mask: torch.BoolTensor) -> torch.LongTensor:
    """Compute sequence lengths from a boolean mask."""
    return mask.sum(dim=-1, dtype=torch.int32)


@tensor_cache
def prepare_cu_seqlens_from_lens(
    lens: torch.LongTensor,
    dtype: torch.dtype | None = torch.int32,
) -> torch.LongTensor:
    """Convert sequence lengths to cumulative sequence lengths."""
    return F.pad(lens.cumsum(dim=0, dtype=dtype), (1, 0))


@tensor_cache
def prepare_cu_seqlens_from_mask(
    mask: torch.BoolTensor,
    dtype: torch.dtype | None = torch.int32,
) -> torch.LongTensor:
    """Convert a boolean mask to cumulative sequence lengths."""
    return prepare_cu_seqlens_from_lens(prepare_lens_from_mask(mask), dtype)


@tensor_cache
def prepare_lens_from_cu_seqlens(
    cu_seqlens: torch.LongTensor,
) -> torch.LongTensor:
    """Extract sequence lengths from cumulative sequence lengths."""
    return torch.diff(cu_seqlens)


@tensor_cache
def prepare_position_ids(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    """Generate position IDs for each sequence."""
    return torch.cat(
        [
            torch.arange(n, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
            for n in prepare_lens(cu_seqlens).unbind()
        ]
    )


@tensor_cache
def prepare_sequence_ids(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    """Generate sequence IDs indicating which sequence each token belongs to."""
    return prepare_position_ids(cu_seqlens).eq(0).cumsum(0) - 1


@tensor_cache
def prepare_token_indices(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    """Generate (sequence_id, position_id) pairs for each token."""
    position_ids = prepare_position_ids(cu_seqlens)
    return torch.stack([prepare_sequence_ids(cu_seqlens), position_ids], 1).to(
        cu_seqlens
    )


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    """
    Prepare chunk indices for variable-length sequences.

    Args:
        cu_seqlens: Cumulative sequence lengths [N+1]
        chunk_size: Size of each chunk

    Returns:
        Tensor of shape [num_chunks, 2] where each row is [sequence_id, chunk_idx_in_seq]
    """
    indices = torch.cat(
        [
            torch.arange(n)
            for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
        ]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    """
    Prepare cumulative chunk offsets for variable-length sequences.

    Args:
        cu_seqlens: Cumulative sequence lengths [N+1]
        chunk_size: Size of each chunk

    Returns:
        Cumulative chunk offsets [N+1]
    """
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)


@tensor_cache
def get_max_num_splits(cu_seqlens: torch.LongTensor, chunk_size: int) -> int:
    """Get maximum number of splits (chunks) across all sequences."""
    return triton.cdiv(int(max(prepare_lens(cu_seqlens))), chunk_size)
