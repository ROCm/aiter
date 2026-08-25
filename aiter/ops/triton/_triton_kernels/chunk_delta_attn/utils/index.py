# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""Index preparation utilities for variable-length sequence processing."""

import torch
import triton

from aiter.ops.triton._triton_kernels.chunk_delta_attn.chunk_delta_attn_utils import (
    tensor_cache,
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
        Tensor of shape [num_chunks, 2] where each row is
        [sequence_id, chunk_idx_in_seq]

    Building this reads ``cu_seqlens`` back to the host and ships a new tensor
    out, so it is cached on the argument identity; see ``tensor_cache``.
    """
    lens = torch.diff(cu_seqlens)
    indices = torch.cat(
        [torch.arange(n) for n in triton.cdiv(lens, chunk_size).tolist()]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)
