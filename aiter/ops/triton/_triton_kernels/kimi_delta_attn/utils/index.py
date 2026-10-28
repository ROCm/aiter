# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""Index preparation utilities for variable-length sequence processing."""

import functools
import itertools

import torch
import triton

from aiter.ops.triton._triton_kernels.kimi_delta_attn.chunk_delta_attn_utils import (
    tensor_cache,
    upload_ints,
    wait_for_upload,
)


def host_seq_bounds(cu_seqlens: torch.Tensor) -> tuple[tuple[int, int], ...]:
    """``(bos, eos)`` per sequence. A device-to-host copy if ``cu_seqlens`` is
    on the device."""
    return tuple(itertools.pairwise(cu_seqlens.tolist()))


# Reading a device cu_seqlens is a device-to-host copy that waits for all work
# queued before it, so it is cached on the tensor: the layers of a forward pass
# share one. See ``tensor_cache``.
_device_seq_bounds = tensor_cache(host_seq_bounds)


def seq_bounds(
    cu_seqlens: torch.Tensor, cu_seqlens_cpu: torch.Tensor | None = None
) -> tuple[tuple[int, int], ...]:
    """``(bos, eos)`` per sequence, from the host copy when the caller has one."""
    if cu_seqlens_cpu is not None:
        return host_seq_bounds(cu_seqlens_cpu)
    return _device_seq_bounds(cu_seqlens)


def chunk_index_pairs(seqs: tuple[tuple[int, int], ...], chunk_size: int) -> list[int]:
    """``prepare_chunk_indices``' rows, flattened, from host bounds."""
    return [
        x
        for i, (bos, eos) in enumerate(seqs)
        for c in range(triton.cdiv(eos - bos, chunk_size))
        for x in (i, c)
    ]


@functools.lru_cache(maxsize=64)
def _chunk_indices_from_bounds(
    seqs: tuple[tuple[int, int], ...],
    chunk_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.cuda.Stream]:
    indices, stream = upload_ints(chunk_index_pairs(seqs, chunk_size), dtype, device)
    return indices.view(-1, 2), stream


def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    cu_seqlens_cpu: torch.LongTensor | None = None,
) -> torch.LongTensor:
    """
    Prepare chunk indices for variable-length sequences.

    Args:
        cu_seqlens: Cumulative sequence lengths [N+1]
        chunk_size: Size of each chunk
        cu_seqlens_cpu: The same values on the host, if the caller has them.
            Without them ``cu_seqlens`` is read back once per tensor.

    Returns:
        Tensor of shape [num_chunks, 2] where each row is
        [sequence_id, chunk_idx_in_seq], with ``cu_seqlens``' dtype and device.
        An empty sequence has no rows and keeps its place in the numbering.

    Cached on the bounds, so the layers of a forward pass share one upload even
    when each passes a new tensor.
    """
    indices, stream = _chunk_indices_from_bounds(
        seq_bounds(cu_seqlens, cu_seqlens_cpu),
        chunk_size,
        cu_seqlens.dtype,
        cu_seqlens.device,
    )
    wait_for_upload(stream, indices)
    return indices
