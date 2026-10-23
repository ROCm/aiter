# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aiter.ops.triton._triton_kernels.moe.q256_schedule import (
    ABI_MAP_ENTRIES as _ABI_MAP_ENTRIES,
)
from aiter.ops.triton._triton_kernels.moe.q256_schedule import (
    ABI_TILE_ROWS as _ABI_TILE_ROWS,
)
from aiter.ops.triton._triton_kernels.moe.q256_schedule import (
    ABI_XCDS as _ABI_XCDS,
)
from aiter.ops.triton._triton_kernels.moe.q256_schedule import (
    make_job_map,
)

ABI_MAP_ENTRIES = _ABI_MAP_ENTRIES.value
ABI_TILE_ROWS = _ABI_TILE_ROWS.value
ABI_XCDS = _ABI_XCDS.value


def q256_make_job_map(
    sorted_ids: torch.Tensor,
    sorted_experts: torch.Tensor,
    counts: torch.Tensor,
    job_map: torch.Tensor,
) -> torch.Tensor:
    """Write and return the caller-owned 1024-entry Q256 ABI128 job map.

    Inputs are contiguous 1-D int32 GPU tensors on the output device. ``counts``
    starts with [padded sorted rows, token count]; ``sorted_experts`` has one
    expert per 256-row physical tile. IDs carry token indices in their low 24
    bits. The caller must provide valid device counts within the input buffers.

    The 8192-token geometry with exactly 32 trailing expert-512 tiles expands
    those tiles to 64 shared jobs of 128 rows each. Other geometry uses identity
    scheduling. Over-capacity geometry preserves the producer's first-1024-slot
    identity fallback; this map does not provide extra visits.

    Validation checks metadata only, without copying counts to the host, so the
    wrapper is suitable for graph capture and preallocated output reuse.
    """
    tensors = {
        "sorted_ids": sorted_ids,
        "sorted_experts": sorted_experts,
        "counts": counts,
        "job_map": job_map,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if (
            tensor.dtype != torch.int32
            or tensor.ndim != 1
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be a contiguous 1-D int32 tensor")
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be on a GPU")
    if any(tensor.device != job_map.device for tensor in tensors.values()):
        raise ValueError("all tensors must be on the same GPU")
    if counts.numel() < 2:
        raise ValueError("counts must contain padded row count and token count")
    if job_map.numel() != ABI_MAP_ENTRIES:
        raise ValueError(f"job_map must contain exactly {ABI_MAP_ENTRIES} entries")
    if sorted_ids.numel() % ABI_TILE_ROWS:
        raise ValueError(f"sorted_ids capacity must be a multiple of {ABI_TILE_ROWS}")
    if sorted_experts.numel() < sorted_ids.numel() // ABI_TILE_ROWS:
        raise ValueError("sorted_experts must cover the sorted_ids physical tiles")
    make_job_map[(ABI_XCDS,)](sorted_ids, sorted_experts, counts, job_map)
    return job_map
