# SPDX-License-Identifier: MIT

"""CPU reference packing helpers for the gfx1201 Q4 group-64 GEMV tests."""

from __future__ import annotations

import torch


def pack_group64(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Pack signed INT4 weights and FP16 row scales into 1088-byte tiles."""

    if q.device.type != "cpu" or scales.device.type != "cpu":
        raise ValueError("reference packer expects CPU tensors")
    if q.dtype != torch.int8 or q.ndim != 2:
        raise ValueError("q must be an INT8 matrix [N,K]")
    n, k = q.shape
    if n <= 0:
        raise ValueError("N must be positive")
    if k <= 0:
        raise ValueError("K must be positive")
    if n % 32:
        raise ValueError("N must be divisible by 32")
    if k % 64:
        raise ValueError("K must be divisible by 64")
    groups = k // 64
    if scales.shape != (n, groups):
        raise ValueError("scales must have shape [N,K/64]")
    if torch.any(q < -8) or torch.any(q > 7):
        raise ValueError("q values must be signed INT4 (-8..7)")

    row_tiles = n // 32
    q_tiles = q.reshape(row_tiles, 32, groups, 64).permute(0, 2, 1, 3)
    low = torch.bitwise_and(q_tiles[..., 0::2], 0xF).to(torch.uint8)
    high = torch.bitwise_and(q_tiles[..., 1::2], 0xF).to(torch.uint8)
    values = (low | (high << 4)).permute(0, 1, 3, 2).contiguous()
    scale_bytes = (
        scales.to(torch.float16)
        .reshape(row_tiles, 32, groups)
        .permute(0, 2, 1)
        .contiguous()
        .view(torch.uint8)
        .reshape(row_tiles, groups, 64)
    )
    packed = torch.empty((row_tiles, groups, 1088), dtype=torch.uint8)
    packed[..., :64] = scale_bytes
    packed[..., 64:] = values.reshape(row_tiles, groups, 1024)
    return packed
