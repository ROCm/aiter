# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Reference cache layout for the paged MXFP4 MQA-logits tests."""

import torch

from aiter.ops.triton.attention.pa_mqa_logits_mxfp4 import (
    K_WIDTH,
    SCALE_MODE_WIDE,
    cache_format,
)


def preshuffle_values(
    x: torch.Tensor, n_per_tile: int, d_per_tile: int = K_WIDTH
) -> torch.Tensor:
    """[P, page, D//2] uint8 into dot-operand order, within each page."""
    p, rows, d = x.shape
    return (
        x.reshape(p, rows // n_per_tile, n_per_tile, d // d_per_tile, d_per_tile)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .reshape(p, rows, d)
    )


def preshuffle_scales(
    x: torch.Tensor, n_per_tile: int, scale_mode: int = SCALE_MODE_WIDE
) -> torch.Tensor:
    """[P, page, D//32] e8m0 into the order the scale load reads. Mode 1 puts
    one MFMA tile's scale run innermost; mode 0 puts the token axis innermost."""
    assert scale_mode in (0, 1), "scale_mode must be 0 or 1"
    WARP_SIZE = 64
    p, rows, ns = x.shape
    if scale_mode == 0:
        return (
            x.reshape(p, rows // n_per_tile, n_per_tile, ns)
            .permute(0, 1, 3, 2)
            .contiguous()
            .reshape(p, rows, ns)
        )
    s_lo = WARP_SIZE // n_per_tile
    s_hi = ns // s_lo
    return (
        x.reshape(p, rows // n_per_tile, n_per_tile, s_hi, s_lo)
        .permute(0, 1, 4, 2, 3)
        .contiguous()
        .reshape(p, rows, ns)
    )


def preshuffle_cache(
    values: torch.Tensor,
    scales: torch.Tensor,
    num_heads: int,
    head_size: int,
    scale_mode: int = SCALE_MODE_WIDE,
):
    """Natural order into the stored order: the reference for what
    k_norm_rope_mxfp4_cache writes."""
    f = cache_format(num_heads, head_size, values.shape[1])
    return (
        preshuffle_values(values, f["n_per_tile"], f["d_per_tile"]),
        preshuffle_scales(scales, f["n_per_tile"], scale_mode),
    )


def unshuffle_values(
    x: torch.Tensor, n_per_tile: int, d_per_tile: int = K_WIDTH
) -> torch.Tensor:
    p, rows, d = x.shape
    return (
        x.reshape(p, rows // n_per_tile, d // d_per_tile, n_per_tile, d_per_tile)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .reshape(p, rows, d)
    )


def unshuffle_scales(
    x: torch.Tensor, n_per_tile: int, scale_mode: int = SCALE_MODE_WIDE
) -> torch.Tensor:
    assert scale_mode in (0, 1), "scale_mode must be 0 or 1"
    WARP_SIZE = 64
    p, rows, ns = x.shape
    if scale_mode == 0:
        return (
            x.reshape(p, rows // n_per_tile, ns, n_per_tile)
            .permute(0, 1, 3, 2)
            .contiguous()
            .reshape(p, rows, ns)
        )
    s_lo = WARP_SIZE // n_per_tile
    s_hi = ns // s_lo
    return (
        x.reshape(p, rows // n_per_tile, s_lo, n_per_tile, s_hi)
        .permute(0, 1, 3, 4, 2)
        .contiguous()
        .reshape(p, rows, ns)
    )


def pack_cache(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """([P, page, D//2], [P, page, D//32]) -> [P, page, 1, D//2 + D//32],
    each page's values then its scales."""
    num_pages, page_size, head_bytes = values.shape
    num_scales = scales.shape[2]
    idim = head_bytes + num_scales
    out = torch.empty(
        num_pages, page_size * idim, dtype=torch.uint8, device=values.device
    )
    out[:, : page_size * head_bytes] = values.reshape(num_pages, -1)
    out[:, page_size * head_bytes :] = scales.reshape(num_pages, -1)
    return out.view(num_pages, page_size, 1, idim)
