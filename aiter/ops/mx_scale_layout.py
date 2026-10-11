# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Public MX scale-buffer layout helpers.

The helpers are pure torch and intentionally reuse the canonical AITER and
OPUS shuffle functions. They are preparation/validation utilities, not GEMM
hot-path conversions.
"""

import torch

from aiter.utility.mx_types import MXScaleLayoutInt

from .shuffle import shuffle_scale, shuffle_scale_f4

_LAYOUT_NAMES = {
    MXScaleLayoutInt.ROW_MAJOR: "ROW_MAJOR",
    MXScaleLayoutInt.AITER_E8M0: "AITER_E8M0",
    MXScaleLayoutInt.OPUS_F4: "OPUS_F4",
}


def normalize_mx_scale_layout(layout: int) -> int:
    try:
        value = int(layout)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid MX scale layout {layout!r}") from exc
    if value not in _LAYOUT_NAMES:
        raise ValueError(
            f"unknown MX scale layout {layout!r}; expected one of {_LAYOUT_NAMES}"
        )
    return value


def resolve_mx_scale_layout(
    scale_layout: int | None,
    shuffle: bool | None,
    *,
    legacy_shuffle_layout: int = MXScaleLayoutInt.AITER_E8M0,
) -> int:
    """Resolve a new explicit layout or a legacy shuffle-bool alias."""
    if scale_layout is not None:
        if shuffle is not None:
            raise ValueError(
                "scale_layout and legacy shuffle flag are mutually exclusive"
            )
        return normalize_mx_scale_layout(scale_layout)
    return (
        normalize_mx_scale_layout(legacy_shuffle_layout)
        if bool(shuffle)
        else MXScaleLayoutInt.ROW_MAJOR
    )


def mx_scale_buffer_shape(rows: int, k_groups: int, layout: int) -> tuple[int, int]:
    """Return the physical 2-D buffer shape for a logical ``[rows, k_groups]`` scale."""
    if rows < 0 or k_groups < 0:
        raise ValueError(
            f"rows and k_groups must be non-negative, got {(rows, k_groups)}"
        )
    layout = normalize_mx_scale_layout(layout)
    if layout == MXScaleLayoutInt.ROW_MAJOR:
        return rows, k_groups
    if layout == MXScaleLayoutInt.AITER_E8M0:
        return ((rows + 255) // 256 * 256, (k_groups + 7) // 8 * 8)
    return ((rows + 31) // 32 * 32, (k_groups + 3) // 4 * 4)


def validate_mx_scale_buffer(
    scale: torch.Tensor,
    rows: int,
    k_groups: int,
    layout: int,
    *,
    exact_shape: bool = True,
) -> tuple[int, int]:
    """Validate a byte-scale buffer and return its canonical physical shape."""
    expected = mx_scale_buffer_shape(rows, k_groups, layout)
    if scale.element_size() != 1:
        raise ValueError(
            f"MX scale buffer must have one-byte elements, got {scale.dtype}"
        )
    if not scale.is_contiguous():
        raise ValueError("MX scale buffer must be contiguous")
    if exact_shape and tuple(scale.shape) != expected:
        raise ValueError(
            f"{_LAYOUT_NAMES[int(layout)]} scale for logical {(rows, k_groups)} "
            f"must have shape {expected}, got {tuple(scale.shape)}"
        )
    if not exact_shape and scale.numel() != expected[0] * expected[1]:
        raise ValueError(
            f"{_LAYOUT_NAMES[int(layout)]} scale for logical {(rows, k_groups)} "
            f"must hold {expected[0] * expected[1]} bytes, got {scale.numel()}"
        )
    return expected


def to_mx_scale_layout(scale: torch.Tensor, layout: int) -> torch.Tensor:
    """Convert a contiguous row-major byte scale to ``layout`` using pure torch."""
    layout = normalize_mx_scale_layout(layout)
    if scale.ndim != 2:
        raise ValueError(f"row-major MX scale must be 2-D, got {scale.ndim}D")
    if scale.element_size() != 1:
        raise ValueError(f"MX scale must have one-byte elements, got {scale.dtype}")
    scale = scale.contiguous()
    if layout == MXScaleLayoutInt.ROW_MAJOR:
        return scale
    if layout == MXScaleLayoutInt.AITER_E8M0:
        return shuffle_scale(scale)
    return shuffle_scale_f4(scale, intype=7)


def _layout_index_map(
    rows: int, k_groups: int, layout: int, device
) -> tuple[torch.Tensor, torch.Tensor]:
    logical = torch.arange(rows * k_groups, dtype=torch.int64, device=device).view(
        rows, k_groups
    )
    packed = torch.zeros(
        mx_scale_buffer_shape(rows, k_groups, layout),
        dtype=torch.int64,
        device=device,
    )
    for shift in range(0, 64, 8):
        plane = ((logical >> shift) & 0xFF).to(torch.uint8)
        packed |= (
            to_mx_scale_layout(plane, layout).view(torch.uint8).to(torch.int64) << shift
        )
    valid = (
        to_mx_scale_layout(
            torch.ones((rows, k_groups), dtype=torch.uint8, device=device), layout
        ).view(torch.uint8)
        == 1
    )
    return packed, valid


def from_mx_scale_layout(
    scale: torch.Tensor, rows: int, k_groups: int, layout: int
) -> torch.Tensor:
    """Convert a validated physical scale buffer back to row-major."""
    layout = normalize_mx_scale_layout(layout)
    validate_mx_scale_buffer(scale, rows, k_groups, layout)
    if layout == MXScaleLayoutInt.ROW_MAJOR:
        return scale.contiguous()

    mapping, valid = _layout_index_map(rows, k_groups, layout, scale.device)
    src = scale.contiguous().view(torch.uint8).reshape(-1)
    out = torch.empty(rows * k_groups, dtype=torch.uint8, device=scale.device)
    out[mapping.reshape(-1)[valid.reshape(-1)]] = src[valid.reshape(-1)]
    return out.view(rows, k_groups).view(scale.dtype)
