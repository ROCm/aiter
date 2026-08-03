# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host-side contracts shared by the DWDP multi-B MoE entry points."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

SUPPORTED_MULTI_B_PARTITIONS = (2, 4, 8)
SUPPORTED_MULTI_B_SCALE_LAYOUTS = ("mx_1x32", "fp8_128x128")


@dataclass(frozen=True)
class MultiBPartitionSpec:
    """Validated logical shape of ordered expert-weight partitions."""

    partition_sizes: tuple[int, ...]
    experts: int
    model_dim: int
    inter_dim: int
    packing_ratio: int
    scale_layout: str


def expert_partition_index(
    expert_id: int, partition_sizes: Sequence[int]
) -> tuple[int, int]:
    """Return ``(partition, local_expert)`` for a global expert id."""

    expert_id = int(expert_id)
    if expert_id < 0:
        raise IndexError(f"expert_id must be non-negative, got {expert_id}")
    base = 0
    for partition, size in enumerate(partition_sizes):
        size = int(size)
        if size <= 0:
            raise ValueError(f"partition sizes must be positive, got {partition_sizes}")
        if expert_id < base + size:
            return partition, expert_id - base
        base += size
    raise IndexError(f"expert_id {expert_id} is outside [0, {base})")


def partition_offsets(partition_sizes: Sequence[int]) -> tuple[int, ...]:
    """Return cumulative starts plus the terminal expert count."""

    offsets = [0]
    for size in partition_sizes:
        size = int(size)
        if size <= 0:
            raise ValueError(f"partition sizes must be positive, got {partition_sizes}")
        offsets.append(offsets[-1] + size)
    return tuple(offsets)


def partition_module_tag(partition_sizes: Sequence[int]) -> str:
    """Return the stable module suffix containing every partition size."""

    sizes = tuple(int(size) for size in partition_sizes)
    partition_offsets(sizes)
    return "_mb" + "x".join(str(size) for size in sizes)


def _as_tensor_tuple(name: str, values) -> tuple[torch.Tensor, ...]:
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} must be an ordered list or tuple of tensors")
    if not values:
        raise ValueError(f"{name} must not be empty")
    if not all(isinstance(value, torch.Tensor) for value in values):
        raise TypeError(f"every {name} entry must be a torch.Tensor")
    return tuple(values)


def _check_mx_scale_partition(
    name: str,
    scale: torch.Tensor,
    *,
    partition: int,
    rows: int,
    groups: int,
    device: torch.device,
    require_e8m0: bool,
) -> None:
    if scale.device != device:
        raise ValueError(f"{name}[{partition}] is on {scale.device}, expected {device}")
    if not scale.is_contiguous():
        raise ValueError(f"{name}[{partition}] must be contiguous")
    if require_e8m0 and scale.element_size() != 1:
        raise TypeError(
            f"{name}[{partition}] must contain one-byte E8M0 scales; "
            f"got dtype={scale.dtype} (element_size={scale.element_size()})"
        )
    padded_groups = (groups + 7) // 8 * 8
    padded_rows = (rows + 255) // 256 * 256
    allowed_shapes = {(rows, padded_groups), (padded_rows, padded_groups)}
    if tuple(scale.shape) not in allowed_shapes:
        expected = " or ".join(str(shape) for shape in sorted(allowed_shapes))
        raise ValueError(
            f"{name}[{partition}] has shape {tuple(scale.shape)}, expected "
            f"{expected} for independently shuffled partition scales"
        )


def _check_fp8_block_scale_partition(
    name: str,
    scale: torch.Tensor,
    *,
    partition: int,
    expected_shape: tuple[int, int, int],
    device: torch.device,
) -> None:
    if scale.device != device:
        raise ValueError(f"{name}[{partition}] is on {scale.device}, expected {device}")
    if scale.dtype != torch.float32:
        raise TypeError(
            f"{name}[{partition}] must be contiguous FP32 for fp8_128x128; "
            f"got {scale.dtype}"
        )
    if scale.ndim != 3 or tuple(scale.shape) != expected_shape:
        raise ValueError(
            f"{name}[{partition}] has shape {tuple(scale.shape)}, expected exact "
            f"rank-3 shape {expected_shape} for fp8_128x128"
        )
    if not scale.is_contiguous():
        raise ValueError(f"{name}[{partition}] must be contiguous")


def _is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype in {
        fp8_dtype
        for fp8_dtype in (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e4m3fnuz", None),
        )
        if fp8_dtype is not None
    }


def validate_multi_b_partitions(
    w1_partitions,
    w2_partitions,
    w1_scale_partitions=None,
    w2_scale_partitions=None,
    *,
    bias1: torch.Tensor | None = None,
    bias2: torch.Tensor | None = None,
    scale_layout: str = "mx_1x32",
    require_e8m0_scales: bool | None = None,
) -> MultiBPartitionSpec:
    """Validate ordered W1/W2 partitions without reading device data.

    ``scale_layout="mx_1x32"`` accepts the existing independently shuffled,
    padded one-byte E8M0 scale matrices. ``scale_layout="fp8_128x128"`` accepts
    only conventional FP8 payloads and exact contiguous FP32 block-scale
    tensors with shapes ``[Ep, N/128, K/128]``.

    ``require_e8m0_scales`` is retained for compatibility with older callers.
    New callers should select an explicit ``scale_layout``.
    """

    scale_layout = str(scale_layout)
    if scale_layout not in SUPPORTED_MULTI_B_SCALE_LAYOUTS:
        raise ValueError(
            f"scale_layout must be one of {SUPPORTED_MULTI_B_SCALE_LAYOUTS}, "
            f"got {scale_layout!r}"
        )
    if require_e8m0_scales is not None and scale_layout != "mx_1x32":
        raise ValueError(
            "require_e8m0_scales is a legacy mx_1x32 option and cannot be "
            f"combined with scale_layout={scale_layout!r}"
        )
    require_e8m0 = True if require_e8m0_scales is None else bool(require_e8m0_scales)

    w1 = _as_tensor_tuple("w1_partitions", w1_partitions)
    w2 = _as_tensor_tuple("w2_partitions", w2_partitions)
    if len(w1) != len(w2):
        raise ValueError(
            "w1_partitions and w2_partitions must have identical cardinality; "
            f"got {len(w1)} and {len(w2)}"
        )
    if len(w1) not in (1, *SUPPORTED_MULTI_B_PARTITIONS):
        raise ValueError(
            "multi-B supports list-of-one delegation or 2/4/8 partitions; "
            f"got {len(w1)}"
        )

    first_w1, first_w2 = w1[0], w2[0]
    if first_w1.ndim != 3 or first_w2.ndim != 3:
        raise ValueError(
            "MoE weight partitions must be rank-3 tensors; "
            f"got W1 {tuple(first_w1.shape)} and W2 {tuple(first_w2.shape)}"
        )
    if first_w1.shape[0] <= 0:
        raise ValueError("weight partitions must contain at least one expert")
    if first_w2.shape[1] % first_w1.shape[2] != 0:
        raise ValueError(
            "cannot infer the packed-weight ratio from W1/W2 shapes: "
            f"{tuple(first_w1.shape)} and {tuple(first_w2.shape)}"
        )
    packing_ratio = first_w2.shape[1] // first_w1.shape[2]
    if packing_ratio not in (1, 2):
        raise ValueError(f"weight packing ratio must be 1 or 2, got {packing_ratio}")

    model_dim = int(first_w2.shape[1])
    inter_dim = int(first_w2.shape[2]) * int(packing_ratio)
    scale_block = 128 if scale_layout == "fp8_128x128" else 32
    if model_dim % scale_block != 0 or inter_dim % scale_block != 0:
        raise ValueError(
            f"multi-B {scale_layout} weight dimensions must be divisible by the "
            f"{scale_block}-element scale block, got model_dim={model_dim}, "
            f"inter_dim={inter_dim}"
        )
    if int(first_w1.shape[1]) != 2 * inter_dim:
        raise ValueError(
            "multi-B mixed MoE requires fused gate+up W1 rows; "
            f"got W1 rows={first_w1.shape[1]}, expected {2 * inter_dim}"
        )

    device = first_w1.device
    dtype = first_w1.dtype
    if scale_layout == "fp8_128x128":
        if packing_ratio != 1:
            raise ValueError(
                "fp8_128x128 requires unpacked FP8 W1/W2 payloads "
                f"(packing_ratio=1), got {packing_ratio}"
            )
        if not _is_fp8_dtype(dtype):
            raise TypeError(
                f"fp8_128x128 requires FP8 E4M3 weight tensors; got dtype={dtype}"
            )
    partition_sizes = []
    expected_w1_tail = tuple(first_w1.shape[1:])
    expected_w2_tail = tuple(first_w2.shape[1:])
    for partition, (w1_part, w2_part) in enumerate(zip(w1, w2)):
        if w1_part.ndim != 3 or w2_part.ndim != 3:
            raise ValueError(f"partition {partition} weights must both be rank-3")
        if w1_part.shape[0] <= 0 or w1_part.shape[0] != w2_part.shape[0]:
            raise ValueError(
                f"partition {partition} W1/W2 expert counts must match and be "
                f"positive; got {w1_part.shape[0]} and {w2_part.shape[0]}"
            )
        if tuple(w1_part.shape[1:]) != expected_w1_tail:
            raise ValueError(
                f"w1_partitions[{partition}] tail shape is "
                f"{tuple(w1_part.shape[1:])}, expected {expected_w1_tail}"
            )
        if tuple(w2_part.shape[1:]) != expected_w2_tail:
            raise ValueError(
                f"w2_partitions[{partition}] tail shape is "
                f"{tuple(w2_part.shape[1:])}, expected {expected_w2_tail}"
            )
        if w1_part.dtype != dtype or w2_part.dtype != dtype:
            raise TypeError(
                f"partition {partition} weight dtype mismatch; all W1/W2 "
                f"partitions must use {dtype}"
            )
        if w1_part.device != device or w2_part.device != device:
            raise ValueError(
                f"partition {partition} device mismatch; all weights must be on {device}"
            )
        if not w1_part.is_contiguous() or not w2_part.is_contiguous():
            raise ValueError(f"partition {partition} weights must be contiguous")
        partition_sizes.append(int(w1_part.shape[0]))

    scales = []
    for name, values in (
        ("w1_scale_partitions", w1_scale_partitions),
        ("w2_scale_partitions", w2_scale_partitions),
    ):
        if values is None:
            scales.append(None)
            continue
        scale_tuple = _as_tensor_tuple(name, values)
        if len(scale_tuple) != len(w1):
            raise ValueError(
                f"{name} must match the weight partition cardinality "
                f"{len(w1)}, got {len(scale_tuple)}"
            )
        scales.append(scale_tuple)

    if scales[0] is None or scales[1] is None:
        raise ValueError(
            f"{scale_layout} multi-B requires matching W1 and W2 scale partitions"
        )

    if scales[0] is not None:
        for partition, (scale, experts) in enumerate(zip(scales[0], partition_sizes)):
            if scale_layout == "fp8_128x128":
                _check_fp8_block_scale_partition(
                    "w1_scale_partitions",
                    scale,
                    partition=partition,
                    expected_shape=(
                        experts,
                        (2 * inter_dim) // 128,
                        model_dim // 128,
                    ),
                    device=device,
                )
            else:
                _check_mx_scale_partition(
                    "w1_scale_partitions",
                    scale,
                    partition=partition,
                    rows=experts * 2 * inter_dim,
                    groups=model_dim // 32,
                    device=device,
                    require_e8m0=require_e8m0,
                )
    if scales[1] is not None:
        for partition, (scale, experts) in enumerate(zip(scales[1], partition_sizes)):
            if scale_layout == "fp8_128x128":
                _check_fp8_block_scale_partition(
                    "w2_scale_partitions",
                    scale,
                    partition=partition,
                    expected_shape=(
                        experts,
                        model_dim // 128,
                        inter_dim // 128,
                    ),
                    device=device,
                )
            else:
                _check_mx_scale_partition(
                    "w2_scale_partitions",
                    scale,
                    partition=partition,
                    rows=experts * model_dim,
                    groups=inter_dim // 32,
                    device=device,
                    require_e8m0=require_e8m0,
                )

    experts = sum(partition_sizes)
    for name, bias, expected_tail in (
        ("bias1", bias1, 2 * inter_dim),
        ("bias2", bias2, model_dim),
    ):
        if bias is None:
            continue
        if bias.device != device:
            raise ValueError(f"{name} is on {bias.device}, expected {device}")
        if bias.dtype != torch.float32:
            raise TypeError(f"{name} must be fp32, got {bias.dtype}")
        if tuple(bias.shape) != (experts, expected_tail):
            raise ValueError(
                f"{name} has shape {tuple(bias.shape)}, expected "
                f"{(experts, expected_tail)} indexed by global expert id"
            )

    return MultiBPartitionSpec(
        partition_sizes=tuple(partition_sizes),
        experts=experts,
        model_dim=model_dim,
        inter_dim=inter_dim,
        packing_ratio=int(packing_ratio),
        scale_layout=scale_layout,
    )


__all__ = [
    "SUPPORTED_MULTI_B_PARTITIONS",
    "SUPPORTED_MULTI_B_SCALE_LAYOUTS",
    "MultiBPartitionSpec",
    "expert_partition_index",
    "partition_module_tag",
    "partition_offsets",
    "validate_multi_b_partitions",
]
