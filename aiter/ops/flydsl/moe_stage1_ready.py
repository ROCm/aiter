# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Structured inputs for the ready-aware FlyDSL MoE Stage1 path.

The production TP AllGather + Stage1 path has one supported scheduling mode.
Keep that mode behind a small structured interface instead of exposing
experimental ready/layout switches at model call sites.
"""

from dataclasses import dataclass

import torch


def _require_int32_contiguous(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must be int32, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


@dataclass(frozen=True, eq=False)
class Stage1ReadyPlan:
    """Expert-major cyclic eligible queue for staged AllGather + Stage1.

    Queue tensors are contiguous views into one packed int32 allocation in the
    order ``cursor[P] / completed[P] / claimed[T,P]``. ``P`` is the Stage1
    N-grid size and ``T`` is the expert-major tile capacity.  Cursors preserve
    the ordinary fast tile order; claim flags let workers skip unavailable
    tiles and safely revisit them after more communication becomes ready.
    """

    ready: torch.Tensor
    tile_source_masks: torch.Tensor
    expert_cursor: torch.Tensor
    completed_tiles: torch.Tensor
    tile_claimed: torch.Tensor
    source_count: int
    chunks_per_source: int
    queue_workers: int

    def __post_init__(self) -> None:
        tensors = {
            "ready": self.ready,
            "tile_source_masks": self.tile_source_masks,
            "expert_cursor": self.expert_cursor,
            "completed_tiles": self.completed_tiles,
            "tile_claimed": self.tile_claimed,
        }
        for name, tensor in tensors.items():
            _require_int32_contiguous(name, tensor)
        device = self.ready.device
        if any(tensor.device != device for tensor in tensors.values()):
            raise ValueError("all Stage1 ready-plan tensors must share one device")
        if self.ready.shape != (1,):
            raise ValueError("ready must be a one-element int32 vector")
        if self.tile_source_masks.ndim != 1 or self.tile_source_masks.numel() == 0:
            raise ValueError("tile_source_masks must be a non-empty vector")
        if self.expert_cursor.ndim != 1 or self.completed_tiles.ndim != 1:
            raise ValueError("cursor/completed tensors must be vectors")
        partitions = self.expert_cursor.numel()
        if partitions <= 0:
            raise ValueError("expert queue must contain at least one N partition")
        if self.completed_tiles.numel() != partitions:
            raise ValueError("cursor/completed must contain one value per N partition")
        if self.tile_claimed.shape != (
            self.tile_source_masks.numel(),
            partitions,
        ):
            raise ValueError("tile_claimed must have shape [tile capacity, N partitions]")
        if self.source_count <= 1 or self.chunks_per_source <= 0:
            raise ValueError("invalid Stage1 ready-plan source geometry")
        if self.dependency_count > 32:
            raise ValueError("Stage1 ready dependencies must fit one int32 mask")
        if self.queue_workers <= 0:
            raise ValueError("queue_workers must be positive")

        base = self.expert_cursor.data_ptr()
        expected_ptrs = (
            base + partitions * 4,
            base + 2 * partitions * 4,
        )
        actual_ptrs = (
            self.completed_tiles.data_ptr(),
            self.tile_claimed.data_ptr(),
        )
        if actual_ptrs != expected_ptrs:
            raise ValueError("Stage1 ready queue views must use the packed ABI")

    @property
    def dependency_count(self) -> int:
        return self.source_count * self.chunks_per_source


@dataclass(frozen=True, eq=False)
class PreparedStage1Input:
    """Prequantized Stage1 payload with physical load-row metadata.

    Stage2 must continue to use its original logical ``sorted_ids``.  Only
    Stage1 consumes ``load_sorted_ids`` from this object.
    """

    values: torch.Tensor
    scales: torch.Tensor
    load_sorted_ids: torch.Tensor
    ready_plan: Stage1ReadyPlan | None = None

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("values must be a 2D Stage1 activation tensor")
        if self.scales.ndim != 2:
            raise ValueError("scales must be a 2D Stage1 scale tensor")
        if self.load_sorted_ids.ndim != 1:
            raise ValueError("load_sorted_ids must be one-dimensional")
        _require_int32_contiguous("load_sorted_ids", self.load_sorted_ids)
        if not self.values.is_contiguous():
            raise ValueError("values must be contiguous")
        if not self.scales.is_contiguous():
            raise ValueError("scales must be contiguous")

        device = self.values.device
        if self.scales.device != device or self.load_sorted_ids.device != device:
            raise ValueError("prepared Stage1 tensors must be on the same device")
        if self.ready_plan is not None and self.ready_plan.ready.device != device:
            raise ValueError(
                "ready plan and prepared Stage1 tensors must share a device"
            )


__all__ = ["PreparedStage1Input", "Stage1ReadyPlan"]
