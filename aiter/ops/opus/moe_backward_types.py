# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Versioned route metadata and output containers shared by backward APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar
from torch import Tensor

@dataclass(frozen=True)
class OpusMoeFixedMetadata:
    """Versioned fixed-top-k sorting metadata emitted by forward."""

    layout_version: ClassVar[int] = 1

    sorted_token_ids: Tensor
    sorted_expert_ids: Tensor
    num_valid_ids: Tensor
    reverse_sorted: Tensor
    expert_padded_offsets: Tensor
    block_m: int


@dataclass(frozen=True)
class OpusMoeVarlenMetadata:
    """Versioned compact-route sorting metadata emitted by forward."""

    layout_version: ClassVar[int] = 1

    sorted_route_ids: Tensor
    sorted_expert_ids: Tensor
    num_valid_ids: Tensor
    route_to_token: Tensor
    token_route_offsets: Tensor
    expert_padded_offsets: Tensor
    block_m: int


@dataclass(frozen=True)
class OpusMoeBackwardOutput:
    """Complete expert-path gradients and reusable K1 intermediates."""

    d_x: Tensor
    d_w1: Tensor
    d_w2: Tensor
    d_scores: Tensor
    d_z_sorted: Tensor
    a_scaled: Tensor
    d_b1: Tensor | None = None
    d_b2: Tensor | None = None


@dataclass(frozen=True)
class OpusMoeDownBackwardOutput:
    d_z_sorted: Tensor
    a_scaled: Tensor
    d_scores: Tensor
    d_scores_workspace: Tensor


@dataclass(frozen=True)
class OpusMoeRouteBackwardOutput:
    d_x_route: Tensor
    d_x: Tensor

    @property
    def d_x_route_sorted(self) -> Tensor:
        """Compatibility alias; the workspace is logical [token, slot] order."""

        return self.d_x_route


@dataclass(frozen=True)
class OpusMoeWeightBackwardOutput:
    d_w1: Tensor
    d_w2: Tensor


@dataclass(frozen=True)
class OpusMoeBiasDownBackwardOutput:
    d_scores: Tensor
    d_b2: Tensor


# Raw fixed-routing JIT bindings.
