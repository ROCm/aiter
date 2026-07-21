# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Common types shared across MoE FlyDSL kernel modules."""

from enum import Enum

import torch


def build_num_valid_token(value: int, device: torch.device) -> torch.Tensor:
    """Return a 1-element int32 device tensor holding ``value``.

    Non-EP dead-tail bound for the grouped MoE kernels. Built with ``fill_``
    (a device kernel) instead of ``torch.tensor([value])`` -- the latter
    allocates on the host and pageable-copies to the device (a Memcpy HtoD on
    every launch), while ``fill_`` writes on the device with no host->device
    memcpy.

    EP mode does not use this helper: there the bound is a dynamic device
    tensor and is sliced/cast in place instead.

    Args:
        value: The scalar count to store (valid routes or valid tokens).
        device: Device the returned tensor lives on.

    Returns:
        A ``(1,)`` int32 tensor holding ``value``.
    """
    tensor = torch.empty(1, dtype=torch.int32, device=device)
    tensor.fill_(int(value))
    return tensor


class GateMode(str, Enum):
    """Gate/Up computation strategy for stage1 GEMM.

    SEPARATED:      Two separate B-tile streams (gate + up), default mode.
    MOCK_GATE_ONLY: Single B-tile stream over full [0, 2*inter_dim), simulates
                    gate-only by doubling grid X on top of SEPARATED layout.
                    Requires split-K (k_batch>1).  NOT true gate-only.
    GATE_ONLY:      Reserved for future true gate-only implementation.
    INTERLEAVE:     Weight rows interleave gate/up (gate[0], up[0], gate[1], ...).
                    pack_N=2 routes even/odd N subtiles.  NOT tied to split-K.
    """

    SEPARATED = "separated"
    MOCK_GATE_ONLY = "mock_gate_only"
    GATE_ONLY = "gate_only"
    INTERLEAVE = "interleave"
