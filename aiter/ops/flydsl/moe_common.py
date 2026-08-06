# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Common types shared across MoE FlyDSL kernel modules."""

from enum import Enum

import torch

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


def apply_gate_up(
    gate: torch.Tensor,
    up: torch.Tensor,
    act: str,
    swiglu_limit: float | None = None,
    situ_beta: float = 1.0,
    situ_linear_beta: float = 1.0,
) -> torch.Tensor:
    """Torch reference for the stage1 gate/up activation.

    ``situv2`` (Kimi-K3 ``hidden_act="situ"``) is the grouped TDM stage1
    epilogue's ``stage1_act=3``; this is what that kernel is checked against.
    """
    lim = 7.0 if swiglu_limit is None else float(swiglu_limit)
    if act == "swiglu":
        gate = gate.clamp(max=lim)
        up = up.clamp(min=-lim, max=lim)
        return gate * torch.sigmoid(1.702 * gate) * (up + 1.0)
    if act == "situv2":
        situ_gate = (
            float(situ_beta) * torch.tanh(gate / float(situ_beta)) * torch.sigmoid(gate)
        )
        up_scaled = float(situ_linear_beta) * torch.tanh(up / float(situ_linear_beta))
        return situ_gate * up_scaled
    if swiglu_limit is not None:
        gate = gate.clamp(max=lim)
        up = up.clamp(min=-lim, max=lim)
    return torch.nn.functional.silu(gate) * up
