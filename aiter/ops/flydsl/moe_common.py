# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Common helpers and types shared across MoE FlyDSL kernel modules."""

from enum import Enum


def xcd_swizzle_workgroup_id(linear_id, num_workgroups, num_xcds, divide, minimum):
    """Map round-robin workgroup IDs into balanced contiguous XCD ranges."""
    workgroups_per_xcd = divide(num_workgroups, num_xcds)
    remainder = num_workgroups % num_xcds
    xcd_id = linear_id % num_xcds
    local_id = divide(linear_id, num_xcds)
    return xcd_id * workgroups_per_xcd + minimum(xcd_id, remainder) + local_id


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
