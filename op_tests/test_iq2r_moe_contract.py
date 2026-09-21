# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest

from aiter.iq2r_moe import IQ2RMoeWorkspace
from aiter.ops.iq2r import iq2r_task_capacity


def test_glm53_workspace_uses_independent_hidden_and_intermediate_widths():
    workspace = IQ2RMoeWorkspace.allocate(
        2,
        8,
        device="meta",
        max_experts=288,
        hidden_size=4096,
        intermediate_size=2048,
    )

    assert workspace.max_routes == 16
    assert tuple(workspace.route_input_fp8.shape) == (16, 4096)
    assert tuple(workspace.route_input_scales.shape) == (16, 128)
    assert tuple(workspace.gate_up.shape) == (16, 4096)
    assert tuple(workspace.intermediate_fp8.shape) == (16, 2048)
    assert tuple(workspace.intermediate_scales.shape) == (16, 64)
    assert tuple(workspace.route_output.shape) == (16, 4096)
    assert tuple(workspace.topk_ids.shape) == (2, 8)


def test_glm53_tile16_workspace_scale_shapes():
    workspace = IQ2RMoeWorkspace.allocate(
        3,
        8,
        device="meta",
        max_experts=288,
        hidden_size=4096,
        intermediate_size=2048,
        scale_layout="tile16",
    )

    assert tuple(workspace.route_input_scales.shape) == (32, 2, 4, 16)
    assert tuple(workspace.intermediate_scales.shape) == (16, 2, 4, 16)


def test_iq2r_task_capacity_supports_glm53_expert_count():
    assert iq2r_task_capacity(16, 288, 16) == 17
    assert iq2r_task_capacity(4096, 288, 16) == 545
    assert iq2r_task_capacity(131072, 288, 16) == 8481
    with pytest.raises(ValueError, match="at most 512"):
        iq2r_task_capacity(16, 513, 16)


def test_glm53_default_atom_workspace_supports_16k_tokens():
    workspace = IQ2RMoeWorkspace.allocate(
        16384,
        8,
        device="meta",
        max_experts=288,
        hidden_size=4096,
        intermediate_size=2048,
    )

    assert workspace.max_routes == 131072
    assert tuple(workspace.tasks.shape) == (8481, 3)
