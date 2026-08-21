# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Test-only A16W16 workspace and resolved-launch helpers."""

from __future__ import annotations

from collections.abc import Callable

import torch

from aiter.ops.opus.launch_plan import A16W16LaunchPlan
from aiter.ops.opus.gemm_op_a16w16 import _check_a16w16_launch_layout


def _init_a16w16_workspace(
    plan: A16W16LaunchPlan,
    XQ: torch.Tensor,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Materialize the immutable workspace spec for a test launch."""
    workspace_spec = plan.workspace_spec
    if workspace_spec is None:
        if workspace is not None:
            raise ValueError(
                "opus_gemm_a16w16_launch: "
                f"kid {plan.resolved_kid} does not use an external workspace"
            )
        return None

    if workspace is not None:
        return workspace
    return torch.empty(
        workspace_spec.shape,
        dtype=workspace_spec.dtype,
        device=XQ.device,
    )


def _launch_a16w16_with_torch_workspace(
    raw_launch: Callable[..., object],
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    bias: torch.Tensor | None,
    plan: A16W16LaunchPlan,
    *,
    workspace: torch.Tensor | None = None,
    _layout_checked: bool = False,
) -> torch.Tensor:
    """Prepare a test workspace and invoke a resolved test launcher."""
    if not _layout_checked:
        _check_a16w16_launch_layout(XQ, WQ, Y)
    workspace = _init_a16w16_workspace(plan, XQ, workspace)
    raw_launch(
        XQ,
        WQ,
        Y,
        bias,
        workspace,
        plan.resolved_kid,
        plan.abi_split_k,
    )
    return Y
