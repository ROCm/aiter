# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MoE sorting kernel — drop-in replacement for CK/Opus moe_sorting_fwd.

Provides `flydsl_moe_sorting_fwd()` with the same signature as
`aiter.moe_sorting_fwd()` so it can be used as a direct dispatch target
in `_moe_sorting_impl()`.

Workspace is pre-allocated here (not inside the kernel) so that CUDA graph
capture sees deterministic allocations.
"""

import torch

_workspace_cache = {}


def flydsl_moe_sorting_fwd(
    topk_ids,
    topk_weights,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    moe_buf,
    num_experts,
    unit_size,
    expert_mask=None,
    num_local_tokens=None,
):
    from .kernels.moe_sorting_kernel import (
        moe_sorting_flydsl,
        moe_sorting_get_workspace_size,
    )

    M = topk_ids.shape[0]
    topk = topk_ids.shape[1]
    device = topk_ids.device

    # Pre-allocate workspace (cached per device for CUDA graph compatibility).
    # A larger workspace can satisfy smaller requests, so we keep the largest seen.
    ws_size = moe_sorting_get_workspace_size(M, num_experts, topk, unit_size)
    workspace = None
    if ws_size > 0:
        workspace = _workspace_cache.get(device)
        if workspace is None or workspace.numel() < ws_size:
            workspace = torch.empty(ws_size, dtype=torch.int32, device=device)
            _workspace_cache[device] = workspace

    moe_sorting_flydsl(
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        num_experts,
        unit_size,
        expert_mask,
        num_local_tokens,
        workspace,
    )
