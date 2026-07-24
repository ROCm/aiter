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
    *,
    compile_context=None,
    launch_context=None,
):
    from .aot_backend import create_runtime_compile_context
    from .kernels.moe_sorting_kernel import (
        SORTING_PATH_ONESHOT,
        moe_sorting_flydsl,
        resolve_moe_sorting_specialization,
    )
    from .launch_context import LaunchContext
    from .moe_compile_plan import (
        MoeSortingCompileCase,
        resolve_moe_sorting_compile_plan,
    )

    max_tokens = int(topk_ids.shape[0])
    topk = int(topk_ids.shape[1])
    device = topk_ids.device
    if compile_context is None:
        compile_context = create_runtime_compile_context(device)
    if launch_context is None:
        launch_context = LaunchContext(torch.cuda.current_stream(device))

    case = MoeSortingCompileCase(
        max_tokens=max_tokens,
        num_experts=int(num_experts),
        topk=topk,
        unit_size=int(unit_size),
        has_mask=expert_mask is not None,
    )
    specialization = resolve_moe_sorting_specialization(
        arch=compile_context.target.arch,
        max_tokens=case.max_tokens,
        num_experts=case.num_experts,
        topk=case.topk,
        unit_size=case.unit_size,
        has_mask=case.has_mask,
        path=case.path,
        k4_block=case.k4_block,
    )
    plan = resolve_moe_sorting_compile_plan(
        case,
        context=compile_context,
        specialization=specialization,
    )
    if len(plan.units) != 1:
        raise RuntimeError(
            f"sorting CompilePlan must contain one unit, got {len(plan.units)}"
        )
    unit = plan.units[0]
    artifact = compile_context.backend.resolve_aot(
        unit,
        context=compile_context,
    )
    launcher = getattr(artifact, "launcher", artifact)

    ws_size = 0
    if specialization.path != SORTING_PATH_ONESHOT:
        mesh_stride = (
            (case.max_tokens + case.unit_size - 1) // case.unit_size
        ) * case.unit_size
        ws_mesh_bytes = case.num_experts * mesh_stride
        ws_size = (ws_mesh_bytes + 3) // 4 + case.num_experts + 1
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
        launcher=launcher,
        specialization=specialization,
        cu_count=compile_context.target.cu_count,
        launch_context=launch_context,
    )
