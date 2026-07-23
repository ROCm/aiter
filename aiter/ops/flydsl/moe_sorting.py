# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MoE sorting kernel — drop-in replacement for CK/Opus moe_sorting_fwd.

Provides `flydsl_moe_sorting_fwd()` with the same signature as
`aiter.moe_sorting_fwd()` so it can be used as a direct dispatch target
in `_moe_sorting_impl()`.

Workspace is pre-allocated here (not inside the kernel) so that CUDA graph
capture sees deterministic allocations.
"""

from dataclasses import dataclass
import os

import torch

from .moe_plan.sorting import (
    MoeSortingCompileCase,
    MoeSortingRole,
    resolve_moe_sorting_operation_plan,
)
from .operation_runtime import (
    ExecutionStep,
    RuntimeAdapterRegistry,
    execute_operation_plan,
)

_workspace_cache = {}


@dataclass
class _SortingExecutionState:
    args: tuple
    case: MoeSortingCompileCase
    device: object


def _sorting_runtime_adapter(step: ExecutionStep, state, *, context) -> None:
    from .kernels.moe_sorting_kernel import (
        SORTING_PATH_4K_FUSED,
        SORTING_PATH_ONESHOT,
        SORTING_PATH_P0V2_P23,
        moe_sorting_flydsl,
    )

    specialization = step.node.runtime_metadata
    expected_paths = {
        MoeSortingRole.ONESHOT.value: SORTING_PATH_ONESHOT,
        MoeSortingRole.P0V2_P23.value: SORTING_PATH_P0V2_P23,
        MoeSortingRole.K4_FUSED.value: SORTING_PATH_4K_FUSED,
    }
    expected_path = expected_paths[step.node.role]
    if specialization.path != expected_path:
        raise RuntimeError("sorting role disagrees with provider specialization")
    launcher = getattr(step.artifact, "launcher", step.artifact)
    if launcher is None:
        raise RuntimeError("sorting node requires a compiled artifact")

    case = state.case
    ws_size = 0
    if expected_path != SORTING_PATH_ONESHOT:
        mesh_stride = (
            (case.max_tokens + case.unit_size - 1) // case.unit_size
        ) * case.unit_size
        ws_mesh_bytes = case.num_experts * mesh_stride
        ws_size = (ws_mesh_bytes + 3) // 4 + case.num_experts + 1
    workspace = None
    if ws_size > 0:
        workspace = _workspace_cache.get(state.device)
        if workspace is None or workspace.numel() < ws_size:
            workspace = torch.empty(
                ws_size,
                dtype=torch.int32,
                device=state.device,
            )
            _workspace_cache[state.device] = workspace

    moe_sorting_flydsl(
        *state.args,
        workspace,
        launcher=launcher,
        specialization=specialization,
        cu_count=step.node.binding.unit.spec.target.cu_count,
        launch_context=context,
    )


_SORTING_RUNTIME_ADAPTERS = RuntimeAdapterRegistry()
for _role in MoeSortingRole:
    _SORTING_RUNTIME_ADAPTERS.register(_role.value, _sorting_runtime_adapter)


def _operation_plan_mode() -> str:
    mode = os.environ.get("AITER_FLYDSL_OPERATION_PLAN", "off").strip().lower()
    if mode not in ("off", "shadow", "on"):
        raise ValueError("AITER_FLYDSL_OPERATION_PLAN must be off|shadow|on")
    return mode


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
    from .launch_context import LaunchContext
    from .moe_compile_plan import resolve_moe_sorting_compile_plan
    from .kernels.moe_sorting_kernel import (
        moe_sorting_flydsl,
        moe_sorting_get_workspace_size,
        resolve_moe_sorting_specialization,
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
    mode = _operation_plan_mode()
    operation_plan = (
        resolve_moe_sorting_operation_plan(case, context=compile_context)
        if mode in ("shadow", "on")
        else None
    )
    if mode == "on":
        state = _SortingExecutionState(
            (
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
            ),
            case,
            device,
        )
        execute_operation_plan(
            operation_plan,
            state,
            compile_context=compile_context,
            launch_context=launch_context,
            adapters=_SORTING_RUNTIME_ADAPTERS,
        )
        return

    plan = resolve_moe_sorting_compile_plan(case, context=compile_context)
    if len(plan.units) != 1:
        raise RuntimeError(
            f"sorting CompilePlan must contain one concrete unit, got {len(plan.units)}"
        )
    unit = plan.units[0]
    artifact = compile_context.backend.resolve_aot(
        unit,
        context=compile_context,
    )
    launcher = getattr(artifact, "launcher", artifact)
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

    # Pre-allocate workspace (cached per device for CUDA graph compatibility).
    # A larger workspace can satisfy smaller requests, so we keep the largest seen.
    ws_size = moe_sorting_get_workspace_size(
        max_tokens,
        num_experts,
        topk,
        unit_size,
        arch=compile_context.target.arch,
        has_mask=case.has_mask,
        path=specialization.path,
        k4_block=specialization.k4_block,
    )
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
    if mode == "shadow":
        expected = operation_plan.nodes[0]
        if (
            expected.binding.unit.spec.op_id != unit.spec.op_id
            or expected.runtime_metadata != specialization
        ):
            raise RuntimeError("sorting OperationPlan shadow mismatch")
