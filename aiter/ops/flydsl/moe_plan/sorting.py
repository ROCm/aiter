# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Single-source explicit FlyDSL sorting specialization and roles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..compile_plan import (
    OperationOutput,
    OperationOutputKind,
    OperationWorkspace,
    PlanBuilder,
    WorkspaceLifetime,
    operation_plan_provider,
)

SORTING_ONESHOT_OP_ID = "aiter.flydsl.moe.sorting.oneshot.v1"
SORTING_P0V2_P23_OP_ID = "aiter.flydsl.moe.sorting.multiphase.p0v2_p23.v1"
SORTING_4K_FUSED_OP_ID = "aiter.flydsl.moe.sorting.multiphase.k4_fused.v1"


class MoeSortingRole(str, Enum):
    ONESHOT = "moe.sorting.oneshot"
    P0V2_P23 = "moe.sorting.p0v2_p23"
    K4_FUSED = "moe.sorting.4k_fused"


@dataclass(frozen=True)
class MoeSortingCompileCase:
    max_tokens: int
    num_experts: int
    topk: int
    has_mask: bool
    unit_size: int = 32
    path: str | None = None
    k4_block: int | None = None


@operation_plan_provider
def resolve_moe_sorting_operation_plan(
    plan: PlanBuilder,
    case: MoeSortingCompileCase,
) -> None:
    if not isinstance(case, MoeSortingCompileCase):
        raise TypeError(
            f"{plan.context}: case must be a MoeSortingCompileCase, "
            f"got {type(case).__name__}"
        )
    from ..kernels.moe_sorting_kernel import (
        SORTING_PATH_4K_FUSED,
        SORTING_PATH_ONESHOT,
        SORTING_PATH_P0V2_P23,
        resolve_moe_sorting_specialization,
    )
    from ..moe_compile_plan import _sorting_operations

    specialization = resolve_moe_sorting_specialization(
        arch=plan.target.arch,
        max_tokens=case.max_tokens,
        num_experts=case.num_experts,
        topk=case.topk,
        unit_size=case.unit_size,
        has_mask=case.has_mask,
        path=case.path,
        k4_block=case.k4_block,
    )
    operations = _sorting_operations()
    if specialization.path == SORTING_PATH_ONESHOT:
        operation = operations[SORTING_ONESHOT_OP_ID]
        role = MoeSortingRole.ONESHOT
        overrides = {"max_tokens": specialization.launcher_max_tokens}
        workspaces = ()
    elif specialization.path == SORTING_PATH_P0V2_P23:
        operation = operations[SORTING_P0V2_P23_OP_ID]
        role = MoeSortingRole.P0V2_P23
        overrides = {"k4_block": specialization.k4_block}
        workspaces = (OperationWorkspace("sorting.workspace", WorkspaceLifetime.PLAN),)
    elif specialization.path == SORTING_PATH_4K_FUSED:
        operation = operations[SORTING_4K_FUSED_OP_ID]
        role = MoeSortingRole.K4_FUSED
        overrides = {"k4_block": specialization.k4_block}
        workspaces = (OperationWorkspace("sorting.workspace", WorkspaceLifetime.PLAN),)
    else:
        raise RuntimeError(
            f"{plan.context}: unhandled sorting path {specialization.path!r}"
        )
    plan.emit_node(
        "sorting",
        role.value,
        operation,
        case,
        outputs=(OperationOutput("sorting.outputs", OperationOutputKind.FINAL),),
        workspaces=workspaces,
        runtime_metadata=specialization,
        compile_overrides=overrides,
    )
