# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Composition of explicit sorting, Stage1, and Stage2 operation cases."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ..compile_plan import CompileContext, OperationPlan
from .sorting import MoeSortingCompileCase, resolve_moe_sorting_operation_plan
from .stage1 import MoeStage1OperationCase, resolve_moe_stage1_operation_plan
from .stage2 import MoeStage2OperationCase, resolve_moe_stage2_operation_plan


@dataclass(frozen=True)
class MoeOperationCase:
    sorting: MoeSortingCompileCase | None
    stage1: MoeStage1OperationCase
    stage2: MoeStage2OperationCase


def _prefix(plan: OperationPlan, prefix: str, first_dependency: str | None):
    nodes = []
    for index, node in enumerate(plan.nodes):
        dependencies = tuple(f"{prefix}.{value}" for value in node.dependencies)
        if index == 0 and first_dependency is not None:
            dependencies = (*dependencies, first_dependency)
        nodes.append(
            replace(
                node,
                node_id=f"{prefix}.{node.node_id}",
                dependencies=dependencies,
            )
        )
    return tuple(nodes)


def resolve_moe_operation_plan(
    case: MoeOperationCase,
    *,
    context: CompileContext[Any],
) -> OperationPlan:
    """Compose explicit child plans without inferring sorting from stage rows."""

    if not isinstance(case, MoeOperationCase):
        raise TypeError(f"case must be a MoeOperationCase, got {type(case).__name__}")
    nodes = ()
    previous = None
    if case.sorting is not None:
        sorting = resolve_moe_sorting_operation_plan(case.sorting, context=context)
        sorting_nodes = _prefix(sorting, "sorting", None)
        nodes += sorting_nodes
        previous = sorting_nodes[-1].node_id
    stage1 = resolve_moe_stage1_operation_plan(case.stage1, context=context)
    stage1_nodes = _prefix(stage1, "stage1", previous)
    nodes += stage1_nodes
    previous = stage1_nodes[-1].node_id
    stage2 = resolve_moe_stage2_operation_plan(case.stage2, context=context)
    nodes += _prefix(stage2, "stage2", previous)
    return OperationPlan(context.target, case, nodes)
