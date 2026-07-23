# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Minimal runtime projection for CPU-resolved FlyDSL operation plans.

This module only resolves compile-bound artifacts and dispatches ordered roles.
Tensor allocation, argument packing, grids, streams, and launches remain owned
by kernel-specific adapters.  It intentionally imports neither torch nor a
generic tensor/scheduling DSL.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .compile_plan import CompileContext, OperationNode, OperationPlan
from .launch_context import LaunchContext

__all__ = [
    "ExecutionStep",
    "RuntimeAdapter",
    "RuntimeAdapterRegistry",
    "execute_operation_plan",
    "resolve_execution_steps",
]


@dataclass(frozen=True)
class ExecutionStep:
    """One operation node paired with its resolved artifact, when any."""

    node: OperationNode
    artifact: Any | None


class RuntimeAdapter(Protocol):
    """Kernel-owned data-plane implementation for one operation role."""

    def __call__(
        self,
        step: ExecutionStep,
        state: Any,
        *,
        context: LaunchContext[Any],
    ) -> None: ...


class RuntimeAdapterRegistry:
    """Explicit role-to-adapter mapping with no graph-selection behavior."""

    def __init__(self) -> None:
        self._adapters: dict[str, RuntimeAdapter] = {}

    def register(
        self,
        role: str,
        adapter: RuntimeAdapter | None = None,
    ) -> Callable[[RuntimeAdapter], RuntimeAdapter] | RuntimeAdapter:
        if not isinstance(role, str) or not role or role != role.strip():
            raise ValueError("adapter role must be a non-empty canonical string")

        def register_one(value: RuntimeAdapter) -> RuntimeAdapter:
            if not callable(value):
                raise TypeError("runtime adapter must be callable")
            if role in self._adapters:
                raise ValueError(f"runtime adapter role {role!r} is already registered")
            self._adapters[role] = value
            return value

        return register_one if adapter is None else register_one(adapter)

    def lookup(self, role: str) -> RuntimeAdapter:
        try:
            return self._adapters[role]
        except KeyError as error:
            raise KeyError(
                f"no runtime adapter registered for role {role!r}"
            ) from error


def resolve_execution_steps(
    plan: OperationPlan,
    *,
    context: CompileContext[Any],
) -> tuple[ExecutionStep, ...]:
    """Resolve all compile-bound artifacts without performing data-plane work."""

    if not isinstance(plan, OperationPlan):
        raise TypeError(f"plan must be an OperationPlan, got {type(plan).__name__}")
    if not isinstance(context, CompileContext):
        raise TypeError(
            f"context must be a CompileContext, got {type(context).__name__}"
        )
    if plan.target != context.target:
        raise ValueError(
            f"operation plan target {plan.target!r} does not match "
            f"compile context target {context.target!r}"
        )

    steps = []
    resolved_bound_nodes = 0
    for node in plan.nodes:
        artifact = None
        if node.binding is not None:
            artifact = context.backend.resolve_aot(
                node.binding.unit,
                context=context,
            )
            resolved_bound_nodes += 1
        steps.append(ExecutionStep(node, artifact))

    if resolved_bound_nodes != len(plan.compile_projection().units):
        raise RuntimeError("execution projection silently lost a compile-bound node")
    return tuple(steps)


def execute_operation_plan(
    plan: OperationPlan,
    state: Any,
    *,
    compile_context: CompileContext[Any],
    launch_context: LaunchContext[Any],
    adapters: RuntimeAdapterRegistry,
) -> tuple[ExecutionStep, ...]:
    """Resolve and execute nodes in provider order through role adapters."""

    if not isinstance(launch_context, LaunchContext):
        raise TypeError(
            "launch_context must be a LaunchContext, "
            f"got {type(launch_context).__name__}"
        )
    if not isinstance(adapters, RuntimeAdapterRegistry):
        raise TypeError(
            f"adapters must be a RuntimeAdapterRegistry, got {type(adapters).__name__}"
        )

    steps = resolve_execution_steps(plan, context=compile_context)
    for step in steps:
        adapter = adapters.lookup(step.node.role)
        adapter(step, state, context=launch_context)
    return steps
