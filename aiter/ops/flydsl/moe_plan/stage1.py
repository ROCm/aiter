# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Single-source Stage1 graph semantics for compile and execution projections."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..compile_plan import (
    CompileOp,
    OperationOutput,
    OperationOutputKind,
    OperationWorkspace,
    PlanBuilder,
    WorkspaceLifetime,
    operation_plan_provider,
)

MIXED_STAGE1_GEMM_OP_ID = "aiter.flydsl.moe.stage1.mixed_gemm.v1"
INT4_STAGE1_GEMM_OP_ID = "aiter.flydsl.moe.stage1.int4_gemm.v1"
FQ_ACTIVATION_OP_ID = "aiter.flydsl.moe.stage1.silu_and_mul_fq.v1"


class MoeStage1Role(str, Enum):
    """Runtime roles selected only by the Stage1 provider."""

    MIXED_GEMM = "moe.stage1.gemm.mixed"
    INT4_GEMM = "moe.stage1.gemm.int4"
    FQ_POSTPROCESS = "moe.stage1.postprocess.fq"
    EXTERNAL_POSTPROCESS = "moe.stage1.postprocess.external"


@dataclass(frozen=True)
class MoeStage1OperationCase:
    """Compact immutable wrapper around existing Stage1 compiler kwargs."""

    compile_kwargs: tuple[tuple[str, Any], ...]

    def __post_init__(self) -> None:
        try:
            items = tuple(tuple(item) for item in self.compile_kwargs)
        except TypeError as error:
            raise TypeError(
                "compile_kwargs must contain (name, value) pairs"
            ) from error
        if any(len(item) != 2 for item in items):
            raise TypeError("compile_kwargs must contain (name, value) pairs")

        names = []
        for name, value in items:
            if not isinstance(name, str) or not name:
                raise ValueError(f"invalid Stage1 compile argument name: {name!r}")
            try:
                hash(value)
            except TypeError as error:
                raise TypeError(
                    f"Stage1 compile argument {name!r} must be hashable"
                ) from error
            names.append(name)
        if len(names) != len(set(names)):
            raise ValueError("Stage1 compile argument names must be unique")
        object.__setattr__(self, "compile_kwargs", items)

    @classmethod
    def from_kwargs(
        cls,
        kwargs: Mapping[str, Any],
    ) -> "MoeStage1OperationCase":
        if not isinstance(kwargs, Mapping):
            raise TypeError("Stage1 compiler kwargs must be a mapping")
        return cls(tuple(sorted(kwargs.items())))

    def as_kwargs(self) -> dict[str, Any]:
        return dict(self.compile_kwargs)


@dataclass(frozen=True)
class Stage1GemmMetadata:
    """Provider-derived facts needed by Stage1 data-plane preparation."""

    split_k: bool
    requested_out_dtype: str


@dataclass(frozen=True)
class Stage1FqMetadata:
    """Compile-time FQ specialization selected for the postprocess node."""

    quant_mode: str
    gui_layout: bool


@dataclass(frozen=True)
class Stage1ExternalPostprocessMetadata:
    """Existing CK/HIP postprocess selection for a runtime-only node."""

    act: str
    enable_bias: bool


def _primary_op(
    plan: PlanBuilder,
    operations: dict[str, CompileOp],
    builder_kwargs: Mapping[str, Any],
) -> CompileOp:
    try:
        a_dtype = builder_kwargs["a_dtype"]
        b_dtype = builder_kwargs["b_dtype"]
    except KeyError as error:
        raise TypeError(
            f"{plan.context}: missing required Stage1 argument: {error.args[0]}"
        ) from error
    if b_dtype in ("fp4", "fp8"):
        return operations[MIXED_STAGE1_GEMM_OP_ID]
    if a_dtype == "bf16" and b_dtype == "int4":
        return operations[INT4_STAGE1_GEMM_OP_ID]
    raise ValueError(
        f"{plan.context}: unsupported Stage1 dtype combination: "
        f"a_dtype={a_dtype!r}, b_dtype={b_dtype!r}"
    )


@operation_plan_provider
def resolve_moe_stage1_operation_plan(
    plan: PlanBuilder,
    case: MoeStage1OperationCase,
) -> None:
    """Resolve the complete Stage1 artifact and runtime postprocess graph."""

    if not isinstance(case, MoeStage1OperationCase):
        raise TypeError(
            f"{plan.context}: case must be a MoeStage1OperationCase, "
            f"got {type(case).__name__}"
        )

    # Imported lazily to keep this operation-specific module independent from
    # torch/FlyDSL imports and avoid duplicating shared op declarations.
    from ..moe_compile_plan import _operations

    builder_kwargs = case.as_kwargs()
    operations = _operations()
    primary = _primary_op(plan, operations, builder_kwargs)
    requested = plan.bind(primary, builder_kwargs)

    k_batch = requested["k_batch"]
    plan.require(
        not isinstance(k_batch, bool) and isinstance(k_batch, int) and k_batch > 0,
        f"k_batch must be a positive integer, got {k_batch!r}",
        operation=primary,
    )
    is_splitk = k_batch > 1

    if primary.op_id == MIXED_STAGE1_GEMM_OP_ID:
        gate_mode = requested["gate_mode"]
        plan.require(
            gate_mode != "gate_only",
            "gate_only is reserved and unsupported",
            operation=primary,
        )
        plan.require(
            gate_mode != "mock_gate_only" or is_splitk,
            "mock_gate_only requires split-K",
            operation=primary,
        )
        plan.require(
            not (
                is_splitk
                and requested["out_dtype"] == "fp8"
                and gate_mode != "interleave"
            ),
            "split-K fp8 output requires an interleaved layout",
            operation=primary,
        )

    requested_out_dtype = requested["out_dtype"]
    primary_role = (
        MoeStage1Role.MIXED_GEMM
        if primary.op_id == MIXED_STAGE1_GEMM_OP_ID
        else MoeStage1Role.INT4_GEMM
    )
    gemm = plan.emit_node(
        "gemm",
        primary_role.value,
        requested,
        outputs=(
            OperationOutput(
                "stage1.gate_up_partials" if is_splitk else "stage1.output",
                (
                    OperationOutputKind.INTERMEDIATE
                    if is_splitk
                    else OperationOutputKind.FINAL
                ),
            ),
        ),
        workspaces=(
            (
                OperationWorkspace(
                    "stage1.splitk_partial",
                    WorkspaceLifetime.PLAN,
                ),
            )
            if is_splitk
            else ()
        ),
        runtime_metadata=Stage1GemmMetadata(
            split_k=is_splitk,
            requested_out_dtype=requested_out_dtype,
        ),
        compile_overrides=(
            {
                "out_dtype": (
                    "bf16"
                    if requested_out_dtype in ("fp4", "fp8")
                    else requested_out_dtype
                ),
                "enable_bias": False,
            }
            if is_splitk
            else None
        ),
    )
    if not is_splitk:
        return

    helper = operations[FQ_ACTIVATION_OP_ID]
    if primary.op_id == MIXED_STAGE1_GEMM_OP_ID:
        gate_mode = requested["gate_mode"]
        if gate_mode == "interleave":
            quant_mode = (
                requested_out_dtype if requested_out_dtype in ("fp4", "fp8") else "none"
            )
            plan.emit_node(
                "postprocess",
                MoeStage1Role.FQ_POSTPROCESS.value,
                helper,
                requested,
                dependencies=(gemm.node_id,),
                outputs=(OperationOutput("stage1.output", OperationOutputKind.FINAL),),
                runtime_metadata=Stage1FqMetadata(
                    quant_mode=quant_mode,
                    gui_layout=True,
                ),
                compile_overrides={
                    "quant_mode": quant_mode,
                    "gui_layout": True,
                },
            )
            return
        if requested_out_dtype == "fp4":
            plan.emit_node(
                "postprocess",
                MoeStage1Role.FQ_POSTPROCESS.value,
                helper,
                requested,
                dependencies=(gemm.node_id,),
                outputs=(OperationOutput("stage1.output", OperationOutputKind.FINAL),),
                runtime_metadata=Stage1FqMetadata(
                    quant_mode="fp4",
                    gui_layout=False,
                ),
                compile_overrides={
                    "quant_mode": "fp4",
                    "gui_layout": False,
                },
            )
            return

    plan.emit_node(
        "postprocess",
        MoeStage1Role.EXTERNAL_POSTPROCESS.value,
        dependencies=(gemm.node_id,),
        outputs=(OperationOutput("stage1.output", OperationOutputKind.FINAL),),
        runtime_metadata=Stage1ExternalPostprocessMetadata(
            act=requested["act"],
            enable_bias=requested["enable_bias"],
        ),
    )
