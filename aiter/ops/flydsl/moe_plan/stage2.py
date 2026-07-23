# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Single-source Stage2 and reduction graph semantics."""

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

MIXED_STAGE2_GEMM_OP_ID = "aiter.flydsl.moe.stage2.mixed_gemm.v1"
INT4_STAGE2_GEMM_OP_ID = "aiter.flydsl.moe.stage2.int4_gemm.v1"
PLAIN_REDUCTION_OP_ID = "aiter.flydsl.moe.stage2.reduction.plain.v1"
MASKED_REDUCTION_OP_ID = "aiter.flydsl.moe.stage2.reduction.masked.v1"


class MoeStage2Role(str, Enum):
    MIXED_GEMM = "moe.stage2.gemm.mixed"
    INT4_GEMM = "moe.stage2.gemm.int4"
    PLAIN_REDUCTION = "moe.stage2.reduction.plain"
    MASKED_REDUCTION = "moe.stage2.reduction.masked"


@dataclass(frozen=True)
class MoeStage2OperationCase:
    compile_kwargs: tuple[tuple[str, Any], ...]
    mode: str
    return_per_slot: bool
    persist: bool | None
    token_num: int
    routing_block_count: int | None
    use_mask: bool
    topk_ids_available: bool
    num_experts: int

    def __post_init__(self) -> None:
        items = tuple(tuple(item) for item in self.compile_kwargs)
        if any(len(item) != 2 for item in items):
            raise TypeError("compile_kwargs must contain (name, value) pairs")
        for name, value in items:
            if not isinstance(name, str) or not name:
                raise ValueError(f"invalid Stage2 compile argument name: {name!r}")
            try:
                hash(value)
            except TypeError as error:
                raise TypeError(
                    f"Stage2 compile argument {name!r} must be hashable"
                ) from error
        if len({name for name, _ in items}) != len(items):
            raise ValueError("Stage2 compile argument names must be unique")
        object.__setattr__(self, "compile_kwargs", items)

    @classmethod
    def from_kwargs(cls, kwargs: Mapping[str, Any], **semantics: Any):
        values = dict(kwargs)
        values.pop("accumulate", None)
        values.pop("persist_m", None)
        return cls(tuple(sorted(values.items())), **semantics)

    def as_kwargs(self) -> dict[str, Any]:
        return dict(self.compile_kwargs)


@dataclass(frozen=True)
class Stage2GemmMetadata:
    accumulate: bool
    return_per_slot: bool
    persist_m: int


@dataclass(frozen=True)
class Stage2ReductionMetadata:
    use_mask: bool


def _primary_op(
    plan: PlanBuilder,
    operations: dict[str, CompileOp],
    kwargs: Mapping[str, Any],
) -> CompileOp:
    try:
        a_dtype, b_dtype = kwargs["a_dtype"], kwargs["b_dtype"]
    except KeyError as error:
        raise TypeError(
            f"{plan.context}: missing required Stage2 argument: {error.args[0]}"
        ) from error
    if a_dtype in ("fp4", "fp8") and b_dtype in ("fp4", "fp8"):
        return operations[MIXED_STAGE2_GEMM_OP_ID]
    if a_dtype == "bf16" and b_dtype == "int4":
        return operations[INT4_STAGE2_GEMM_OP_ID]
    raise ValueError(
        f"{plan.context}: unsupported Stage2 dtype combination: "
        f"a_dtype={a_dtype!r}, b_dtype={b_dtype!r}"
    )


@operation_plan_provider
def resolve_moe_stage2_operation_plan(
    plan: PlanBuilder,
    case: MoeStage2OperationCase,
) -> None:
    if not isinstance(case, MoeStage2OperationCase):
        raise TypeError(
            f"{plan.context}: case must be a MoeStage2OperationCase, "
            f"got {type(case).__name__}"
        )
    from ..moe_compile_plan import _operations
    from ..moe_kernels import resolve_stage2_persist_m

    kwargs = case.as_kwargs()
    operations = _operations()
    primary = _primary_op(plan, operations, kwargs)
    requested = plan.bind(primary, kwargs, persist_m=1)
    for name in (
        "model_dim",
        "inter_dim",
        "experts",
        "topk",
        "tile_m",
        "tile_n",
        "tile_k",
    ):
        value = requested[name]
        plan.require(
            not isinstance(value, bool) and isinstance(value, int) and value > 0,
            f"{name} must be a positive integer, got {value!r}",
            operation=primary,
        )
    plan.require(
        isinstance(requested["doweight_stage2"], bool),
        "doweight_stage2 must explicitly identify route-weight ownership",
        operation=primary,
    )
    plan.require(
        isinstance(requested["enable_bias"], bool),
        f"enable_bias must be a bool, got {requested['enable_bias']!r}",
        operation=primary,
    )
    plan.require(
        case.mode in ("atomic", "reduce"),
        f"unsupported Stage2 mode: {case.mode!r}",
    )
    plan.require(
        isinstance(case.return_per_slot, bool),
        f"return_per_slot must be a bool, got {case.return_per_slot!r}",
    )
    accumulate = case.mode != "reduce" and not case.return_per_slot
    needs_reduction = not accumulate and not case.return_per_slot
    reduction_dtype = {
        "bf16": "bf16",
        "bfloat16": "bf16",
        "f16": "f16",
        "fp16": "f16",
        "half": "f16",
    }.get(str(requested["out_dtype"]).lower())
    plan.require(
        reduction_dtype is not None,
        f"unsupported Stage2 output dtype: {requested['out_dtype']!r}",
        operation=primary,
    )
    plan.require(
        not (primary.op_id == INT4_STAGE2_GEMM_OP_ID and requested["enable_bias"]),
        "bf16×int4 Stage2 does not support bias",
        operation=primary,
    )
    if case.use_mask:
        plan.require(needs_reduction, "masked reduction requires reduction output")
        plan.require(
            case.topk_ids_available,
            "masked reduction requires top-k-id semantics",
        )
        plan.require(
            case.num_experts > 0,
            "masked reduction requires a positive global expert count",
        )
    else:
        plan.require(
            not case.topk_ids_available,
            "top-k-id semantics are only valid for masked reduction",
        )
        plan.require(
            case.num_experts == 0,
            "plain reduction requires num_experts=0",
        )
    persist_m = resolve_stage2_persist_m(
        token_num=case.token_num,
        topk=requested["topk"],
        experts=case.num_experts if case.use_mask else requested["experts"],
        tile_m=requested["tile_m"],
        sort_block_m=requested["sort_block_m"],
        routing_block_count=case.routing_block_count,
        persist=case.persist,
        a_dtype=requested["a_dtype"],
    )
    primary_role = (
        MoeStage2Role.MIXED_GEMM
        if primary.op_id == MIXED_STAGE2_GEMM_OP_ID
        else MoeStage2Role.INT4_GEMM
    )
    gemm = plan.emit_node(
        "gemm",
        primary_role.value,
        requested,
        outputs=(
            OperationOutput(
                "stage2.per_slot" if needs_reduction else "stage2.output",
                (
                    OperationOutputKind.INTERMEDIATE
                    if needs_reduction
                    else OperationOutputKind.FINAL
                ),
            ),
        ),
        workspaces=(
            (OperationWorkspace("stage2.per_slot", WorkspaceLifetime.PLAN),)
            if needs_reduction
            else ()
        ),
        runtime_metadata=Stage2GemmMetadata(
            accumulate=accumulate,
            return_per_slot=case.return_per_slot,
            persist_m=persist_m,
        ),
        compile_overrides={"accumulate": accumulate, "persist_m": persist_m},
    )
    if not needs_reduction:
        return
    masked = case.use_mask
    plan.emit_node(
        "reduction",
        (
            MoeStage2Role.MASKED_REDUCTION.value
            if masked
            else MoeStage2Role.PLAIN_REDUCTION.value
        ),
        operations[MASKED_REDUCTION_OP_ID if masked else PLAIN_REDUCTION_OP_ID],
        requested,
        dependencies=(gemm.node_id,),
        outputs=(OperationOutput("stage2.output", OperationOutputKind.FINAL),),
        runtime_metadata=Stage2ReductionMetadata(use_mask=masked),
        compile_overrides={
            "dtype_str": reduction_dtype,
            "use_mask": masked,
            "num_experts": case.num_experts,
        },
    )
