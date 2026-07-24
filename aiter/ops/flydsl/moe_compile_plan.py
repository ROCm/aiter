# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MoE FlyDSL CompilePlan resolution using existing compiler signatures.

There is deliberately no typed mirror of MoE builder parameters here.
``CompileOpRegistry.make_unit`` binds directly to the existing compiler
wrappers, so their signatures and defaults remain the only callable schema.
This module owns only compile-unit ordering and the launch ABI that FlyDSL cannot
currently expose.

Importing this module does not compile or launch anything. The existing
``moe_kernels`` host module (and therefore torch/FlyDSL Python modules) is
imported lazily when an operation set is first registered.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .compile_plan import (
    ArgumentKind,
    CompileContext,
    CompileOp,
    CompileOpRegistry,
    CompilePlan,
    DEFAULT_COMPILE_OP_REGISTRY,
    KernelSignature,
    PlanBuilder,
    SignatureArg,
    op,
    plan_provider,
)
from .moe_compile_decisions import (
    Stage1CompileDecision,
    Stage2CompileDecision,
    resolve_stage1_compile_decision,
    resolve_stage2_compile_decision,
)

MIXED_STAGE1_GEMM_OP_ID = "aiter.flydsl.moe.stage1.mixed_gemm.v1"
INT4_STAGE1_GEMM_OP_ID = "aiter.flydsl.moe.stage1.int4_gemm.v1"
FQ_ACTIVATION_OP_ID = "aiter.flydsl.moe.stage1.silu_and_mul_fq.v1"
CKTILE_SWIGLU_AND_MUL_OP_ID = "aiter.flydsl.moe.stage1.cktile_swiglu_and_mul.v1"
MIXED_STAGE2_GEMM_OP_ID = "aiter.flydsl.moe.stage2.mixed_gemm.v1"
INT4_STAGE2_GEMM_OP_ID = "aiter.flydsl.moe.stage2.int4_gemm.v1"
PLAIN_REDUCTION_OP_ID = "aiter.flydsl.moe.stage2.reduction.plain.v1"
MASKED_REDUCTION_OP_ID = "aiter.flydsl.moe.stage2.reduction.masked.v1"
SORTING_ONESHOT_OP_ID = "aiter.flydsl.moe.sorting.oneshot.v1"
SORTING_P0V2_P23_OP_ID = "aiter.flydsl.moe.sorting.multiphase.p0v2_p23.v1"
SORTING_4K_FUSED_OP_ID = "aiter.flydsl.moe.sorting.multiphase.k4_fused.v1"

__all__ = [
    "CKTILE_SWIGLU_AND_MUL_OP_ID",
    "FQ_ACTIVATION_OP_ID",
    "INT4_STAGE1_GEMM_OP_ID",
    "INT4_STAGE2_GEMM_OP_ID",
    "MASKED_REDUCTION_OP_ID",
    "MIXED_STAGE1_GEMM_OP_ID",
    "MIXED_STAGE2_GEMM_OP_ID",
    "MoeCompilePlanCase",
    "MoeSortingCompileCase",
    "PLAIN_REDUCTION_OP_ID",
    "SORTING_4K_FUSED_OP_ID",
    "SORTING_ONESHOT_OP_ID",
    "SORTING_P0V2_P23_OP_ID",
    "register_moe_stage1_ops",
    "register_moe_stage2_ops",
    "register_moe_sorting_ops",
    "resolve_cktile_stage1_compile_plan",
    "resolve_moe_compile_plan",
    "resolve_moe_sorting_compile_plan",
    "resolve_moe_stage1_compile_plan",
    "resolve_moe_stage2_compile_plan",
    "sorting_abi",
    "stage1_abi",
    "stage2_abi",
]


@dataclass(frozen=True)
class MoeSortingCompileCase:
    """Explicit CPU metadata for one independently cached sorting launcher."""

    max_tokens: int
    num_experts: int
    topk: int
    has_mask: bool
    unit_size: int = 32
    path: str | None = None
    k4_block: int | None = None


@dataclass(frozen=True)
class MoeCompilePlanCase:
    """Already-bound stage plans plus an explicitly optional sorting case."""

    sorting: MoeSortingCompileCase | None
    stage1: CompilePlan
    stage2: CompilePlan


def _abi(
    pointers: str = "",
    i32: str = "",
    f32: str = "",
    tensors: tuple[SignatureArg, ...] = (),
) -> KernelSignature:
    # ptr_arg() materializes an fx.Pointer<Uint8>, regardless of tensor dtype.
    groups = (
        (pointers, ArgumentKind.POINTER, "u8"),
        (i32, ArgumentKind.SCALAR, "i32"),
        (f32, ArgumentKind.SCALAR, "f32"),
    )
    arguments = tuple(
        SignatureArg(name, kind, dtype)
        for names, kind, dtype in groups
        for name in names.split()
    )
    return KernelSignature(
        tensors + arguments + (SignatureArg("stream", ArgumentKind.STREAM),)
    )


_MIXED_GEMM_ABI = _abi(
    pointers="""
        arg_out arg_x arg_w arg_scale_x arg_scale_w arg_sorted_token_ids
        arg_expert_ids arg_sorted_weights arg_max_token_ids arg_bias
        arg_out_scale_sorted
    """,
    i32="i32_tokens_in i32_inter_in i32_k_in i32_size_expert_ids_in",
    f32="f32_swiglu_limit",
)

_INT4_GEMM_ABI = _abi(
    pointers="""
        arg_out arg_x arg_w arg_scale_x arg_scale_w arg_sorted_token_ids
        arg_expert_ids arg_sorted_weights arg_max_token_ids
    """,
    i32="i32_tokens_in i32_inter_in i32_k_in i32_size_expert_ids_in",
)

_FQ_ACTIVATION_ABI = _abi(
    pointers="""
        x out_buf out_scale_sorted sorted_ids num_valid_ids topk_ids bias
    """,
    i32="token_num num_sorted_rows",
    f32="swiglu_limit_f",
)

_SWIGLU_EPILOGUE_ABI = _abi(
    i32="num_rows",
    tensors=(
        SignatureArg("x", ArgumentKind.TENSOR, "bf16", (None, None), (None, 1)),
        SignatureArg("out", ArgumentKind.TENSOR, "bf16", (None, None), (None, 1)),
    ),
)

_MIXED_STAGE2_GEMM_ABI = _abi(
    pointers="""
        arg_out arg_x arg_w arg_scale_x arg_scale_w arg_sorted_token_ids
        arg_expert_ids arg_sorted_weights arg_num_valid_ids arg_bias
    """,
    i32="i32_tokens_in i32_n_in i32_k_in i32_size_expert_ids_in",
)

_INT4_STAGE2_GEMM_ABI = _abi(
    pointers="""
        arg_out arg_x arg_w arg_scale_x arg_scale_w arg_sorted_token_ids
        arg_expert_ids arg_sorted_weights arg_num_valid_ids
    """,
    i32="i32_tokens_in i32_n_in i32_k_in i32_size_expert_ids_in",
)

_REDUCTION_ABI = _abi(
    pointers="X Y expert_mask topk_ids",
    i32="i32_m_tokens",
)


def _tensor(name: str, dtype: str, rank: int) -> SignatureArg:
    if rank == 1:
        return SignatureArg(name, ArgumentKind.TENSOR, dtype, (None,), (1,))
    if rank == 2:
        return SignatureArg(
            name,
            ArgumentKind.TENSOR,
            dtype,
            (None, None),
            (None, 1),
        )
    raise ValueError(f"unsupported sorting tensor rank: {rank}")


_SORTING_COMMON_TENSORS = (
    _tensor("topk_ids_tensor", "i32", 2),
    _tensor("topk_weights_tensor", "f32", 2),
    _tensor("sorted_token_ids", "i32", 1),
    _tensor("sorted_weights_out", "f32", 1),
    _tensor("sorted_expert_ids", "i32", 1),
    _tensor("num_valid_ids_out", "i32", 1),
    _tensor("moe_buf", "i32", 2),
    _tensor("expert_mask_tensor", "i32", 1),
)

_SORTING_ONESHOT_ABI = _abi(
    tensors=_SORTING_COMMON_TENSORS,
    i32="i32_tokens i32_moe_buf_elems n_grid_blocks",
)

_SORTING_MULTIPHASE_TENSORS = (
    _tensor("topk_ids", "i32", 2),
    _tensor("workspace", "i32", 1),
    *_SORTING_COMMON_TENSORS[1:],
)

_SORTING_P0V2_P23_ABI = _abi(
    tensors=_SORTING_MULTIPHASE_TENSORS,
    i32="""
        i32_tokens i32_mesh_stride i32_mesh_size i32_moe_buf_elems
        n_grid_p23
    """,
)

_SORTING_4K_FUSED_ABI = _abi(
    tensors=_SORTING_MULTIPHASE_TENSORS,
    i32="""
        i32_tokens i32_mesh_stride i32_mesh_size i32_moe_buf_elems
        i32_ws_total i32_p0_niters n_grid_k1 n_grid_k2 n_grid_p23
    """,
)

_ABIS = {
    MIXED_STAGE1_GEMM_OP_ID: _MIXED_GEMM_ABI,
    INT4_STAGE1_GEMM_OP_ID: _INT4_GEMM_ABI,
    FQ_ACTIVATION_OP_ID: _FQ_ACTIVATION_ABI,
    CKTILE_SWIGLU_AND_MUL_OP_ID: _SWIGLU_EPILOGUE_ABI,
    MIXED_STAGE2_GEMM_OP_ID: _MIXED_STAGE2_GEMM_ABI,
    INT4_STAGE2_GEMM_OP_ID: _INT4_STAGE2_GEMM_ABI,
    PLAIN_REDUCTION_OP_ID: _REDUCTION_ABI,
    MASKED_REDUCTION_OP_ID: _REDUCTION_ABI,
    SORTING_ONESHOT_OP_ID: _SORTING_ONESHOT_ABI,
    SORTING_P0V2_P23_OP_ID: _SORTING_P0V2_P23_ABI,
    SORTING_4K_FUSED_OP_ID: _SORTING_4K_FUSED_ABI,
}


def stage1_abi(op_id: str) -> KernelSignature:
    """Return the isolated manual ABI for one stable Stage1 operation."""

    try:
        return _ABIS[op_id]
    except KeyError as error:
        raise KeyError(f"unknown Stage1 op_id: {op_id!r}") from error


def stage2_abi(op_id: str) -> KernelSignature:
    """Return the isolated manual ABI for one stable Stage2 operation."""

    if op_id not in (
        MIXED_STAGE2_GEMM_OP_ID,
        INT4_STAGE2_GEMM_OP_ID,
        PLAIN_REDUCTION_OP_ID,
        MASKED_REDUCTION_OP_ID,
    ):
        raise KeyError(f"unknown Stage2 op_id: {op_id!r}")
    return _ABIS[op_id]


def sorting_abi(op_id: str) -> KernelSignature:
    """Return the exact ABI for one concrete sorting launcher family."""

    if op_id not in (
        SORTING_ONESHOT_OP_ID,
        SORTING_P0V2_P23_OP_ID,
        SORTING_4K_FUSED_OP_ID,
    ):
        raise KeyError(f"unknown sorting op_id: {op_id!r}")
    return _ABIS[op_id]


@lru_cache(maxsize=1)
def _declared_ops() -> tuple[CompileOp, ...]:
    """Declare MoE ops lazily so importing this module stays CPU-light."""

    from .moe_kernels import (
        compile_flydsl_moe_reduction,
        _get_compiled_silu_fused,
        _get_compiled_swiglu,
        compile_flydsl_moe_stage1,
        compile_flydsl_moe_stage2,
    )

    declarations = (
        (MIXED_STAGE1_GEMM_OP_ID, compile_flydsl_moe_stage1, _MIXED_GEMM_ABI),
        (INT4_STAGE1_GEMM_OP_ID, compile_flydsl_moe_stage1, _INT4_GEMM_ABI),
        (FQ_ACTIVATION_OP_ID, _get_compiled_silu_fused, _FQ_ACTIVATION_ABI),
        (
            CKTILE_SWIGLU_AND_MUL_OP_ID,
            _get_compiled_swiglu,
            _SWIGLU_EPILOGUE_ABI,
        ),
        (
            MIXED_STAGE2_GEMM_OP_ID,
            compile_flydsl_moe_stage2,
            _MIXED_STAGE2_GEMM_ABI,
        ),
        (
            INT4_STAGE2_GEMM_OP_ID,
            compile_flydsl_moe_stage2,
            _INT4_STAGE2_GEMM_ABI,
        ),
        (
            PLAIN_REDUCTION_OP_ID,
            compile_flydsl_moe_reduction,
            _REDUCTION_ABI,
        ),
        (
            MASKED_REDUCTION_OP_ID,
            compile_flydsl_moe_reduction,
            _REDUCTION_ABI,
        ),
    )
    return tuple(op(op_id, compiler, abi=abi) for op_id, compiler, abi in declarations)


def _operations() -> dict[str, CompileOp]:
    return {operation.op_id: operation for operation in _declared_ops()}


@lru_cache(maxsize=1)
def _declared_sorting_ops() -> tuple[CompileOp, ...]:
    """Declare concrete sorting launchers without invoking their builders."""

    from .kernels.moe_sorting_kernel import (
        compile_moe_sorting_4k_fused,
        compile_moe_sorting_oneshot,
        compile_moe_sorting_p0v2_p23,
    )

    declarations = (
        (
            SORTING_ONESHOT_OP_ID,
            compile_moe_sorting_oneshot,
            _SORTING_ONESHOT_ABI,
        ),
        (
            SORTING_P0V2_P23_OP_ID,
            compile_moe_sorting_p0v2_p23,
            _SORTING_P0V2_P23_ABI,
        ),
        (
            SORTING_4K_FUSED_OP_ID,
            compile_moe_sorting_4k_fused,
            _SORTING_4K_FUSED_ABI,
        ),
    )
    return tuple(op(op_id, compiler, abi=abi) for op_id, compiler, abi in declarations)


def _sorting_operations() -> dict[str, CompileOp]:
    return {operation.op_id: operation for operation in _declared_sorting_ops()}


def register_moe_stage1_ops(
    registry: CompileOpRegistry = DEFAULT_COMPILE_OP_REGISTRY,
) -> CompileOpRegistry:
    """Register all Stage1 declarations with ``registry``."""

    stage1_op_ids = {
        MIXED_STAGE1_GEMM_OP_ID,
        INT4_STAGE1_GEMM_OP_ID,
        FQ_ACTIVATION_OP_ID,
        CKTILE_SWIGLU_AND_MUL_OP_ID,
    }
    for operation in _declared_ops():
        if operation.op_id in stage1_op_ids:
            operation.register(registry)
    return registry


def register_moe_stage2_ops(
    registry: CompileOpRegistry = DEFAULT_COMPILE_OP_REGISTRY,
) -> CompileOpRegistry:
    """Register all Stage2 declarations with ``registry``."""

    stage2_op_ids = {
        MIXED_STAGE2_GEMM_OP_ID,
        INT4_STAGE2_GEMM_OP_ID,
        PLAIN_REDUCTION_OP_ID,
        MASKED_REDUCTION_OP_ID,
    }
    for operation in _declared_ops():
        if operation.op_id in stage2_op_ids:
            operation.register(registry)
    return registry


def register_moe_sorting_ops(
    registry: CompileOpRegistry = DEFAULT_COMPILE_OP_REGISTRY,
) -> CompileOpRegistry:
    """Register all concrete sorting launcher declarations."""

    for operation in _declared_sorting_ops():
        operation.register(registry)
    return registry


@plan_provider
def _resolve_moe_sorting_compile_plan(
    plan: PlanBuilder,
    case: MoeSortingCompileCase,
    *,
    specialization: Any,
) -> None:
    from .kernels.moe_sorting_kernel import (
        SORTING_PATH_4K_FUSED,
        SORTING_PATH_ONESHOT,
        SORTING_PATH_P0V2_P23,
        MoeSortingSpecialization,
    )

    if not isinstance(case, MoeSortingCompileCase):
        raise TypeError(
            f"{plan.context}: case must be a MoeSortingCompileCase, "
            f"got {type(case).__name__}"
        )
    if not isinstance(specialization, MoeSortingSpecialization):
        raise TypeError("specialization must be a MoeSortingSpecialization")
    if (
        specialization.max_tokens != case.max_tokens
        or specialization.has_mask != case.has_mask
    ):
        raise ValueError("sorting specialization disagrees with compile case")

    operations = _sorting_operations()
    if specialization.path == SORTING_PATH_ONESHOT:
        operation = operations[SORTING_ONESHOT_OP_ID]
        overrides = {"max_tokens": specialization.launcher_max_tokens}
    elif specialization.path == SORTING_PATH_P0V2_P23:
        operation = operations[SORTING_P0V2_P23_OP_ID]
        overrides = {"k4_block": specialization.k4_block}
    elif specialization.path == SORTING_PATH_4K_FUSED:
        operation = operations[SORTING_4K_FUSED_OP_ID]
        overrides = {"k4_block": specialization.k4_block}
    else:
        raise RuntimeError(
            f"{plan.context}: unhandled sorting path {specialization.path!r}"
        )
    plan.emit(operation, case, **overrides)


def resolve_moe_sorting_compile_plan(
    case: MoeSortingCompileCase,
    *,
    context: CompileContext[Any],
    specialization: Any = None,
) -> CompilePlan:
    """Resolve one explicit sorting specialization into one CompileUnit."""

    if not isinstance(case, MoeSortingCompileCase):
        raise TypeError(
            f"case must be a MoeSortingCompileCase, got {type(case).__name__}"
        )
    if specialization is None:
        from .kernels.moe_sorting_kernel import resolve_moe_sorting_specialization

        specialization = resolve_moe_sorting_specialization(
            arch=context.target.arch,
            max_tokens=case.max_tokens,
            num_experts=case.num_experts,
            topk=case.topk,
            unit_size=case.unit_size,
            has_mask=case.has_mask,
            path=case.path,
            k4_block=case.k4_block,
        )
    return _resolve_moe_sorting_compile_plan(
        case,
        context=context,
        specialization=specialization,
    )


def resolve_moe_compile_plan(
    case: MoeCompilePlanCase,
    *,
    context: CompileContext[Any],
) -> CompilePlan:
    """Compose optional sorting, Stage1, and Stage2 plans in launch order."""

    if not isinstance(context, CompileContext):
        raise TypeError(
            f"context must be a CompileContext, got {type(context).__name__}"
        )
    if not isinstance(case, MoeCompilePlanCase):
        raise TypeError(f"case must be a MoeCompilePlanCase, got {type(case).__name__}")
    if case.sorting is not None and not isinstance(
        case.sorting,
        MoeSortingCompileCase,
    ):
        raise TypeError(
            "sorting must be an explicit MoeSortingCompileCase or None, "
            f"got {type(case.sorting).__name__}"
        )
    for name, subplan in (("stage1", case.stage1), ("stage2", case.stage2)):
        if not isinstance(subplan, CompilePlan):
            raise TypeError(f"{name} must be a CompilePlan")
        for unit in subplan.units:
            if unit.spec.target != context.target:
                raise ValueError(
                    f"{name} unit {unit.spec.op_id!r} targets "
                    f"{unit.spec.target!r}, expected {context.target!r}"
                )

    sorting_plan = (
        CompilePlan(())
        if case.sorting is None
        else resolve_moe_sorting_compile_plan(case.sorting, context=context)
    )
    return CompilePlan(
        sorting_plan.units + case.stage1.units + case.stage2.units,
    )


@plan_provider
def _resolve_moe_stage1_compile_plan(
    plan: PlanBuilder,
    *,
    decision: Stage1CompileDecision,
    **builder_kwargs: Any,
) -> None:
    if not isinstance(decision, Stage1CompileDecision):
        raise TypeError("decision must be a Stage1CompileDecision")

    operations = _operations()
    primary = operations[
        (
            MIXED_STAGE1_GEMM_OP_ID
            if decision.primary_family == "mixed"
            else INT4_STAGE1_GEMM_OP_ID
        )
    ]
    requested = plan.bind(primary)
    plan.emit(
        requested,
        out_dtype=decision.main_out_dtype,
        enable_bias=decision.main_enable_bias,
    )
    if decision.postprocess_kind != "fq":
        return
    plan.emit(
        operations[FQ_ACTIVATION_OP_ID],
        requested,
        quant_mode=decision.fq_quant_mode,
        gui_layout=decision.fq_gui_layout,
        enable_bias=decision.fq_enable_bias,
    )


def resolve_moe_stage1_compile_plan(
    *,
    context: CompileContext[Any],
    decision: Stage1CompileDecision | None = None,
    **builder_kwargs: Any,
) -> CompilePlan:
    """Resolve a Stage1 GEMM and its optional FlyDSL split-K helper.

    ``builder_kwargs`` are bound directly to ``compile_flydsl_moe_stage1``.
    Adding a normal compile parameter therefore requires changing that existing
    wrapper signature/default and its real call site, not a parallel Spec type.
    """

    if decision is None:
        decision = resolve_stage1_compile_decision(builder_kwargs)
    return _resolve_moe_stage1_compile_plan(
        context=context,
        decision=decision,
        **builder_kwargs,
    )


@plan_provider
def _resolve_moe_stage2_compile_plan(
    plan: PlanBuilder,
    *,
    decision: Stage2CompileDecision,
    **builder_kwargs: Any,
) -> None:
    if not isinstance(decision, Stage2CompileDecision):
        raise TypeError("decision must be a Stage2CompileDecision")

    operations = _operations()
    primary = operations[
        (
            MIXED_STAGE2_GEMM_OP_ID
            if decision.primary_family == "mixed"
            else INT4_STAGE2_GEMM_OP_ID
        )
    ]
    requested = plan.bind(primary, persist_m=1)
    plan.emit(
        requested,
        accumulate=decision.accumulate,
        persist_m=decision.persist_m,
    )
    if decision.reduction_kind == "none":
        return
    masked = decision.reduction_kind == "masked"
    plan.emit(
        operations[MASKED_REDUCTION_OP_ID if masked else PLAIN_REDUCTION_OP_ID],
        requested,
        dtype_str=decision.reduction_dtype,
        use_mask=masked,
        num_experts=decision.reduction_num_experts,
    )


def resolve_moe_stage2_compile_plan(
    *,
    context: CompileContext[Any],
    mode: str,
    accumulate: bool,
    return_per_slot: bool,
    persist: bool | None,
    token_num: int,
    routing_block_count: int | None,
    dtype_str: str,
    use_mask: bool,
    topk_ids_available: bool,
    num_experts: int,
    decision: Stage2CompileDecision | None = None,
    **builder_kwargs: Any,
) -> CompilePlan:
    """Resolve one Stage2 GEMM and its optional top-k reduction."""

    if "persist_m" in builder_kwargs:
        raise TypeError("persist_m is owned by the Stage2 provider")
    if decision is None:
        decision = resolve_stage2_compile_decision(
            builder_kwargs,
            mode=mode,
            accumulate=accumulate,
            return_per_slot=return_per_slot,
            persist=persist,
            token_num=token_num,
            routing_block_count=routing_block_count,
            dtype_str=dtype_str,
            use_mask=use_mask,
            topk_ids_available=topk_ids_available,
            num_experts=num_experts,
        )
    return _resolve_moe_stage2_compile_plan(
        context=context,
        decision=decision,
        **builder_kwargs,
    )


@plan_provider
def resolve_cktile_stage1_compile_plan(
    plan: PlanBuilder,
    *,
    inter_dim: int,
    topk: int,
    split_k: int,
    act: str,
    post_activation_layout: str,
    enable_bias: bool = False,
) -> None:
    """Resolve the optional CK-Tile interleaved Stage1 FlyDSL epilogue."""

    operations = _operations()
    plan.require(
        post_activation_layout in ("auto", "standard", "interleaved"),
        f"unsupported post_activation_layout: {post_activation_layout}",
    )
    plan.require(
        post_activation_layout != "auto",
        "post_activation_layout='auto' is ambiguous",
    )
    plan.require(
        not isinstance(split_k, bool) and isinstance(split_k, int) and split_k > 0,
        f"split_k must be a positive integer, got {split_k!r}",
    )
    if split_k == 1 or post_activation_layout == "standard" or act == "gelu":
        return
    plan.require(
        not enable_bias,
        "CK-Tile interleaved split-K bias is unsupported",
    )
    if act == "silu":
        plan.emit(
            operations[FQ_ACTIVATION_OP_ID],
            quant_mode="none",
            gui_layout=True,
        )
    elif act == "swiglu":
        plan.emit(operations[CKTILE_SWIGLU_AND_MUL_OP_ID])
    else:
        plan.require(False, f"unsupported CK-Tile activation: {act!r}")
