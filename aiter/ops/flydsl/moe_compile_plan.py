# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Stage1 FlyDSL graph resolution using existing compiler signatures.

There is deliberately no typed mirror of Stage1 builder parameters here.
``CompileOpRegistry.make_unit`` binds directly to the existing compiler
wrappers, so their signatures and defaults remain the only callable schema.
This module owns only graph semantics and the launch ABI that FlyDSL cannot
currently expose.

Importing this module does not compile or launch anything. The existing
``moe_kernels`` host module (and therefore torch/FlyDSL Python modules) is
imported lazily when an operation set is first registered.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from .compile_plan import (
    ArgumentKind,
    CompileOp,
    CompileOpRegistry,
    DEFAULT_COMPILE_OP_REGISTRY,
    KernelSignature,
    PlanBuilder,
    SignatureArg,
    op,
    plan_provider,
)

MIXED_STAGE1_GEMM_OP_ID = "aiter.flydsl.moe.stage1.mixed_gemm.v1"
INT4_STAGE1_GEMM_OP_ID = "aiter.flydsl.moe.stage1.int4_gemm.v1"
FQ_ACTIVATION_OP_ID = "aiter.flydsl.moe.stage1.silu_and_mul_fq.v1"
CKTILE_SWIGLU_AND_MUL_OP_ID = "aiter.flydsl.moe.stage1.cktile_swiglu_and_mul.v1"

__all__ = [
    "CKTILE_SWIGLU_AND_MUL_OP_ID",
    "FQ_ACTIVATION_OP_ID",
    "INT4_STAGE1_GEMM_OP_ID",
    "MIXED_STAGE1_GEMM_OP_ID",
    "register_moe_stage1_ops",
    "resolve_cktile_stage1_compile_plan",
    "resolve_moe_stage1_compile_plan",
    "stage1_abi",
]


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

_ABIS = {
    MIXED_STAGE1_GEMM_OP_ID: _MIXED_GEMM_ABI,
    INT4_STAGE1_GEMM_OP_ID: _INT4_GEMM_ABI,
    FQ_ACTIVATION_OP_ID: _FQ_ACTIVATION_ABI,
    CKTILE_SWIGLU_AND_MUL_OP_ID: _SWIGLU_EPILOGUE_ABI,
}


def stage1_abi(op_id: str) -> KernelSignature:
    """Return the isolated manual ABI for one stable Stage1 operation."""

    try:
        return _ABIS[op_id]
    except KeyError as error:
        raise KeyError(f"unknown Stage1 op_id: {op_id!r}") from error


@lru_cache(maxsize=1)
def _declared_ops() -> tuple[CompileOp, ...]:
    """Declare Stage1 ops lazily so importing this module stays CPU-light."""

    from .moe_kernels import (
        _get_compiled_silu_fused,
        _get_compiled_swiglu,
        compile_flydsl_moe_stage1,
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
    )
    return tuple(
        op(op_id, compiler, abi=abi, target_kw="compile_target")
        for op_id, compiler, abi in declarations
    )


def _operations() -> dict[str, CompileOp]:
    return {operation.op_id: operation for operation in _declared_ops()}


def register_moe_stage1_ops(
    registry: CompileOpRegistry = DEFAULT_COMPILE_OP_REGISTRY,
) -> CompileOpRegistry:
    """Register all Stage1 declarations with ``registry``."""

    for operation in _declared_ops():
        operation.register(registry)
    return registry


def _primary_op(
    plan: PlanBuilder,
    operations: dict[str, CompileOp],
    builder_kwargs: dict[str, Any],
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


@plan_provider
def resolve_moe_stage1_compile_plan(
    plan: PlanBuilder,
    **builder_kwargs: Any,
) -> None:
    """Resolve a FlyDSL Stage1 GEMM and any split-K FlyDSL helper.

    ``builder_kwargs`` are bound directly to ``compile_flydsl_moe_stage1``.
    Adding a normal compile parameter therefore requires changing that existing
    wrapper signature/default and its real call site, not a parallel Spec type.
    """

    operations = _operations()
    primary = _primary_op(plan, operations, builder_kwargs)
    # ``plan_provider`` installed the provider kwargs as a reflected source.
    # Bind only fields accepted by the real compiler; AOT job metadata such as
    # token counts and CSV labels must not become a mirrored callable schema.
    requested = plan.bind(primary)

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
    if is_splitk:
        plan.emit(
            requested,
            out_dtype=(
                "bf16" if requested_out_dtype in ("fp4", "fp8") else requested_out_dtype
            ),
            enable_bias=False,
        )
    else:
        plan.emit(requested)
    if primary.op_id != MIXED_STAGE1_GEMM_OP_ID or not is_splitk:
        return

    helper = operations[FQ_ACTIVATION_OP_ID]
    gate_mode = requested["gate_mode"]
    if gate_mode == "interleave":
        quant_mode = (
            requested_out_dtype if requested_out_dtype in ("fp4", "fp8") else "none"
        )
        plan.emit(
            helper,
            requested,
            quant_mode=quant_mode,
            gui_layout=True,
        )
    elif requested_out_dtype == "fp4":
        plan.emit(
            helper,
            requested,
            quant_mode="fp4",
            gui_layout=False,
        )
    # Other separated split-K paths use a CK/HIP activation and add no
    # FlyDSL compile unit.


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
