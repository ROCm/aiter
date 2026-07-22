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

from typing import Any

from .compile_plan import (
    ArgumentKind,
    CompileOpRegistry,
    CompilePlan,
    DEFAULT_COMPILE_OP_REGISTRY,
    KernelSignature,
    RocmTarget,
    SignatureArg,
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


def _pointer(name: str) -> SignatureArg:
    # ptr_arg() materializes an fx.Pointer<Uint8>, regardless of tensor dtype.
    return SignatureArg(name, ArgumentKind.POINTER, "u8")


def _scalar(name: str, dtype: str) -> SignatureArg:
    return SignatureArg(name, ArgumentKind.SCALAR, dtype)


_MIXED_GEMM_ABI = KernelSignature(
    tuple(
        _pointer(name)
        for name in (
            "arg_out",
            "arg_x",
            "arg_w",
            "arg_scale_x",
            "arg_scale_w",
            "arg_sorted_token_ids",
            "arg_expert_ids",
            "arg_sorted_weights",
            "arg_max_token_ids",
            "arg_bias",
            "arg_out_scale_sorted",
        )
    )
    + tuple(
        _scalar(name, dtype)
        for name, dtype in (
            ("i32_tokens_in", "i32"),
            ("i32_inter_in", "i32"),
            ("i32_k_in", "i32"),
            ("i32_size_expert_ids_in", "i32"),
            ("f32_swiglu_limit", "f32"),
        )
    )
    + (SignatureArg("stream", ArgumentKind.STREAM),)
)

_INT4_GEMM_ABI = KernelSignature(
    tuple(
        _pointer(name)
        for name in (
            "arg_out",
            "arg_x",
            "arg_w",
            "arg_scale_x",
            "arg_scale_w",
            "arg_sorted_token_ids",
            "arg_expert_ids",
            "arg_sorted_weights",
            "arg_max_token_ids",
        )
    )
    + tuple(
        _scalar(name, "i32")
        for name in (
            "i32_tokens_in",
            "i32_inter_in",
            "i32_k_in",
            "i32_size_expert_ids_in",
        )
    )
    + (SignatureArg("stream", ArgumentKind.STREAM),)
)

_FQ_ACTIVATION_ABI = KernelSignature(
    tuple(
        _pointer(name)
        for name in (
            "x",
            "out_buf",
            "out_scale_sorted",
            "sorted_ids",
            "num_valid_ids",
            "topk_ids",
            "bias",
        )
    )
    + (
        _scalar("token_num", "i32"),
        _scalar("num_sorted_rows", "i32"),
        _scalar("swiglu_limit_f", "f32"),
        SignatureArg("stream", ArgumentKind.STREAM),
    )
)

_SWIGLU_EPILOGUE_ABI = KernelSignature(
    (
        SignatureArg("x", ArgumentKind.TENSOR, "bf16", (None, None), (None, 1)),
        SignatureArg("out", ArgumentKind.TENSOR, "bf16", (None, None), (None, 1)),
        _scalar("num_rows", "i32"),
        SignatureArg("stream", ArgumentKind.STREAM),
    )
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


def register_moe_stage1_ops(
    registry: CompileOpRegistry = DEFAULT_COMPILE_OP_REGISTRY,
) -> CompileOpRegistry:
    """Register existing cached compiler wrappers without field adapters."""

    from .moe_kernels import (
        _get_compiled_silu_fused,
        _get_compiled_swiglu,
        compile_flydsl_moe_stage1,
    )

    entries = (
        (MIXED_STAGE1_GEMM_OP_ID, compile_flydsl_moe_stage1),
        (INT4_STAGE1_GEMM_OP_ID, compile_flydsl_moe_stage1),
        (FQ_ACTIVATION_OP_ID, _get_compiled_silu_fused),
        (CKTILE_SWIGLU_AND_MUL_OP_ID, _get_compiled_swiglu),
    )
    for op_id, compiler in entries:
        try:
            registered = registry.lookup(op_id)
        except KeyError:
            registry.register(op_id)(compiler)
        else:
            if registered is not compiler:
                raise RuntimeError(f"{op_id!r} is registered to another callable")
    return registry


def _require_target(target: object) -> RocmTarget:
    if not isinstance(target, RocmTarget):
        raise TypeError(f"target must be a RocmTarget, got {type(target).__name__}")
    return target


def _primary_op_id(builder_kwargs: dict[str, Any]) -> str:
    try:
        a_dtype = builder_kwargs["a_dtype"]
        b_dtype = builder_kwargs["b_dtype"]
    except KeyError as error:
        raise TypeError(f"missing required Stage1 argument: {error.args[0]}") from error
    if b_dtype in ("fp4", "fp8"):
        return MIXED_STAGE1_GEMM_OP_ID
    if a_dtype == "bf16" and b_dtype == "int4":
        return INT4_STAGE1_GEMM_OP_ID
    raise ValueError(
        "unsupported Stage1 dtype combination: "
        f"a_dtype={a_dtype!r}, b_dtype={b_dtype!r}"
    )


def _fq_unit(
    registry: CompileOpRegistry,
    target: RocmTarget,
    *,
    inter_dim: int,
    topk: int,
    quant_mode: str,
    gui_layout: bool,
    act: str,
    enable_bias: bool,
):
    return registry.make_unit(
        FQ_ACTIVATION_OP_ID,
        target=target,
        signature=stage1_abi(FQ_ACTIVATION_OP_ID),
        inter_dim=inter_dim,
        topk=topk,
        quant_mode=quant_mode,
        gui_layout=gui_layout,
        act=act,
        enable_bias=enable_bias,
        compile_target=target,
    )


def resolve_moe_stage1_compile_plan(
    *,
    target: RocmTarget,
    registry: CompileOpRegistry = DEFAULT_COMPILE_OP_REGISTRY,
    **builder_kwargs: Any,
) -> CompilePlan:
    """Resolve a FlyDSL Stage1 GEMM and any split-K FlyDSL helper.

    ``builder_kwargs`` are bound directly to ``compile_flydsl_moe_stage1``.
    Adding a normal compile parameter therefore requires changing that existing
    wrapper signature/default and its real call site, not a parallel Spec type.
    """

    target = _require_target(target)
    register_moe_stage1_ops(registry)
    op_id = _primary_op_id(builder_kwargs)

    # The first bind applies the wrapper's defaults. Graph decisions and the
    # final primary unit then consume that one normalized source.
    requested = registry.make_unit(
        op_id,
        target=target,
        signature=stage1_abi(op_id),
        compile_target=target,
        **builder_kwargs,
    )
    normalized = requested.spec.call.as_kwargs()
    k_batch = normalized["k_batch"]
    if isinstance(k_batch, bool) or not isinstance(k_batch, int) or k_batch <= 0:
        raise ValueError("k_batch must be a positive integer")
    is_splitk = k_batch > 1

    if op_id == MIXED_STAGE1_GEMM_OP_ID:
        gate_mode = normalized["gate_mode"]
        if gate_mode == "gate_only":
            raise ValueError("gate_only is reserved and unsupported")
        if gate_mode == "mock_gate_only" and not is_splitk:
            raise ValueError("mock_gate_only requires split-K")
        if is_splitk and normalized["out_dtype"] == "fp8" and gate_mode != "interleave":
            raise ValueError("split-K fp8 output requires an interleaved layout")

    requested_out_dtype = normalized["out_dtype"]
    if is_splitk and requested_out_dtype in ("fp4", "fp8"):
        normalized["out_dtype"] = "bf16"
    if is_splitk:
        normalized["enable_bias"] = False

    units = [
        registry.make_unit(
            op_id,
            target=target,
            signature=stage1_abi(op_id),
            **normalized,
        )
    ]
    if op_id != MIXED_STAGE1_GEMM_OP_ID or not is_splitk:
        return CompilePlan(tuple(units))

    helper_kwargs = {
        "inter_dim": normalized["inter_dim"],
        "topk": normalized["topk"],
        "act": normalized["act"],
        # Bias is intentionally taken from the requested call, not the
        # split-K primary GEMM where it was disabled above.
        "enable_bias": requested.spec.call.as_kwargs()["enable_bias"],
    }
    gate_mode = normalized["gate_mode"]
    if gate_mode == "interleave":
        quant_mode = (
            requested_out_dtype if requested_out_dtype in ("fp4", "fp8") else "none"
        )
        units.append(
            _fq_unit(
                registry,
                target,
                quant_mode=quant_mode,
                gui_layout=True,
                **helper_kwargs,
            )
        )
    elif requested_out_dtype == "fp4":
        units.append(
            _fq_unit(
                registry,
                target,
                quant_mode="fp4",
                gui_layout=False,
                **helper_kwargs,
            )
        )
    # Other separated split-K paths use a CK/HIP activation and add no
    # FlyDSL compile unit.
    return CompilePlan(tuple(units))


def resolve_cktile_stage1_compile_plan(
    *,
    target: RocmTarget,
    inter_dim: int,
    topk: int,
    split_k: int,
    act: str,
    post_activation_layout: str,
    enable_bias: bool = False,
    registry: CompileOpRegistry = DEFAULT_COMPILE_OP_REGISTRY,
) -> CompilePlan:
    """Resolve the optional CK-Tile interleaved Stage1 FlyDSL epilogue."""

    target = _require_target(target)
    register_moe_stage1_ops(registry)
    if post_activation_layout not in ("auto", "standard", "interleaved"):
        raise ValueError(
            f"unsupported post_activation_layout: {post_activation_layout}"
        )
    if post_activation_layout == "auto":
        raise ValueError("post_activation_layout='auto' is ambiguous")
    if isinstance(split_k, bool) or not isinstance(split_k, int) or split_k <= 0:
        raise ValueError("split_k must be a positive integer")
    if split_k == 1 or post_activation_layout == "standard" or act == "gelu":
        return CompilePlan(())
    if enable_bias:
        raise ValueError("CK-Tile interleaved split-K bias is unsupported")
    if act == "silu":
        return CompilePlan(
            (
                _fq_unit(
                    registry,
                    target,
                    inter_dim=inter_dim,
                    topk=topk,
                    quant_mode="none",
                    gui_layout=True,
                    act="silu",
                    enable_bias=False,
                ),
            )
        )
    if act == "swiglu":
        return CompilePlan(
            (
                registry.make_unit(
                    CKTILE_SWIGLU_AND_MUL_OP_ID,
                    target=target,
                    signature=stage1_abi(CKTILE_SWIGLU_AND_MUL_OP_ID),
                    inter_dim=inter_dim,
                    compile_target=target,
                ),
            )
        )
    raise ValueError(f"unsupported CK-Tile activation: {act!r}")
