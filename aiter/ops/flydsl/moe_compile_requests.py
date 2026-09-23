# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pure compile-request factories for standard FlyDSL MoE kernels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .compile_request import (
    DEFAULT_COMPILE_OP_REGISTRY,
    ArgumentKind,
    CompileOpRegistry,
    CompileRequest,
    KernelSignature,
    RocmTarget,
    SignatureArg,
)
from .moe_compile_decisions import (
    Stage1CompileDecision,
    Stage2CompileDecision,
    resolve_stage1_compile_decision,
    resolve_stage2_compile_decision,
)

MIXED_STAGE1_GEMM_OP_ID = "aiter.flydsl.moe.stage1.mixed_gemm.v1"
INT4_STAGE1_GEMM_OP_ID = "aiter.flydsl.moe.stage1.int4_gemm.v1"
A16_STAGE1_GEMM_OP_ID = "aiter.flydsl.moe.stage1.a16w_mix_gemm.v1"
FHMOE_STAGE1_GEMM_OP_ID = "aiter.flydsl.fhmoe.stage1.mixed_gemm.v1"
FQ_ACTIVATION_OP_ID = "aiter.flydsl.moe.stage1.silu_and_mul_fq.v1"
CKTILE_SWIGLU_AND_MUL_OP_ID = "aiter.flydsl.moe.stage1.cktile_swiglu_and_mul.v1"
MIXED_STAGE2_GEMM_OP_ID = "aiter.flydsl.moe.stage2.mixed_gemm.v1"
INT4_STAGE2_GEMM_OP_ID = "aiter.flydsl.moe.stage2.int4_gemm.v1"
A16_STAGE2_GEMM_OP_ID = "aiter.flydsl.moe.stage2.a16w_mix_gemm.v1"
FHMOE_STAGE2_GEMM_OP_ID = "aiter.flydsl.fhmoe.stage2.mixed_gemm.v1"
PLAIN_REDUCTION_OP_ID = "aiter.flydsl.moe.stage2.reduction.plain.v1"
MASKED_REDUCTION_OP_ID = "aiter.flydsl.moe.stage2.reduction.masked.v1"
SORTING_ONESHOT_OP_ID = "aiter.flydsl.moe.sorting.oneshot.v1"
SORTING_P0V2_P23_OP_ID = "aiter.flydsl.moe.sorting.multiphase.p0v2_p23.v1"
SORTING_4K_FUSED_OP_ID = "aiter.flydsl.moe.sorting.multiphase.k4_fused.v1"

__all__ = [
    "A16_STAGE1_GEMM_OP_ID",
    "A16_STAGE2_GEMM_OP_ID",
    "CKTILE_SWIGLU_AND_MUL_OP_ID",
    "FHMOE_STAGE1_GEMM_OP_ID",
    "FHMOE_STAGE2_GEMM_OP_ID",
    "FQ_ACTIVATION_OP_ID",
    "INT4_STAGE1_GEMM_OP_ID",
    "INT4_STAGE2_GEMM_OP_ID",
    "MASKED_REDUCTION_OP_ID",
    "MIXED_STAGE1_GEMM_OP_ID",
    "MIXED_STAGE2_GEMM_OP_ID",
    "PLAIN_REDUCTION_OP_ID",
    "SORTING_4K_FUSED_OP_ID",
    "SORTING_ONESHOT_OP_ID",
    "SORTING_P0V2_P23_OP_ID",
    "MoeSortingCompileCase",
    "Stage2RuntimeMetadata",
    "cktile_epilogue_compile_requests",
    "get_kernel_signature",
    "register_moe_sorting_ops",
    "register_moe_stage1_ops",
    "register_moe_stage2_ops",
    "sorting_compile_request",
    "stage1_compile_requests",
    "stage2_compile_requests",
]


@dataclass(frozen=True)
class MoeSortingCompileCase:
    """Explicit CPU metadata for one independently cached sorting launcher."""

    max_tokens: int
    num_experts: int
    topk: int
    has_mask: bool
    has_local_tokens: bool = False
    unit_size: int = 32
    path: str | None = None
    k4_block: int | None = None


@dataclass(frozen=True)
class Stage2RuntimeMetadata:
    """Runtime facts used to select Stage2 compile requests.

    These values are intentionally separate from builder kwargs: they may
    choose a kernel or derive a constexpr, but they are never blindly added to
    the builder cache key.
    """

    mode: str
    accumulate: bool | None
    return_per_slot: bool
    persist: bool | None
    token_num: int
    routing_block_count: int | None
    dtype_str: str | None
    use_mask: bool
    topk_ids_available: bool
    num_experts: int
    fp8_intermediate: bool = False
    out_dtype_str: str | None = None
    use_weight: bool = False
    scale_blk: int | None = None
    pitch_align: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("atomic", "reduce"):
            raise ValueError(f"unsupported Stage2 mode: {self.mode!r}")
        for name in (
            "return_per_slot",
            "use_mask",
            "topk_ids_available",
            "fp8_intermediate",
            "use_weight",
        ):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool, got {value!r}")
        for name in ("accumulate", "persist"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} must be bool or None, got {value!r}")
        if isinstance(self.token_num, bool) or not isinstance(self.token_num, int):
            raise TypeError(f"token_num must be an integer, got {self.token_num!r}")
        if self.token_num <= 0:
            raise ValueError(
                f"token_num must be a positive integer, got {self.token_num!r}"
            )
        if self.routing_block_count is not None:
            if isinstance(self.routing_block_count, bool) or not isinstance(
                self.routing_block_count, int
            ):
                raise TypeError(
                    "routing_block_count must be an integer or None, got "
                    f"{self.routing_block_count!r}"
                )
            if self.routing_block_count <= 0:
                raise ValueError(
                    "routing_block_count must be a positive integer or None, got "
                    f"{self.routing_block_count!r}"
                )
        if isinstance(self.num_experts, bool) or not isinstance(self.num_experts, int):
            raise TypeError(f"num_experts must be an integer, got {self.num_experts!r}")
        if self.num_experts < 0:
            raise ValueError(
                f"num_experts must be a non-negative integer, got {self.num_experts!r}"
            )
        for name in ("dtype_str", "out_dtype_str"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise TypeError(f"{name} must be a non-empty string or None")
        if self.scale_blk is not None:
            if isinstance(self.scale_blk, bool) or not isinstance(self.scale_blk, int):
                raise TypeError(
                    f"scale_blk must be an integer or None, got {self.scale_blk!r}"
                )
            if self.scale_blk <= 0:
                raise ValueError(
                    "scale_blk must be a positive integer or None, got "
                    f"{self.scale_blk!r}"
                )
        if self.pitch_align is not None:
            if isinstance(self.pitch_align, bool) or not isinstance(
                self.pitch_align, int
            ):
                raise TypeError(
                    "pitch_align must be an integer or None, got "
                    f"{self.pitch_align!r}"
                )
            if self.pitch_align < 0:
                raise ValueError(
                    "pitch_align must be a non-negative integer or None, got "
                    f"{self.pitch_align!r}"
                )


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
    f32="""
        f32_situ_beta f32_situ_beta_rcp f32_situ_linear_beta
        f32_situ_linear_beta_rcp f32_swiglu_limit
    """,
)

_FHMOE_STAGE1_GEMM_ABI = _abi(
    pointers="""
        arg_out arg_x arg_w arg_scale_x arg_scale_w arg_shared_w
        arg_shared_scale_w arg_sorted_token_ids arg_expert_ids
        arg_sorted_weights arg_max_token_ids arg_bias arg_out_scale_sorted
    """,
    i32="i32_tokens_in i32_inter_in i32_k_in i32_size_expert_ids_in",
    f32="""
        f32_situ_beta f32_situ_beta_rcp f32_situ_linear_beta
        f32_situ_linear_beta_rcp f32_swiglu_limit
    """,
)

_A16_STAGE1_GEMM_ABI = KernelSignature(
    tuple(
        SignatureArg(name, ArgumentKind.SCALAR, dtype)
        for name, dtype in (
            ("arg_x", "i64"),
            ("arg_bq", "i64"),
            ("arg_bscale", "i64"),
            ("arg_eids", "i64"),
            ("arg_cumsum", "i64"),
            ("arg_mind", "i64"),
            ("i32_ntok", "i32"),
            ("i32_grid", "i32"),
            ("f32_situ_beta", "f32"),
            ("f32_situ_beta_rcp", "f32"),
            ("f32_situ_linbeta", "f32"),
            ("f32_situ_linbeta_rcp", "f32"),
            ("f32_swiglu_limit", "f32"),
            ("arg_out", "i64"),
        )
    )
    + (SignatureArg("stream", ArgumentKind.STREAM),)
)

_FQ_ACTIVATION_ABI = _abi(
    pointers="""
        x out_buf out_scale_sorted sorted_ids num_valid_ids topk_ids bias
    """,
    i32="token_num num_sorted_rows",
    f32="""
        situ_beta_f situ_beta_rcp_f situ_linear_beta_f
        situ_linear_beta_rcp_f swiglu_limit_f
    """,
)

_SWIGLU_EPILOGUE_ABI = _abi(
    pointers="x out",
    i32="num_rows",
)

_MIXED_STAGE2_GEMM_ABI = _abi(
    pointers="""
        arg_out arg_x arg_w arg_scale_x arg_scale_w arg_sorted_token_ids
        arg_expert_ids arg_sorted_weights arg_num_valid_ids arg_bias
    """,
    i32="""
        i32_tokens_in i32_x_rows i32_n_in i32_k_in
        i32_size_expert_ids_in
    """,
)

_FHMOE_STAGE2_GEMM_ABI = _abi(
    pointers="""
        arg_out arg_x arg_w arg_scale_x arg_scale_w arg_shared_w
        arg_shared_scale_w arg_sorted_token_ids arg_expert_ids
        arg_sorted_weights arg_num_valid_ids arg_bias
    """,
    i32="""
        i32_tokens_in i32_x_rows i32_n_in i32_k_in
        i32_size_expert_ids_in
    """,
)

_A16_STAGE2_GEMM_ABI = KernelSignature(
    tuple(
        SignatureArg(name, ArgumentKind.SCALAR, dtype)
        for name, dtype in (
            ("arg_a", "i64"),
            ("arg_bq", "i64"),
            ("arg_bscale", "i64"),
            ("arg_eids", "i64"),
            ("arg_cumsum", "i64"),
            ("arg_stids", "i64"),
            ("arg_sweights", "i64"),
            ("i32_M", "i32"),
            ("i32_max_m_blocks", "i32"),
            ("i32_grid", "i32"),
            ("arg_out", "i64"),
        )
    )
    + (SignatureArg("stream", ArgumentKind.STREAM),)
)

_REDUCTION_ABI = _abi(
    pointers="X Y expert_mask topk_ids topk_weights",
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
    _tensor("local_tokens_tensor", "i32", 1),
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
    INT4_STAGE1_GEMM_OP_ID: _A16_STAGE1_GEMM_ABI,
    A16_STAGE1_GEMM_OP_ID: _A16_STAGE1_GEMM_ABI,
    FHMOE_STAGE1_GEMM_OP_ID: _FHMOE_STAGE1_GEMM_ABI,
    FQ_ACTIVATION_OP_ID: _FQ_ACTIVATION_ABI,
    CKTILE_SWIGLU_AND_MUL_OP_ID: _SWIGLU_EPILOGUE_ABI,
    MIXED_STAGE2_GEMM_OP_ID: _MIXED_STAGE2_GEMM_ABI,
    INT4_STAGE2_GEMM_OP_ID: _A16_STAGE2_GEMM_ABI,
    A16_STAGE2_GEMM_OP_ID: _A16_STAGE2_GEMM_ABI,
    FHMOE_STAGE2_GEMM_OP_ID: _FHMOE_STAGE2_GEMM_ABI,
    PLAIN_REDUCTION_OP_ID: _REDUCTION_ABI,
    MASKED_REDUCTION_OP_ID: _REDUCTION_ABI,
    SORTING_ONESHOT_OP_ID: _SORTING_ONESHOT_ABI,
    SORTING_P0V2_P23_OP_ID: _SORTING_P0V2_P23_ABI,
    SORTING_4K_FUSED_OP_ID: _SORTING_4K_FUSED_ABI,
}


def get_kernel_signature(op_id: str) -> KernelSignature:
    """Return the manually declared launch ABI for one stable operation."""

    try:
        return _ABIS[op_id]
    except KeyError as error:
        raise KeyError(f"unknown MoE compile op_id: {op_id!r}") from error


def _load_stage1_builder():
    from .moe_kernels import compile_flydsl_moe_stage1

    return compile_flydsl_moe_stage1


def _load_stage2_builder():
    from .moe_kernels import compile_flydsl_moe_stage2

    return compile_flydsl_moe_stage2


def _load_a16_stage1_builder():
    from .moe_kernels import compile_flydsl_moe_stage1_a16w_mix

    return compile_flydsl_moe_stage1_a16w_mix


def _load_a16_stage2_builder():
    from .moe_kernels import compile_flydsl_moe_stage2_a16w_mix

    return compile_flydsl_moe_stage2_a16w_mix


def _load_fhmoe_stage1_builder():
    from .fhmoe import compile_flydsl_fhmoe_stage1

    return compile_flydsl_fhmoe_stage1


def _load_fhmoe_stage2_builder():
    from .fhmoe import compile_flydsl_fhmoe_stage2

    return compile_flydsl_fhmoe_stage2


def _load_fq_builder():
    from .moe_kernels import _get_compiled_silu_fused

    return _get_compiled_silu_fused


def _load_swiglu_builder():
    from .moe_kernels import _get_compiled_swiglu

    return _get_compiled_swiglu


def _load_reduction_builder():
    from .kernels.moe_reduce import compile_moe_reduction

    return compile_moe_reduction


def _load_sorting_oneshot_builder():
    from .kernels.moe_sorting_kernel import compile_moe_sorting_oneshot

    return compile_moe_sorting_oneshot


def _load_sorting_p0v2_p23_builder():
    from .kernels.moe_sorting_kernel import compile_moe_sorting_p0v2_p23

    return compile_moe_sorting_p0v2_p23


def _load_sorting_4k_builder():
    from .kernels.moe_sorting_kernel import compile_moe_sorting_4k_fused

    return compile_moe_sorting_4k_fused


_STAGE1_LOADERS = {
    MIXED_STAGE1_GEMM_OP_ID: _load_stage1_builder,
    INT4_STAGE1_GEMM_OP_ID: _load_a16_stage1_builder,
    A16_STAGE1_GEMM_OP_ID: _load_a16_stage1_builder,
    FHMOE_STAGE1_GEMM_OP_ID: _load_fhmoe_stage1_builder,
    FQ_ACTIVATION_OP_ID: _load_fq_builder,
    CKTILE_SWIGLU_AND_MUL_OP_ID: _load_swiglu_builder,
}
_STAGE2_LOADERS = {
    MIXED_STAGE2_GEMM_OP_ID: _load_stage2_builder,
    INT4_STAGE2_GEMM_OP_ID: _load_a16_stage2_builder,
    A16_STAGE2_GEMM_OP_ID: _load_a16_stage2_builder,
    FHMOE_STAGE2_GEMM_OP_ID: _load_fhmoe_stage2_builder,
    PLAIN_REDUCTION_OP_ID: _load_reduction_builder,
    MASKED_REDUCTION_OP_ID: _load_reduction_builder,
}
_SORTING_LOADERS = {
    SORTING_ONESHOT_OP_ID: _load_sorting_oneshot_builder,
    SORTING_P0V2_P23_OP_ID: _load_sorting_p0v2_p23_builder,
    SORTING_4K_FUSED_OP_ID: _load_sorting_4k_builder,
}


def _register(
    declarations: Mapping[str, Any],
    registry: CompileOpRegistry,
) -> CompileOpRegistry:
    for op_id, loader in declarations.items():
        registry.ensure_lazy(op_id, loader)
    return registry


def register_moe_stage1_ops(
    registry: CompileOpRegistry = DEFAULT_COMPILE_OP_REGISTRY,
) -> CompileOpRegistry:
    """Lazily register Stage1 GEMM and epilogue builders."""

    return _register(_STAGE1_LOADERS, registry)


def register_moe_stage2_ops(
    registry: CompileOpRegistry = DEFAULT_COMPILE_OP_REGISTRY,
) -> CompileOpRegistry:
    """Lazily register Stage2 GEMM and reduction builders."""

    return _register(_STAGE2_LOADERS, registry)


def register_moe_sorting_ops(
    registry: CompileOpRegistry = DEFAULT_COMPILE_OP_REGISTRY,
) -> CompileOpRegistry:
    """Lazily register concrete sorting builders."""

    return _register(_SORTING_LOADERS, registry)


def _mapping(metadata: object, name: str) -> Mapping[str, Any]:
    if not isinstance(metadata, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(metadata).__name__}")
    return metadata


def _request(
    op_id: str,
    metadata: Mapping[str, Any],
    target: RocmTarget,
    registry: CompileOpRegistry,
    *,
    signature: KernelSignature | None = None,
    **overrides: Any,
) -> CompileRequest:
    parameter_names = registry.parameter_names(op_id)
    kwargs = {name: metadata[name] for name in parameter_names if name in metadata}
    kwargs.update(overrides)
    return registry.make_request(
        op_id,
        target=target,
        signature=signature or get_kernel_signature(op_id),
        **kwargs,
    )


def stage1_compile_requests(
    metadata: Mapping[str, Any],
    target: RocmTarget,
    *,
    decision: Stage1CompileDecision | None = None,
    registry: CompileOpRegistry = DEFAULT_COMPILE_OP_REGISTRY,
) -> tuple[CompileRequest, ...]:
    """Return the primary Stage1 request and optional FlyDSL FQ request."""

    metadata = _mapping(metadata, "Stage1 metadata")
    register_moe_stage1_ops(registry)
    resolved_decision = resolve_stage1_compile_decision(metadata)
    if decision is None:
        decision = resolved_decision
    elif not isinstance(decision, Stage1CompileDecision):
        raise TypeError("decision must be a Stage1CompileDecision")
    elif decision != resolved_decision:
        raise ValueError("Stage1 decision disagrees with compile metadata")

    shared_expert_id = metadata.get("shared_expert_id", -1)
    is_fhmoe = isinstance(shared_expert_id, int) and shared_expert_id >= 0
    is_a16w_mix = metadata.get("a_dtype") == "bf16" and metadata.get("b_dtype") in (
        "fp4",
        "int4",
    )
    if is_a16w_mix:
        if metadata.get("enable_bias", False):
            raise ValueError("a16w-mix Stage1 does not support expert bias")
        if metadata.get("k_batch", 1) != 1:
            raise ValueError("a16w-mix Stage1 does not support split-K")
        if metadata.get("gate_mode", "separated") != "separated":
            raise ValueError("a16w-mix Stage1 supports only separated gate mode")
        act = metadata.get("act", "silu")
        if act == "situ":
            act = "situv2"
        if act not in ("silu", "swiglu", "situv2"):
            raise ValueError(f"unsupported a16w-mix Stage1 activation: {act!r}")
    if is_fhmoe:
        primary_op_id = FHMOE_STAGE1_GEMM_OP_ID
    elif is_a16w_mix and metadata.get("b_dtype") == "fp4":
        primary_op_id = A16_STAGE1_GEMM_OP_ID
    elif decision.primary_family == "mixed":
        primary_op_id = MIXED_STAGE1_GEMM_OP_ID
    else:
        primary_op_id = INT4_STAGE1_GEMM_OP_ID

    primary_overrides = {
        "out_dtype": decision.main_out_dtype,
    }
    if not is_a16w_mix:
        primary_overrides["enable_bias"] = decision.main_enable_bias
    if is_a16w_mix:
        waves_per_eu = metadata.get("waves_per_eu")
        if waves_per_eu is not None and int(waves_per_eu) <= 1:
            primary_overrides["waves_per_eu"] = None
        primary_overrides["act"] = act
    requests = [
        _request(
            primary_op_id,
            metadata,
            target,
            registry,
            **primary_overrides,
        )
    ]
    if decision.postprocess_kind == "fq":
        requests.append(
            _request(
                FQ_ACTIVATION_OP_ID,
                metadata,
                target,
                registry,
                quant_mode=decision.fq_quant_mode,
                gui_layout=decision.fq_gui_layout,
                enable_bias=decision.fq_enable_bias,
            )
        )
    return tuple(requests)


def _required(values: Mapping[str, Any], name: str) -> Any:
    try:
        return values[name]
    except KeyError as error:
        raise TypeError(f"missing required Stage2 runtime metadata: {name}") from error


def _stage2_runtime_metadata(
    metadata: Stage2RuntimeMetadata | Mapping[str, Any],
) -> Stage2RuntimeMetadata:
    if isinstance(metadata, Stage2RuntimeMetadata):
        return metadata
    values = _mapping(metadata, "Stage2 runtime metadata")
    return Stage2RuntimeMetadata(
        mode=_required(values, "mode"),
        accumulate=values.get("accumulate"),
        return_per_slot=_required(values, "return_per_slot"),
        persist=_required(values, "persist"),
        token_num=_required(values, "token_num"),
        routing_block_count=_required(values, "routing_block_count"),
        dtype_str=values.get("dtype_str"),
        use_mask=_required(values, "use_mask"),
        topk_ids_available=_required(values, "topk_ids_available"),
        num_experts=_required(values, "num_experts"),
        fp8_intermediate=values.get("fp8_intermediate", False),
        out_dtype_str=values.get("out_dtype_str"),
        use_weight=values.get("use_weight", False),
        scale_blk=values.get("scale_blk"),
        pitch_align=values.get("pitch_align"),
    )


def stage2_compile_requests(
    metadata: Mapping[str, Any],
    runtime_metadata: Stage2RuntimeMetadata | Mapping[str, Any],
    target: RocmTarget,
    *,
    decision: Stage2CompileDecision | None = None,
    registry: CompileOpRegistry = DEFAULT_COMPILE_OP_REGISTRY,
) -> tuple[CompileRequest, ...]:
    """Return the Stage2 GEMM request and optional reduction request."""

    metadata = _mapping(metadata, "Stage2 metadata")
    runtime_metadata = _stage2_runtime_metadata(runtime_metadata)
    register_moe_stage2_ops(registry)
    if "persist_m" in metadata:
        raise TypeError("persist_m is owned by the Stage2 compile decision")
    resolved_decision = resolve_stage2_compile_decision(
        metadata,
        mode=runtime_metadata.mode,
        accumulate=runtime_metadata.accumulate,
        return_per_slot=runtime_metadata.return_per_slot,
        persist=runtime_metadata.persist,
        token_num=runtime_metadata.token_num,
        routing_block_count=runtime_metadata.routing_block_count,
        dtype_str=runtime_metadata.dtype_str,
        use_mask=runtime_metadata.use_mask,
        topk_ids_available=runtime_metadata.topk_ids_available,
        num_experts=runtime_metadata.num_experts,
        fp8_intermediate=runtime_metadata.fp8_intermediate,
    )
    if decision is None:
        decision = resolved_decision
    elif not isinstance(decision, Stage2CompileDecision):
        raise TypeError("decision must be a Stage2CompileDecision")
    elif decision != resolved_decision:
        raise ValueError("Stage2 decision disagrees with compile/runtime metadata")

    shared_expert_id = metadata.get("shared_expert_id", -1)
    is_fhmoe = isinstance(shared_expert_id, int) and shared_expert_id >= 0
    is_a16w_mix = metadata.get("a_dtype") == "bf16" and metadata.get("b_dtype") in (
        "fp4",
        "int4",
    )
    if is_a16w_mix:
        if metadata.get("enable_bias", False):
            raise ValueError("a16w-mix Stage2 does not support expert bias")
        if runtime_metadata.return_per_slot:
            raise ValueError("a16w-mix Stage2 does not support per-slot output")
        if metadata.get("b_dtype") == "fp4" and runtime_metadata.mode != "atomic":
            raise ValueError("a16w4 Stage2 supports only atomic mode")
    if is_fhmoe:
        primary_op_id = FHMOE_STAGE2_GEMM_OP_ID
    elif is_a16w_mix and metadata.get("b_dtype") == "fp4":
        primary_op_id = A16_STAGE2_GEMM_OP_ID
    elif decision.primary_family == "mixed":
        primary_op_id = MIXED_STAGE2_GEMM_OP_ID
    else:
        primary_op_id = INT4_STAGE2_GEMM_OP_ID

    primary_metadata = dict(metadata)
    if is_a16w_mix:
        tile_n = int(primary_metadata["tile_n"])
        model_dim = int(primary_metadata["model_dim"])
        if model_dim % tile_n:
            primary_metadata["tile_n"] = 256 if model_dim % 256 == 0 else 128
        tile_k = int(primary_metadata["tile_k"])
        inter_dim = int(primary_metadata["inter_dim"])
        if inter_dim % tile_k:
            primary_metadata["tile_k"] = 128 if inter_dim % 128 == 0 else 64
    primary_overrides = {
        "persist_m": (
            (-1 if runtime_metadata.persist is True else 1)
            if is_a16w_mix
            else decision.persist_m
        ),
        "out_dtype": decision.gemm_out_dtype,
    }
    if not is_a16w_mix:
        primary_overrides["accumulate"] = decision.accumulate
    if not is_fhmoe:
        primary_overrides["mode"] = runtime_metadata.mode
    requests = [
        _request(
            primary_op_id,
            primary_metadata,
            target,
            registry,
            **primary_overrides,
        )
    ]
    if decision.reduction_kind != "none":
        reduction_op_id = (
            MASKED_REDUCTION_OP_ID
            if decision.reduction_kind == "masked"
            else PLAIN_REDUCTION_OP_ID
        )
        reduction_metadata = {**metadata, **vars(runtime_metadata)}
        requests.append(
            _request(
                reduction_op_id,
                reduction_metadata,
                target,
                registry,
                dtype_str=decision.reduction_input_dtype,
                use_mask=decision.reduction_kind == "masked",
                num_experts=decision.reduction_num_experts,
                out_dtype_str=(
                    runtime_metadata.out_dtype_str or decision.reduction_out_dtype
                ),
                use_weight=runtime_metadata.use_weight,
                scale_blk=runtime_metadata.scale_blk,
                pitch_align=runtime_metadata.pitch_align,
            )
        )
    return tuple(requests)


def sorting_compile_request(
    case: MoeSortingCompileCase,
    target: RocmTarget,
    *,
    specialization: Any = None,
    registry: CompileOpRegistry = DEFAULT_COMPILE_OP_REGISTRY,
) -> CompileRequest:
    """Return one concrete oneshot, P0v2/P23, or four-kernel request."""

    if not isinstance(case, MoeSortingCompileCase):
        raise TypeError(
            f"case must be a MoeSortingCompileCase, got {type(case).__name__}"
        )
    register_moe_sorting_ops(registry)
    from .kernels.moe_sorting_kernel import (
        SORTING_PATH_4K_FUSED,
        SORTING_PATH_ONESHOT,
        SORTING_PATH_P0V2_P23,
        MoeSortingSpecialization,
        resolve_moe_sorting_specialization,
    )

    if specialization is None:
        specialization = resolve_moe_sorting_specialization(
            arch=target.arch,
            max_tokens=case.max_tokens,
            num_experts=case.num_experts,
            topk=case.topk,
            unit_size=case.unit_size,
            has_mask=case.has_mask,
            has_local_tokens=case.has_local_tokens,
            path=case.path,
            k4_block=case.k4_block,
        )
    elif not isinstance(specialization, MoeSortingSpecialization):
        raise TypeError("specialization must be a MoeSortingSpecialization")
    if (
        specialization.max_tokens != case.max_tokens
        or specialization.has_mask != case.has_mask
        or specialization.has_local_tokens != case.has_local_tokens
    ):
        raise ValueError("sorting specialization disagrees with compile case")

    metadata = vars(case)
    if specialization.path == SORTING_PATH_ONESHOT:
        return _request(
            SORTING_ONESHOT_OP_ID,
            metadata,
            target,
            registry,
            max_tokens=specialization.launcher_max_tokens,
        )
    if specialization.path == SORTING_PATH_P0V2_P23:
        return _request(
            SORTING_P0V2_P23_OP_ID,
            metadata,
            target,
            registry,
            k4_block=specialization.k4_block,
        )
    if specialization.path == SORTING_PATH_4K_FUSED:
        return _request(
            SORTING_4K_FUSED_OP_ID,
            metadata,
            target,
            registry,
            k4_block=specialization.k4_block,
        )
    raise RuntimeError(f"unhandled sorting path {specialization.path!r}")


def cktile_epilogue_compile_requests(
    metadata: Mapping[str, Any],
    target: RocmTarget,
    *,
    registry: CompileOpRegistry = DEFAULT_COMPILE_OP_REGISTRY,
) -> tuple[CompileRequest, ...]:
    """Return the optional CK-Tile interleaved Stage1 FlyDSL epilogue."""

    metadata = _mapping(metadata, "CK-Tile epilogue metadata")
    register_moe_stage1_ops(registry)
    layout = _required_cktile(metadata, "post_activation_layout")
    split_k = _required_cktile(metadata, "split_k")
    act = _required_cktile(metadata, "act")
    enable_bias = metadata.get("enable_bias", False)

    if layout not in ("auto", "standard", "interleaved"):
        raise ValueError(f"unsupported post_activation_layout: {layout}")
    if layout == "auto":
        raise ValueError("post_activation_layout='auto' is ambiguous")
    if isinstance(split_k, bool) or not isinstance(split_k, int) or split_k <= 0:
        raise ValueError(f"split_k must be a positive integer, got {split_k!r}")
    if split_k == 1 or layout == "standard" or act == "gelu":
        return ()
    if enable_bias:
        raise ValueError("CK-Tile interleaved split-K bias is unsupported")
    if act == "silu":
        return (
            _request(
                FQ_ACTIVATION_OP_ID,
                metadata,
                target,
                registry,
                quant_mode="none",
                gui_layout=True,
            ),
        )
    if act == "swiglu":
        return (
            _request(
                CKTILE_SWIGLU_AND_MUL_OP_ID,
                metadata,
                target,
                registry,
            ),
        )
    raise ValueError(f"unsupported CK-Tile activation: {act!r}")


def _required_cktile(values: Mapping[str, Any], name: str) -> Any:
    try:
        return values[name]
    except KeyError as error:
        raise TypeError(f"missing required CK-Tile metadata: {name}") from error
