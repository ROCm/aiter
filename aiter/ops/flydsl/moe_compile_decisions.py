# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pure MoE compile decisions shared by runtime and direct AOT."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

Stage1Family = Literal["mixed", "int4"]
Stage1PostprocessKind = Literal["fq", "external"]
Stage2Family = Literal["mixed", "int4"]
Stage2TargetLayout = Literal["accumulate", "per_slot", "reduction"]
Stage2ReductionKind = Literal["plain", "masked", "none"]


@dataclass(frozen=True)
class Stage1CompileDecision:
    """Compile and launch choices derived from Stage1 builder arguments."""

    primary_family: Stage1Family
    split_k: bool
    requested_out_dtype: str
    main_out_dtype: str
    main_enable_bias: bool
    postprocess_kind: Stage1PostprocessKind | None
    fq_quant_mode: str | None
    fq_gui_layout: bool | None
    fq_enable_bias: bool | None


@dataclass(frozen=True)
class Stage2CompileDecision:
    """Compile and target-layout choices for Stage2 and its reduction."""

    primary_family: Stage2Family
    m_blocks: int
    accumulate: bool
    target_layout: Stage2TargetLayout
    persist_m: int
    reduction_kind: Stage2ReductionKind
    reduction_dtype: str
    reduction_num_experts: int


def _required(values: Mapping[str, Any], name: str, stage: str) -> Any:
    try:
        return values[name]
    except KeyError as error:
        raise TypeError(f"missing required {stage} argument: {name}") from error


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def resolve_stage1_compile_decision(
    builder_kwargs: Mapping[str, Any],
) -> Stage1CompileDecision:
    """Resolve Stage1 family, effective GEMM output, and split-K postprocess."""

    if not isinstance(builder_kwargs, Mapping):
        raise TypeError("Stage1 builder kwargs must be a mapping")

    a_dtype = _required(builder_kwargs, "a_dtype", "Stage1")
    b_dtype = _required(builder_kwargs, "b_dtype", "Stage1")
    requested_out_dtype = _required(builder_kwargs, "out_dtype", "Stage1")
    if b_dtype in ("fp4", "fp8"):
        primary_family: Stage1Family = "mixed"
    elif a_dtype == "bf16" and b_dtype == "int4":
        primary_family = "int4"
    else:
        raise ValueError(
            "unsupported Stage1 dtype combination: "
            f"a_dtype={a_dtype!r}, b_dtype={b_dtype!r}"
        )

    k_batch = builder_kwargs.get("k_batch", 1)
    _positive_int("k_batch", k_batch)
    split_k = k_batch > 1
    gate_mode = builder_kwargs.get("gate_mode", "separated")
    enable_bias = builder_kwargs.get("enable_bias", False)
    if not isinstance(enable_bias, bool):
        raise ValueError(f"enable_bias must be a bool, got {enable_bias!r}")

    if primary_family == "mixed":
        if gate_mode == "gate_only":
            raise ValueError("gate_only is reserved and unsupported")
        if gate_mode == "mock_gate_only" and not split_k:
            raise ValueError("mock_gate_only requires split-K")
        if split_k and requested_out_dtype == "fp8" and gate_mode != "interleave":
            raise ValueError("split-K fp8 output requires an interleaved layout")

    main_out_dtype = (
        "bf16"
        if split_k and requested_out_dtype in ("fp4", "fp8")
        else requested_out_dtype
    )
    main_enable_bias = enable_bias and not split_k

    postprocess_kind: Stage1PostprocessKind | None = None
    fq_quant_mode = None
    fq_gui_layout = None
    fq_enable_bias = None
    if split_k:
        if primary_family == "mixed" and gate_mode == "interleave":
            postprocess_kind = "fq"
            fq_quant_mode = (
                requested_out_dtype if requested_out_dtype in ("fp4", "fp8") else "none"
            )
            fq_gui_layout = True
            fq_enable_bias = enable_bias
        elif primary_family == "mixed" and requested_out_dtype == "fp4":
            postprocess_kind = "fq"
            fq_quant_mode = "fp4"
            fq_gui_layout = False
            fq_enable_bias = enable_bias
        else:
            postprocess_kind = "external"

    return Stage1CompileDecision(
        primary_family=primary_family,
        split_k=split_k,
        requested_out_dtype=requested_out_dtype,
        main_out_dtype=main_out_dtype,
        main_enable_bias=main_enable_bias,
        postprocess_kind=postprocess_kind,
        fq_quant_mode=fq_quant_mode,
        fq_gui_layout=fq_gui_layout,
        fq_enable_bias=fq_enable_bias,
    )


def resolve_stage2_m_blocks(
    *,
    token_num: int,
    topk: int,
    experts: int,
    tile_m: int,
    sort_block_m: int,
    routing_block_count: int | None,
) -> int:
    """Resolve the Stage2 grid's M-block count from CPU routing metadata."""

    token_num = _positive_int("token_num", token_num)
    topk = _positive_int("topk", topk)
    experts = _positive_int("experts", experts)
    tile_m = _positive_int("tile_m", tile_m)
    if isinstance(sort_block_m, bool) or not isinstance(sort_block_m, int):
        raise TypeError(f"sort_block_m must be an integer, got {sort_block_m!r}")
    if sort_block_m < 0:
        raise ValueError(f"sort_block_m must be non-negative, got {sort_block_m}")

    sorting_block_m = sort_block_m if sort_block_m > 0 else tile_m
    if routing_block_count is None:
        route_capacity = token_num * topk + experts * sorting_block_m - topk
        routing_block_count = (route_capacity + sorting_block_m - 1) // sorting_block_m
    else:
        routing_block_count = _positive_int(
            "routing_block_count",
            routing_block_count,
        )

    if sorting_block_m == tile_m:
        return min(routing_block_count, token_num * topk)
    total_sorted = routing_block_count * sorting_block_m
    return (total_sorted + tile_m - 1) // tile_m


def resolve_stage2_persist_m(
    *,
    m_blocks: int,
    persist: bool | None,
    a_dtype: str,
) -> int:
    """Resolve Stage2 persistence after the shared M-block calculation."""

    m_blocks = _positive_int("m_blocks", m_blocks)
    if persist is not None and not isinstance(persist, bool):
        raise TypeError(f"persist must be bool or None, got {persist!r}")
    if a_dtype not in ("fp4", "fp8", "bf16"):
        raise ValueError(f"unsupported Stage2 activation dtype: {a_dtype!r}")

    if persist is True:
        persist_m = -1
    elif persist is False:
        persist_m = 4 if m_blocks > 256 else 1
    else:
        persist_m = -1 if m_blocks > 256 else 1
    return 1 if a_dtype == "fp8" else persist_m


def resolve_stage2_compile_decision(
    builder_kwargs: Mapping[str, Any],
    *,
    mode: str,
    return_per_slot: bool,
    persist: bool | None,
    token_num: int,
    routing_block_count: int | None,
    use_mask: bool,
    topk_ids_available: bool,
    num_experts: int,
    accumulate: bool | None = None,
    dtype_str: str | None = None,
) -> Stage2CompileDecision:
    """Resolve Stage2 family, layout, persistence, and optional reduction."""

    if not isinstance(builder_kwargs, Mapping):
        raise TypeError("Stage2 builder kwargs must be a mapping")

    a_dtype = _required(builder_kwargs, "a_dtype", "Stage2")
    b_dtype = _required(builder_kwargs, "b_dtype", "Stage2")
    if a_dtype in ("fp4", "fp8") and b_dtype in ("fp4", "fp8"):
        primary_family: Stage2Family = "mixed"
    elif a_dtype == "bf16" and b_dtype == "int4":
        primary_family = "int4"
    else:
        raise ValueError(
            "unsupported Stage2 dtype combination: "
            f"a_dtype={a_dtype!r}, b_dtype={b_dtype!r}"
        )

    for name in (
        "model_dim",
        "inter_dim",
        "experts",
        "topk",
        "tile_m",
        "tile_n",
        "tile_k",
    ):
        _positive_int(name, _required(builder_kwargs, name, "Stage2"))
    if not isinstance(builder_kwargs.get("doweight_stage2"), bool):
        raise ValueError(
            "doweight_stage2 must explicitly identify route-weight ownership"
        )
    enable_bias = builder_kwargs.get("enable_bias", False)
    if not isinstance(enable_bias, bool):
        raise ValueError(f"enable_bias must be a bool, got {enable_bias!r}")
    if primary_family == "int4" and enable_bias:
        raise ValueError("bf16×int4 Stage2 does not support bias")

    if mode not in ("atomic", "reduce"):
        raise ValueError(f"unsupported Stage2 mode: {mode!r}")
    if not isinstance(return_per_slot, bool):
        raise ValueError(f"return_per_slot must be a bool, got {return_per_slot!r}")
    expected_accumulate = mode != "reduce" and not return_per_slot
    if accumulate is not None:
        if not isinstance(accumulate, bool):
            raise ValueError(f"accumulate must be a bool, got {accumulate!r}")
        if accumulate != expected_accumulate:
            raise ValueError(
                "accumulate disagrees with mode/return_per_slot: "
                f"expected {expected_accumulate}, got {accumulate}"
            )

    if expected_accumulate:
        target_layout: Stage2TargetLayout = "accumulate"
    elif return_per_slot:
        target_layout = "per_slot"
    else:
        target_layout = "reduction"

    out_dtype = str(_required(builder_kwargs, "out_dtype", "Stage2")).lower()
    reduction_dtype = {
        "bf16": "bf16",
        "bfloat16": "bf16",
        "f16": "f16",
        "fp16": "f16",
        "half": "f16",
    }.get(out_dtype)
    if reduction_dtype is None:
        raise ValueError(
            f"unsupported Stage2 output dtype: {builder_kwargs['out_dtype']!r}"
        )
    if dtype_str is not None and dtype_str != reduction_dtype:
        raise ValueError(
            "reduction dtype must match Stage2 output dtype: "
            f"expected {reduction_dtype!r}, got {dtype_str!r}"
        )

    if not isinstance(use_mask, bool):
        raise ValueError(f"use_mask must be a bool, got {use_mask!r}")
    if not isinstance(topk_ids_available, bool):
        raise ValueError(
            f"topk_ids_available must be a bool, got {topk_ids_available!r}"
        )
    if isinstance(num_experts, bool) or not isinstance(num_experts, int):
        raise ValueError(f"num_experts must be an integer, got {num_experts!r}")

    needs_reduction = target_layout == "reduction"
    if use_mask:
        if not needs_reduction:
            raise ValueError("masked reduction requires reduction output")
        if not topk_ids_available:
            raise ValueError("masked reduction requires top-k-id semantics")
        if num_experts <= 0:
            raise ValueError("masked reduction requires a positive global expert count")
    else:
        if topk_ids_available:
            raise ValueError("top-k-id semantics are only valid for masked reduction")
        if num_experts != 0:
            raise ValueError("plain reduction requires num_experts=0")

    routing_experts = (
        num_experts if use_mask else _required(builder_kwargs, "experts", "Stage2")
    )
    m_blocks = resolve_stage2_m_blocks(
        token_num=token_num,
        topk=_required(builder_kwargs, "topk", "Stage2"),
        experts=routing_experts,
        tile_m=_required(builder_kwargs, "tile_m", "Stage2"),
        sort_block_m=builder_kwargs.get("sort_block_m", 0),
        routing_block_count=routing_block_count,
    )
    persist_m = resolve_stage2_persist_m(
        m_blocks=m_blocks,
        persist=persist,
        a_dtype=a_dtype,
    )
    if not needs_reduction:
        reduction_kind: Stage2ReductionKind = "none"
    elif use_mask:
        reduction_kind = "masked"
    else:
        reduction_kind = "plain"

    return Stage2CompileDecision(
        primary_family=primary_family,
        m_blocks=m_blocks,
        accumulate=expected_accumulate,
        target_layout=target_layout,
        persist_m=persist_m,
        reduction_kind=reduction_kind,
        reduction_dtype=reduction_dtype,
        reduction_num_experts=num_experts,
    )
