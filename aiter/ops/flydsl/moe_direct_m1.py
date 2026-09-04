# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Raw-topk M=1 A4W4 SiTUv2 executor selected by ordinary FMoE metadata."""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any

import torch

from aiter.jit.core import compile_ops
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import (
    DEFAULT_SITUV2_BETA,
    DEFAULT_SITUV2_LINEAR_BETA,
    GateMode,
)
from aiter.ops.flydsl.mxfp4_moe_capability import (
    LOWM_ACTIVATIONS,
    MoeCall,
    check_a4w4_lowm,
    enum_name,
    metadata_kernel_name,
    scale_cols,
)
from aiter.utility import dtypes

DIRECT_M1_STAGE1_PREFIX = "flydsl_moe1_direct_m1_"
DIRECT_M1_STAGE2_PREFIX = "flydsl_moe2_direct_m1_"


@compile_ops("module_quant", develop=True)
def _dynamic_per_group_scaled_quant_fp4_direct_m1_internal(
    out: torch.Tensor,
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_output: torch.Tensor,
) -> None:
    """Private direct-M1 launch: canonical MXFP4 quant plus output zero."""


def _kernel_names_from_cfg(config: dict[str, Any]) -> tuple[str, str]:
    return tuple(str(config.get(f"kernelName{i}", "") or "").strip() for i in (1, 2))


def is_direct_kernel_name(kernel_name: str) -> bool:
    """Whether a name belongs to the direct-M1 executor, at either stage."""
    return kernel_name.startswith((DIRECT_M1_STAGE1_PREFIX, DIRECT_M1_STAGE2_PREFIX))


def is_direct_kernel_pair(kernel_name1: str, kernel_name2: str) -> bool:
    return kernel_name1.startswith(DIRECT_M1_STAGE1_PREFIX) and kernel_name2.startswith(
        DIRECT_M1_STAGE2_PREFIX
    )


def _base_name(kernel_name: str, stage: int) -> str:
    prefix = DIRECT_M1_STAGE1_PREFIX if stage == 1 else DIRECT_M1_STAGE2_PREFIX
    if not kernel_name.startswith(prefix):
        raise ValueError(f"invalid direct-M1 stage{stage} kernel {kernel_name!r}")
    return f"flydsl_moe{stage}_" + kernel_name.removeprefix(prefix)


def _integer(config: dict[str, Any], key: str, default: int = 0) -> int:
    return int(float(config.get(key, default) or default))


@dataclass(frozen=True)
class DirectM1Plan:
    """The stage1/stage2 kernel properties the direct executor is compiled for."""

    stage1: dict[str, Any]
    stage2: dict[str, Any]

    @property
    def block_m(self) -> int:
        return self.stage2["tile_m"]


def _require(stage: int, properties) -> None:
    for label, actual, expected in properties:
        if actual != expected:
            raise ValueError(
                f"direct-M1 stage{stage} {label} must be {expected}, got {actual}"
            )


@functools.cache
def parse_stage1(kernel_name: str) -> dict[str, Any]:
    """Parse a direct stage1 kernel name, or raise naming what it violates.

    Direct execution addresses raw ``(token, topk)`` routes instead of sorted
    rows, so the emitter assumes the compact BM32/PM1 A4W4 shape below.
    """
    from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params

    params = get_flydsl_kernel_params(_base_name(kernel_name, 1))
    if params is None:
        raise ValueError(f"unknown direct-M1 stage1 kernel {kernel_name!r}")
    _require(
        1,
        (
            ("stage", params.get("stage"), 1),
            ("dtypes", (params.get("a_dtype"), params.get("b_dtype")), ("fp4", "fp4")),
            ("out dtype", params.get("out_dtype"), "fp4"),
            # tile_n only sizes grid.x; the route index rides grid.y, so it is
            # free. tile_m is the compact route-block the emitter assumes.
            ("tile_m", params.get("tile_m"), 32),
            ("k_batch", params.get("k_batch", 1), 1),
            ("b_nt", params.get("b_nt", 2), 2),
            ("xcd_swizzle", params.get("xcd_swizzle", 0), 0),
            ("gate_mode", params.get("gate_mode", "separated"), "separated"),
        ),
    )
    return params


@functools.cache
def parse_stage2(kernel_name: str) -> dict[str, Any]:
    """Parse a direct stage2 kernel name, or raise naming what it violates."""
    from aiter.ops.flydsl.mxfp4_kname import parse_flydsl_v2_gemm2_kernel

    params = parse_flydsl_v2_gemm2_kernel(_base_name(kernel_name, 2))
    if params is None:
        raise ValueError(f"unknown direct-M1 stage2 kernel {kernel_name!r}")
    tile = (params["tile_m"], params["tile_n"], params["tile_k"])
    _require(
        2,
        (
            ("dtypes", (params["a_dtype"], params["b_dtype"]), ("fp4", "fp4")),
            ("out dtype", params["out_dtype"], "bf16"),
            ("tile", tile, (32, 128, 128)),
            ("epilog", params["epilog"], "atomic"),
            ("sort_block_m", params["sort_block_m"] or tile[0], tile[0]),
            ("persist", bool(params["persist"]), False),
            ("spart", params["spart"] or 0, 0),
        ),
    )
    return params


def parse_direct_plan(kernel_name1: str, kernel_name2: str) -> DirectM1Plan:
    """Resolve the canonical direct kernel pair into one typed plan."""
    if not is_direct_kernel_pair(kernel_name1, kernel_name2):
        raise ValueError("direct-M1 requires both canonical kernel prefixes")
    return DirectM1Plan(
        stage1=parse_stage1(kernel_name1), stage2=parse_stage2(kernel_name2)
    )


def _check_shapes(plan: DirectM1Plan, hidden: int, inter: int) -> str:
    """Return the first tiling invariant the shapes violate, or ``''``."""
    p1, p2 = plan.stage1, plan.stage2
    for label, size, tile in (
        # k_wave splits the contraction; each wave covers model_dim / k_wave.
        ("model_dim", hidden, p1["tile_k"] * int(p1.get("k_wave", 1) or 1)),
        ("2*inter_dim", 2 * inter, p1["tile_n"]),
        ("model_dim", hidden, p2["tile_n"]),
        ("inter_dim", inter, p2["tile_k"]),
    ):
        if size <= 0 or size % tile:
            return f"{label}={size} is not a positive multiple of {tile}"
    return ""


def candidate_kernel_pairs(model_dim: int, inter_dim: int) -> list[tuple[str, str]]:
    """Enumerate the direct kernel pairs legal for one shape, for the tuner.

    ``parse_stage2`` pins stage2 down to the NT flag, so the space is stage1's
    tiling, which is shape dependent -- hence per-model tuning.
    """
    from aiter.ops.flydsl.moe_kernels import (
        build_flydslv2_gemm2_name,
        get_flydsl_stage1_kernels,
    )

    stage2 = [
        DIRECT_M1_STAGE2_PREFIX
        + build_flydslv2_gemm2_name(
            "fp4",
            "fp4",
            "bf16",
            tm=32,
            tn=128,
            tk=128,
            epilog="atomic",
            persist=False,
            use_nt=use_nt,
            sbm=32,
        ).removeprefix("flydsl_moe2_")
        for use_nt in (False, True)
    ]
    pairs = []
    # Stage1's packed fp4 output is spelled as a ``_fp4`` suffix on a bf16 name.
    for base in get_flydsl_stage1_kernels("fp4", "fp4", "bf16"):
        kn1 = DIRECT_M1_STAGE1_PREFIX + base.removeprefix("flydsl_moe1_") + "_fp4"
        for kn2 in stage2:
            try:
                plan = parse_direct_plan(kn1, kn2)
            except ValueError:
                break  # stage1 rejected; kn2 cannot rescue it
            if not _check_shapes(plan, model_dim, inter_dim):
                pairs.append((kn1, kn2))
    return pairs


def cfg_is_supported(
    config: dict[str, Any],
    *,
    gfx: Any = None,
    token: Any = None,
    model_dim: Any = None,
    inter_dim: Any = None,
    expert: Any = None,
    topk: Any = None,
    activation: Any = None,
    dtype: Any = None,
    q_dtype_a: Any = None,
    q_dtype_w: Any = None,
    q_type: Any = None,
    use_g1u1: Any = None,
    doweight_stage1: Any = None,
) -> tuple[bool, str]:
    """Validate a tuned direct row against the shape key it was resolved for.

    Every keyword defaults to the matching CSV column so AOT job discovery can
    validate a row on its own.
    """
    lookup = {
        "gfx": gfx,
        "token": token,
        "model_dim": model_dim,
        "inter_dim": inter_dim,
        "expert": expert,
        "topk": topk,
        "act_type": activation,
        "dtype": dtype,
        "q_dtype_a": q_dtype_a,
        "q_dtype_w": q_dtype_w,
        "q_type": q_type,
        "use_g1u1": use_g1u1,
        "doweight_stage1": doweight_stage1,
    }
    key = {
        name: config.get(name) if value is None else value
        for name, value in lookup.items()
    }
    try:
        plan = parse_direct_plan(*_kernel_names_from_cfg(config))
        token, hidden, inter, experts, topk = (
            int(key[name])
            for name in ("token", "model_dim", "inter_dim", "expert", "topk")
        )
        # The kernel-name pair is what selects this executor; the row only has
        # to agree on the launch flags it implies.
        flags = tuple(
            _integer(config, name) for name in ("block_m", "run_1stage", "ksplit")
        )
    except (KeyError, TypeError, ValueError) as exc:
        return False, str(exc) if isinstance(exc, ValueError) else "malformed CSV row"

    if enum_name(key["gfx"]) != "gfx950":
        return False, f"requires gfx950, got {key['gfx']}"
    if token != 1:
        return False, f"direct-M1 only serves the token=1 row, got {token}"
    if flags != (plan.block_m, 0, 0):
        return False, (
            f"requires (block_m, run_1stage, ksplit) == "
            f"({plan.block_m}, 0, 0), got {flags}"
        )
    if enum_name(key["act_type"]) not in LOWM_ACTIVATIONS:
        return False, f"unsupported activation {key['act_type']}"
    if enum_name(key["q_type"]) != "per_1x32":
        return False, f"unsupported quant type {key['q_type']}"
    if "bfloat16" not in str(key["dtype"]):
        return False, f"unsupported dtype {key['dtype']}"
    if "float4" not in str(key["q_dtype_a"]) or "float4" not in str(key["q_dtype_w"]):
        return False, "requires MXFP4 activations and weights"
    if int(key["use_g1u1"]) != 1 or int(key["doweight_stage1"]) != 0:
        return False, "requires g1u1 weights with doweight_stage1=0"
    if not 0 < topk <= experts:
        return False, f"topk={topk} is out of range for expert={experts}"
    shape_reason = _check_shapes(plan, hidden, inter)
    return (False, shape_reason) if shape_reason else (True, "")


def _situ_values(act: str, beta: Any, linear_beta: Any) -> tuple[float, float]:
    """The two SiTUv2 kernel arguments; silu ignores them but the ABI keeps them."""
    if act != "situv2":
        return DEFAULT_SITUV2_BETA, DEFAULT_SITUV2_LINEAR_BETA
    beta = DEFAULT_SITUV2_BETA if beta is None else float(beta)
    linear_beta = (
        DEFAULT_SITUV2_LINEAR_BETA if linear_beta is None else float(linear_beta)
    )
    if not (
        math.isfinite(beta)
        and math.isfinite(linear_beta)
        and min(beta, linear_beta) > 0
    ):
        raise ValueError("SiTUv2 beta values must be finite and positive")
    return beta, linear_beta


def runtime_is_supported(metadata: Any, call: MoeCall) -> tuple[bool, str]:
    """Make a host-only dispatch decision; tensor values are never read."""
    if get_gfx() != "gfx950":
        return False, f"requires gfx950, got {get_gfx()}"
    if not metadata.flat or metadata.run_1stage or metadata.ksplit:
        return False, "requires flat=1, run_1stage=0, ksplit=0 metadata"
    try:
        plan = parse_direct_plan(
            metadata_kernel_name(metadata, 1), metadata_kernel_name(metadata, 2)
        )
        _situ_values(enum_name(call.activation), call.beta, call.linear_beta)
    except ValueError as exc:
        return False, str(exc)

    supported, reason = check_a4w4_lowm(call)
    if not supported:
        return False, reason

    hidden, inter, experts, topk = call.hidden, call.inter, call.experts, call.topk
    if call.tokens != 1:
        return False, f"direct-M1 serves a single token, got {call.tokens}"
    if call.w1.dtype != dtypes.fp4x2 or call.w2.dtype != dtypes.fp4x2:
        return False, "packed weights must be MXFP4"
    if call.topk_ids.dtype != torch.int32:
        return False, f"topk_ids must be int32, got {call.topk_ids.dtype}"
    if call.topk_weight.dtype != torch.float32:
        return False, f"topk_weight must be fp32, got {call.topk_weight.dtype}"
    shapes = (
        (call.topk_weight.shape, (1, topk)),
        (call.topk_ids.shape, (1, topk)),
        (call.w1.shape, (experts, 2 * inter, hidden // 2)),
        (call.w2.shape, (experts, hidden, inter // 2)),
    )
    for actual, expected in shapes:
        if tuple(actual) != expected:
            return False, f"expected shape {expected}, got {tuple(actual)}"
    shape_reason = _check_shapes(plan, hidden, inter)
    if shape_reason:
        return False, shape_reason
    if call.block_size_M is not None and int(call.block_size_M) != int(
        metadata.block_m
    ):
        return False, "caller block_size_M overrides the tuned block_m"
    return True, ""


@functools.cache
def _stage1_launcher(
    hidden: int, inter: int, experts: int, topk: int, kernel: str, act: str
):
    from aiter.ops.flydsl.kernels.mixed_moe_gemm_2stage_common import (
        compile_mixed_moe_gemm1_common,
    )

    p = parse_stage1(kernel)
    return compile_mixed_moe_gemm1_common(
        model_dim=hidden,
        inter_dim=inter,
        experts=experts,
        topk=topk,
        tile_m=p["tile_m"],
        tile_n=p["tile_n"],
        tile_k=p["tile_k"],
        doweight_stage1=False,
        a_dtype="fp4",
        b_dtype="fp4",
        out_dtype="fp4",
        act=act,
        persist_m=1,
        use_async_copy=p.get("use_async_copy", False),
        waves_per_eu=p.get("waves_per_eu", 4),
        k_batch=p.get("k_batch", 1),
        b_nt=p.get("b_nt", 2),
        gate_mode=GateMode(p.get("gate_mode", "separated")),
        a_scale_one=p.get("a_scale_one", False),
        xcd_swizzle=p.get("xcd_swizzle", 0),
        k_wave=p.get("k_wave", 1),
        v2_output_layout=True,
        route_centric_m1=True,
    )


@functools.cache
def _stage2_launcher(hidden: int, inter: int, experts: int, kernel: str):
    from aiter.ops.flydsl.kernels.mxmoe_dispatcher import compile_gemm2_a4w4_port

    p = parse_stage2(kernel)
    return compile_gemm2_a4w4_port(
        p["tile_m"],
        p["tile_n"],
        p["tile_k"],
        p["use_nt"],
        hidden,
        p["epilog"],
        inter,
        "fp4",
        "fp4",
        1,
        p["sort_block_m"] or p["tile_m"],
        False,
        0,
        g2_spart=0,
        g2_bf16_lds=p["bf16_lds"],
        g2_kstatic=True,
        out_dtype="bf16",
        route_centric_m1=True,
    )


def run(metadata: Any, call: MoeCall) -> torch.Tensor:
    """Execute compact quant, direct stage1, and direct stage2 kernels."""
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_arg

    x, w1, w2 = call.hidden_states, call.w1, call.w2
    hidden, inter, experts, topk = call.hidden, call.inter, call.experts, call.topk
    kernel1 = metadata_kernel_name(metadata, 1)
    kernel2 = metadata_kernel_name(metadata, 2)
    act = enum_name(call.activation)
    beta, linear_beta = _situ_values(act, call.beta, call.linear_beta)
    device = x.device
    stream = torch.cuda.current_stream(device)

    # The quant kernel also zeroes `out`, which stage2 accumulates into.
    out = torch.empty((1, hidden), dtype=torch.bfloat16, device=device)
    aq = torch.empty((1, hidden // 2), dtype=dtypes.fp4x2, device=device)
    ascale = torch.empty(
        (256, scale_cols(hidden)), dtype=dtypes.fp8_e8m0, device=device
    )
    _dynamic_per_group_scaled_quant_fp4_direct_m1_internal(aq, x, ascale, out)

    interq = torch.empty((topk, inter // 2), dtype=dtypes.fp4x2, device=device)
    inters = torch.empty(
        (topk, scale_cols(inter)), dtype=dtypes.fp8_e8m0, device=device
    )
    _run_compiled(
        _stage1_launcher(hidden, inter, experts, topk, kernel1, act),
        *(
            ptr_arg(t)
            for t in (interq, aq, w1, ascale, call.w1_scale, call.topk_ids, inters)
        ),
        1,
        2 * inter,
        hidden,
        topk,
        beta,
        1 / beta,
        linear_beta,
        1 / linear_beta,
        float("inf"),
        stream,
    )
    _run_compiled(
        _stage2_launcher(hidden, inter, experts, kernel2),
        *(
            t.data_ptr()
            for t in (
                interq,
                inters,
                w2,
                call.w2_scale,
                call.topk_ids,
                call.topk_weight,
                out,
            )
        ),
        topk,
        experts,
        inter,
        hidden,
        stream,
    )
    return out


def aot_jobs(row: dict[str, str]) -> list[dict[str, Any]]:
    if not cfg_is_supported(row)[0]:
        return []
    common = {
        "BM": _integer(row, "block_m"),
        "D_HIDDEN": _integer(row, "model_dim"),
        "D_INTER": _integer(row, "inter_dim"),
        "NE": _integer(row, "expert"),
        "topk": _integer(row, "topk"),
        "act": enum_name(row["act_type"]),
        "direct_m1": True,
    }
    return [
        {**common, "stage": stage, "kernel_name": row[f"kernelName{stage}"].strip()}
        for stage in (1, 2)
    ]


def compile_aot_job(**job: Any) -> None:
    """Warm the exact direct launcher under COMPILE_ONLY."""
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_arg

    buffers = [torch.zeros(256, dtype=torch.uint8, device="cpu") for _ in range(7)]
    hidden, inter = job["D_HIDDEN"], job["D_INTER"]
    if job["stage"] == 1:
        launcher = _stage1_launcher(
            hidden, inter, job["NE"], job["topk"], job["kernel_name"], job["act"]
        )
        args = (
            *(ptr_arg(buffer) for buffer in buffers),
            1,
            2 * inter,
            hidden,
            job["topk"],
            DEFAULT_SITUV2_BETA,
            1 / DEFAULT_SITUV2_BETA,
            DEFAULT_SITUV2_LINEAR_BETA,
            1 / DEFAULT_SITUV2_LINEAR_BETA,
            float("inf"),
            0,
        )
    else:
        launcher = _stage2_launcher(hidden, inter, job["NE"], job["kernel_name"])
        args = (
            *(buffer.data_ptr() for buffer in buffers),
            job["topk"],
            job["NE"],
            inter,
            hidden,
            0,
        )
    _run_compiled(launcher, *args)
