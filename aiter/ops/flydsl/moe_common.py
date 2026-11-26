# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared types and activation configuration for MoE FlyDSL modules."""

import math
import os
from enum import Enum

import torch

DEFAULT_SITUV2_BETA = 4.0
DEFAULT_SITUV2_LINEAR_BETA = 25.0


#: ``(model_dim, inter_dim)`` gate/up widths whose interleaved a8w4 prefill
#: launches the wide-tile, XCD-grouping and CShuffle-budget rules were measured
#: on (DeepSeek-V4.1-Flash and DeepSeek-V4-Pro on MI355X).  Other widths keep
#: the stock launch and the stock CShuffle budget: the token boundaries are
#: tuning results, not arithmetic, so they are not extrapolated to models the
#: evidence did not time.
FLYDSL_PREFILL_SHAPES = frozenset(
    {
        (5120, 384),
        (5120, 640),
        (5120, 1152),
        (7168, 384),
    }
)

#: Architectures the rules were measured on.  Other archs (gfx942, gfx1250)
#: keep the stock launch and the stock CShuffle budget until timed.
FLYDSL_PREFILL_ARCHS = frozenset({"gfx950"})


def flydsl_stage1_measured_prefill(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    act: str,
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    k_batch: int,
    gate_up_interleave: bool,
    tile_m: int,
    inter_dim_pad: int,
    model_dim_pad: int,
    gpu_arch: str,
    has_bias: bool = False,
    has_sorted_weights: bool = False,
) -> bool:
    """Whether this call is one of the interleaved a8w4 prefill shapes measured
    for the wide-tile, XCD-grouping and CShuffle-budget rules.  Keep this in one
    place so the three rules cannot drift apart; every other call keeps the
    stock launch and the stock LDS budget.  The rules are MI355X/gfx950
    measurements; gfx942/gfx1250 are not extrapolated.

    The predicate names exactly the dimensions the timing evidence covers:
    the four measured ``(model_dim, inter_dim)`` pairs at E 384/385, topk 6/7,
    a 64/128-row sorted block, and the bf16 or fp8 epilogue.  Other experts,
    topk, block sizes, epilogues and widths keep the stock launch and the stock
    LDS budget until they are timed."""
    out_s = str(out_dtype).strip().lower()
    return (
        bool(gate_up_interleave)
        and a_dtype == "fp8"
        and b_dtype == "fp4"
        and out_s in ("bf16", "bfloat16", "fp8")
        and int(k_batch) == 1
        and int(topk) in (6, 7)
        and str(act) == "silu"
        and int(experts) in (384, 385)
        and int(tile_m) in (64, 128)
        and (int(model_dim), int(inter_dim)) in FLYDSL_PREFILL_SHAPES
        and int(model_dim_pad) == 0
        and int(inter_dim_pad) == 0
        and str(gpu_arch) in FLYDSL_PREFILL_ARCHS
        and not has_bias
        and not has_sorted_weights
    )


def _flydsl_stage2_fp8_enabled():
    return os.environ.get("AITER_FLYDSL_STAGE2_FP8", "0") == "1"


def _mxfp4_intermediate_enabled():
    # Lossy before-sum 4-bit quant of the scatter intermediate (ok for gsm8k, degrades other evals).
    return os.environ.get("AITER_MXFP4_INTERMEDIATE", "0") == "1"


def resolve_swiglu_limit(
    swiglu_limit: "object" = None,
) -> "tuple[float | None, str]":
    """Return the validated SiLU/SwiGLU clamp limit and the name it came from.

    ``None`` falls back to AITER_MXFP4_TUNE_SWIGLU_LIMIT; an unset or empty
    variable keeps the activation default (``None``).
    """
    source = "swiglu_limit"
    if swiglu_limit is None:
        swiglu_limit = os.environ.get("AITER_MXFP4_TUNE_SWIGLU_LIMIT") or None
        source = "AITER_MXFP4_TUNE_SWIGLU_LIMIT"
    if swiglu_limit is None:
        return None, source
    try:
        limit = float(swiglu_limit)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source} must be a finite positive number") from error
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError(f"{source} must be a finite positive number")
    return limit, source


def get_flydsl_activation_name(activation) -> str:
    # ActivationType is backed by module_aiter_core. Keep this import lazy because
    # wheel AOT job discovery imports moe_common before that extension is importable.
    from aiter.ops.enum import ActivationType

    activation_names = {
        ActivationType.Silu: "silu",
        ActivationType.Swiglu: "swiglu",
        ActivationType.Situv2: "situv2",
    }
    try:
        return activation_names[activation]
    except KeyError as error:
        raise ValueError(
            f"Unsupported FlyDSL MoE activation: {activation!r}"
        ) from error


class GateMode(str, Enum):
    """Gate/Up computation strategy for stage1 GEMM.

    SEPARATED:      Two separate B-tile streams (gate + up), default mode.
    MOCK_GATE_ONLY: Single B-tile stream over full [0, 2*inter_dim), simulates
                    gate-only by doubling grid X on top of SEPARATED layout.
                    Requires split-K (k_batch>1).  NOT true gate-only.
    GATE_ONLY:      Reserved for future true gate-only implementation.
    INTERLEAVE:     Weight rows interleave gate/up (gate[0], up[0], gate[1], ...).
                    pack_N=2 routes even/odd N subtiles.  NOT tied to split-K.
    """

    SEPARATED = "separated"
    MOCK_GATE_ONLY = "mock_gate_only"
    GATE_ONLY = "gate_only"
    INTERLEAVE = "interleave"


def apply_gate_up(
    gate: torch.Tensor,
    up: torch.Tensor,
    act: str,
    swiglu_limit: float | None = None,
    situ_beta: float = 1.0,
    situ_linear_beta: float = 1.0,
) -> torch.Tensor:
    """Torch reference for the stage1 gate/up activation.

    ``situv2`` (Kimi-K3 ``hidden_act="situ"``) is the grouped TDM stage1
    epilogue's ``stage1_act=3``; this is what that kernel is checked against.
    """
    lim = 7.0 if swiglu_limit is None else float(swiglu_limit)
    if act == "swiglu":
        gate = gate.clamp(max=lim)
        up = up.clamp(min=-lim, max=lim)
        return gate * torch.sigmoid(1.702 * gate) * (up + 1.0)
    if act == "situv2":
        situ_gate = (
            float(situ_beta) * torch.tanh(gate / float(situ_beta)) * torch.sigmoid(gate)
        )
        up_scaled = float(situ_linear_beta) * torch.tanh(up / float(situ_linear_beta))
        return situ_gate * up_scaled
    if swiglu_limit is not None:
        gate = gate.clamp(max=lim)
        up = up.clamp(min=-lim, max=lim)
    return torch.nn.functional.silu(gate) * up
