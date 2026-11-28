# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Activation quantization policy: the activation dtype fused_moe quantizes to.

Fixed before dispatch.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
This module moved here from aiter/fused_moe.py and is shared with the legacy MoE path:
aiter.fused_moe re-exports it and legacy MoE calls run it. Changes must keep legacy
behavior (`python tests/moe_dispatch/snapshot.py --check`) and should loop in the moev2
owners (@qli88). See aiter/moev2/README.md.
"""

import os

import torch

from aiter import ActivationType, QuantType
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.moe_common import (
    DEFAULT_SITUV2_BETA,
    DEFAULT_SITUV2_LINEAR_BETA,
    GateMode,
)
from aiter.utility import dtypes

_SWIGLU_MXFP4_BF16_BOUND = int(os.environ.get("GPTOSS_SWIGLU_MXFP4_BF16_BOUND", "256"))
quant_remap = {QuantType.per_128x128: QuantType.per_1x128}


def _bound_split(M, bound, below, at_or_above):
    """`below if M < bound else at_or_above`, deferring when M is unknown."""
    if M is not None:
        return below if M < bound else at_or_above
    # M >= 0, so a non-positive bound settles the branch without it.
    return at_or_above if bound <= 0 else None


def resolve_activation_dtype(
    quant_type: QuantType | int,
    q_dtype_w: torch.dtype,
    *,
    activation: ActivationType | int = ActivationType.Silu,
    gate_mode: GateMode | str = GateMode.SEPARATED,
    M: int | None = None,
    hidden_dtype: torch.dtype | None = None,
    has_a1_scale: bool = False,
    gfx: str | None = None,
    gate_up_interleave: bool | None = None,
) -> torch.dtype | None:
    """The activation dtype fused_moe() quantizes its input to.

    Lets a caller that produces the activations itself -- an all-to-all
    dispatch, a fused epilogue -- emit that dtype directly rather than one
    fused_moe() converts.

    `M` is the token count. Omit it when the dtype has to be chosen before any
    token exists: the result is then None wherever the answer would have needed
    M, and the caller should fall back to bf16, which fused_moe() accepts from
    any path.

    `q_dtype_w` is `w1.dtype`; `hidden_dtype` / `has_a1_scale` only matter for
    the per_1x128 fp8 passthrough.
    """
    quant_type = QuantType(quant_type)
    quant_type = quant_remap.get(quant_type, quant_type)
    activation = ActivationType(activation)
    gate_mode = GateMode(gate_mode)
    gfx = get_gfx_runtime() if gfx is None else gfx

    q_dtype_a = q_dtype_w if q_dtype_w != torch.uint32 else dtypes.fp8
    # If input is already FP8-quantized (e.g. from FP8 dispatch) with block scale,
    # use FP8 as activation dtype to skip redundant re-quantization
    if (
        quant_type == QuantType.per_1x128
        and hidden_dtype == dtypes.fp8
        and has_a1_scale
    ):
        q_dtype_a = dtypes.fp8
    bf16_fp8_bound = int(os.environ.get("AITER_BF16_FP8_MOE_BOUND", "256"))
    if quant_type == QuantType.per_1x32 and q_dtype_w == dtypes.i4x2:
        # a16wi4: bf16 activations, int4 weights with groupwise scale
        q_dtype_a = dtypes.bf16
    elif quant_type == QuantType.per_1x32 and q_dtype_w == dtypes.fp8:
        # mxfp8: both activation and weight are fp8 (per-1x32 e8m0 microscale).
        q_dtype_a = dtypes.fp8
    elif quant_type == QuantType.per_1x32:
        if activation == ActivationType.Situv2:
            # SiTUv2 defaults to a16w4 (bf16 activation x mxfp4 weight) on the
            # mixed_moe kernels. AITER_SITUV2_A8W4 / AITER_SITUV2_A4W4 select the
            # fp8 / fp4 activation instead; each has its own tuned config
            # (kimik3_{a8w4,a4w4}_tuned_fmoe.csv). Tested before the INTERLEAVE
            # branch below, which would otherwise claim SiTUv2 and pick the
            # activation dtype itself.
            if os.environ.get("AITER_SITUV2_A8W4", "0") == "1":
                q_dtype_a = dtypes.fp8
            elif os.environ.get("AITER_SITUV2_A4W4", "0") == "1":
                q_dtype_a = dtypes.fp4x2
            else:
                q_dtype_a = dtypes.bf16
        elif activation == ActivationType.Swiglu and gate_mode == GateMode.SEPARATED:
            q_dtype_a = _bound_split(
                M, _SWIGLU_MXFP4_BF16_BOUND, dtypes.bf16, dtypes.fp4x2
            )
        elif activation == ActivationType.Relu2:
            q_dtype_a = dtypes.bf16
        elif activation == ActivationType.Swiglu or gate_mode == GateMode.INTERLEAVE:
            if gfx != "gfx950" or gate_up_interleave is False:
                q_dtype_a = dtypes.bf16
            else:
                q_dtype_a = _bound_split(M, bf16_fp8_bound, dtypes.bf16, dtypes.fp8)
        else:
            q_dtype_a = dtypes.fp4x2

    if gfx in ("gfx1250", "gfx1250-strict"):
        if os.environ.get("AITER_FORCE_A8W4", "0") in ("1"):
            q_dtype_a = dtypes.fp8
        else:
            q_dtype_a = dtypes.fp4x2

    # gfx942 (CDNA3) has no FP4 MFMA, so an fp4x2 activation quant aborts in
    # fused_dynamic_mx_quant_moe_sort ("not support output type: fp4x2"). The
    # gfx950 / gfx1250 paths are arch-gated above; the Swiglu+SEPARATED large-M
    # split and the fall-through else still resolve fp4x2 on gfx942. Mirror that
    # gate for gfx942 by falling back to a bf16 activation (a16w4: bf16
    # activation x MXFP4 weight), the intended gfx942 MoE path.
    if gfx == "gfx942" and q_dtype_a == dtypes.fp4x2:
        q_dtype_a = dtypes.bf16

    return q_dtype_a


def _normalize_mxfp4_activation_params(
    activation,
    beta: float | None,
    linear_beta: float | None,
    swiglu_limit: float | None,
) -> tuple[float, float, float | None]:
    situ_beta = 1.0
    situ_linear_beta = 1.0
    if activation == ActivationType.Situv2:
        situ_beta = DEFAULT_SITUV2_BETA if beta is None else float(beta)
        situ_linear_beta = (
            DEFAULT_SITUV2_LINEAR_BETA if linear_beta is None else float(linear_beta)
        )
    normalized_swiglu_limit = (
        swiglu_limit
        if activation in (ActivationType.Silu, ActivationType.Swiglu)
        else None
    )
    return situ_beta, situ_linear_beta, normalized_swiglu_limit
