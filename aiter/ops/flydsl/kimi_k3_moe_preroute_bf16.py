# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exact-BF16 Kimi-K3 B1 MoE pre-route fusion for gfx950."""

import functools
import math

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.utils import is_flydsl_available

_BATCH_SIZE = 1
_HIDDEN_SIZE = 7168
_ROUTED_SIZE = 3584
_SHARED_GATE_UP_SIZE = 1536
_SHARED_INTERMEDIATE_SIZE = 768

# Frozen from the MI355X shape sweep. Keeping these values here makes the
# production dispatch deterministic; builder-level parameters remain
# available for future architecture-specific tuning.
_DUAL_ROWS_PER_WAVE = 3
_DUAL_CU_COUNT = 256
_DUAL_WAVES_PER_EU = 0
_DUAL_WEIGHT_CACHE_MODIFIER = 2
_DUAL_HIDDEN_TO_LDS = True
_SHARED_ROWS_PER_WAVE = 1
_SHARED_CU_COUNT = 256
_SHARED_WAVES_PER_EU = 0
_SHARED_WEIGHT_CACHE_MODIFIER = 2


def is_kimi_k3_moe_preroute_bf16_available() -> bool:
    """Return whether the fixed-shape gfx950 backend can be compiled."""

    return is_flydsl_available() and get_gfx_runtime() == "gfx950"


def supports_kimi_k3_moe_preroute_bf16(
    hidden: torch.Tensor,
    routed_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
) -> bool:
    """Return whether tensors satisfy the fixed Kimi-K3 B1 contract."""

    tensors = (
        hidden,
        routed_weight,
        shared_gate_up_weight,
        shared_down_weight,
    )
    return (
        all(tensor.is_cuda for tensor in tensors)
        and all(tensor.dtype == torch.bfloat16 for tensor in tensors)
        and all(tensor.is_contiguous() for tensor in tensors)
        and len({tensor.device for tensor in tensors}) == 1
        and hidden.shape == (_BATCH_SIZE, _HIDDEN_SIZE)
        and routed_weight.shape == (_ROUTED_SIZE, _HIDDEN_SIZE)
        and shared_gate_up_weight.shape == (_SHARED_GATE_UP_SIZE, _HIDDEN_SIZE)
        and shared_down_weight.shape == (_HIDDEN_SIZE, _SHARED_INTERMEDIATE_SIZE)
        and is_kimi_k3_moe_preroute_bf16_available()
    )


@functools.cache
def _compiled_dual_projection():
    from aiter.ops.flydsl.kernels.kimi_k3_dual_projection_bf16_gfx950 import (
        build_kimi_k3_b1_dual_projection_bf16_module,
    )

    return build_kimi_k3_b1_dual_projection_bf16_module(
        rows_per_wave=_DUAL_ROWS_PER_WAVE,
        cu_count=_DUAL_CU_COUNT,
        waves_per_eu=_DUAL_WAVES_PER_EU,
        weight_cache_modifier=_DUAL_WEIGHT_CACHE_MODIFIER,
        hidden_to_lds=_DUAL_HIDDEN_TO_LDS,
    )


@functools.cache
def _compiled_shared_down(
    situ_beta: float,
    situ_linear_beta: float,
):
    from aiter.ops.flydsl.kernels.kimi_k3_shared_down_bf16_gfx950 import (
        build_kimi_k3_b1_shared_down_bf16_module,
    )

    return build_kimi_k3_b1_shared_down_bf16_module(
        rows_per_wave=_SHARED_ROWS_PER_WAVE,
        cu_count=_SHARED_CU_COUNT,
        waves_per_eu=_SHARED_WAVES_PER_EU,
        weight_cache_modifier=_SHARED_WEIGHT_CACHE_MODIFIER,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )


def kimi_k3_moe_preroute_bf16(
    hidden: torch.Tensor,
    routed_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
    situ_beta: float = 4.0,
    situ_linear_beta: float = 25.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return routed latent output and the shared-expert partial output."""

    if (
        not math.isfinite(situ_beta)
        or not math.isfinite(situ_linear_beta)
        or situ_beta <= 0.0
        or situ_linear_beta <= 0.0
    ):
        raise ValueError("SiTU beta values must be finite and positive")
    if not supports_kimi_k3_moe_preroute_bf16(
        hidden,
        routed_weight,
        shared_gate_up_weight,
        shared_down_weight,
    ):
        raise ValueError("unsupported Kimi-K3 B1 pre-route BF16 inputs")

    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

    routed_output = torch.empty(
        (_BATCH_SIZE, _ROUTED_SIZE),
        dtype=hidden.dtype,
        device=hidden.device,
    )
    shared_gate_up = torch.empty(
        (_BATCH_SIZE, _SHARED_GATE_UP_SIZE),
        dtype=hidden.dtype,
        device=hidden.device,
    )
    shared_output = torch.empty(
        (_BATCH_SIZE, _HIDDEN_SIZE),
        dtype=hidden.dtype,
        device=hidden.device,
    )
    stream = torch.cuda.current_stream(hidden.device)

    _compiled_dual_projection()(
        ptr_arg(hidden),
        ptr_arg(routed_weight),
        ptr_arg(shared_gate_up_weight),
        ptr_arg(routed_output),
        ptr_arg(shared_gate_up),
        stream=stream,
    )
    _compiled_shared_down(
        float(situ_beta),
        float(situ_linear_beta),
    )(
        ptr_arg(shared_gate_up),
        ptr_arg(shared_down_weight),
        ptr_arg(shared_output),
        stream=stream,
    )
    return routed_output, shared_output
