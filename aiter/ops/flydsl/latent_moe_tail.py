# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Narrow BF16 latent-MoE local-tail primitive."""

import functools
import math

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.utils import is_flydsl_available

_LATENT_DIM = 3584
_HIDDEN_DIM = 7168
_B1_ROWS_PER_BLOCK = 14
_B1_WAVES_PER_EU = 4
# Policy 2 bypasses cache levels that would otherwise retain the one-use
# 49 MiB projection matrix. It is the same FlyDSL cache modifier used by
# existing streamed mixed-MoE weight loads.
_B1_WEIGHT_CACHE_MODIFIER = 2


def supports_latent_moe_tail(
    routed: torch.Tensor,
    shared: torch.Tensor,
    rms_weight: torch.Tensor,
    up_weight: torch.Tensor,
    epsilon: float,
) -> bool:
    """Return whether the fixed gfx950 BF16 primitive supports these tensors."""

    tensors = (routed, shared, rms_weight, up_weight)
    return (
        all(tensor.is_cuda for tensor in tensors)
        and len({tensor.device for tensor in tensors}) == 1
        and all(tensor.dtype == torch.bfloat16 for tensor in tensors)
        and all(tensor.is_contiguous() for tensor in tensors)
        and tuple(routed.shape) == (1, _LATENT_DIM)
        and tuple(shared.shape) == (1, _HIDDEN_DIM)
        and tuple(rms_weight.shape) == (_LATENT_DIM,)
        and tuple(up_weight.shape) == (_HIDDEN_DIM, _LATENT_DIM)
        and math.isfinite(epsilon)
        and epsilon > 0.0
        and is_flydsl_available()
        and get_gfx_runtime() == "gfx950"
    )


@functools.cache
def _compiled_b1_latent_moe_tail(
    rows_per_block: int,
    waves_per_eu: int,
    normalize_in_kernel: bool,
    elements_per_thread: int,
    use_dot2: bool,
    weight_cache_modifier: int,
):
    from aiter.ops.flydsl.kernels.latent_moe_tail_gfx950 import (
        build_b1_latent_moe_tail_module,
    )

    return build_b1_latent_moe_tail_module(
        rows_per_block,
        waves_per_eu,
        normalize_in_kernel,
        elements_per_thread,
        use_dot2,
        weight_cache_modifier,
    )


def _launch_b1_latent_moe_tail(
    routed: torch.Tensor,
    shared: torch.Tensor,
    rms_weight: torch.Tensor,
    up_weight: torch.Tensor,
    epsilon: float,
    *,
    out: torch.Tensor,
    rows_per_block: int,
    waves_per_eu: int,
    normalize_in_kernel: bool = True,
    elements_per_thread: int = 8,
    use_dot2: bool = True,
    weight_cache_modifier: int = 0,
) -> torch.Tensor:
    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

    _compiled_b1_latent_moe_tail(
        rows_per_block,
        waves_per_eu,
        normalize_in_kernel,
        elements_per_thread,
        use_dot2,
        weight_cache_modifier,
    )(
        ptr_arg(routed),
        ptr_arg(shared),
        ptr_arg(rms_weight),
        ptr_arg(up_weight),
        ptr_arg(out),
        float(epsilon),
        stream=torch.cuda.current_stream(routed.device),
    )
    return out


def latent_moe_tail(
    routed: torch.Tensor,
    shared: torch.Tensor,
    rms_weight: torch.Tensor,
    up_weight: torch.Tensor,
    epsilon: float,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse BF16 RMSNorm, FP32-accumulated projection, and BF16 shared add."""

    if not supports_latent_moe_tail(routed, shared, rms_weight, up_weight, epsilon):
        raise NotImplementedError(
            "latent_moe_tail requires contiguous gfx950 BF16 tensors with "
            "shapes (1,3584), (1,7168), (3584,), and (7168,3584)"
        )
    if out is None:
        out = torch.empty_like(shared)
    elif (
        out.device != routed.device
        or out.dtype != torch.bfloat16
        or not out.is_contiguous()
        or tuple(out.shape) != (1, _HIDDEN_DIM)
    ):
        raise ValueError(
            "out must be contiguous BF16 shape (1, 7168) on the input device"
        )
    return _launch_b1_latent_moe_tail(
        routed,
        shared,
        rms_weight,
        up_weight,
        epsilon,
        out=out,
        rows_per_block=_B1_ROWS_PER_BLOCK,
        waves_per_eu=_B1_WAVES_PER_EU,
        weight_cache_modifier=_B1_WEIGHT_CACHE_MODIFIER,
    )
