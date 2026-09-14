# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Explicit Kimi-K3 latent FHMoE API.

This contract is intentionally separate from :mod:`aiter.fhmoe`: K3 has two
activation tensors, two model widths, two intermediate widths, BF16 shared
weights, and two outputs. Treating it as the DSV4 homogeneous contract would
make pointer arithmetic and output ownership ambiguous.
"""

from __future__ import annotations

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.jit.utils.torch_guard import torch_compile_guard

K3_ROUTED_MODEL_DIM = 3584
K3_ROUTED_INTER_DIM = 384
K3_SHARED_MODEL_DIM = 7168
K3_SHARED_INTER_DIM = 768
K3_SHARED_GATE_UP_DIM = 1536
K3_SITUV2_BETA = 4.0
K3_SITUV2_LINEAR_BETA = 25.0
_SUPPORTED_DECODE_M = frozenset((1, 8, 16))


def _expect_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"Expected {name} shape {shape}, got {tuple(tensor.shape)}")


def _validate_latent_fhmoe_contract(
    routed_input: torch.Tensor,
    routed_w1: torch.Tensor,
    routed_w2: torch.Tensor,
    routed_w1_scale: torch.Tensor,
    routed_w2_scale: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_input: torch.Tensor,
    shared_w1: torch.Tensor,
    shared_w2: torch.Tensor,
    activation: ActivationType,
    quant_type: QuantType,
    beta: float,
    linear_beta: float,
) -> None:
    if get_gfx() != "gfx950":
        raise NotImplementedError("K3 latent FHMoE currently requires gfx950")

    m = routed_input.shape[0]
    if m not in _SUPPORTED_DECODE_M:
        raise ValueError(f"K3 latent FHMoE supports M in {sorted(_SUPPORTED_DECODE_M)}")
    _expect_shape("routed_input", routed_input, (m, K3_ROUTED_MODEL_DIM))
    _expect_shape("shared_input", shared_input, (m, K3_SHARED_MODEL_DIM))

    if routed_input.dtype != torch.bfloat16 or shared_input.dtype != torch.bfloat16:
        raise ValueError("K3 latent FHMoE inputs must be BF16")
    if routed_w1.dtype != dtypes.fp4x2 or routed_w2.dtype != dtypes.fp4x2:
        raise ValueError("K3 latent FHMoE routed weights must use MXFP4")
    if routed_w1.ndim != 3 or routed_w2.ndim != 3:
        raise ValueError("K3 latent FHMoE routed weights must be rank 3")

    experts = routed_w1.shape[0]
    _expect_shape(
        "routed_w1",
        routed_w1,
        (experts, 2 * K3_ROUTED_INTER_DIM, K3_ROUTED_MODEL_DIM // 2),
    )
    _expect_shape(
        "routed_w2",
        routed_w2,
        (experts, K3_ROUTED_MODEL_DIM, K3_ROUTED_INTER_DIM // 2),
    )
    _expect_shape(
        "routed_w1_scale",
        routed_w1_scale,
        (experts, 2 * K3_ROUTED_INTER_DIM, K3_ROUTED_MODEL_DIM // 32),
    )
    _expect_shape(
        "routed_w2_scale",
        routed_w2_scale,
        (
            experts,
            K3_ROUTED_MODEL_DIM,
            ((K3_ROUTED_INTER_DIM + 255) // 256) * 256 // 32,
        ),
    )
    if routed_w1_scale.dtype not in (dtypes.fp8_e8m0, torch.uint8):
        raise ValueError("K3 routed_w1_scale must use E8M0 storage")
    if routed_w2_scale.dtype not in (dtypes.fp8_e8m0, torch.uint8):
        raise ValueError("K3 routed_w2_scale must use E8M0 storage")

    _expect_shape("shared_w1", shared_w1, (1, 1536, 7168))
    _expect_shape("shared_w2", shared_w2, (1, 7168, 768))
    if shared_w1.dtype != torch.bfloat16 or shared_w2.dtype != torch.bfloat16:
        raise ValueError("K3 latent FHMoE shared weights must be BF16")

    if topk_ids.ndim != 2 or topk_ids.shape[0] != m:
        raise ValueError("topk_ids must have shape [M, topk]")
    if topk_weight.shape != topk_ids.shape:
        raise ValueError("topk_weight and topk_ids must have identical shapes")
    if topk_ids.shape[1] > 16:
        raise ValueError("K3 latent FHMoE routed topk must not exceed 16")
    if topk_ids.dtype != torch.int32:
        raise ValueError("topk_ids must be int32")
    if topk_weight.dtype != torch.float32:
        raise ValueError("topk_weight must be float32")

    if activation != ActivationType.Situv2:
        raise ValueError("K3 latent FHMoE requires SiTUv2")
    if quant_type != QuantType.per_1x32:
        raise ValueError("K3 latent FHMoE requires per_1x32 routed quantization")
    if (beta, linear_beta) != (K3_SITUV2_BETA, K3_SITUV2_LINEAR_BETA):
        raise ValueError("K3 latent FHMoE requires beta=4 and linear_beta=25")

    tensors = (
        routed_w1,
        routed_w2,
        routed_w1_scale,
        routed_w2_scale,
        topk_weight,
        topk_ids,
        shared_input,
        shared_w1,
        shared_w2,
    )
    if any(t.device != routed_input.device for t in tensors):
        raise ValueError("All K3 latent FHMoE tensors must be on one device")
    if any(not t.is_contiguous() for t in (routed_input, *tensors)):
        raise ValueError("All K3 latent FHMoE tensors must be contiguous")


def latent_fhmoe_fake(
    routed_input: torch.Tensor,
    routed_w1: torch.Tensor,
    routed_w2: torch.Tensor,
    routed_w1_scale: torch.Tensor,
    routed_w2_scale: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_input: torch.Tensor,
    shared_w1: torch.Tensor,
    shared_w2: torch.Tensor,
    activation: int = ActivationType.Situv2.value,
    quant_type: int = QuantType.per_1x32.value,
    beta: float = K3_SITUV2_BETA,
    linear_beta: float = K3_SITUV2_LINEAR_BETA,
) -> tuple[torch.Tensor, torch.Tensor]:
    del (
        routed_w1,
        routed_w2,
        routed_w1_scale,
        routed_w2_scale,
        topk_weight,
        topk_ids,
        shared_w1,
        shared_w2,
        activation,
        quant_type,
        beta,
        linear_beta,
    )
    m = routed_input.shape[0]
    return (
        torch.empty(
            (m, K3_ROUTED_MODEL_DIM),
            dtype=routed_input.dtype,
            device=routed_input.device,
        ),
        torch.empty(
            (m, K3_SHARED_MODEL_DIM),
            dtype=shared_input.dtype,
            device=shared_input.device,
        ),
    )


@torch_compile_guard(gen_fake=latent_fhmoe_fake)
def latent_fhmoe_(
    routed_input: torch.Tensor,
    routed_w1: torch.Tensor,
    routed_w2: torch.Tensor,
    routed_w1_scale: torch.Tensor,
    routed_w2_scale: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_input: torch.Tensor,
    shared_w1: torch.Tensor,
    shared_w2: torch.Tensor,
    activation: int = ActivationType.Situv2.value,
    quant_type: int = QuantType.per_1x32.value,
    beta: float = K3_SITUV2_BETA,
    linear_beta: float = K3_SITUV2_LINEAR_BETA,
) -> tuple[torch.Tensor, torch.Tensor]:
    activation_enum = ActivationType(activation)
    quant_type_enum = QuantType(quant_type)
    _validate_latent_fhmoe_contract(
        routed_input,
        routed_w1,
        routed_w2,
        routed_w1_scale,
        routed_w2_scale,
        topk_weight,
        topk_ids,
        shared_input,
        shared_w1,
        shared_w2,
        activation_enum,
        quant_type_enum,
        beta,
        linear_beta,
    )
    from aiter.ops.flydsl.latent_fhmoe import run_latent_fhmoe

    return run_latent_fhmoe(
        routed_input,
        routed_w1,
        routed_w2,
        routed_w1_scale,
        routed_w2_scale,
        topk_weight,
        topk_ids,
        shared_input,
        shared_w1,
        shared_w2,
        beta=beta,
        linear_beta=linear_beta,
    )


def latent_fhmoe(
    routed_input: torch.Tensor,
    routed_w1: torch.Tensor,
    routed_w2: torch.Tensor,
    routed_w1_scale: torch.Tensor,
    routed_w2_scale: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_input: torch.Tensor,
    shared_w1: torch.Tensor,
    shared_w2: torch.Tensor,
    *,
    beta: float = K3_SITUV2_BETA,
    linear_beta: float = K3_SITUV2_LINEAR_BETA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run K3 latent FHMoE and return ``(routed_output, shared_output)``."""
    return latent_fhmoe_(
        routed_input,
        routed_w1,
        routed_w2,
        routed_w1_scale,
        routed_w2_scale,
        topk_weight,
        topk_ids,
        shared_input,
        shared_w1,
        shared_w2,
        ActivationType.Situv2.value,
        QuantType.per_1x32.value,
        beta,
        linear_beta,
    )
