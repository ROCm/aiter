# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.common.activation import _get_activation_from_str
from aiter.ops.triton._triton_kernels.normalization.channel_rmsnorm import (
    _channel_rmsnorm_kernel,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.config_utils import select_leq_config
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.normalization_config_utils import get_normalization_config

_LOGGER = AiterTritonLogger()

__all__ = ["channel_rmsnorm"]


def channel_rmsnorm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    RMS norm over the channel dim (dim 1) of a channel-first tensor, with an
    optional activation, in one kernel.

    Computes ``act(F.normalize(x, dim=1) * sqrt(C) * gamma + bias)``. This is
    the ``WanRMS_norm(channel_first=True)`` of the diffusers Wan VAE, followed
    by its SiLU. ``F.normalize`` divides by ``max(||x||_2, eps)``, so ``eps``
    is not inside the square root.

    Args:
        x (torch.Tensor): Contiguous input with shape (N, C, ...), for example
            (N, C, T, H, W). fp32, fp16 or bf16.
        gamma (torch.Tensor): Scale with C elements, for example shape
            (C, 1, 1, 1). Same dtype as x.
        bias (Optional[torch.Tensor]): Bias with C elements. Same dtype as x.
        activation (Optional[str]): Activation after the norm, for example
            "silu". See ``_get_activation_from_str``.
        eps (float): Lower bound of the L2 norm.

    Returns:
        torch.Tensor: Output with the shape and dtype of x.

    Limits: for fp16 and bf16, the norm is in fp32 and each later step rounds
    to the input dtype, as in eager PyTorch. The activation is in fp32.
    """
    assert x.dim() >= 2, f"x must be (N, C, ...), got {tuple(x.shape)}"
    assert x.is_contiguous(), "x must be contiguous"
    assert x.dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ), f"unsupported dtype: {x.dtype}"
    N, C = x.shape[0], x.shape[1]
    assert gamma.numel() == C, f"gamma must have {C} elements, got {gamma.numel()}"
    assert gamma.dtype == x.dtype, f"dtype mismatch: {gamma.dtype} vs {x.dtype}"
    assert gamma.device == x.device, "gamma must be on the device of x"
    if bias is not None:
        assert bias.numel() == C, f"bias must have {C} elements, got {bias.numel()}"
        assert bias.dtype == x.dtype, f"dtype mismatch: {bias.dtype} vs {x.dtype}"
        assert bias.device == x.device, "bias must be on the device of x"
        bias = bias.contiguous()
    gamma = gamma.contiguous()

    _LOGGER.info(
        "CHANNEL_RMSNORM: x=%s dtype=%s activation=%s",
        tuple(x.shape),
        x.dtype,
        activation,
    )

    out = torch.empty_like(x)
    S = x.numel() // (N * C) if N * C > 0 else 0
    if out.numel() == 0:
        return out

    config = select_leq_config(
        get_normalization_config("channel_rmsnorm", get_arch()), C, prefix="C_LEQ_"
    )
    block_c_max = config.pop("BLOCK_C_MAX")
    BLOCK_S = config.pop("BLOCK_S")
    one_pass = triton.next_power_of_2(C) <= block_c_max
    BLOCK_C = triton.next_power_of_2(C) if one_pass else block_c_max

    grid = (triton.cdiv(S, BLOCK_S), N)
    _channel_rmsnorm_kernel[grid](
        x,
        gamma,
        bias,
        out,
        C,
        S,
        x.stride(0),
        x.stride(1),
        float(C) ** 0.5,
        eps,
        activation=_get_activation_from_str(activation) if activation else "",
        USE_ACTIVATION=activation is not None,
        HAS_BIAS=bias is not None,
        ONE_PASS=one_pass,
        BLOCK_C=BLOCK_C,
        BLOCK_S=BLOCK_S,
        **config,
    )
    return out
