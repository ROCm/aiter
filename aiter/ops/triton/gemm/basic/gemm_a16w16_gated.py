# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.common.activation import _get_activation_from_str
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16w16_gated import (
    _gemm_a16_w16_gated_kernel,
    _get_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def gemm_a16w16_gated(
    x,
    w,
    dtype: float | None = torch.bfloat16,
    y: torch.Tensor | None = None,
    config: dict | None = None,
    activation: str | None = None,
    bias: torch.Tensor | None = None,
    gate_first: bool = True,
):
    """
    Computes 16 bit gated matrix multiplication Y = X @ W^T with gating mechanism (e.g., SwiGLU).
    Splits the N outputs of X @ W^T + bias in two halves, applies the activation
    to the gate half and multiplies it with the other half, producing (M, N//2) output.

    With gate_first=True (default), the first half is the gate:
    out = act(Y[:, :N//2]) * Y[:, N//2:].
    With gate_first=False, the second half is the gate, as in diffusers GEGLU:
    out = Y[:, :N//2] * act(Y[:, N//2:]).

    Args:
        x (torch.Tensor): Input matrix with shape (M, K).
        w (torch.Tensor): Weight matrix with shape (N, K), internally transposed. N must be even.
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N//2).
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M).
        activation (Optional[str]): Activation function applied to gate ("gelu", "gelu_tanh",
            "silu", "silu_exp2", "relu").
        bias (Optional[torch.Tensor]): Bias vector with shape (N,), added to both halves
            before the activation.
        gate_first (bool): True if the first half of the output is the gate, False if the
            second half is the gate.

    Returns:
        torch.Tensor: Gated output with shape (M, N//2).
    """
    _LOGGER.info("GEMM_A16W16_GATED: x=%s w=%s", tuple(x.shape), tuple(w.shape))

    # Shape checks
    assert x.shape[1] == w.shape[1], "Incompatible matrix shapes."
    M, K = x.shape
    N, K = w.shape

    assert N % 2 == 0, "Weight shape incompatible with gating (N not divisible by 2)"
    if bias is not None:
        assert bias.shape == (
            N,
        ), f"bias must have shape ({N},), got {tuple(bias.shape)}"
        bias = bias.contiguous()

    w = w.T

    if y is None:
        y = torch.empty((M, N // 2), dtype=dtype, device=x.device)

    if config is None:
        config, _ = _get_config(M, N, K)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _gemm_a16_w16_gated_kernel[grid](
        x,
        w,
        bias,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0),
        y.stride(1),
        activation=_get_activation_from_str(activation) if activation else "",
        use_activation=activation is not None,
        ADD_BIAS=bias is not None,
        GATE_FIRST=gate_first,
        **config,
    )

    return y
