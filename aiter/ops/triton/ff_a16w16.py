# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import torch
from aiter.ops.triton.gemm_a16w16 import gemm_a16w16
from aiter.ops.triton.gemm_a16w16_gated import gemm_a16w16_gated


def ff_a16w16_nogate(
    x,
    w_up,
    w_down,
    dtype: Optional[float] = torch.bfloat16,
    intermediate: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    activation: Optional[str] = None,
):
    """
    Full feed-forward block with gating (e.g swiglu).

    x: torch.Tensor (M, K)
    w_up: torch.Tensor (N, K)
    w_down: torch.Tensor (N, K)
    intermediate: torch.Tensor (M, N)
    y: torch.Tensor (M, K)
    activation: One of ("silu", "relu", "gelu", "gelu_tanh", "silu_exp2", None)
    """
    # Shape checks
    assert (
        x.shape[1] == w_up.shape[1] == w_down.shape[1]
    ), f"Incompatible matrix shapes: x:{x.shape}, w_up:{w_up.shape}, w_down:{w_down.shape}"
    assert (
        w_up.shape[0] == w_down.shape[0]
    ), f"Incompatible matrix shapes: w_up:{w_up.shape}, w_down:{w_down.shape}"

    M, K = x.shape
    N = w_up.shape[0]

    if intermediate is None:
        intermediate = torch.empty((M, N), dtype=dtype, device=x.device)

    intermediate = gemm_a16w16(
        x,
        w_up,
        dtype=dtype,
        y=intermediate,
        config=config,
        activation=activation,
    )

    if y is None:
        y = torch.empty((M, K), dtype=dtype, device=x.device)
    y = gemm_a16w16(intermediate, w_down.T, dtype=dtype, config=config, y=y)

    return y


def ff_a16w16_gated(
    x,
    w_up,
    w_down,
    dtype: Optional[float] = torch.bfloat16,
    intermediate: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    activation: Optional[str] = None,
):
    """
    Full feed-forward block with gating (e.g swiglu).

    x: torch.Tensor (M, K)
    w_up: torch.Tensor (N, K)
    w_down: torch.Tensor (N//2, K)
    intermediate: torch.Tensor (M, N//2)
    y: torch.Tensor (M, K)
    activation: One of ("silu", "relu", "gelu", "gelu_tanh", "silu_exp2", None)
    """
    # Shape checks
    assert (
        x.shape[1] == w_up.shape[1] == w_down.shape[1]
    ), f"Incompatible matrix shapes: x:{x.shape}, w_up:{w_up.shape}, w_down:{w_down.shape}"
    assert (
        w_up.shape[0] == w_down.shape[0] * 2
    ), f"Incompatible matrix shapes: w_up:{w_up.shape}, w_down:{w_down.shape}"
    assert w_up.shape[0] % 2 == 0, "Shape incompatible with gating"

    M, K = x.shape
    N = w_up.shape[0]

    if intermediate is None:
        intermediate = torch.empty((M, N // 2), dtype=dtype, device=x.device)
    intermediate = gemm_a16w16_gated(
        x,
        w_up,
        y=intermediate,
        dtype=dtype,
        config=config,
        activation=activation,
    )
    if y is None:
        y = torch.empty((M, K), dtype=dtype, device=x.device)
    y = gemm_a16w16(intermediate, w_down.T, dtype=dtype, config=config, y=y)

    return y
