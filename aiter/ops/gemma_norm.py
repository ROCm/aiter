# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Gemma normalization ops.

This module exposes Python bindings for Gemma-style RMSNorm and fused
add-RMSNorm operations via `compile_ops("module_gemma_norm", ...)`.

The underlying CUDA kernel implementations live in:
  - csrc/kernels/gemma_norm_kernels.cu
  - csrc/pybind/gemma_norm_pybind.cu
"""
import torch
from torch import Tensor
from ..jit.core import compile_ops


def gemma_rmsnorm(
    input: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """
    Gemma-style RMSNorm: returns (input / RMS(input)) * (1 + weight).
    Only fp16 and bf16.
    """
    output = torch.empty_like(input, dtype=input.dtype, device=input.device)
    _gemma_rmsnorm(output, input, weight, eps)
    return output


@compile_ops("module_gemma_norm", fc_name="gemma_rmsnorm")
def _gemma_rmsnorm(
    output: Tensor,
    input: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
) -> None:
    """Internal: output = (input / RMS(input)) * (1 + weight)."""
    ...


@compile_ops("module_gemma_norm", fc_name="gemma_fused_add_rmsnorm")
def gemma_fused_add_rmsnorm(
    input: Tensor,
    residual: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
) -> None:
    """
    residual += input; input = (residual / RMS(residual)) * (1 + weight). In-place.
    Only fp16 and bf16.
    """
    ...
