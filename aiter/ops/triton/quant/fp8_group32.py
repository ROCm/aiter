# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._gluon_kernels.gfx1250.quant.fp8_group32 import (
    _quantize_fp8_group32_kernel,
)
from aiter.ops.triton.quant.mono_utils import check_output, require_wave32

__all__ = ["quantize_fp8_group32"]


def quantize_fp8_group32(
    x: torch.Tensor,
    *,
    persistent: bool = False,
    dequantize: bool = False,
    out: torch.Tensor | None = None,
    scales: torch.Tensor | None = None,
):
    """Quantize BF16 values with ATOM's group32 E4M3/E8M0 ceil-scale recipe.

    Args:
        x: Contiguous BF16 tensor on gfx1250. Its last width is divisible by 32.
        persistent: Use 256 one-dimensional workers, four waves, and stride 256.
        dequantize: Return QAT values in BF16 instead of FP8 and scale tensors.
        out: Optional contiguous output with x.shape and the result dtype.
        scales: Optional contiguous E8M0 tensor, with last width x.shape[-1]/32.

    Returns:
        (FP8 values, E8M0 scales), or BF16 values when dequantize is True.
        One task processes 1024 elements. This is a standalone stage.

    Outputs must not overlap inputs or each other. Empty inputs do not launch.
    This recipe differs from dynamic_mxfp8_quant. It uses a fixed 1e-10 floor.
    """
    if x.ndim < 1 or x.shape[-1] % 32:
        raise ValueError("The last dimension must be divisible by 32")
    if not x.is_cuda or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("x must be a contiguous BF16 GPU tensor")
    require_wave32()
    if dequantize and scales is not None:
        raise ValueError("scales is not used with dequantize=True")
    dtype = x.dtype if dequantize else torch.float8_e4m3fn
    shape_s = (*x.shape[:-1], x.shape[-1] // 32)
    check_output(out, x.shape, dtype, x.device, "out", contiguous=True)
    check_output(
        scales, shape_s, torch.float8_e8m0fnu, x.device, "scales", contiguous=True
    )
    if out is None:
        out = torch.empty(x.shape, device=x.device, dtype=dtype)
    if not dequantize and scales is None:
        scales = torch.empty(shape_s, device=x.device, dtype=torch.float8_e8m0fnu)
    if x.numel():
        grid = (256,) if persistent else (triton.cdiv(x.numel(), 1024),)
        _quantize_fp8_group32_kernel[grid](
            x,
            out,
            None if scales is None else scales.view(torch.uint8),
            x.numel(),
            dequantize,
            persistent,
            num_warps=4,
        )
    return out if dequantize else (out, scales)
