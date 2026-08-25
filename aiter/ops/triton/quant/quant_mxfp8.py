# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.quant.quant_mxfp8 import (
    _convert_from_mxfp8_kernel,
    _convert_to_mxfp8_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

__all__ = [
    "convert_from_mxfp8",
    "convert_to_mxfp8",
]

_LOGGER = AiterTritonLogger()


def convert_to_mxfp8(
    x: torch.Tensor,
    fp8_dtype: torch.dtype,
    quant_block_size: int = 32,
    is_2d_block: bool = False,
    use_sr: bool = False,
    use_asm: bool = True,
    block_m: int = 64,
    block_n: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to MXFP8 format.

    Args:
        x: Input tensor of shape (M, N), dtype float32 or bfloat16.
        fp8_dtype: Target FP8 dtype (torch.float8_e4m3fn or torch.float8_e5m2).
        quant_block_size: Block size for quantization scaling (default 32).
        is_2d_block: Whether to use 2D block scaling.
        use_sr: Whether to use stochastic rounding.
        use_asm: Whether to use inline assembly for conversion.
        block_m: Tile size along M dimension.
        block_n: Tile size along N dimension.

    Returns:
        Tuple of (quantized_tensor, scales) where quantized_tensor has fp8_dtype
        and scales has uint8 dtype with e8m0 format.
    """
    _LOGGER.info(f"CONVERT_TO_MXFP8: x={tuple(x.shape)}")
    assert x.ndim == 2, "Input must be 2D"
    M, N = x.shape

    y = torch.empty((M, N), dtype=fp8_dtype, device=x.device)
    if is_2d_block:
        scale_m = triton.cdiv(M, quant_block_size)
    else:
        scale_m = M
    scale_n = triton.cdiv(N, quant_block_size)
    s = torch.empty((scale_m, scale_n), dtype=torch.uint8, device=x.device)

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _convert_to_mxfp8_kernel[grid](
        x,
        y,
        s,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        s.stride(0),
        s.stride(1),
        0,
        0,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        QUANT_BLOCK_SIZE=quant_block_size,
        IS_2D_BLOCK=is_2d_block,
        USE_SR=use_sr,
        USE_ASM=use_asm,
    )
    return y, s


def convert_from_mxfp8(
    x: torch.Tensor,
    s: torch.Tensor,
    output_dtype: torch.dtype,
    quant_block_size: int = 32,
    is_2d_block: bool = False,
    use_asm: bool = True,
    block_m: int = 64,
    block_n: int = 64,
) -> torch.Tensor:
    """
    Dequantize a tensor from MXFP8 format.

    Args:
        x: Quantized input tensor of shape (M, N), dtype float8.
        s: Scale tensor with uint8 dtype (e8m0 format).
        output_dtype: Target output dtype (torch.float32 or torch.bfloat16).
        quant_block_size: Block size for quantization scaling (default 32).
        is_2d_block: Whether 2D block scaling was used.
        use_asm: Whether to use inline assembly for conversion.
        block_m: Tile size along M dimension.
        block_n: Tile size along N dimension.

    Returns:
        Dequantized tensor with output_dtype.
    """
    _LOGGER.info(f"CONVERT_FROM_MXFP8: x={tuple(x.shape)}")
    assert x.ndim == 2, "Input must be 2D"
    M, N = x.shape

    y = torch.empty((M, N), dtype=output_dtype, device=x.device)

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _convert_from_mxfp8_kernel[grid](
        x,
        y,
        s,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        s.stride(0),
        s.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        QUANT_BLOCK_SIZE=quant_block_size,
        IS_2D_BLOCK=is_2d_block,
        USE_ASM=use_asm,
    )
    return y
