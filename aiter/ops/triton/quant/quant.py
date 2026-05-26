# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import torch
from aiter.ops.triton._triton_kernels.quant.quant import (
    _static_per_tensor_quant_fp8_i8_kernel,
    _dynamic_per_tensor_quant_fp8_i8_kernel,
    _dynamic_per_token_quant_fp8_i8_kernel,
    _dynamic_mxfp4_quant_kernel,
    _mxfp4_quant_op,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

__all__ = [
    "static_per_tensor_quant_fp8_i8",
    "dynamic_per_tensor_quant_fp8_i8",
    "dynamic_per_token_quant_fp8_i8",
    "dynamic_mxfp4_quant",
    "_mxfp4_quant_op",
]


_LOGGER = AiterTritonLogger()


def static_per_tensor_quant_fp8_i8(
    qx: torch.Tensor, x_in: torch.Tensor, scale_in: torch.Tensor
):
    """
    Quantizes tensor using the provided scale to int8 or fp8

    Parameters:
    - qx: Output tensor of same shape as x_in. Must be fp8 or int8 dtype and allocated by the caller
    - x_in: Input tensor of shape (M, N).
    - scale_in: Input Scale tensor of shape (1,) and dtype fp32

    Returns:
    - qx: Quantized output values.
    """
    _LOGGER.info(f"STAIC_PER_TENSOR_QUANT_FP8_I8: x={tuple(x_in.shape)}")
    assert scale_in.numel() == 1  # only single scale value
    rows = x_in.shape[0]
    cols = x_in.shape[1]
    NUM_COL_POW2 = triton.next_power_of_2(cols)
    grid = lambda meta: (rows,)  # noqa: E731
    _static_per_tensor_quant_fp8_i8_kernel[grid](
        qx, x_in, scale_in, cols, x_in.stride(0), NUM_COL_POW2=NUM_COL_POW2
    )

    return qx


def dynamic_per_tensor_quant_fp8_i8(
    qx: torch.Tensor, x_in: torch.Tensor, scale_out: torch.Tensor
):
    """
    Calculate per tensor scale and then uses the scale to quantize input tensor to fp8 or int8

    Parameters:
    - x_in: Input tensor of shape (M, N).
    - qx: Output tensor of same shape as x_in. Must be fp8 or int8 dtype and allocated by the caller
    - scale_out: Output scale tensor of shape (1,), dtype fp32 and allocated by the caller

    Returns:
    - qx: Quantized output values of shape (M, N) with dtype fp8 or int8
    - scale_out: Single scale value of shape (1,)
    """
    _LOGGER.info(f"DYNAMIC_PER_TENSOR_QUANT_FP8_I8: x={tuple(x_in.shape)}")
    rows = x_in.shape[0]
    cols = x_in.shape[1]
    NUM_COL_POW2 = triton.next_power_of_2(cols)
    grid = lambda meta: (rows,)  # noqa: E731
    _dynamic_per_tensor_quant_fp8_i8_kernel[grid](
        x_in,
        scale_out,
        cols,
        x_in.stride(0),
        NUM_COL_POW2=NUM_COL_POW2,
        DTYPE_MAX=(
            torch.finfo(qx.dtype).max
            if torch.is_floating_point(qx)
            else torch.iinfo(qx.dtype).max
        ),
    )

    _static_per_tensor_quant_fp8_i8_kernel[grid](
        qx, x_in, scale_out, cols, x_in.stride(0), NUM_COL_POW2=NUM_COL_POW2
    )

    return qx, scale_out


def dynamic_per_token_quant_fp8_i8(
    qx: torch.Tensor,
    x_in: torch.Tensor,
    scale_out: torch.Tensor,
):
    """
    Quantizes tensor using the provided scale

    Parameters:
    - x_in: Input tensor of shape (M, N).
    - dtype_max: Optional parameter which specifies the max value of the dtype of x_in.
    - qx: Output tensor of same shape as x_in. Must be fp8 dtype and allocated by the caller
    - scale_out: Output scale tensor of shape (M,) dtype fp32 and allocated by the caller

    Returns:
    - qx: Quantized output values.
    - scale_out: Scale tensor of shape (M, )
    """
    _LOGGER.info(f"DYNAMIC_PER_TOKEN_QUANT_FP8_I8: x={tuple(x_in.shape)}")
    rows = x_in.shape[0]
    cols = x_in.shape[1]
    NUM_COL_POW2 = triton.next_power_of_2(cols)
    grid = lambda meta: (rows,)  # noqa: E731
    _dynamic_per_token_quant_fp8_i8_kernel[grid](
        qx,
        scale_out,
        x_in,
        cols,
        x_in.stride(0),
        NUM_COL_POW2=NUM_COL_POW2,
        DTYPE_MAX=(
            torch.finfo(qx.dtype).max
            if torch.is_floating_point(qx)
            else torch.iinfo(qx.dtype).max
        ),
    )

    return qx, scale_out


def dynamic_mxfp4_quant(
    x: torch.Tensor, scaling_mode: str = "even"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to MX FP4 format.

    Args:
        x: The input tensor, typically fp16 or bf16.
        scaling_mode: The method to calculate MX block scaling.
            - "even" (default): `even_round` in `quark.torch.quantization.utils`.
            - etc.
    Returns:
        A tuple of (x_fp4, blockscale_e8m0).
    """
    _LOGGER.info(f"DYNAMIC_MXFP4_QUANT: x={tuple(x.shape)}")
    # Assume x is 2D-Tensor for now
    M, N = x.shape

    assert (N // 2) % 2 == 0

    # This is fixed by spec for MXFP4. Do not tune this.
    MXFP4_QUANT_BLOCK_SIZE = 32
    x_fp4 = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
    blockscale_e8m0 = torch.empty(
        ((N + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE, M),
        dtype=torch.uint8,
        device=x.device,
    ).T

    # for large N values
    if M <= 32:
        NUM_ITER = 1
        BLOCK_SIZE_M = triton.next_power_of_2(M)
        BLOCK_SIZE_N = 32
        NUM_WARPS = 1
        NUM_STAGES = 1
    else:
        NUM_ITER = 4
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        NUM_WARPS = 4
        NUM_STAGES = 2

        if N <= 16384:
            BLOCK_SIZE_M = 32
            BLOCK_SIZE_N = 128

    # for small N values
    if N <= 1024:
        NUM_ITER = 1
        NUM_STAGES = 1
        NUM_WARPS = 4
        BLOCK_SIZE_N = min(256, triton.next_power_of_2(N))
        # BLOCK_SIZE_N needs to be multiple of 32
        BLOCK_SIZE_N = max(32, BLOCK_SIZE_N)
        BLOCK_SIZE_M = min(8, triton.next_power_of_2(M))

    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N * NUM_ITER),
    )

    _dynamic_mxfp4_quant_kernel[grid](
        x,
        x_fp4,
        blockscale_e8m0,
        *x.stride(),
        *x_fp4.stride(),
        *blockscale_e8m0.stride(),
        M=M,
        N=N,
        MXFP4_QUANT_BLOCK_SIZE=MXFP4_QUANT_BLOCK_SIZE,
        SCALING_MODE=0,
        NUM_ITER=NUM_ITER,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        NUM_STAGES=NUM_STAGES,
        num_warps=NUM_WARPS,
        waves_per_eu=0,
        num_stages=1,
    )

    return (x_fp4, blockscale_e8m0)
