# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional, Tuple

import triton
import torch
from aiter.ops.triton._triton_kernels.quant.quant import (
    _static_per_tensor_quant_fp8_i8_kernel,
    _dynamic_per_tensor_quant_fp8_i8_kernel,
    _dynamic_per_token_quant_fp8_i8_kernel,
    _dynamic_mxfp4_quant_kernel,
    _mxfp4_quant_op,
    _dynamic_mxfp8_quant_kernel,
    _mxfp8_quant_op,
    _fp8_legacy_to_mxfp8_kernel,
    _dynamic_nvfp4_quant_kernel,
    _nvfp4_quant_op,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.types import e4m3_dtype

__all__ = [
    "static_per_tensor_quant_fp8_i8",
    "dynamic_per_tensor_quant_fp8_i8",
    "dynamic_per_token_quant_fp8_i8",
    "dynamic_mxfp4_quant",
    "_mxfp4_quant_op",
    "dynamic_mxfp8_quant",
    "fp8_legacy_to_mxfp8",
    "_mxfp8_quant_op",
    "dynamic_nvfp4_quant",
    "_nvfp4_quant_op",
]

_MXFP8_QUANT_BLOCK_SIZE = 32
_MXFP8_LEGACY_BLOCK_SIZE = 128


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


def dynamic_mxfp8_quant(
    x: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    quant_dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-1x32 MXFP8 quantization (e8m0 scale + FP8 e4m3 values).

    Args:
        x: Input tensor (..., K). Typically bf16 or fp16. K % 32 == 0.
        scale: Pre-allocated scale tensor (M, K // 32) uint8. Optional.
        quant_dtype: FP8 dtype to cast quantized values to. On MI3xx
            torch.float8_e4m3fnuz is the canonical FP8 e4m3 type. torch.float8_e4m3fn
            is acceptable on hardware that supports it.

    Returns:
        Tuple of:
            y: FP8 tensor of shape x.shape.
            s: e8m0 (uint8) scale tensor of shape (..., K // 32).
    """
    assert x.dim() >= 2, f"x must be at least 2D, got {x.dim()}"
    orig_shape = x.shape
    K = orig_shape[-1]
    assert (
        K % _MXFP8_QUANT_BLOCK_SIZE == 0
    ), f"last dim K={K} must be a multiple of {_MXFP8_QUANT_BLOCK_SIZE}"

    x2d = x.reshape(-1, K).contiguous()
    M = x2d.shape[0]
    Ns = K // _MXFP8_QUANT_BLOCK_SIZE  # number of scales per row

    y = torch.empty((M, K), dtype=quant_dtype, device=x.device)
    if scale is None:
        scale = torch.empty((M, Ns), dtype=torch.uint8, device=x.device)
    else:
        assert scale.shape == (M, Ns), f"scale shape {scale.shape} != ({M},{Ns})"
        assert scale.dtype == torch.uint8

    BLOCK_SIZE_N = triton.next_power_of_2(K)
    NUM_PRGMS = M
    grid = (NUM_PRGMS,)

    _dynamic_mxfp8_quant_kernel[grid](
        x2d,
        y,
        scale,
        M,
        K,
        x2d.stride(0),
        x2d.stride(1),
        y.stride(0),
        y.stride(1),
        scale.stride(0),
        scale.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        QUANT_BLOCK_SIZE=_MXFP8_QUANT_BLOCK_SIZE,
        NUM_PRGMS=NUM_PRGMS,
    )

    y = y.view(*orig_shape[:-1], K)
    s = scale.view(*orig_shape[:-1], Ns)
    return y, s


def fp8_legacy_to_mxfp8(
    x_fnuz: torch.Tensor,
    x_scale_fp32: torch.Tensor,
    y_fn: Optional[torch.Tensor] = None,
    y_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transcode (FP8 e4m3fnuz, fp32 1x128 scale) -> (FP8 e4m3fn, e8m0 1x32 scale)
    in a single Triton launch. Replaces the Python dequant+requant cascade
    used when MXFP8 path receives legacy-formatted (FP8 + fp32 1x128) inputs.

    Args:
        x_fnuz: FP8 e4m3fnuz tensor of shape (M, N), N % 32 == 0.
        x_scale_fp32: fp32 scale of shape (M, N // 128).
        y_fn: optional preallocated output FP8 e4m3fn tensor.
        y_scale: optional preallocated uint8 e8m0 scale tensor.

    Returns:
        y_fn (M, N) fp8 e4m3fn, y_scale (M, N // 32) uint8 e8m0.
    """
    assert x_fnuz.dim() == 2, f"x must be 2D, got {x_fnuz.dim()}"
    M, N = x_fnuz.shape
    assert N % _MXFP8_QUANT_BLOCK_SIZE == 0
    assert N % _MXFP8_LEGACY_BLOCK_SIZE == 0
    assert x_scale_fp32.shape == (
        M,
        N // _MXFP8_LEGACY_BLOCK_SIZE,
    ), f"x_scale_fp32 shape {x_scale_fp32.shape} != ({M},{N // _MXFP8_LEGACY_BLOCK_SIZE})"

    Ns = N // _MXFP8_QUANT_BLOCK_SIZE
    if y_fn is None:
        y_fn = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x_fnuz.device)
    if y_scale is None:
        y_scale = torch.empty((M, Ns), dtype=torch.uint8, device=x_fnuz.device)

    BLOCK_SIZE_M = 1
    grid = (triton.cdiv(M, BLOCK_SIZE_M), Ns)

    _fp8_legacy_to_mxfp8_kernel[grid](
        x_fnuz,
        x_scale_fp32,
        y_fn,
        y_scale,
        M,
        N,
        x_fnuz.stride(0),
        x_fnuz.stride(1),
        x_scale_fp32.stride(0),
        x_scale_fp32.stride(1),
        y_fn.stride(0),
        y_fn.stride(1),
        y_scale.stride(0),
        y_scale.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        QUANT_BLOCK_SIZE=_MXFP8_QUANT_BLOCK_SIZE,
        LEGACY_BLOCK_SIZE=_MXFP8_LEGACY_BLOCK_SIZE,
    )

    return y_fn, y_scale


def dynamic_nvfp4_quant(
    x: torch.Tensor,
    global_scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to MX FP4 format.

    Args:
        x: The input tensor, typically fp16 or bf16.
    Returns:
        A tuple of (x_fp4, blockscale_e4m3).
    """
    _LOGGER.info(f"DYNAMIC_NVFP4_QUANT: x={tuple(x.shape)}")
    # Assume x is 2D-Tensor for now
    M, N = x.shape

    assert (N // 2) % 2 == 0

    # This is fixed by spec for MXFP4. Do not tune this.
    NVFP4_QUANT_BLOCK_SIZE = 16
    x_fp4 = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
    blockscale_e4m3 = torch.empty(
        ((N + NVFP4_QUANT_BLOCK_SIZE - 1) // NVFP4_QUANT_BLOCK_SIZE, M),
        dtype=e4m3_dtype,
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

    _dynamic_nvfp4_quant_kernel[grid](
        x,
        x_fp4,
        blockscale_e4m3,
        *x.stride(),
        *x_fp4.stride(),
        *blockscale_e4m3.stride(),
        M=M,
        N=N,
        NVFP4_QUANT_BLOCK_SIZE=NVFP4_QUANT_BLOCK_SIZE,
        NUM_ITER=NUM_ITER,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        NUM_STAGES=NUM_STAGES,
        num_warps=NUM_WARPS,
        waves_per_eu=0,
        num_stages=1,
    )

    return x_fp4, blockscale_e4m3
