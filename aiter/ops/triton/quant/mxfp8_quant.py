# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import torch
from aiter.ops.triton._triton_kernels.quant.mxfp8_quant import (
    _downcast_to_mxfp8,
    _upcast_from_mxfp8,
    _downcast_to_mxfp8_2d,
    _upcast_from_mxfp8_2d,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

__all__ = [
    "downcast_to_mxfp8",
    "upcast_from_mxfp8",
    "downcast_to_mxfp8_2d",
    "upcast_from_mxfp8_2d",
]


_LOGGER = AiterTritonLogger()


def downcast_to_mxfp8(
    src_tensor: torch.Tensor,
    out_quant_type: torch.dtype,
    axis: int,
    SCALE_ROUNDING_MODE: int = 0,
):
    """
    Convert the src weights to mx8 format. The src weight is quantized along the axis dimension.

    If weight_quant_type is torch.uint8, we output mxfp4 where two e2m1 values are packed into a single byte.
    Note that this means the k_dim of the tensor will be half of the logical k_dim.

    If weight_quant_type is torch.float8_e4m3fn or torch.float8_e5m2, we output mxfp8 with the float8s are stored
    in their respective formats.
    """
    ndim = src_tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid axis {axis=}"
    axis = axis if axis >= 0 else axis + ndim
    # downcast
    src_tensor = src_tensor.transpose(axis, src_tensor.ndim - 1)
    L = src_tensor.shape[-1]
    out_shape = src_tensor.shape[:-1] + (L,)
    out_scale_shape = src_tensor.shape[:-1] + (triton.cdiv(L, 32),)

    out_quant_tensor = src_tensor.new_empty(out_shape, dtype=out_quant_type)
    out_scale = src_tensor.new_empty(out_scale_shape, dtype=torch.uint8)

    kernel_src_tensor = src_tensor.reshape(-1, src_tensor.shape[-1])
    kernel_quant_tensor = out_quant_tensor.view(-1, out_quant_tensor.shape[-1])
    kernel_scale = out_scale.view(-1, out_scale.shape[-1])

    BLOCK_OUT_DIM = 128
    BLOCK_QUANT_DIM = 32
    grid_out = triton.cdiv(kernel_src_tensor.shape[0], BLOCK_OUT_DIM)
    grid_quant = triton.cdiv(kernel_src_tensor.shape[1], BLOCK_QUANT_DIM)

    _downcast_to_mxfp8[(grid_out, grid_quant)](
        kernel_quant_tensor,
        *kernel_quant_tensor.stride(),
        kernel_scale,
        *kernel_scale.stride(),
        kernel_src_tensor,
        *kernel_src_tensor.stride(),
        *kernel_src_tensor.shape,
        BLOCK_OUT_DIM,
        BLOCK_QUANT_DIM,
        SCALE_ROUNDING_MODE,
        num_warps=8,
    )

    out_quant_tensor = out_quant_tensor.transpose(axis, src_tensor.ndim - 1)
    out_scale = out_scale.transpose(axis, src_tensor.ndim - 1)
    return out_quant_tensor, out_scale


def upcast_from_mxfp8(
    tensor: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype, axis: int
):
    """
    Upcasts an mxfp8 weight tensor back to float16 or bfloat16.

    The function assumes that the tensors were quantized along the given axis.
    It permutes the tensor so that the quantized axis is last, reshapes to 2D,
    launches the Triton upcast kernel, and then unpermutes back to the original order.
    """
    ndim = tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid axis {axis=}"
    axis = axis if axis >= 0 else axis + ndim
    assert tensor.ndim == scale.ndim, (
        f"Weight and scale must have the same number of dimensions. "
        f"Got {tensor.ndim=} and {scale.ndim=}"
    )
    # dtype checks
    assert tensor.dtype in {
        torch.float8_e5m2,
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    }, f"Invalid tensor dtype {tensor.dtype=}"
    assert scale.dtype == torch.uint8, f"Invalid scale dtype {scale.dtype=}"
    assert dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), f"Invalid output dtype {dtype=}"
    # upcast
    logical_quant_dim = tensor.shape[axis]
    tensor = tensor.transpose(axis, tensor.ndim - 1).contiguous()
    scale = scale.transpose(axis, scale.ndim - 1).contiguous()
    out = torch.empty(
        (*tensor.shape[:-1], logical_quant_dim), dtype=dtype, device=tensor.device
    )
    reshaped_out = out.view(-1, out.shape[-1])
    reshaped_tensor = tensor.view(-1, tensor.shape[-1])
    reshaped_scale = scale.view(-1, scale.shape[-1])
    BLOCK_OUT_DIM = 128
    BLOCK_QUANT_DIM = 32
    blocks_out_dim = triton.cdiv(reshaped_out.shape[0], BLOCK_OUT_DIM)
    blocks_quant_dim = triton.cdiv(reshaped_out.shape[1], BLOCK_QUANT_DIM)
    _upcast_from_mxfp8[(blocks_out_dim, blocks_quant_dim)](
        reshaped_out,
        *reshaped_out.stride(),
        reshaped_scale,
        *reshaped_scale.stride(),
        reshaped_tensor,
        *reshaped_tensor.stride(),
        *reshaped_out.shape,
        BLOCK_OUT_DIM,
        BLOCK_QUANT_DIM,
        num_warps=8,
    )
    out = out.transpose(axis, scale.ndim - 1).contiguous()
    return out


def downcast_to_mxfp8_2d(
    src_tensor: torch.Tensor,
    out_quant_type: torch.dtype,
    SCALE_ROUNDING_MODE: int = 0,
):
    """
    Convert the last two dimensions of ``src_tensor`` to the mxfp8 format,
    where each 32x32 block of those two dimensions shares a single scale.

    The quantized tensor preserves the input shape; the scale tensor has shape
    ``(..., cdiv(M, 32), cdiv(N, 32))`` and dtype uint8, with ``M`` and ``N``
    being the last two dims of ``src_tensor``.

    ``out_quant_type`` must be ``torch.float8_e4m3fn`` or ``torch.float8_e5m2``.
    """
    assert (
        src_tensor.ndim >= 2
    ), f"src_tensor must have at least 2 dimensions, got {src_tensor.ndim}"
    assert out_quant_type in {
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    }, f"Invalid out_quant_type {out_quant_type=}"

    src_tensor = src_tensor.contiguous()
    M = src_tensor.shape[-2]
    N = src_tensor.shape[-1]
    M_scale = triton.cdiv(M, 32)
    N_scale = triton.cdiv(N, 32)

    out_shape = src_tensor.shape
    out_scale_shape = src_tensor.shape[:-2] + (M_scale, N_scale)

    out_quant_tensor = src_tensor.new_empty(out_shape, dtype=out_quant_type)
    out_scale = src_tensor.new_empty(out_scale_shape, dtype=torch.uint8)

    kernel_src = src_tensor.reshape(-1, M, N)
    kernel_quant = out_quant_tensor.view(-1, M, N)
    kernel_scale = out_scale.view(-1, M_scale, N_scale)

    BLOCK_M = 128
    BLOCK_N = 128
    B = kernel_src.shape[0]
    grid = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _downcast_to_mxfp8_2d[grid](
        kernel_quant,
        *kernel_quant.stride(),
        kernel_scale,
        *kernel_scale.stride(),
        kernel_src,
        *kernel_src.stride(),
        M,
        N,
        BLOCK_M,
        BLOCK_N,
        SCALE_ROUNDING_MODE,
        num_warps=8,
    )

    return out_quant_tensor, out_scale


def upcast_from_mxfp8_2d(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
):
    """
    Inverse of :func:`downcast_to_mxfp8_2d`. ``scale`` is expected to be a
    ``(..., cdiv(M, 32), cdiv(N, 32))`` uint8 tensor where each entry
    corresponds to a 32x32 block of ``tensor``'s last two dimensions.
    """
    assert tensor.ndim >= 2 and scale.ndim >= 2
    assert tensor.dtype in {
        torch.float8_e5m2,
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    }, f"Invalid tensor dtype {tensor.dtype=}"
    assert scale.dtype == torch.uint8, f"Invalid scale dtype {scale.dtype=}"
    assert dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), f"Invalid output dtype {dtype=}"
    assert tensor.ndim == scale.ndim, (
        f"tensor and scale must have the same number of dimensions. "
        f"Got {tensor.ndim=} and {scale.ndim=}"
    )

    tensor = tensor.contiguous()
    scale = scale.contiguous()
    M = tensor.shape[-2]
    N = tensor.shape[-1]
    M_scale = scale.shape[-2]
    N_scale = scale.shape[-1]
    assert M_scale == triton.cdiv(
        M, 32
    ), f"scale shape mismatch: got {M_scale=} expected {triton.cdiv(M, 32)}"
    assert N_scale == triton.cdiv(
        N, 32
    ), f"scale shape mismatch: got {N_scale=} expected {triton.cdiv(N, 32)}"

    out = torch.empty(tensor.shape, dtype=dtype, device=tensor.device)

    kernel_out = out.view(-1, M, N)
    kernel_tensor = tensor.view(-1, M, N)
    kernel_scale = scale.view(-1, M_scale, N_scale)

    BLOCK_M = 128
    BLOCK_N = 128
    B = kernel_out.shape[0]
    grid = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _upcast_from_mxfp8_2d[grid](
        kernel_out,
        *kernel_out.stride(),
        kernel_scale,
        *kernel_scale.stride(),
        kernel_tensor,
        *kernel_tensor.stride(),
        M,
        N,
        BLOCK_M,
        BLOCK_N,
        num_warps=8,
    )

    return out
