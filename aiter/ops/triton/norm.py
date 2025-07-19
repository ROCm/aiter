# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
from typing import Optional
from aiter.ops.triton.utils.types import get_dtype_max


@triton.jit
def _per_token_quant(
    x,
    row_max,
    DTYPE_MAX: tl.constexpr,
):
    """
    #TODO: Add Doc
    """

    scale_out = row_max / DTYPE_MAX
    scale_out = tl.where(scale_out == 0, 1.0, scale_out)

    scale_recip = 1 / scale_out

    qx = x * scale_recip

    return qx, scale_out


@triton.jit
def _layernorm_kernel(
    # Pointers to matrices
    x_ptr,
    y_ptr,
    w_ptr,
    b_ptr,
    mean_ptr,
    rstd_ptr,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `x_row_stride` is
    # how much to increase `x_ptr` by to get the element one row down.
    x_row_stride,
    y_row_stride,
    # Matrix dimensions
    n_rows,
    n_cols,
    # Epsilon to avoid division by zero
    eps,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call layer_norm function
    below

    Applies Layer Normalization over a mini-batch of inputs.

    Key parameters:
    - X: The input tensor to be normalized with shape (M, N).
    - Y: The output tensor with the same shape as the input one.
    - W: The learnable weights tensor with shape (N, ).
    - B: The learnable bias tensor with shape (N, ).
    """
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    x_ptr_start = x_ptr + (row * x_row_stride)
    y_ptr_start = y_ptr + (row * y_row_stride)

    loop_num = tl.cdiv(n_cols, BLOCK_SIZE) - 1

    # Calculate mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    loop_num_l = loop_num
    for b in range(0, loop_num_l):
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x_block = tl.load(x_ptr_start + col_offsets).to(tl.float32)  # Unmasked loads
        _mean += x_block

    # For last iteration, do masked load
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_block = tl.load(
        x_ptr_start + col_offsets, mask=col_offsets < n_cols, other=0.0
    ).to(tl.float32)
    _mean += x_block
    mean = tl.sum(_mean, axis=0) / n_cols

    # Calculate variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    loop_num_l = loop_num
    for b in range(0, loop_num_l):
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x_block = tl.load(x_ptr_start + col_offsets).to(tl.float32)  # Unmasked loads
        x_block = x_block - mean
        _var += x_block * x_block

    # For last iteration, do masked load
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_block = tl.load(
        x_ptr_start + col_offsets, mask=col_offsets < n_cols, other=0.0
    ).to(tl.float32)
    x_block = tl.where(col_offsets < n_cols, x_block - mean, 0.0)
    _var += x_block * x_block

    var = tl.sum(_var, axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)

    # Write mean / rstd
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)

    # Normalize and store
    loop_num_l = loop_num
    for b in range(0, loop_num_l):
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        w_block = tl.load(w_ptr + col_offsets)
        b_block = tl.load(b_ptr + col_offsets)
        x_block = tl.load(x_ptr_start + col_offsets).to(tl.float32)
        y_block = (x_block - mean) * rstd
        y_block = y_block * w_block + b_block
        tl.store(y_ptr_start + col_offsets, y_block)

    # For last iteration, do masked load and store
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    w_block = tl.load(w_ptr + col_offsets, mask=mask, other=0.0)
    b_block = tl.load(b_ptr + col_offsets, mask=mask, other=0.0)
    x_block = tl.load(x_ptr_start + col_offsets, mask=mask, other=0.0).to(tl.float32)
    y_block = (x_block - mean) * rstd
    y_block = y_block * w_block + b_block
    tl.store(y_ptr_start + col_offsets, y_block, mask=mask)


@triton.jit
def _fused_add_layernorm_kernel(
    # Pointers to matrices
    x_ptr,
    y_ptr,
    res_in_ptr,
    res_out_ptr,
    w_ptr,
    b_ptr,
    mean_ptr,
    rstd_ptr,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `x_row_stride` is
    # how much to increase `x_ptr` by to get the element one row down.
    x_row_stride,
    y_row_stride,
    # Matrix dimensions
    n_rows,
    n_cols,
    # Epsilon to avoid division by zero
    eps,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call layernorm2d_fwd_with_add function
    below

    Performs an addition between two inputs and then applies Layer Normalization over
    the addition result.

    Key parameters:
    - X: The input tensor to be normalized with shape (M, N).
    - Y: The output tensor with the same shape as the input one.
    - Res_in: The tensor to be added to the X tensor with shape (M, N).
    - Res_out: The tensor in which the addition result will be stored with shape (M, N).
    - W: The learnable weights tensor with shape (N, ).
    - B: The learnable bias tensor with shape (N, ).
    """
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    x_ptr_start = x_ptr + (row * x_row_stride)
    y_ptr_start = y_ptr + (row * y_row_stride)
    res_in_ptr_start = res_in_ptr + (row * x_row_stride)
    res_out_ptr_start = res_out_ptr + (row * x_row_stride)

    loop_num = tl.cdiv(n_cols, BLOCK_SIZE) - 1

    # Calculate mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    loop_num_l = loop_num
    for b in range(0, loop_num_l):
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        _x_block = tl.load(x_ptr_start + col_offsets)  # Unmasked loads
        res_in_block = tl.load(res_in_ptr_start + col_offsets)
        _x_block += res_in_block
        tl.store(res_out_ptr_start + col_offsets, _x_block)  # Stores residual_out
        _mean += _x_block.to(tl.float32)

    # For last iteration, do masked load
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    _x_block = tl.load(x_ptr_start + col_offsets, mask=col_offsets < n_cols, other=0.0)
    res_in_block = tl.load(
        res_in_ptr_start + col_offsets, mask=col_offsets < n_cols, other=0.0
    )
    _x_block += res_in_block
    tl.store(
        res_out_ptr_start + col_offsets, _x_block, mask=col_offsets < n_cols
    )  # Stores residual_out
    _mean += _x_block.to(tl.float32)
    mean = tl.sum(_mean, axis=0) / n_cols

    # Calculate variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    loop_num_l = loop_num
    for b in range(0, loop_num_l):
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x_block = tl.load(res_out_ptr_start + col_offsets).to(
            tl.float32
        )  # Unmasked loads
        x_block = x_block - mean
        _var += x_block * x_block

    # For last iteration, do masked load
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_block = tl.load(
        res_out_ptr_start + col_offsets, mask=col_offsets < n_cols, other=0.0
    ).to(tl.float32)
    x_block = tl.where(col_offsets < n_cols, x_block - mean, 0.0)
    _var += x_block * x_block

    var = tl.sum(_var, axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)

    # Write mean / rstd
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)

    # Normalize and store
    loop_num_l = loop_num
    for b in range(0, loop_num_l):
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        w_block = tl.load(w_ptr + col_offsets)
        b_block = tl.load(b_ptr + col_offsets)
        x_block = tl.load(res_out_ptr_start + col_offsets).to(tl.float32)
        y_block = (x_block - mean) * rstd
        y_block = y_block * w_block + b_block
        tl.store(y_ptr_start + col_offsets, y_block)

    # For last iteration, do masked load and store
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    w_block = tl.load(w_ptr + col_offsets, mask=mask, other=0.0)
    b_block = tl.load(b_ptr + col_offsets, mask=mask, other=0.0)
    x_block = tl.load(res_out_ptr_start + col_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    y_block = (x_block - mean) * rstd
    y_block = y_block * w_block + b_block
    tl.store(y_ptr_start + col_offsets, y_block, mask=mask)


@triton.jit
def _quant_layernorm_kernel(
    # Pointers to matrices
    x_ptr,
    y_ptr,
    w_ptr,
    b_ptr,
    x_scale_ptr,
    y_scale_ptr,
    # Auxiliary tensor to store intermediate data
    aux_ptr,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `x_row_stride` is
    # how much to increase `x_ptr` by to get the element one row down.
    x_row_stride,
    y_row_stride,
    aux_row_stride,
    # Matrix dimensions
    n_rows,
    n_cols,
    # Epsilon to avoid division by zero
    eps,
    # Dtype max for quantization
    DTYPE_MAX: tl.constexpr,
    # Meta-parameters
    IS_SMOOTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call layer_norm function
    below

    Applies Layer Normalization over a mini-batch of inputs and quantizes the result.

    Key parameters:
    - X: The input tensor to be normalized with shape (M, N).
    - Y: The output tensor with the same shape as the input one.
    - W: The learnable weights tensor with shape (N, ).
    - B: The learnable bias tensor with shape (N, ).
    - X_scale: The tensor to be multiplied by the LayerNorm output if IS_SMOOTH is true, with shape (n_cols, ).
    - Y_scale: The tensor where the scale for each row will be stored with shape (n_rows, ).
    """
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    x_ptr_start = x_ptr + (row * x_row_stride)
    y_ptr_start = y_ptr + (row * y_row_stride)
    aux_ptr_start = aux_ptr + (row * aux_row_stride)

    loop_num = tl.cdiv(n_cols, BLOCK_SIZE) - 1

    # Calculate mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    loop_num_l = loop_num
    for b in range(0, loop_num_l):
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x_block = tl.load(x_ptr_start + col_offsets).to(tl.float32)  # Unmasked loads
        _mean += x_block

    # For last iteration, do masked load
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_block = tl.load(
        x_ptr_start + col_offsets, mask=col_offsets < n_cols, other=0.0
    ).to(tl.float32)
    _mean += x_block
    mean = tl.sum(_mean, axis=0) / n_cols

    # Calculate variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    loop_num_l = loop_num
    for b in range(0, loop_num_l):
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x_block = tl.load(x_ptr_start + col_offsets).to(tl.float32)  # Unmasked loads
        x_block = x_block - mean
        _var += x_block * x_block

    # For last iteration, do masked load
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_block = tl.load(
        x_ptr_start + col_offsets, mask=col_offsets < n_cols, other=0.0
    ).to(tl.float32)
    x_block = tl.where(col_offsets < n_cols, x_block - mean, 0.0)
    _var += x_block * x_block

    var = tl.sum(_var, axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)

    row_max: tl.float32 = 0.0

    # Normalize and write output temporarily as fp32
    loop_num_l = loop_num
    for b in range(0, loop_num_l):
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        w_block = tl.load(w_ptr + col_offsets)
        b_block = tl.load(b_ptr + col_offsets)
        x_block = tl.load(x_ptr_start + col_offsets).to(tl.float32)
        y_block = (x_block - mean) * rstd
        y_block = y_block * w_block + b_block

        if IS_SMOOTH:
            x_scale_ptrs = x_scale_ptr + col_offsets
            x_scale = tl.load(x_scale_ptrs)
            y_block *= x_scale

        # Computes the max value for each row
        blk_max = tl.max(tl.abs(y_block), axis=-1)
        row_max = max(row_max, blk_max)

        aux_ptrs = aux_ptr_start + col_offsets
        tl.store(aux_ptrs, y_block)

    # For last iteration, do masked load
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    w_block = tl.load(w_ptr + col_offsets, mask=mask, other=0.0)
    b_block = tl.load(b_ptr + col_offsets, mask=mask, other=0.0)
    x_block = tl.load(x_ptr_start + col_offsets, mask=mask, other=0.0).to(tl.float32)
    y_block = tl.where(mask, (x_block - mean) * rstd, 0.0)
    y_block = y_block * w_block + b_block

    if IS_SMOOTH:
        x_scale_ptrs = x_scale_ptr + col_offsets
        x_scale = tl.load(x_scale_ptrs, mask=mask, other=0.0)
        y_block *= x_scale

    # Computes the max value for each row
    blk_max = tl.max(tl.abs(y_block), axis=-1)
    row_max = max(row_max, blk_max)

    tl.store(aux_ptr_start + col_offsets, y_block, mask=mask)

    # Apply quantization and write output
    loop_num_l = loop_num
    for b in range(0, loop_num_l):
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        aux_block = tl.load(aux_ptr_start + col_offsets)  # Unmasked loads

        y_block, _ = _per_token_quant(aux_block, row_max, DTYPE_MAX)

        tl.store(y_ptr_start + col_offsets, y_block.to(y_ptr.type.element_ty))

    # For last iteration, do masked load and store
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    aux_block = tl.load(aux_ptr_start + col_offsets, mask=mask, other=0.0)

    y_block, y_scale = _per_token_quant(aux_block, row_max, DTYPE_MAX)

    # Store scale
    tl.store(y_scale_ptr + row, y_scale.to(y_scale_ptr.type.element_ty))

    tl.store(y_ptr_start + col_offsets, y_block.to(y_ptr.type.element_ty), mask=mask)


@triton.jit
def _quant_fused_add_layernorm_kernel(
    # Pointers to matrices
    x_ptr,
    y_ptr,
    res_in_ptr,
    res_out_ptr,
    w_ptr,
    b_ptr,
    x_scale_ptr,
    y_scale_ptr,
    # Auxiliary tensor to store intermediate data
    aux_ptr,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `x_row_stride` is
    # how much to increase `x_ptr` by to get the element one row down.
    x_row_stride,
    y_row_stride,
    aux_row_stride,
    # Matrix dimensions
    n_rows,
    n_cols,
    # Epsilon to avoid division by zero
    eps,
    # Dtype max for quantization
    DTYPE_MAX: tl.constexpr,
    # Meta-parameters
    IS_SMOOTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call layernorm2d_fwd_with_add function
    below

    Performs an addition between two inputs, applies Layer Normalization over the result and then quantizes it.

    Key parameters:
    - X: The input tensor to be normalized with shape (M, N).
    - Y: The output tensor with the same shape as the input one.
    - Res_in: The tensor to be added to the X tensor with shape (M, N).
    - Res_out: The tensor in which the addition result will be stored with shape (M, N).
    - W: The learnable weights tensor with shape (N, ).
    - B: The learnable bias tensor with shape (N, ).
    - X_scale: The tensor to be multiplied by the LayerNorm output if IS_SMOOTH is true, with shape (n_cols, ).
    - Y_scale: The tensor where the scale for each row will be stored with shape (n_rows, ).
    """
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    x_ptr_start = x_ptr + (row * x_row_stride)
    y_ptr_start = y_ptr + (row * y_row_stride)
    res_in_ptr_start = res_in_ptr + (row * x_row_stride)
    res_out_ptr_start = res_out_ptr + (row * x_row_stride)
    aux_ptr_start = aux_ptr + (row * aux_row_stride)

    loop_num = tl.cdiv(n_cols, BLOCK_SIZE) - 1

    # Calculate mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    loop_num_l = loop_num
    for b in range(0, loop_num_l):
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        _x_block = tl.load(x_ptr_start + col_offsets)  # Unmasked loads
        res_in_block = tl.load(res_in_ptr_start + col_offsets)
        _x_block += res_in_block
        tl.store(res_out_ptr_start + col_offsets, _x_block)  # Stores residual_out
        _mean += _x_block.to(tl.float32)

    # For last iteration, do masked load
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    _x_block = tl.load(x_ptr_start + col_offsets, mask=col_offsets < n_cols, other=0.0)
    res_in_block = tl.load(
        res_in_ptr_start + col_offsets, mask=col_offsets < n_cols, other=0.0
    )
    _x_block += res_in_block
    tl.store(
        res_out_ptr_start + col_offsets, _x_block, mask=col_offsets < n_cols
    )  # Stores residual_out
    _mean += _x_block.to(tl.float32)
    mean = tl.sum(_mean, axis=0) / n_cols

    # Calculate variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    loop_num_l = loop_num
    for b in range(0, loop_num_l):
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x_block = tl.load(res_out_ptr_start + col_offsets).to(
            tl.float32
        )  # Unmasked loads
        x_block = x_block - mean
        _var += x_block * x_block

    # For last iteration, do masked load
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_block = tl.load(
        res_out_ptr_start + col_offsets, mask=col_offsets < n_cols, other=0.0
    ).to(tl.float32)
    x_block = tl.where(col_offsets < n_cols, x_block - mean, 0.0)
    _var += x_block * x_block

    var = tl.sum(_var, axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)

    row_max: tl.float32 = 0.0

    # Normalize and write output temporarily as fp32
    loop_num_l = loop_num
    for b in range(0, loop_num_l):
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        w_block = tl.load(w_ptr + col_offsets)
        b_block = tl.load(b_ptr + col_offsets)
        x_block = tl.load(res_out_ptr_start + col_offsets).to(tl.float32)
        y_block = (x_block - mean) * rstd
        y_block = y_block * w_block + b_block

        if IS_SMOOTH:
            x_scale_ptrs = x_scale_ptr + col_offsets
            x_scale = tl.load(x_scale_ptrs)
            y_block *= x_scale

        # Computes the max value for each row
        blk_max = tl.max(tl.abs(y_block), axis=-1)
        row_max = max(row_max, blk_max)

        aux_ptrs = aux_ptr_start + col_offsets
        tl.store(aux_ptrs, y_block)

    # For last iteration, do masked load and store
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    w_block = tl.load(w_ptr + col_offsets, mask=mask, other=0.0)
    b_block = tl.load(b_ptr + col_offsets, mask=mask, other=0.0)
    x_block = tl.load(res_out_ptr_start + col_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    y_block = tl.where(mask, (x_block - mean) * rstd, 0.0)
    y_block = y_block * w_block + b_block

    if IS_SMOOTH:
        x_scale_ptrs = x_scale_ptr + col_offsets
        x_scale = tl.load(x_scale_ptrs, mask=mask, other=0.0)
        y_block *= x_scale

    # Computes the max value for each row
    blk_max = tl.max(tl.abs(y_block), axis=-1)
    row_max = max(row_max, blk_max)

    tl.store(aux_ptr_start + col_offsets, y_block, mask=mask)

    # Apply quantization and write output
    loop_num_l = loop_num
    for b in range(0, loop_num_l):
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        aux_block = tl.load(aux_ptr_start + col_offsets)  # Unmasked loads

        y_block, _ = _per_token_quant(aux_block, row_max, DTYPE_MAX)

        tl.store(y_ptr_start + col_offsets, y_block.to(y_ptr.type.element_ty))

    # For last iteration, do masked load and store
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    aux_block = tl.load(aux_ptr_start + col_offsets, mask=mask, other=0.0)

    y_block, y_scale = _per_token_quant(aux_block, row_max, DTYPE_MAX)

    # Store scale
    tl.store(y_scale_ptr + row, y_scale.to(y_scale_ptr.type.element_ty))

    tl.store(y_ptr_start + col_offsets, y_block.to(y_ptr.type.element_ty), mask=mask)


def _layernorm_forward(
    y: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    eps: float = 1e-5,
):

    M, N = x.shape

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))

    _layernorm_kernel[(M,)](
        x, y, weight, bias, mean, rstd, x.stride(0), y.stride(0), M, N, eps, BLOCK_SIZE
    )

    return


def _layernorm_forward_with_add(
    y: torch.Tensor,
    x: torch.Tensor,
    res_in: torch.Tensor,
    res_out: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    epsilon: float,
):

    M, N = x.shape

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))

    _fused_add_layernorm_kernel[(M,)](
        x,
        y,
        res_in,
        res_out,
        weight,
        bias,
        mean,
        rstd,
        x.stride(0),
        y.stride(0),
        M,
        N,
        epsilon,
        BLOCK_SIZE,
    )

    return


@triton.jit
def _layernorm_bwd_dx_fused_triton(
    DX,  # pointer to the input gradient
    DY,  # pointer to the output gradient
    DW,  # pointer to the partial sum of weights gradient
    DB,  # pointer to the partial sum of biases gradient
    X,  # pointer to the input
    W,  # pointer to the weights
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    NUM_ROWS: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    USE_BLOCKED: tl.constexpr,
    IGNORE_DW_DB: tl.constexpr = False,
):
    # Map the program id to the elements of X, DX, and DY it should compute.
    pid = tl.program_id(0)
    tile_num = tl.num_programs(0)
    rows_per_tile = NUM_ROWS // tile_num
    if pid < NUM_ROWS % tile_num:
        rows_per_tile += 1

    if USE_BLOCKED:
        # Blocked approach:

        col_offsets = tl.arange(0, BLOCK_SIZE_N)
        num_col_blocks = tl.cdiv(N, BLOCK_SIZE_N) - 1
        row = pid

        for _ in range(0, rows_per_tile):
            # Load row statistics:
            mean = tl.load(Mean + row)
            rstd = tl.load(Rstd + row)

            # Accumulate c1 and c2 sums:

            x_row_ptr = X + row * stride
            dy_row_ptr = DY + row * stride

            c1 = 0.0
            c2 = 0.0

            for block_idx in tl.range(0, num_col_blocks):
                cols = block_idx * BLOCK_SIZE_N + col_offsets

                x = tl.load(x_row_ptr + cols).to(tl.float32)
                dy = tl.load(dy_row_ptr + cols).to(tl.float32)
                w = tl.load(W + cols).to(tl.float32)

                xhat = (x - mean) * rstd
                wdy = w * dy
                c1 += tl.sum(xhat * wdy, axis=0)
                c2 += tl.sum(wdy, axis=0)

            cols = num_col_blocks * BLOCK_SIZE_N + col_offsets
            mask = cols < N

            x = tl.load(x_row_ptr + cols, mask=mask, other=0).to(tl.float32)
            dy = tl.load(dy_row_ptr + cols, mask=mask, other=0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0).to(tl.float32)

            xhat = (x - mean) * rstd
            wdy = w * dy
            wdy = tl.where(mask, wdy, 0)
            c1 += tl.sum(xhat * wdy, axis=0)
            c2 += tl.sum(wdy, axis=0)

            c1 /= N
            c2 /= N

            # Compute dx and partial sums for dw and db:

            dx_row_ptr = DX + row * stride
            if not IGNORE_DW_DB:
                dw_row_ptr = DW + pid * N
                db_row_ptr = DB + pid * N

            for block_idx in tl.range(0, num_col_blocks):
                cols = block_idx * BLOCK_SIZE_N + col_offsets

                x = tl.load(x_row_ptr + cols).to(tl.float32)
                dy = tl.load(dy_row_ptr + cols).to(tl.float32)
                w = tl.load(W + cols).to(tl.float32)

                xhat = (x - mean) * rstd
                wdy = w * dy

                dx = (wdy - (xhat * c1 + c2)) * rstd
                tl.store(dx_row_ptr + cols, dx.to(DX.type.element_ty))
                if not IGNORE_DW_DB:
                    partial_dw = dy * xhat
                    dw_ptrs = dw_row_ptr + cols
                    partial_dw += tl.load(dw_ptrs).to(tl.float32)
                    tl.store(dw_ptrs, partial_dw.to(DW.type.element_ty))

                    partial_db = dy
                    db_ptrs = db_row_ptr + cols
                    partial_db += tl.load(db_ptrs).to(tl.float32)
                    tl.store(db_ptrs, partial_db.to(DB.type.element_ty))

            cols = num_col_blocks * BLOCK_SIZE_N + col_offsets
            mask = cols < N

            x = tl.load(x_row_ptr + cols, mask=mask, other=0).to(tl.float32)
            dy = tl.load(dy_row_ptr + cols, mask=mask, other=0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0).to(tl.float32)

            xhat = (x - mean) * rstd
            wdy = w * dy

            dx = (wdy - (xhat * c1 + c2)) * rstd
            tl.store(dx_row_ptr + cols, dx.to(DX.type.element_ty), mask=mask)
            if not IGNORE_DW_DB:
                partial_dw = dy * xhat
                dw_ptrs = dw_row_ptr + cols
                partial_dw += tl.load(dw_ptrs, mask=mask).to(tl.float32)
                tl.store(dw_ptrs, partial_dw.to(DW.type.element_ty), mask=mask)

                partial_db = dy
                db_ptrs = db_row_ptr + cols
                partial_db += tl.load(db_ptrs, mask=mask).to(tl.float32)
                tl.store(db_ptrs, partial_db.to(DB.type.element_ty), mask=mask)

            # Advance to next row.
            row += tile_num

    else:
        # Unblocked approach:

        cols = tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        row = pid
        if not IGNORE_DW_DB:
            dw_row = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
            db_row = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

        for _ in range(0, rows_per_tile):
            # Compute pointers:
            x_ptrs = X + row * stride
            dy_ptrs = DY + row * stride
            dx_ptrs = DX + row * stride

            # Load data to SRAM:
            x = tl.load(x_ptrs + cols, mask=mask, other=0).to(tl.float32)
            dy = tl.load(dy_ptrs + cols, mask=mask, other=0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0).to(tl.float32)
            mean = tl.load(Mean + row)
            rstd = tl.load(Rstd + row)

            # Compute dx:
            xhat = (x - mean) * rstd
            wdy = w * dy
            wdy = tl.where(mask, wdy, 0)
            c1 = tl.sum(xhat * wdy, axis=0) / N
            c2 = tl.sum(wdy, axis=0) / N
            dx = (wdy - (xhat * c1 + c2)) * rstd

            # Write dx:
            tl.store(dx_ptrs + cols, dx.to(DX.type.element_ty), mask=mask)
            if not IGNORE_DW_DB:
                # Accumulate partial sums for dw and db:
                dw_row += dy * xhat
                db_row += dy

            # Advance to next row:
            row += tile_num
        if not IGNORE_DW_DB:
            tl.store(DW + pid * N + cols, dw_row.to(DW.type.element_ty), mask=mask)
            tl.store(DB + pid * N + cols, db_row.to(DB.type.element_ty), mask=mask)


@triton.jit
def _layernorm_bwd_dwdb_triton(
    DW,  # pointer to the partial sum of weights gradient
    DB,  # pointer to the partial sum of biases gradient
    FINAL_DW,  # pointer to the weights gradient
    FINAL_DB,  # pointer to the biases gradient
    M,  # GROUP_SIZE_M
    N,  # number of columns
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Map the program id to the elements of DW and DB it should compute.
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Iterate through the rows of DW and DB to sum the partial sums.
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0)
        db += tl.load(DB + offs, mask=mask, other=0.0)
    # Write the final sum to the output.
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw.to(FINAL_DW.type.element_ty), mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db.to(FINAL_DB.type.element_ty), mask=cols < N)


@triton.jit
def _layernorm_bwd_dwdb_triton_v2(
    X,  # pointer to the input
    DY,  # pointer to the output gradient
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,
    FINAL_DW,  # pointer to the weights gradient
    FINAL_DB,  # pointer to the biases gradient
    M,  # GROUP_SIZE_M
    N,  # number of columns
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Iterate through the rows of x and dy to compute dw and db
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        means = tl.load(Mean + rows, mask=rows < M, other=0.0).to(tl.float32)
        rstds = tl.load(Rstd + rows, mask=rows < M, other=0.0).to(tl.float32)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * stride + cols[None, :]
        x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
        xhat = (x - means[:, None]) * rstds[:, None]
        dw += dy * xhat
        db += dy
    # Write the final sum to the output.
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw.to(FINAL_DW.type.element_ty), mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db.to(FINAL_DB.type.element_ty), mask=cols < N)


def _layernorm_backward(
    dy: torch.Tensor,
    dx: torch.Tensor,
    dw: torch.Tensor,
    db: torch.Tensor,
    x: torch.Tensor,
    gamma: torch.Tensor,
    mu: torch.Tensor,
    rsigma: torch.Tensor,
):

    M, N = x.shape
    # calculate dw and db separately when M is small
    IGNORE_DW_DB_IN_FUSED = M <= 512
    tile_num = max(min(256, M // 4), 1)
    if M <= 512 and M * N < 64 * 1024 * 1024:
        tile_num = M
    elif M >= 8192:
        tile_num = 2048
    max_fused_size = 32768 // x.element_size()
    next_power = triton.next_power_of_2(N)
    BLOCK_SIZE = min(max_fused_size, next_power)
    # For cases with small M and large N, decrease block size to help with occupancy and register spill
    if tile_num == M:
        if tile_num > 256:
            BLOCK_SIZE = min(BLOCK_SIZE, 2048)
        else:
            BLOCK_SIZE = min(BLOCK_SIZE, 4096)
    USE_BLOCKED = N > BLOCK_SIZE
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

    if not IGNORE_DW_DB_IN_FUSED:
        _dw = torch.zeros((tile_num, N), dtype=torch.float32, device=gamma.device)
        _db = torch.zeros((tile_num, N), dtype=torch.float32, device=gamma.device)
    else:
        _dw = None
        _db = None

    grid_bwd = (tile_num,)
    _layernorm_bwd_dx_fused_triton[grid_bwd](
        dx,
        dy,
        _dw,
        _db,
        x,
        gamma,
        mu,
        rsigma,
        x.stride(0),
        N,
        NUM_ROWS=M,
        BLOCK_SIZE_N=BLOCK_SIZE,
        USE_BLOCKED=USE_BLOCKED,
        num_warps=num_warps,
        IGNORE_DW_DB=IGNORE_DW_DB_IN_FUSED,
    )
    grid_reduce = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE_N"]),)  # noqa: E731
    if not IGNORE_DW_DB_IN_FUSED:
        dwdb_block_n = max(16, N // 256)
        dwdb_block_n = triton.next_power_of_2(dwdb_block_n)
        dwdb_block_m = (64 * 128) // dwdb_block_n
        dwdb_block_m = min(triton.next_power_of_2(tile_num), dwdb_block_m)
        _layernorm_bwd_dwdb_triton[grid_reduce](
            _dw,
            _db,
            dw,
            db,
            min(tile_num, M),
            N,
            BLOCK_SIZE_M=dwdb_block_m,
            BLOCK_SIZE_N=dwdb_block_n,
        )
    else:
        dwdb_block_n = max(16, N // 256)
        dwdb_block_n = triton.next_power_of_2(dwdb_block_n)
        dwdb_block_m = (64 * 128) // dwdb_block_n
        dwdb_block_m = min(triton.next_power_of_2(M), dwdb_block_m)
        _layernorm_bwd_dwdb_triton_v2[grid_reduce](
            x,
            dy,
            mu,
            rsigma,
            x.stride(0),
            dw,
            db,
            M,
            N,
            BLOCK_SIZE_M=dwdb_block_m,
            BLOCK_SIZE_N=dwdb_block_n,
        )

    return dx, dw, db


class _LayerNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps, is_grad_enabled):

        is_grad = is_grad_enabled and any(
            tensor.requires_grad for tensor in [x, weight, bias]
        )

        y = torch.empty_like(x)
        M = x.shape[0]
        mean = torch.empty((M,), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

        _layernorm_forward(y, x, weight, bias, mean, rstd, eps)

        if is_grad:
            ctx.save_for_backward(x, weight, mean, rstd)

        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, m, v = ctx.saved_tensors
        N = w.shape[0]

        dw = torch.empty((N,), dtype=w.dtype, device=w.device)
        db = torch.empty((N,), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)

        _layernorm_backward(dy, dx, dw, db, x, w, m, v)

        return dx, dw, db, None, None


class _Layernorm2dFwdWithAdd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, y, x, res_in, res_out, weight, bias, eps, is_grad_enabled):

        is_grad = is_grad_enabled and any(
            tensor.requires_grad for tensor in [x, weight, bias]
        )

        M = x.shape[0]
        mean = torch.empty((M,), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

        _layernorm_forward_with_add(
            y, x, res_in, res_out, weight, bias, mean, rstd, eps
        )

        if is_grad:
            ctx.save_for_backward(res_out, weight, mean, rstd)

        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, m, v = ctx.saved_tensors
        N = w.shape[0]

        dw = torch.empty((N,), dtype=w.dtype, device=w.device)
        db = torch.empty((N,), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)

        _layernorm_backward(dy, dx, dw, db, x, w, m, v)

        return None, dx, None, None, dw, db, None, None


def layer_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    x_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Applies Layer Normalization over a mini-batch of inputs.

    Key parameters:
    - input: The input tensor to be normalized with shape (M, N).
    - weight: The learnable weights tensor with shape (N, ).
    - bias: The learnable bias tensor with shape (N, )
    - eps: A value added to the denominator for numerical stability.

    Returns:
    - Output: The output tensor with shape (M, N).
    """
    return _LayerNorm.apply(input, weight, bias, eps, torch.is_grad_enabled())


def layernorm2d_fwd_with_add(
    out: torch.Tensor,
    input: torch.Tensor,
    residual_in: torch.Tensor,
    residual_out: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    epsilon: float,
    x_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Adds two inputs and then applies Layer Normalization

    Key parameters:
    - out: The output of layer normalization with shape (M, N). Allocated by the caller
    - input: The input tensor to be normalized with shape (M, N).
    - residual_in: Tensor added to the input and same shape as input (M, N)
    - residual_out: Output tensor that is input + residual_in with shape (M, N). Must be allocated by the caller
    - weight: The learnable weights tensor with shape (N, ).
    - bias: Bias added to the result of layer norm with shape (N,)
    - epsilon: A value added to the denominator for numerical stability.

    Returns:
    - out: The output tensor with shape (M, N).
    - residual_out: Output tensor that is input + residual_in with shape (M, N).
    """
    return _Layernorm2dFwdWithAdd.apply(
        out,
        input,
        residual_in,
        residual_out,
        weight,
        bias,
        epsilon,
        torch.is_grad_enabled(),
    )


def layernorm2d_fwd_with_dynamicquant(
    out: torch.Tensor,
    input: torch.Tensor,
    yscale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    epsilon: float = 1e-5,
    x_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Applies Layer Normalization and then quantizes the output

    Key parameters:
    - out: The output of layer normalization with shape (M, N). Allocated by the caller
    - input: The input tensor to be normalized with shape (M, N) and dtype in (fp32, fp16 or bf16)
    - yscale: Output scale tensor with shape (M,) and dtype fp32. Allocated by the caller
    - weight: The learnable weights tensor with shape (N, ).
    - bias: Bias added to the result of layer norm with shape (N,)
    - eps: A value added to the denominator for numerical stability.

    Returns:
    - out: The output tensor with shape (M, N).
    - yscale: Output scale tensor with shape (M,). Allocated by the caller
    """
    M, N = input.shape

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // input.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))

    xscale = None
    IS_SMOOTH = False
    DTYPE_MAX = get_dtype_max(out.dtype)

    # Auxiliary tensor to store the RMSNorm output as fp32 before applying the quantization when using the blocked approach
    aux = torch.empty(M, N, dtype=torch.float32, device=input.device)

    _quant_layernorm_kernel[(M,)](
        input,
        out,
        weight,
        bias,
        xscale,
        yscale,
        aux,
        input.stride(0),
        out.stride(0),
        aux.stride(0),
        M,
        N,
        epsilon,
        DTYPE_MAX,
        IS_SMOOTH,
        BLOCK_SIZE,
    )

    return


def layernorm2d_fwd_with_smoothquant(
    out: torch.Tensor,
    input: torch.Tensor,
    xscale: torch.Tensor,
    yscale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    epsilon: float = 1e-5,
    x_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Applies Layer Normalization and then quantizes the output

    Key parameters:
    - input: The input tensor to be normalized with shape (M, N).
    - xscale: Input scale tensor which is multiplied with the output of layer normalization before quantization.
    - yscale: Output scale tensor with shape (M,) and dtype fp32. Allocated by the caller
    - weight: The learnable weights tensor with shape (N, ).
    - bias: Bias added to the result of layer norm with shape (N,)
    - eps: A value added to the denominator for numerical stability.

    Returns:
    - Output: The output tensor with shape (M, N).
    """
    M, N = input.shape

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // input.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))

    IS_SMOOTH = True
    DTYPE_MAX = get_dtype_max(out.dtype)

    # Auxiliary tensor to store the RMSNorm output as fp32 before applying the quantization when using the blocked approach
    aux = torch.empty(M, N, dtype=torch.float32, device=input.device)

    _quant_layernorm_kernel[(M,)](
        input,
        out,
        weight,
        bias,
        xscale,
        yscale,
        aux,
        input.stride(0),
        out.stride(0),
        aux.stride(0),
        M,
        N,
        epsilon,
        DTYPE_MAX,
        IS_SMOOTH,
        BLOCK_SIZE,
    )

    return


def layernorm2d_fwd_with_add_dynamicquant(
    out: torch.Tensor,
    input: torch.Tensor,
    residual_in: torch.Tensor,
    residual_out: torch.Tensor,
    yscale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    epsilon: float = 1e-5,
    x_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Adds two input toegether, then does layer Normalization before quantizing the final output

    Key parameters:
    - out: The output of layer normalization with shape (M, N). Allocated by the caller
    - input: The input tensor to be normalized with shape (M, N) and dtype in (fp32, fp16 or bf16)
    - residual_in: Tensor added to the input and same shape as input (M, N)
    - residual_out: Output tensor that is input + residual_in with shape (M, N). Must be allocated by the caller
    - yscale: Output scale tensor with shape (M,) and dtype fp32. Allocated by the caller
    - weight: The learnable weights tensor with shape (N, ).
    - bias: Bias added to the result of layer norm with shape (N,)
    - eps: A value added to the denominator for numerical stability.

    Returns:
    - out: The output tensor with shape (M, N).
    - yscale: Output scale tensor with shape (M,). Allocated by the caller
    """
    M, N = input.shape

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // input.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))

    xscale = None
    IS_SMOOTH = False
    DTYPE_MAX = get_dtype_max(out.dtype)

    # Auxiliary tensor to store the RMSNorm output as fp32 before applying the quantization when using the blocked approach
    aux = torch.empty(M, N, dtype=torch.float32, device=input.device)

    _quant_fused_add_layernorm_kernel[(M,)](
        input,
        out,
        residual_in,
        residual_out,
        weight,
        bias,
        xscale,
        yscale,
        aux,
        input.stride(0),
        out.stride(0),
        aux.stride(0),
        M,
        N,
        epsilon,
        DTYPE_MAX,
        IS_SMOOTH,
        BLOCK_SIZE,
    )

    return


def layernorm2d_fwd_with_add_smoothquant(
    out: torch.Tensor,
    input: torch.Tensor,
    residual_in: torch.Tensor,
    residual_out: torch.Tensor,
    xscale: torch.Tensor,
    yscale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    epsilon: float = 1e-5,
    x_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Applies Layer Normalization and then quantizes the output

    Key parameters:
    - input: The input tensor to be normalized with shape (M, N).
    - residual_in: Tensor added to the input and same shape as input (M, N)
    - residual_out: Output tensor that is input + residual_in with shape (M, N). Must be allocated by the caller
    - xscale: Input scale tensor which is multiplied with the output of layer normalization before quantization.
    - yscale: Output scale tensor with shape (M,) and dtype fp32. Allocated by the caller
    - weight: The learnable weights tensor with shape (N, ).
    - bias: Bias added to the result of layer norm with shape (N,)
    - eps: A value added to the denominator for numerical stability.

    Returns:
    - Output: The output tensor with shape (M, N).
    - yscale: Output scale tensor with shape (M,). Allocated by the caller
    """

    M, N = input.shape

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // input.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))

    IS_SMOOTH = True
    DTYPE_MAX = get_dtype_max(out.dtype)

    # Auxiliary tensor to store the RMSNorm output as fp32 before applying the quantization when using the blocked approach
    aux = torch.empty(M, N, dtype=torch.float32, device=input.device)

    _quant_fused_add_layernorm_kernel[(M,)](
        input,
        out,
        residual_in,
        residual_out,
        weight,
        bias,
        xscale,
        yscale,
        aux,
        input.stride(0),
        out.stride(0),
        aux.stride(0),
        M,
        N,
        epsilon,
        DTYPE_MAX,
        IS_SMOOTH,
        BLOCK_SIZE,
    )

    return
