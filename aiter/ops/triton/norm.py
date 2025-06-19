# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
from typing import Optional


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


def get_dtype_max(dtype):
    if torch.is_floating_point(torch.tensor([], dtype=dtype)):
        return torch.finfo(dtype).max
    else:
        return torch.iinfo(dtype).max


def layer_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    x_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    out = torch.empty_like(input)
    M, N = input.shape

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // input.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))

    _layernorm_kernel[(M,)](
        input, out, weight, bias, input.stride(0), out.stride(0), M, N, eps, BLOCK_SIZE
    )

    return out


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

    M, N = input.shape

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // input.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))

    _fused_add_layernorm_kernel[(M,)](
        input,
        out,
        residual_in,
        residual_out,
        weight,
        bias,
        input.stride(0),
        out.stride(0),
        M,
        N,
        epsilon,
        BLOCK_SIZE,
    )

    return


def layernorm2d_fwd_with_dynamicquant(
    out: torch.Tensor,
    input: torch.Tensor,
    yscale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    epsilon: float = 1e-5,
    x_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:

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

    return out


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

    return out


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

    return out


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

    return out
