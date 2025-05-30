# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
from typing import Optional


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
