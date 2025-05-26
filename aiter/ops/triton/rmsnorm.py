# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def _per_token_quant(
    x,
    y_scale_ptr,
    row_max,
    row_idx,
    DTYPE_MAX: tl.constexpr,
    scale_ub_ptr=None,
    EPS_8BIT: tl.constexpr = 1e-12,
    CLAMP_MAX: tl.constexpr = False,
    CLAMP_OUT: tl.constexpr = False,
):
    """
    #TODO: Add Doc
    """

    if CLAMP_MAX:
        ub = tl.load(scale_ub_ptr)
        row_max = tl.clamp(row_max, EPS_8BIT, ub)

    scale_out = row_max / DTYPE_MAX
    scale_out = tl.where(scale_out == 0, 1.0, scale_out)

    scale_recip = 1 / scale_out

    qx = x * scale_recip

    if CLAMP_OUT:
        qx = tl.clamp(qx, -DTYPE_MAX, DTYPE_MAX)

    tl.store(y_scale_ptr + row_idx, scale_out.to(y_scale_ptr.dtype.element_ty))

    return qx


@triton.jit
def _rms_norm_kernel(
    # Pointers to matrices
    input_ptr,
    output_ptr,
    g_ptr,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `input_row_stride` is
    # how much to increase `input_ptr` by to get the element one row down.
    input_row_stride,
    output_row_stride,
    # Matrix dimensions
    n_rows,
    n_cols,
    # Epsilon to avoid division by zero
    epsilon,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
    USE_BLOCKED: tl.constexpr,
    NUM_PRGMS: tl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call rms_norm function
    below.

    Applies Root Mean Square Layer Normalization over a mini-batch of inputs.

    Key parameters:
    - Input: The input tensor to be normalized with shape (n_rows, n_cols).
    - Output: The output tensor with shape (n_rows, n_cols).
    - G: The learnable weights tensor with shape (n_cols, ).
    """
    # Map the program id to the first row of input and output it should compute.
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    if USE_BLOCKED:
        # Persistent loop for rows
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=1):
            row_input_ptr = input_ptr + row_idx * input_row_stride
            row_output_ptr = output_ptr + row_idx * output_row_stride

            # Accumulate sum of squares
            n_cols_blks = tl.cdiv(n_cols, BLOCK_SIZE) - 1
            sum_squares = 0.0
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                input_ptrs = tl.multiple_of(input_ptrs, (16,))
                x = tl.load(input_ptrs).to(tl.float32)
                sum_squares += tl.sum(x * x, axis=0)

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            input_ptrs = tl.multiple_of(input_ptrs, (16,))
            x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(
                tl.float32
            )
            sum_squares += tl.sum(x * x, axis=0)

            # Compute normalization factor
            mean_square = sum_squares / n_cols
            norm_factor = tl.rsqrt(mean_square + epsilon)

            # Normalize and write output
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                input_ptrs = tl.multiple_of(input_ptrs, (16,))
                x = tl.load(input_ptrs).to(tl.float32)
                g_ptrs = g_ptr + cols
                g = tl.load(g_ptrs).to(tl.float32)
                rms_norm = x * norm_factor * g
                output_ptrs = row_output_ptr + cols
                tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty))

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(
                tl.float32
            )
            g_ptrs = g_ptr + cols
            g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
            rms_norm = x * norm_factor * g
            output_ptrs = row_output_ptr + cols
            tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty), mask=mask)

    else:
        mask = col_offsets < n_cols
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=2):
            input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
            input_ptrs = tl.multiple_of(input_ptrs, (16,))
            row = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(
                tl.float32
            )
            g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            row_norm = row * row
            row_norm = tl.sum(row_norm, axis=-1)
            norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)

            rms_norm = row * norm_factor * g

            output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
            output_ptrs = tl.multiple_of(output_ptrs, (16,))
            tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty), mask=mask)


@triton.jit
def _quant_rms_norm_kernel(
    # Pointers to matrices
    input_ptr,
    output_ptr,
    x_scale_ptr,
    y_scale_ptr,
    g_ptr,
    # Auxiliary tensor to store intermediate data
    aux_ptr,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `input_row_stride` is
    # how much to increase `input_ptr` by to get the element one row down.
    input_row_stride,
    output_row_stride,
    aux_row_stride,
    # Matrix dimensions
    n_rows,
    n_cols,
    # Epsilon to avoid division by zero
    epsilon,
    # Optional pointers
    scale_ub_ptr,  # Pointer to the scale upper bound tensor
    out_intermediate_ptr,  # Pointer to the intermediate output tensor
    # Dtype max for quantization
    DTYPE_MAX: tl.constexpr,
    # Meta-parameters
    IS_SMOOTH: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    CLAMP_OUT: tl.constexpr,
    DUMP_INTERMEDIATE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_BLOCKED: tl.constexpr,
    NUM_PRGMS: tl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call rmsnorm2d_fwd_with_smoothquant or
    rmsnorm2d_fwd_with_dynamicquant functions below.

    Applies Root Mean Square Layer Normalization over a mini-batch of inputs and quantizes the result.

    Key parameters:
    - Input: The input tensor to be normalized with shape (n_rows, n_cols).
    - Output: The output tensor with shape (n_rows, n_cols).
    - X_scale: The tensor to be multiplied by the RMSNorm output if IS_SMOOTH is true, with shape (n_cols, ).
    - Y_scale: The tensor where the scale for each row will be stored with shape (n_rows, ).
    - G: The learnable weights tensor with shape (n_cols, ).
    """
    # Map the program id to the first row of input and output it should compute.
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    if USE_BLOCKED:
        # Persistent loop for rows
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=1):
            row_input_ptr = input_ptr + row_idx * input_row_stride
            row_output_ptr = output_ptr + row_idx * output_row_stride
            row_aux_ptr = aux_ptr + row_idx * aux_row_stride

            # Accumulate sum of squares
            n_cols_blks = tl.cdiv(n_cols, BLOCK_SIZE) - 1
            sum_squares = 0.0
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                input_ptrs = tl.multiple_of(input_ptrs, (16,))
                x = tl.load(input_ptrs).to(tl.float32)
                sum_squares += tl.sum(x * x, axis=0)

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            input_ptrs = tl.multiple_of(input_ptrs, (16,))
            x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(
                tl.float32
            )
            sum_squares += tl.sum(x * x, axis=0)

            # Compute normalization factor
            mean_square = sum_squares / n_cols
            norm_factor = tl.rsqrt(mean_square + epsilon)

            row_max = 0.0

            # Normalize and write output temporarily as fp32
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                input_ptrs = tl.multiple_of(input_ptrs, (16,))
                x = tl.load(input_ptrs).to(tl.float32)
                g_ptrs = g_ptr + cols
                g = tl.load(g_ptrs).to(tl.float32)
                rms_norm = x * norm_factor * g

                if DUMP_INTERMEDIATE:
                    tl.store(
                        out_intermediate_ptr + row_idx * n_cols + cols,
                        rms_norm.to(out_intermediate_ptr.type.element_ty),
                    )

                if IS_SMOOTH:
                    x_scale_ptrs = x_scale_ptr + cols
                    x_scale_ptrs = tl.multiple_of(x_scale_ptrs, (16,))
                    x_scale = tl.load(x_scale_ptrs)
                    rms_norm *= x_scale

                blk_max = tl.max(tl.abs(rms_norm), axis=-1)
                row_max = max(row_max, blk_max)

                aux_ptrs = row_aux_ptr + cols
                tl.store(aux_ptrs, rms_norm)

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(
                tl.float32
            )
            g_ptrs = g_ptr + cols
            g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
            rms_norm = x * norm_factor * g

            if DUMP_INTERMEDIATE:
                tl.store(
                    out_intermediate_ptr + row_idx * n_cols + cols,
                    rms_norm.to(out_intermediate_ptr.type.element_ty),
                    mask=mask,
                )

            if IS_SMOOTH:
                x_scale_ptrs = x_scale_ptr + cols
                x_scale = tl.load(
                    x_scale_ptrs, mask=mask, other=0.0, cache_modifier=".cg"
                )
                rms_norm *= x_scale

            blk_max = tl.max(tl.abs(rms_norm), axis=-1)
            row_max = max(row_max, blk_max)

            aux_ptrs = row_aux_ptr + cols
            tl.store(aux_ptrs, rms_norm, mask=mask)

            # Apply quantization and write output
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                aux_ptrs = row_aux_ptr + cols
                aux_ptrs = tl.multiple_of(aux_ptrs, (16,))
                aux = tl.load(aux_ptrs)

                output = _per_token_quant(
                    aux,
                    y_scale_ptr,
                    row_max,
                    row_idx,
                    DTYPE_MAX,
                    scale_ub_ptr=scale_ub_ptr,
                    CLAMP_MAX=CLAMP_MAX,
                    CLAMP_OUT=CLAMP_OUT,
                )

                output_ptrs = row_output_ptr + cols
                tl.store(output_ptrs, output.to(output_ptr.dtype.element_ty))

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            aux_ptrs = row_aux_ptr + cols
            aux = tl.load(aux_ptrs, mask=mask, other=0.0, cache_modifier=".cg")

            output = _per_token_quant(
                aux,
                y_scale_ptr,
                row_max,
                row_idx,
                DTYPE_MAX,
                scale_ub_ptr=scale_ub_ptr,
                CLAMP_MAX=CLAMP_MAX,
                CLAMP_OUT=CLAMP_OUT,
            )

            output_ptrs = row_output_ptr + cols
            tl.store(output_ptrs, output.to(output_ptr.dtype.element_ty), mask=mask)
    else:
        mask = col_offsets < n_cols
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=2):
            input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
            input_ptrs = tl.multiple_of(input_ptrs, (16,))
            row = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(
                tl.float32
            )
            g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            row_norm = row * row
            row_norm = tl.sum(row_norm, axis=-1)
            norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)

            rms_norm = row * norm_factor * g

            if DUMP_INTERMEDIATE:
                tl.store(
                    out_intermediate_ptr + row_idx * n_cols + col_offsets,
                    rms_norm.to(out_intermediate_ptr.type.element_ty),
                    mask=mask,
                )

            if IS_SMOOTH:
                x_scale_ptrs = x_scale_ptr + col_offsets
                x_scale_ptrs = tl.multiple_of(x_scale_ptrs, (16,))
                x_scale = tl.load(
                    x_scale_ptrs, mask=mask, other=0.0, cache_modifier=".cg"
                )
                rms_norm *= x_scale

            row_max = tl.max(tl.abs(rms_norm), axis=-1)
            rms_norm = _per_token_quant(
                rms_norm,
                y_scale_ptr,
                row_max,
                row_idx,
                DTYPE_MAX,
                scale_ub_ptr=scale_ub_ptr,
                CLAMP_MAX=CLAMP_MAX,
                CLAMP_OUT=CLAMP_OUT,
            )

            output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
            output_ptrs = tl.multiple_of(output_ptrs, (16,))
            tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty), mask=mask)


@triton.jit
def _fused_add_rmsnorm_kernel(
    # Pointers to matrices
    input_ptr,
    output_ptr,
    res_in_ptr,
    res_out_ptr,
    g_ptr,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `input_row_stride` is
    # how much to increase `input_ptr` by to get the element one row down.
    input_row_stride,
    output_row_stride,
    # Matrix dimensions
    n_rows,
    n_cols,
    # Epsilon to avoid division by zero
    epsilon,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
    USE_BLOCKED: tl.constexpr,
    NUM_PRGMS: tl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call
    rmsnorm2d_fwd_with_add function below.

    Performs an addition between two inputs and then applies Root Mean Square Layer Normalization over
    the addition result.

    Key parameters:
    - Input: The input tensor to be normalized with shape (n_rows, n_cols).
    - Output: The output tensor with shape (n_rows, n_cols).
    - Res_in: The tensor to be added to the Input tensor with shape (n_rows, n_cols).
    - Res_out: The tensor in which the addition result will be stored with shape (n_rows, n_cols).
    - G: The learnable weights tensor with shape (n_cols, ).
    """
    # Map the program id to the first row of input and output it should compute.
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    if USE_BLOCKED:
        # Persistent loop for rows
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=1):
            row_input_ptr = input_ptr + row_idx * input_row_stride
            row_output_ptr = output_ptr + row_idx * output_row_stride
            row_res_in_ptr = res_in_ptr + row_idx * input_row_stride
            row_res_out_ptr = res_out_ptr + row_idx * input_row_stride

            # Accumulate sum of squares
            n_cols_blks = tl.cdiv(n_cols, BLOCK_SIZE) - 1
            sum_squares = 0.0
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                input_ptrs = tl.multiple_of(input_ptrs, (16,))
                x = tl.load(input_ptrs)
                res_in_ptrs = row_res_in_ptr + cols
                res_in_ptrs = tl.multiple_of(res_in_ptrs, (16,))
                res_in = tl.load(res_in_ptrs)
                x += res_in
                # Stores residual_out
                res_out_ptrs = row_res_out_ptr + cols
                tl.store(res_out_ptrs, x.to(res_out_ptr.type.element_ty))

                x = x.to(tl.float32)
                sum_squares += tl.sum(x * x, axis=0)

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            input_ptrs = tl.multiple_of(input_ptrs, (16,))
            x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
            res_in_ptrs = row_res_in_ptr + cols
            res_in_ptrs = tl.multiple_of(res_in_ptrs, (16,))
            res_in = tl.load(res_in_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
            x += res_in
            # Stores residual_out
            res_out_ptrs = row_res_out_ptr + cols
            tl.store(res_out_ptrs, x.to(res_out_ptr.type.element_ty), mask=mask)

            x = x.to(tl.float32)
            sum_squares += tl.sum(x * x, axis=0)

            # Compute normalization factor
            mean_square = sum_squares / n_cols
            norm_factor = tl.rsqrt(mean_square + epsilon)

            # Normalize and write output
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                res_out_ptrs = row_res_out_ptr + cols
                res_out_ptrs = tl.multiple_of(res_out_ptrs, (16,))
                x = tl.load(res_out_ptrs).to(tl.float32)
                g_ptrs = g_ptr + cols
                g = tl.load(g_ptrs).to(tl.float32)
                rms_norm = x * norm_factor * g
                output_ptrs = row_output_ptr + cols
                tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty))

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            res_out_ptrs = row_res_out_ptr + cols
            x = tl.load(res_out_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(
                tl.float32
            )
            g_ptrs = g_ptr + cols
            g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
            rms_norm = x * norm_factor * g
            output_ptrs = row_output_ptr + cols
            tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty), mask=mask)

    else:
        mask = col_offsets < n_cols
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=2):
            input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
            input_ptrs = tl.multiple_of(input_ptrs, (16,))
            row = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
            res_in_ptrs = res_in_ptr + row_idx * input_row_stride + col_offsets
            res_in_ptrs = tl.multiple_of(res_in_ptrs, (16,))
            res_in = tl.load(res_in_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
            row += res_in
            # Stores residual_out
            res_out_ptrs = res_out_ptr + row_idx * input_row_stride + col_offsets
            res_out_ptrs = tl.multiple_of(res_out_ptrs, (16,))
            tl.store(res_out_ptrs, row.to(res_out_ptr.type.element_ty), mask=mask)
            row = row.to(tl.float32)

            g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            row_norm = row * row
            row_norm = tl.sum(row_norm, axis=-1)
            norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)

            rms_norm = row * norm_factor * g

            output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
            output_ptrs = tl.multiple_of(output_ptrs, (16,))
            tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty), mask=mask)


@triton.jit
def _quant_fused_add_rmsnorm_kernel(
    # Pointers to matrices
    input_ptr,
    output_ptr,
    res_in_ptr,
    res_out_ptr,
    x_scale_ptr,
    y_scale_ptr,
    g_ptr,
    # Auxiliary tensor to store intermediate data
    aux_ptr,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `input_row_stride` is
    # how much to increase `input_ptr` by to get the element one row down.
    input_row_stride,
    output_row_stride,
    aux_row_stride,
    # Matrix dimensions
    n_rows,
    n_cols,
    # Epsilon to avoid division by zero
    epsilon,
    # Dtype max for quantization
    DTYPE_MAX: tl.constexpr,
    # Meta-parameters
    IS_SMOOTH: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
    USE_BLOCKED: tl.constexpr,
    NUM_PRGMS: tl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call
    rmsnorm2d_fwd_with_add_smoothquant or rmsnorm2d_fwd_with_add_dynamicquant functions below.

    Performs an addition between two inputs and then applies Root Mean Square Layer Normalization over
    the addition result followed by a quantization.

    Key parameters:
    - Input: The input tensor to be normalized with shape (n_rows, n_cols).
    - Output: The output tensor with shape (n_rows, n_cols).
    - Res_in: The tensor to be added to the Input tensor with shape (n_rows, n_cols).
    - Res_out: The tensor in which the addition result will be stored with shape (n_rows, n_cols).
    - X_scale: The tensor to be multiplied by the RMSNorm output if IS_SMOOTH is true, with shape (n_cols, ).
    - Y_scale: The tensor where the scale for each row will be stored with shape (n_rows, ).
    - G: The learnable weights tensor with shape (n_cols, ).
    """
    # Map the program id to the first row of input and output it should compute.
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    if USE_BLOCKED:
        # Persistent loop for rows
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=1):
            row_input_ptr = input_ptr + row_idx * input_row_stride
            row_output_ptr = output_ptr + row_idx * output_row_stride
            row_res_in_ptr = res_in_ptr + row_idx * input_row_stride
            row_res_out_ptr = res_out_ptr + row_idx * input_row_stride
            row_aux_ptr = aux_ptr + row_idx * aux_row_stride

            # Accumulate sum of squares
            n_cols_blks = tl.cdiv(n_cols, BLOCK_SIZE) - 1
            sum_squares = 0.0
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                input_ptrs = tl.multiple_of(input_ptrs, (16,))
                x = tl.load(input_ptrs)
                res_in_ptrs = row_res_in_ptr + cols
                res_in_ptrs = tl.multiple_of(res_in_ptrs, (16,))
                res_in = tl.load(res_in_ptrs)
                x += res_in
                # Stores residual_out
                res_out_ptrs = row_res_out_ptr + cols
                tl.store(res_out_ptrs, x.to(res_out_ptr.type.element_ty))

                x = x.to(tl.float32)
                sum_squares += tl.sum(x * x, axis=0)

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            input_ptrs = tl.multiple_of(input_ptrs, (16,))
            x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
            res_in_ptrs = row_res_in_ptr + cols
            res_in_ptrs = tl.multiple_of(res_in_ptrs, (16,))
            res_in = tl.load(res_in_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
            x += res_in
            # Stores residual_out
            res_out_ptrs = row_res_out_ptr + cols
            tl.store(res_out_ptrs, x.to(res_out_ptr.type.element_ty), mask=mask)

            x = x.to(tl.float32)
            sum_squares += tl.sum(x * x, axis=0)

            # Compute normalization factor
            mean_square = sum_squares / n_cols
            norm_factor = tl.rsqrt(mean_square + epsilon)

            row_max = 0.0

            # Normalize and write output temporarily as fp32
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                res_out_ptrs = row_res_out_ptr + cols
                res_out_ptrs = tl.multiple_of(res_out_ptrs, (16,))
                x = tl.load(res_out_ptrs).to(tl.float32)
                g_ptrs = g_ptr + cols
                g = tl.load(g_ptrs).to(tl.float32)
                rms_norm = x * norm_factor * g

                if IS_SMOOTH:
                    x_scale_ptrs = x_scale_ptr + cols
                    x_scale_ptrs = tl.multiple_of(x_scale_ptrs, (16,))
                    x_scale = tl.load(x_scale_ptrs)
                    rms_norm *= x_scale

                blk_max = tl.max(tl.abs(rms_norm), axis=-1)
                row_max = max(row_max, blk_max)

                aux_ptrs = row_aux_ptr + cols
                tl.store(aux_ptrs, rms_norm)

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            res_out_ptrs = row_res_out_ptr + cols
            x = tl.load(res_out_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(
                tl.float32
            )
            g_ptrs = g_ptr + cols
            g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
            rms_norm = x * norm_factor * g

            if IS_SMOOTH:
                x_scale_ptrs = x_scale_ptr + cols
                x_scale = tl.load(
                    x_scale_ptrs, mask=mask, other=0.0, cache_modifier=".cg"
                )
                rms_norm *= x_scale

            blk_max = tl.max(tl.abs(rms_norm), axis=-1)
            row_max = max(row_max, blk_max)

            aux_ptrs = row_aux_ptr + cols
            tl.store(aux_ptrs, rms_norm, mask=mask)

            # Apply quantization and write output
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                aux_ptrs = row_aux_ptr + cols
                aux_ptrs = tl.multiple_of(aux_ptrs, (16,))
                aux = tl.load(aux_ptrs)

                output = _per_token_quant(
                    aux,
                    y_scale_ptr,
                    row_max,
                    row_idx,
                    DTYPE_MAX,
                )

                output_ptrs = row_output_ptr + cols
                tl.store(output_ptrs, output.to(output_ptr.dtype.element_ty))

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            aux_ptrs = row_aux_ptr + cols
            aux = tl.load(aux_ptrs, mask=mask, other=0.0, cache_modifier=".cg")

            output = _per_token_quant(
                aux,
                y_scale_ptr,
                row_max,
                row_idx,
                DTYPE_MAX,
            )

            output_ptrs = row_output_ptr + cols
            tl.store(output_ptrs, output.to(output_ptr.dtype.element_ty), mask=mask)

    else:
        mask = col_offsets < n_cols
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=2):
            input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
            input_ptrs = tl.multiple_of(input_ptrs, (16,))
            row = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
            res_in_ptrs = res_in_ptr + row_idx * input_row_stride + col_offsets
            res_in_ptrs = tl.multiple_of(res_in_ptrs, (16,))
            res_in = tl.load(res_in_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
            row += res_in
            # Stores residual_out
            res_out_ptrs = res_out_ptr + row_idx * input_row_stride + col_offsets
            res_out_ptrs = tl.multiple_of(res_out_ptrs, (16,))
            tl.store(res_out_ptrs, row.to(res_out_ptr.type.element_ty), mask=mask)
            row = row.to(tl.float32)

            g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            row_norm = row * row
            row_norm = tl.sum(row_norm, axis=-1)
            norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)

            rms_norm = row * norm_factor * g

            if IS_SMOOTH:
                x_scale_ptrs = x_scale_ptr + col_offsets
                x_scale_ptrs = tl.multiple_of(x_scale_ptrs, (16,))
                x_scale = tl.load(
                    x_scale_ptrs, mask=mask, other=0.0, cache_modifier=".cg"
                )
                rms_norm *= x_scale

            row_max = tl.max(tl.abs(rms_norm), axis=-1)
            rms_norm = _per_token_quant(
                rms_norm,
                y_scale_ptr,
                row_max,
                row_idx,
                DTYPE_MAX,
            )

            output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
            output_ptrs = tl.multiple_of(output_ptrs, (16,))
            tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty), mask=mask)


def get_num_sms():
    # Returns the Compute Unit count of the current device
    current_device_index = torch.cuda.current_device()
    current_device = torch.cuda.get_device_properties(current_device_index)
    num_sms = current_device.multi_processor_count
    return num_sms


def get_dtype_max(dtype):
    if torch.is_floating_point(torch.tensor([], dtype=dtype)):
        return torch.finfo(dtype).max
    else:
        return torch.iinfo(dtype).max


def rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float = 1e-6):

    n_rows, n_cols = x.shape
    y = torch.empty_like(x, device="cuda", dtype=x.dtype)

    MAX_FUSED_SIZE = 65536 // x.element_size()
    blk_size = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
    USE_BLOCKED = n_cols > blk_size
    NUM_PRGMS = min(n_rows, get_num_sms())

    grid = lambda meta: (NUM_PRGMS,)  # noqa: E731
    _rms_norm_kernel[grid](
        x,
        y,
        weight,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        epsilon,
        blk_size,
        USE_BLOCKED,
        NUM_PRGMS,
    )

    return y


def rmsnorm2d_fwd_with_add(
    out: torch.Tensor,
    input: torch.Tensor,
    residual_in: torch.Tensor,
    residual_out: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
):

    n_rows, n_cols = input.shape

    MAX_FUSED_SIZE = 65536 // input.element_size()
    blk_size = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
    USE_BLOCKED = n_cols > blk_size
    NUM_PRGMS = min(n_rows, get_num_sms())

    grid = lambda meta: (NUM_PRGMS,)  # noqa: E731
    _fused_add_rmsnorm_kernel[grid](
        input,
        out,
        residual_in,
        residual_out,
        weight,
        input.stride(0),
        out.stride(0),
        n_rows,
        n_cols,
        epsilon,
        blk_size,
        USE_BLOCKED,
        NUM_PRGMS,
    )

    return out, residual_out


def rmsnorm2d_fwd_with_smoothquant(
    out: torch.Tensor,
    input: torch.Tensor,
    xscale: torch.Tensor,
    yscale: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
):

    n_rows, n_cols = input.shape

    MAX_FUSED_SIZE = 65536 // input.element_size()
    blk_size = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
    USE_BLOCKED = n_cols > blk_size
    NUM_PRGMS = min(n_rows, get_num_sms())

    IS_SMOOTH = True
    DTYPE_MAX = get_dtype_max(out.dtype)

    scale_ub = None
    out_rmsnorm = None
    CLAMP_MAX = False
    clamp_out = False
    dump_rms_norm = False

    # Auxiliary tensor to store the RMSNorm output as fp32 before applying the quantization when using the blocked approach
    aux = None
    if USE_BLOCKED:
        aux = torch.empty(n_rows, n_cols, dtype=torch.float32, device=input.device)

    grid = lambda meta: (NUM_PRGMS,)  # noqa: E731
    _quant_rms_norm_kernel[grid](
        input,
        out,
        xscale,
        yscale,
        weight,
        aux,
        input.stride(0),
        out.stride(0),
        aux.stride(0) if USE_BLOCKED else None,
        n_rows,
        n_cols,
        epsilon,
        scale_ub,
        out_rmsnorm,
        DTYPE_MAX,
        IS_SMOOTH,
        CLAMP_MAX,
        clamp_out,
        dump_rms_norm,
        blk_size,
        USE_BLOCKED,
        NUM_PRGMS,
    )


def rmsnorm2d_fwd_with_dynamicquant(
    out: torch.Tensor,
    input: torch.Tensor,
    yscale: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
    scale_ub: Optional[torch.Tensor] = None,
    clamp_out: bool = False,
    dump_rms_norm: bool = False,
):

    n_rows, n_cols = input.shape

    MAX_FUSED_SIZE = 65536 // input.element_size()
    blk_size = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
    USE_BLOCKED = n_cols > blk_size
    NUM_PRGMS = min(n_rows, get_num_sms())

    xscale = None
    IS_SMOOTH = False
    DTYPE_MAX = get_dtype_max(out.dtype)
    CLAMP_MAX = scale_ub is not None

    out_rms_norm = None
    if dump_rms_norm:
        out_rms_norm = torch.empty_like(input)

    # Auxiliary tensor to store the RMSNorm output as fp32 before applying the quantization when using the blocked approach
    aux = None
    if USE_BLOCKED:
        aux = torch.empty(n_rows, n_cols, dtype=torch.float32, device=input.device)

    grid = lambda meta: (NUM_PRGMS,)  # noqa: E731
    _quant_rms_norm_kernel[grid](
        input,
        out,
        xscale,
        yscale,
        weight,
        aux,
        input.stride(0),
        out.stride(0),
        aux.stride(0) if USE_BLOCKED else None,
        n_rows,
        n_cols,
        epsilon,
        scale_ub,
        out_rms_norm,
        DTYPE_MAX,
        IS_SMOOTH,
        CLAMP_MAX,
        clamp_out,
        dump_rms_norm,
        blk_size,
        USE_BLOCKED,
        NUM_PRGMS,
    )

    return out_rms_norm


def rmsnorm2d_fwd_with_add_smoothquant(
    out: torch.Tensor,
    input: torch.Tensor,
    residual_in: torch.Tensor,
    residual_out: torch.Tensor,
    xscale: torch.Tensor,
    yscale: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
):

    n_rows, n_cols = input.shape

    MAX_FUSED_SIZE = 65536 // input.element_size()
    blk_size = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
    USE_BLOCKED = n_cols > blk_size
    NUM_PRGMS = min(n_rows, get_num_sms())

    IS_SMOOTH = True
    DTYPE_MAX = get_dtype_max(out.dtype)

    # Auxiliary tensor to store the RMSNorm output as fp32 before applying the quantization when using the blocked approach
    aux = None
    if USE_BLOCKED:
        aux = torch.empty(n_rows, n_cols, dtype=torch.float32, device=input.device)

    grid = lambda meta: (NUM_PRGMS,)  # noqa: E731
    _quant_fused_add_rmsnorm_kernel[grid](
        input,
        out,
        residual_in,
        residual_out,
        xscale,
        yscale,
        weight,
        aux,
        input.stride(0),
        out.stride(0),
        aux.stride(0) if USE_BLOCKED else None,
        n_rows,
        n_cols,
        epsilon,
        DTYPE_MAX,
        IS_SMOOTH,
        blk_size,
        USE_BLOCKED,
        NUM_PRGMS,
    )


def rmsnorm2d_fwd_with_add_dynamicquant(
    out: torch.Tensor,
    input: torch.Tensor,
    residual_in: torch.Tensor,
    residual_out: torch.Tensor,
    yscale: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
):

    n_rows, n_cols = input.shape

    MAX_FUSED_SIZE = 65536 // input.element_size()
    blk_size = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
    USE_BLOCKED = n_cols > blk_size
    NUM_PRGMS = min(n_rows, get_num_sms())

    xscale = None
    IS_SMOOTH = False
    DTYPE_MAX = get_dtype_max(out.dtype)

    # Auxiliary tensor to store the RMSNorm output as fp32 before applying the quantization when using the blocked approach
    aux = None
    if USE_BLOCKED:
        aux = torch.empty(n_rows, n_cols, dtype=torch.float32, device=input.device)

    grid = lambda meta: (NUM_PRGMS,)  # noqa: E731
    _quant_fused_add_rmsnorm_kernel[grid](
        input,
        out,
        residual_in,
        residual_out,
        xscale,
        yscale,
        weight,
        aux,
        input.stride(0),
        out.stride(0),
        aux.stride(0) if USE_BLOCKED else None,
        n_rows,
        n_cols,
        epsilon,
        DTYPE_MAX,
        IS_SMOOTH,
        blk_size,
        USE_BLOCKED,
        NUM_PRGMS,
    )
