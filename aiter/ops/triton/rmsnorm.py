# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


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


def get_num_sms():
    # Returns the Compute Unit count of the current device
    current_device_index = torch.cuda.current_device()
    current_device = torch.cuda.get_device_properties(current_device_index)
    num_sms = current_device.multi_processor_count
    return num_sms


def rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float = 1e-6):

    n_rows, n_cols = x.shape
    y = torch.empty_like(x, device="cuda", dtype=x.dtype)

    MAX_FUSED_SIZE = 65536 // x.element_size()
    blk_size = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
    USE_BLOCKED = n_cols > blk_size
    NUM_PRGMS = min(n_rows, get_num_sms())

    grid = lambda meta: (NUM_PRGMS,)
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

    grid = lambda meta: (NUM_PRGMS,)
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


@triton.jit
def _rms_norm_dynamic_per_token_fp8_quant_kernel(
    x_ptr,
    o_ptr,
    w_ptr,
    o_scale_ptr,
    scale_ub,
    o_rms_norm_ptr,
    EPS: tl.constexpr,
    EPS_FP8: tl.constexpr,
    D: tl.constexpr,
    MAX_FP8: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    TL_FP8_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DUMP_INTERMEDIATE: tl.constexpr,
):
    row_idx = tl.program_id(axis=0).to(tl.int64)
    variance = tl.zeros([1], dtype=tl.float32)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < D

    row_start_ptr = x_ptr + row_idx * D
    out_start_ptr = o_ptr + row_idx * D
    input_ptrs = row_start_ptr + col_offsets
    output_ptrs = out_start_ptr + col_offsets

    row = tl.load(input_ptrs, mask=mask, other=0.0, eviction_policy="evict_last").to(
        tl.float32
    )
    variance = tl.sum(row * row, axis=0) / D
    rstd = tl.rsqrt(variance + EPS)

    w = tl.load(w_ptr + col_offsets, mask=mask, eviction_policy="evict_first").to(
        tl.float32
    )
    o_rms_norm = row * rstd * w

    # FP8 rowwise quantize
    cur_max = tl.max(tl.abs(o_rms_norm))

    if CLAMP_MAX:
        ub = tl.load(scale_ub)
        cur_max = tl.clamp(cur_max, EPS_FP8, ub)
    else:
        cur_max = tl.maximum(cur_max, EPS_FP8)
    o_scale = MAX_FP8 / cur_max
    tl.store(o_scale_ptr + row_idx, 1.0 / o_scale)
    o_fp8 = tl.clamp(o_rms_norm * o_scale, -MAX_FP8, MAX_FP8)
    o_fp8.to(TL_FP8_DTYPE)
    tl.store(output_ptrs, o_fp8, mask=mask)

    if DUMP_INTERMEDIATE:
        tl.store(
            o_rms_norm_ptr + row_idx * D + col_offsets,
            o_rms_norm,
            mask=mask,
        )


def rms_norm_dynamic_per_token_fp8_quant(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float = 1.0e-5,
    scale_ub: Optional[torch.Tensor] = None,
    dump_rms_norm: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    assert x.is_contiguous()
    assert w.is_contiguous()
    x_shape = x.shape
    x = x.view(-1, x.size(-1))
    (B_T, D) = x.shape

    pt_dtype = torch.float8_e4m3fnuz
    tl_dtype = tl.float8e4b8
    max_fp8 = torch.finfo(pt_dtype).max
    eps_fp8 = 1e-12

    out = torch.empty((B_T, D), device=x.device, dtype=pt_dtype)
    out_scale = torch.empty((B_T), device=x.device, dtype=torch.float32)
    assert w.shape == (D,)
    assert out.shape == x.shape

    out_rms_norm = None
    if dump_rms_norm:
        out_rms_norm = torch.empty_like(x)

    _rms_norm_dynamic_per_token_fp8_quant_kernel[(B_T,)](
        x,
        out,
        w,
        out_scale,
        scale_ub,
        out_rms_norm,
        eps,
        eps_fp8,
        D,
        MAX_FP8=max_fp8,
        CLAMP_MAX=scale_ub is not None,
        TL_FP8_DTYPE=tl_dtype,
        BLOCK_SIZE=triton.next_power_of_2(D),
        DUMP_INTERMEDIATE=dump_rms_norm,
    )

    if dump_rms_norm:
        assert out_rms_norm is not None
        out_rms_norm = out_rms_norm.view(x_shape)

    return out.view(x_shape), out_scale, out_rms_norm
