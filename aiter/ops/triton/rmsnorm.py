# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
from typing import Optional
from aiter.ops.triton.utils.types import get_dtype_max
from aiter.ops.triton.utils.device_info import get_num_sms
from aiter.ops.triton._triton_kernels.rmsnorm import (
    _per_token_quant,
    _rms_norm_kernel,
    _quant_rms_norm_kernel,
    _fused_add_rmsnorm_kernel,
    _quant_fused_add_rmsnorm_kernel,
    _rmsnorm_bwd_triton,
    _rmsnorm_bwd_dg_reduce_triton,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def num_programs(x):
    return min(x.shape[0], get_num_sms())


def block_size(x):
    return min(65536 // x.element_size(), triton.next_power_of_2(x.shape[1]))


def use_blocked(x):
    return x.shape[1] > block_size(x)


def dg_tmp_rows(x):
    return x.shape[0] if use_blocked(x) else num_programs(x)


def _rmsnorm_forward(x: torch.Tensor, weight: torch.Tensor, epsilon: float):

    n_rows, n_cols = x.shape

    y = torch.empty_like(x)
    rsigma = torch.empty((n_rows,), dtype=torch.float32, device=x.device)

    blk_size = block_size(x)
    USE_BLOCKED = use_blocked(x)
    NUM_PRGMS = num_programs(x)

    grid = lambda meta: (NUM_PRGMS,)  # noqa: E731
    _rms_norm_kernel[grid](
        x,
        y,
        weight,
        rsigma,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        epsilon,
        blk_size,
        USE_BLOCKED,
        NUM_PRGMS,
    )

    return y, rsigma


def _rmsnorm_forward_with_add(
    out: torch.Tensor,
    x: torch.Tensor,
    residual_in: torch.Tensor,
    residual_out: torch.Tensor,
    weight: torch.Tensor,
    rsigma: torch.Tensor,
    epsilon: float,
):

    n_rows, n_cols = x.shape

    blk_size = block_size(x)
    USE_BLOCKED = use_blocked(x)
    NUM_PRGMS = num_programs(x)

    grid = lambda meta: (NUM_PRGMS,)  # noqa: E731
    _fused_add_rmsnorm_kernel[grid](
        x,
        out,
        residual_in,
        residual_out,
        weight,
        rsigma,
        x.stride(0),
        out.stride(0),
        n_rows,
        n_cols,
        epsilon,
        blk_size,
        USE_BLOCKED,
        NUM_PRGMS,
    )


def _rmsnorm_backward(dz, x, gamma, rsigma):
    dz_ = dz.contiguous()
    x_ = x.contiguous()
    gamma_ = gamma.contiguous()
    rsigma_ = rsigma.contiguous()

    dx = torch.empty_like(x_)
    dgamma = torch.empty_like(gamma_)

    M, N = x_.shape
    blk_size = block_size(x_)
    USE_BLOCKED = use_blocked(x_)
    NUM_PRGMS = num_programs(x_)
    need_reduction = N > 1

    dg_tmp = (
        torch.empty(
            dg_tmp_rows(x_), N, device="cuda", dtype=torch.float32, requires_grad=False
        )
        if need_reduction
        else None
    )

    grid_bwd = lambda meta: (NUM_PRGMS,)  # noqa: E731
    _rmsnorm_bwd_triton[grid_bwd](
        dz_,
        x_,
        gamma_,
        rsigma_,
        dx,
        dg_tmp if need_reduction else dgamma,
        x_.stride(0),
        dz_.stride(0),
        M,
        N,
        blk_size,
        USE_BLOCKED,
        NUM_PRGMS,
        num_warps=8,
    )

    if need_reduction:
        grid_reduce = lambda meta: [triton.cdiv(N, meta["BLOCK_SIZE_N"])]  # noqa: E731
        _rmsnorm_bwd_dg_reduce_triton[grid_reduce](
            dg_tmp,
            dgamma,
            dg_tmp.stride(0),
            dg_tmp.shape[0],
            dg_tmp.shape[1],
            BLOCK_SIZE_M=128,
            BLOCK_SIZE_N=64,
        )

    return dx, dgamma


class _RMSNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, epsilon, is_grad_enabled):

        is_grad = is_grad_enabled and any(
            tensor.requires_grad for tensor in [x, weight]
        )

        y, rsigma = _rmsnorm_forward(x, weight, epsilon)

        if is_grad:
            ctx.save_for_backward(x, weight, rsigma)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, rsigma = ctx.saved_tensors

        dx, dg = _rmsnorm_backward(grad_output, x, weight, rsigma)

        return dx, dg, None, None


class _RMSNorm2dFwdWithAdd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, y, x, res_in, res_out, weight, epsilon, is_grad_enabled):

        is_grad = is_grad_enabled and any(
            tensor.requires_grad for tensor in [x, weight]
        )

        M = x.shape[0]
        rsigma = torch.empty((M,), dtype=torch.float32, device=x.device)

        _rmsnorm_forward_with_add(y, x, res_in, res_out, weight, rsigma, epsilon)

        if is_grad:
            ctx.save_for_backward(res_out, weight, rsigma)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, rsigma = ctx.saved_tensors

        dx, dg = _rmsnorm_backward(grad_output, x, weight, rsigma)

        return None, dx, None, None, dg, None, None


def rms_norm(input: torch.Tensor, weight: torch.Tensor, epsilon: float):
    """
    Applies Root Mean Square Layer Normalization over a mini-batch of inputs.

    Key parameters:
    - Input: The input tensor to be normalized with shape (M, N).
    - Weight: The learnable weights tensor with shape (N, ).
    - Epsilon: A value added to the denominator for numerical stability.

    Returns:
    - Output: The output tensor with shape (M, N).
    """
    _LOGGER.info(f"RMSNORM: input={tuple(input.shape)} weight={tuple(weight.shape)} ")
    return _RMSNorm.apply(input, weight, epsilon, torch.is_grad_enabled())


def rmsnorm2d_fwd_with_add(
    out: torch.Tensor,
    input: torch.Tensor,
    residual_in: torch.Tensor,
    residual_out: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
):
    """
    Performs an addition between two inputs and then applies Root Mean Square Layer Normalization over
    the addition result.

    Key parameters:
    - Out: The tensor where the output will be stored with shape (M, N).
    - Input: The input tensor to be normalized with shape (M, N).
    - Residual_in: The tensor to be added to the Input tensor with shape (M, N).
    - Residual_out: The tensor in which the addition result will be stored with shape (M, N).
    - Weight: The learnable weights tensor with shape (N, ).
    - Epsilon: A value added to the denominator for numerical stability.

    Returns:
    - Output: The output tensor with shape (M, N).
    """
    _LOGGER.info(
        f"RMSNORM_2D_FWD_ADD: input={tuple(input.shape)} weight={tuple(weight.shape)} residual_in={tuple(residual_in.shape)}  "
    )
    return _RMSNorm2dFwdWithAdd.apply(
        out, input, residual_in, residual_out, weight, epsilon, torch.is_grad_enabled()
    )


def rmsnorm2d_fwd_with_smoothquant(
    out: torch.Tensor,
    input: torch.Tensor,
    xscale: torch.Tensor,
    yscale: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
):
    """
    Applies Root Mean Square Layer Normalization over a mini-batch of inputs and quantizes the result.

    Key parameters:
    - Out: The tensor where the output will be stored with shape (M, N).
    - Input: The input tensor to be normalized with shape (M, N).
    - Xscale: The tensor to be multiplied by the RMSNorm output, with shape (N, ).
    - Yscale: The tensor where the scale for each row will be stored with shape (M, ).
    - Weight: The learnable weights tensor with shape (N, ).
    - Epsilon: A value added to the denominator for numerical stability.
    """
    _LOGGER.info(
        f"RMSNORM_2D_FWD_SMOOTHQUANT: input={tuple(input.shape)} weight={tuple(weight.shape)} "
        + f"xscale={tuple(xscale.shape)} yscale={tuple(yscale.shape)}  "
    )
    n_rows, n_cols = input.shape

    blk_size = block_size(input)
    USE_BLOCKED = use_blocked(input)
    NUM_PRGMS = num_programs(input)

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
    epsilon: float,
    scale_ub: Optional[torch.Tensor] = None,
    clamp_out: bool = False,
    dump_rms_norm: bool = False,
):
    """
    Applies Root Mean Square Layer Normalization over a mini-batch of inputs and quantizes the result.

    Key parameters:
    - Out: The tensor where the output will be stored with shape (M, N).
    - Input: The input tensor to be normalized with shape (M, N).
    - Yscale: The tensor where the scale for each row will be stored with shape (M, ).
    - Weight: The learnable weights tensor with shape (N, ).
    - Epsilon: A value added to the denominator for numerical stability.
    """
    _LOGGER.info(
        f"RMSNORM_2D_FWD_DYNAMICQUANT: input={tuple(input.shape)} weight={tuple(weight.shape)} yscale={tuple(yscale.shape)}  "
    )
    n_rows, n_cols = input.shape

    blk_size = block_size(input)
    USE_BLOCKED = use_blocked(input)
    NUM_PRGMS = num_programs(input)

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
    """
    Performs an addition between two inputs and then applies Root Mean Square Layer Normalization over
    the addition result followed by a quantization.

    Key parameters:
    - Out: The tensor where the output will be stored with shape (M, N).
    - Input: The input tensor to be normalized with shape (M, N).
    - Residual_in: The tensor to be added to the Input tensor with shape (M, N).
    - Residual_out: The tensor in which the addition result will be stored with shape (M, N).
    - Xscale: The tensor to be multiplied by the RMSNorm output, with shape (N, ).
    - Yscale: The tensor where the scale for each row will be stored with shape (M, ).
    - Weight: The learnable weights tensor with shape (N, ).
    - Epsilon: A value added to the denominator for numerical stability.
    """
    _LOGGER.info(
        f"RMSNORM_2D_FWD_ADD_SMOOTHQUANT: input={tuple(input.shape)} weight={tuple(weight.shape)} "
        + f"residual_in={tuple(residual_in.shape)} xscale={tuple(xscale.shape)} yscale={tuple(yscale.shape)}  "
    )
    n_rows, n_cols = input.shape

    blk_size = block_size(input)
    USE_BLOCKED = use_blocked(input)
    NUM_PRGMS = num_programs(input)

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
    """
    Performs an addition between two inputs and then applies Root Mean Square Layer Normalization over
    the addition result followed by a quantization.

    Key parameters:
    - Out: The tensor where the output will be stored with shape (M, N).
    - Input: The input tensor to be normalized with shape (M, N).
    - Residual_in: The tensor to be added to the Input tensor with shape (M, N).
    - Residual_out: The tensor in which the addition result will be stored with shape (M, N).
    - Yscale: The tensor where the scale for each row will be stored with shape (M, ).
    - Weight: The learnable weights tensor with shape (N, ).
    - Epsilon: A value added to the denominator for numerical stability.
    """
    _LOGGER.info(
        f"RMSNORM_2D_FWD_ADD_DYNAMICQUANT: input={input.shape} weight={weight.shape} residual_in={residual_in.shape} yscale={yscale.shape}  "
    )
    n_rows, n_cols = input.shape

    blk_size = block_size(input)
    USE_BLOCKED = use_blocked(input)
    NUM_PRGMS = num_programs(input)

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

from itertools import product
def get_hip_autotune_config():
    return [triton.Config({'waves_per_eu': we}, num_warps=nw) for (we, nw) in product([0, 1, 2, 4], [2, 4, 8, 16])]

def get_autotune_config():
    get_hip_autotune_config()


@triton.autotune(configs=get_autotune_config(), key=['n_rows1', 'n_cols'], use_cuda_graph=False)
@triton.jit
def fused_rmsnorm_rope_kernel(
    # Pointers to matrices
    input_ptr1,
    input_ptr2,
    out_ptr1,
    out_ptr2,
    g_ptr1,
    g_ptr2,
    pos_id_ptr,
    cos_sin_cache_ptr,
    input_stride_1,
    input_stride_2,
    nhead_1,
    nhead_2,
    n_rows1,
    n_rows2,
    n_cols,
    epsilon,
    ROPE_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCKING: tl.constexpr,
    NUM_PRGMS: tl.constexpr,
    IS_NEOX_STYPE: tl.constexpr,
):

    # Map the program id to the first row of input and output it should compute.
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    total_rows = n_rows1
    # tl.device_print("pid is: ", row_start)

    if not BLOCKING:
        mask = col_offsets < n_cols
        for row_idx in tl.range(row_start, total_rows, NUM_PRGMS, num_stages=2):
            if row_idx < n_rows1:
                pos_idx = row_idx // nhead_1
                # tl.device_print("row_idx: ", row_idx)
                input_ptrs = input_ptr1 + row_idx * input_stride_1 + col_offsets
                output_ptrs = out_ptr1 + row_idx * input_stride_1
                g_ptr = g_ptr1 + col_offsets
            else:
                new_row_idx = row_idx - n_rows1
                # tl.device_print("row_idx: ", new_row_idx)
                pos_idx = new_row_idx // nhead_2
                input_ptrs = input_ptr2 + new_row_idx * input_stride_2 + col_offsets
                output_ptrs = out_ptr2 + new_row_idx * input_stride_2
                g_ptr = g_ptr2 + col_offsets

            tl.multiple_of(input_ptrs, 16)
            tl.multiple_of(output_ptrs, 16)
            tl.multiple_of(g_ptr, 16)

            # rms_norm
            row = tl.load(input_ptrs, mask=mask)
            dtype = row.dtype
            row = row.to(tl.float32)
            g = tl.load(g_ptr, mask=mask)
            row_norm = row * row
            # row_norm = tl.where(mask, row_norm, 0.0)
            row_norm = tl.sum(row_norm, axis=-1)
            norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)
            rms_norm = (row * norm_factor).to(dtype) * g


            # rope
            rope_mask = (col_offsets < 2 * ROPE_DIM) & mask
            pos_ids = tl.load(pos_id_ptr + pos_idx)
            if BLOCK_SIZE == ROPE_DIM * 2:
                cos_sin = tl.load(cos_sin_cache_ptr + pos_ids * 2 * ROPE_DIM + col_offsets, mask=rope_mask)
                if IS_NEOX_STYPE:
                    rms_norm_x1, rms_norm_x2 = rms_norm.reshape(2, BLOCK_SIZE // 2).permute(1, 0).split()
                else:
                    rms_norm_x1, rms_norm_x2 = rms_norm.reshape(BLOCK_SIZE // 2, 2).split()
                cos, sin = cos_sin.reshape(2, BLOCK_SIZE // 2).permute(1, 0).split()
                o1 = rms_norm_x1 * cos - rms_norm_x2 * sin
                o2 = rms_norm_x2 * cos + rms_norm_x1 * sin
                rms_norm_out = tl.join(o1, o2).permute(1, 0).reshape(BLOCK_SIZE)
                output_ptrs = output_ptrs + col_offsets
                tl.multiple_of(output_ptrs, 16)
                tl.store(output_ptrs, rms_norm_out, mask=mask)
            else:
                # tl.device_print("in other path")
                rope_idx = tl.arange(0, ROPE_DIM)
                rope_load_idx = tl.arange(0, ROPE_DIM * 2)
                cos_sin = tl.load(cos_sin_cache_ptr + pos_ids * ROPE_DIM * 2 + rope_load_idx)
                if IS_NEOX_STYPE:
                    rms_norm_x1 = tl.gather(rms_norm, rope_idx, axis=0)
                    rms_norm_x2 = tl.gather(rms_norm, rope_idx + ROPE_DIM, axis=0)
                else:
                    rms_norm_x1 = tl.gather(rms_norm, rope_idx * 2, axis=0)
                    rms_norm_x2 = tl.gather(rms_norm, rope_idx * 2 + 1, axis=0)
                cos, sin = cos_sin.reshape(2, ROPE_DIM).permute(1, 0).split()
                o1 = rms_norm_x1 * cos - rms_norm_x2 * sin
                o2 = rms_norm_x2 * cos + rms_norm_x1 * sin
                rms_norm_out = tl.join(o1, o2).permute(1, 0).reshape(ROPE_DIM * 2)
                output_rope_ptrs = output_ptrs + rope_load_idx
                output_non_rope_ptrs = output_ptrs + col_offsets
                rope_store_mask = col_offsets >= 2 * ROPE_DIM
                tl.store(output_rope_ptrs, rms_norm_out)
                tl.store(output_non_rope_ptrs, rms_norm, mask=rope_store_mask)
    else:
        # tl.device_print("in other path")
        for row_idx in tl.range(row_start, total_rows, NUM_PRGMS, num_stages=1):
            if row_idx < n_rows1:
                row_base_input_ptr = input_ptr1 + row_idx * input_stride_1
                row_base_output_ptr = out_ptr1 + row_idx * input_stride_1
                g_base_ptr = g_ptr1
                pos_id_ptr = pos_id_ptr + row_idx
            else:
                new_row_idx = row_idx - n_rows1
                row_base_input_ptr = input_ptr2 + new_row_idx * input_stride_2
                row_base_output_ptr = out_ptr2 + new_row_idx * input_stride_2
                g_base_ptr = g_ptr2
                pos_id_ptr = pos_id_ptr + new_row_idx
            n_cols_blks = n_cols // BLOCK_SIZE
            sum_sq = 0.0
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                mask = col_offsets < (n_cols - blk_idx * BLOCK_SIZE)
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_base_input_ptr + cols
                tl.multiple_of(input_ptrs, 16)
                x = tl.load(input_ptrs, mask=mask, cache_modifier=".cg").to(tl.float32)
                sum_sq += tl.sum(x * x, axis=0)
            norm_factor = tl.rsqrt(sum_sq / cols + epsilon)
            pos_idx = tl.load(pos_id_ptr)
            cos_ptr = cos_sin_cache_ptr + pos_idx * 2 * ROPE_DIM
            sin_ptr = cos_sin_cache_ptr + pos_idx * 2 * ROPE_DIM + ROPE_DIM

            # process the rope part
            for blk_idx in tl.range(0, ROPE_DIM, BLOCK_SIZE, num_stages=2):
                # We assume the rope mask is valid on both cos and sin
                rope_mask = col_offsets + blk_idx < ROPE_DIM
                input_x1_ptrs = row_base_input_ptr + blk_idx + col_offsets
                input_x2_ptrs = row_base_input_ptr + blk_idx + ROPE_DIM + col_offsets
                cos_ptrs = cos_ptr + blk_idx + col_offsets
                sin_ptrs = sin_ptr + blk_idx + col_offsets
                g1_ptrs = g_base_ptr + col_offsets
                g2_ptrs = g_base_ptr + col_offsets + ROPE_DIM
                output_o1_ptrs = row_base_output_ptr + blk_idx + col_offsets
                output_o2_ptrs = row_base_output_ptr + blk_idx + ROPE_DIM + col_offsets
                tl.multiple_of(input_x1_ptrs, 16)
                tl.multiple_of(g1_ptrs, 16)
                tl.multiple_of(input_x2_ptrs, 16)
                tl.multiple_of(g2_ptrs, 16)
                tl.multiple_of(cos_ptr, 16)
                tl.multiple_of(sin_ptr, 16)
                x1 = tl.load(input_x1_ptrs, mask=rope_mask).to(tl.float32)
                g1 = tl.load(g1_ptrs, mask=rope_mask)
                x2 = tl.load(input_x2_ptrs, mask=rope_mask).to(tl.float32)
                g2 = tl.load(g2_ptrs, mask=rope_mask)
                cos = tl.load(cos_ptrs, mask=rope_mask)
                sin = tl.load(sin_ptrs, mask=rope_mask)

                rms_norm_x1 = x1 * norm_factor * g1
                rms_norm_x2 = x2 * norm_factor * g2
                o1 = rms_norm_x1 * cos - rms_norm_x2 * sin
                o2 = rms_norm_x2 * cos + rms_norm_x1 * sin

                tl.store(output_o1_ptrs, o1, mask=rope_mask)
                tl.store(output_o2_ptrs, o2, mask=rope_mask)

            # process the remaining part of rmsnorm
            for blk_idx in tl.range(ROPE_DIM * 2, n_cols, BLOCK_SIZE, num_stages=2):
                mask = col_offsets + blk_idx < n_cols
                cols = blk_idx + col_offsets
                input_ptrs = row_base_input_ptr + cols
                g_ptrs = g_base_ptr + cols
                output_ptrs = row_base_output_ptr + cols
                tl.multiple_of(input_ptrs, 16)
                tl.multiple_of(g_ptrs, 16)
                x = tl.load(input_ptrs, mask=mask).to(tl.float32)
                g = tl.load(g_ptrs, mask=mask).to(tl.float32)
                rms_norm = x * norm_factor * g

                tl.store(output_ptrs, rms_norm, mask=mask)


def rmsnorm3d_fwd_with_rope(
    input1: torch.Tensor,
    input2: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    epsilon: float,
    pos_embedding: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool
):
    """
    Performs an addition between two inputs and then applies Root Mean Square Layer Normalization over
    the addition result followed by a quantization.

    Key parameters:
    - Out: The tensor where the output will be stored with shape (M, N, K).
    - Input: The input tensor to be normalized with shape (M, N).
    - Residual_in: The tensor to be added to the Input tensor with shape (M, N).
    - Residual_out: The tensor in which the addition result will be stored with shape (M, N).
    - Yscale: The tensor where the scale for each row will be stored with shape (M, ).
    - Weight: The learnable weights tensor with shape (N, ).
    - Epsilon: A value added to the denominator for numerical stability.
    """
    import numpy as np
    # input1 = input1.view(-1, input1.shape[-1])
    # 
    # # add = torch.empty_like(input1)
    # rsigma = torch.empty([input1.shape[-1]], dtype=input1.dtype, device="cuda")
    # _rmsnorm_forward_with_add(input1, input1, input1, input1, weight1, rsigma, epsilon)
    # return input1, None
    n_row1 = int(np.prod(input1.shape[:-1]))
    n_row2 = int(np.prod(input2.shape[:-1]))
    head_dim = input1.shape[-1]
    n_head = input1.shape[-2]
    n_kv_head = input2.shape[-2]
    query_stride = input1.stride(-2)
    key_stride = input2.stride(-2)
    ROPE_DIM = cos_sin_cache.size(-1) // 2
    # def get_blk_size(head_dim):
    #     return min(1024, triton.next_power_of_2(head_dim))
    # WARP_SIZE = 64
    # NUM_WARPS = triton.next_power_of_2(head_dim // WARP_SIZE)
    input1 = input1.view(-1, input1.shape[-1])
    input2 = input2.view(-1, input2.shape[-1])
    out1 = torch.empty_like(input1)
    out2 = torch.empty_like(input2)
    BLOCK_SIZE = block_size(input1)
    USE_BLOCKED = use_blocked(input1)
    NUM_PRGMS = num_programs(input1)
    # BLOCKING = False
    # if BLOCK_SIZE < head_dim:
    #     BLOCKING = False
    # NUM_PRGMS = min(n_row1, get_num_sms())
    # print("num prog: ", NUM_PRGMS, n_row1 + n_row2)
    # print("num programs: ", NUM_PRGMS)
    grid = lambda meta: (NUM_PRGMS, 1, 1)  # noqa: E731
    assert input1.shape[-1] == head_dim
    assert input2.shape[-1] == head_dim
    assert weight1.shape[-1] == input1.shape[-1]
    assert weight2.shape[-1] == input1.shape[-1]
    rsigma = torch.empty((n_row1, ), device=input1.device, dtype=torch.float32)

    rms_kernel[grid](out1, input1, weight1, rsigma, input1.stride(0), out1.stride(0), n_row1, head_dim, epsilon, False,
                        BLOCK_SIZE, USE_BLOCKED, NUM_PRGMS)
    # print("blocking: ", BLOCKING)
    # fused_rmsnorm_rope_kernel[grid](
    #     input1,
    #     input2,
    #     out1,
    #     out2,
    #     weight1,
    #     weight2,
    #     pos_embedding,
    #     cos_sin_cache,
    #     query_stride,
    #     key_stride,
    #     n_head,
    #     n_kv_head,
    #     n_row1,
    #     0,
    #     head_dim,
    #     epsilon,
    #     ROPE_DIM=ROPE_DIM,
    #     BLOCK_SIZE=BLOCK_SIZE,
    #     BLOCKING=USE_BLOCKED,
    #     NUM_PRGMS=NUM_PRGMS,
    #     IS_NEOX_STYPE=is_neox_style
    # )
    return out1, out2




@triton.jit
def rms_kernel(output_ptr, input_ptr, g_ptr, rsigma_ptr, input_row_stride, output_row_stride, n_rows, n_cols, epsilon,
               ZERO_CENTERED_GAMMA: tl.constexpr, BLOCK_SIZE: tl.constexpr, USE_BLOCKED: tl.constexpr,
               NUM_PRGMS: tl.constexpr):
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # as older version Triton doesn't support tl.assume and BUFF OPS, comment out for now
    # tl.assume(input_row_stride >= 0)
    # tl.assume(output_row_stride >= 0)
    # tl.assume(row_start >= 0)

    if USE_BLOCKED:

        # Persistent loop for rows
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=1):
            row_input_ptr = input_ptr + row_idx * input_row_stride
            row_output_ptr = output_ptr + row_idx * output_row_stride

            # Accumulate sum of squares
            n_cols_blks = tl.cdiv(n_cols, BLOCK_SIZE) - 1
            # older version of triton doesn't accept below init
            # sum_squares: tl.float32 = 0.
            # however, with type promoting rule in triton, sum_squares should be always fp32 with below init
            sum_squares = 0.
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                input_ptrs = tl.multiple_of(input_ptrs, (16, ))
                x = tl.load(input_ptrs).to(tl.float32)
                sum_squares += tl.sum(x * x, axis=0)

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            input_ptrs = tl.multiple_of(input_ptrs, (16, ))
            x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
            sum_squares += tl.sum(x * x, axis=0)

            # Compute normalization factor
            mean_square = sum_squares / n_cols
            norm_factor = tl.rsqrt(mean_square + epsilon)

            # Store rsigma (norm_factor)
            tl.store(rsigma_ptr + row_idx, norm_factor)

            # Normalize and write output
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                input_ptrs = tl.multiple_of(input_ptrs, (16, ))
                x = tl.load(input_ptrs).to(tl.float32)
                g_ptrs = g_ptr + cols
                g = tl.load(g_ptrs).to(tl.float32)
                if (ZERO_CENTERED_GAMMA):
                    g += 1
                rms_norm = x * norm_factor * g
                output_ptrs = row_output_ptr + cols
                tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty))

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
            g_ptrs = g_ptr + cols
            g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
            if (ZERO_CENTERED_GAMMA):
                g += 1
            rms_norm = x * norm_factor * g
            output_ptrs = row_output_ptr + cols
            tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty), mask=mask)

    else:
        mask = col_offsets < n_cols
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=2):
            input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
            input_ptrs = tl.multiple_of(input_ptrs, (16, ))
            row = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
            g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            row_norm = row * row
            row_norm = tl.sum(row_norm, axis=-1)
            norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)

            # Store rsigma (norm_factor)
            rsigma_output_ptr = rsigma_ptr + row_idx
            tl.store(rsigma_output_ptr, norm_factor)

            if (ZERO_CENTERED_GAMMA):
                g += 1
            rms_norm = row * norm_factor * g

            output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
            output_ptrs = tl.multiple_of(output_ptrs, (16, ))
            tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty), mask=mask)

def forward(x, g, y, rsigma, dx, dg, dg_tmp, ZERO_CENTERED_GAMMA, epsilon=1e-6):
        n_rows, n_cols = x.shape
        blk_size = block_size(x)
        USE_BLOCKED = use_blocked(x)
        NUM_PRGMS = num_programs(x)
        # heuristics for number of warps:
        # num_warps = min(max(blk_size // 256, 1), 8)
        num_warps = 8
        grid = lambda meta: (NUM_PRGMS, )
        rsigma = torch.empty((n_rows, ), device=x.device, dtype=torch.float32)
        rms_kernel[grid](y, x, g, rsigma, x.stride(0), y.stride(0), n_rows, n_cols, epsilon, False,
                         blk_size, USE_BLOCKED, NUM_PRGMS)

        # ctx.save_for_backward(x, g, rsigma)
        # ctx.n_rows = n_rows
        # ctx.n_cols = n_cols
        # ctx.ZERO_CENTERED_GAMMA = ZERO_CENTERED_GAMMA
        # ctx.blk_size = blk_size
        # ctx.USE_BLOCKED = USE_BLOCKED
        # ctx.NUM_PRGMS = NUM_PRGMS
        # ctx.num_warps = num_warps

        # ctx.dx = dx
        # ctx.dg_tmp = dg_tmp
        # ctx.dg = dg

        return y