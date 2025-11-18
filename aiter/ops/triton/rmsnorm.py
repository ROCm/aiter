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
    _rmsnorm_kernel_large_m_small_n,
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


def _should_use_large_m_small_n(M: int, N: int) -> bool:

    if M > 8192 and N <= 2048:
        return True

    return False


def rmsnorm_forward_inference(x: torch.Tensor, weight: torch.Tensor, eps: float):
    assert x.ndim == 2 and weight.ndim == 1 and x.shape[1] == weight.shape[0]
    x = x.contiguous()
    weight = weight.contiguous()
    M, N = x.shape

    if _should_use_large_m_small_n(M, N):
        return _rmsnorm_forward_large_m_small_n(x, weight, eps, return_rsigma=False)
    else:
        y, _ = _rmsnorm_forward(
            x, weight, eps
        )  # always returns rsigma, but we discard it
        return y


class _RMSNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, epsilon, is_grad_enabled):

        is_grad = is_grad_enabled and any(
            tensor.requires_grad for tensor in [x, weight]
        )
        M, N = x.shape
        if _should_use_large_m_small_n(M, N):
            out = _rmsnorm_forward_large_m_small_n(
                x, weight, epsilon, return_rsigma=is_grad
            )
            if is_grad:
                y, rsigma = out
            else:
                y = out
                rsigma = None
        else:
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


def _rmsnorm_forward_large_m_small_n(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    return_rsigma: bool = False,
):
    assert x.ndim == 2 and weight.ndim == 1 and x.shape[1] == weight.shape[0]
    x, weight = x.contiguous(), weight.contiguous()
    M, N = x.shape
    y = torch.empty_like(x)
    rsigma = (
        torch.empty(M, dtype=torch.float32, device=x.device) if return_rsigma else None
    )

    BLOCK_N = triton.next_power_of_2(N)
    BLOCK_M = min(16384 // BLOCK_N, 32)
    BLOCK_M = max(BLOCK_M, 8)

    grid = (triton.cdiv(M, BLOCK_M),)
    _rmsnorm_kernel_large_m_small_n[grid](
        x,
        y,
        weight,
        rsigma,
        M,
        N,
        eps,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=2,
    )
    return (y, rsigma) if return_rsigma else y
