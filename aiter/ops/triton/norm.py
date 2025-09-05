# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
from typing import Optional
from aiter.ops.triton._triton_kernels.norm import (
    _per_token_quant,
    _layernorm_kernel,
    _fused_add_layernorm_kernel,
    _quant_layernorm_kernel,
    _quant_fused_add_layernorm_kernel,
    _layernorm_bwd_dx_fused_triton,
    _layernorm_bwd_dwdb_triton,
    _layernorm_bwd_dwdb_triton_v2,
)
from aiter.ops.triton.utils.types import get_dtype_max
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


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
    _LOGGER.info(f"LAYERNORM: input={tuple(input.shape)} weight={tuple(weight.shape)} ")
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
    _LOGGER.info(
        f"LAYERNORM_2D_FWD_ADD: input={tuple(input.shape)} weight={tuple(weight.shape)} residual_in={tuple(residual_in.shape)}  "
    )
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
    _LOGGER.info(
        f"LAYERNORM_2D_FWD_DYNAMICQUANT: input={tuple(input.shape)} weight={tuple(weight.shape)} yscale={tuple(yscale.shape)}  "
    )
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
    _LOGGER.info(
        f"RMSNORM_2D_FWD_SMOOTHQUANT: input={tuple(input.shape)} weight={tuple(weight.shape)} xscale={tuple(xscale.shape)} yscale={tuple(yscale.shape)}  "
    )
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
    _LOGGER.info(
        f"LAYERNORM_2D_FWD_ADD_DYNAMICQUANT: input={input.shape} weight={weight.shape} residual_in={residual_in.shape} yscale={yscale.shape}  "
    )
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

    _LOGGER.info(
        f"LAYERNORM_2D_FWD_ADD_SMOOTHQUANT: input={tuple(input.shape)} weight={tuple(weight.shape)} "
        + f"residual_in={tuple(residual_in.shape)} xscale={tuple(xscale.shape)} yscale={tuple(yscale.shape)}  "
    )
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
