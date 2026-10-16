# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.quant.fused_fp8_quant import (
    _fused_reduce_rms_fp8_group_quant_kernel,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.normalization_config_utils import get_normalization_config

_LOGGER = AiterTritonLogger()


def _num_warps(block_size_n: int) -> int:
    raw = get_normalization_config("fused_reduce_add_rmsnorm", get_arch())
    for bound in sorted(int(k[len("N_LEQ_") :]) for k in raw if k.startswith("N_LEQ_")):
        if block_size_n <= bound:
            return raw[f"N_LEQ_{bound}"]["num_warps"]
    return raw["any"]["num_warps"]


def fused_reduce_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    gemma_norm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split-K reduce + residual add + RMSNorm in one kernel (the no-quant form of
    fused_reduce_rms_fp8_group_quant).

    Parameters:
        x (torch.Tensor): [num_ksplit, M, N] fp32 split-K partials, or a [M, N] GEMM output.
        residual (torch.Tensor): [M, N] residual, fp16 or bf16.
        weight (torch.Tensor): [N] norm weight, applied as (1 + weight) when gemma_norm.
        epsilon (float): RMSNorm epsilon.
        gemma_norm (bool): GemmaRMSNorm weight.

    Returns:
        out (torch.Tensor): [M, N] RMSNorm(sum(x) + residual), in residual.dtype.
        residual_out (torch.Tensor): [M, N] sum(x) + residual, in residual.dtype.
    """
    if x.dim() == 3 and x.shape[0] == 1:
        x = x[0]
    if x.dim() == 3:
        num_ksplit, M, N = x.shape
        x_spk_stride, x_row_stride, x_col_stride = x.stride()
    elif x.dim() == 2:
        M, N = x.shape
        num_ksplit, x_spk_stride = 1, 0
        x_row_stride, x_col_stride = x.stride()
    else:
        raise ValueError(f"x must be 2D or 3D, got shape {tuple(x.shape)}")
    assert residual.shape == (
        M,
        N,
    ), f"residual {tuple(residual.shape)} vs ({M}, {N})"
    assert weight.shape == (N,), f"weight {tuple(weight.shape)} vs ({N},)"
    assert residual.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"residual must be fp16 or bf16, got {residual.dtype}"

    _LOGGER.info(
        "FUSED_REDUCE_ADD_RMSNORM: x=%s residual=%s gemma_norm=%s",
        tuple(x.shape),
        tuple(residual.shape),
        gemma_norm,
    )

    out = torch.empty((M, N), dtype=residual.dtype, device=residual.device)
    residual_out = torch.empty((M, N), dtype=residual.dtype, device=residual.device)
    if M == 0:
        return out, residual_out

    BLOCK_SIZE_N = triton.next_power_of_2(N)
    _fused_reduce_rms_fp8_group_quant_kernel[(M,)](
        x,
        weight,
        None,
        None,
        None,
        residual,
        None,
        None,
        None,
        residual_out,
        out,
        None,
        epsilon,
        0.0,
        M,
        N,
        0,
        0,
        x_spk_stride,
        0,
        0,
        x_row_stride,
        0,
        0,
        x_col_stride,
        0,
        0,
        residual.stride(0),
        residual.stride(1),
        0,
        0,
        0,
        0,
        0,
        0,
        residual_out.stride(0),
        residual_out.stride(1),
        out.stride(0),
        out.stride(1),
        0,
        0,
        BLOCK_SIZE_N1=BLOCK_SIZE_N,
        BLOCK_SIZE_N2=1,
        BLOCK_SIZE_N3=1,
        N_MASK1=(BLOCK_SIZE_N != N),
        N_MASK2=False,
        N_MASK3=False,
        QUANT_BLOCK_SIZE=BLOCK_SIZE_N,
        DTYPE_MAX=0.0,
        DTYPE_MIN=0.0,
        HAVE_SECOND_INPUT=False,
        FIRST_INPUT_RES=True,
        FIRST_INPUT_OUT=True,
        HAS_SPLITK=num_ksplit > 1,
        NUM_SPLITK=num_ksplit,
        NUM_SPLITK_POW2=triton.next_power_of_2(num_ksplit),
        HAS_QUANT=False,
        GEMMA_NORM=gemma_norm,
        HAVE_THIRD_INPUT=False,
        num_warps=_num_warps(BLOCK_SIZE_N),
    )
    return out, residual_out
