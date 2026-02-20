# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Pure torch reference implementation of fused AllReduce + RMSNorm + Quant.

Uses only standard PyTorch ops (no vLLM or AITER dependencies).
Portable to aiter as-is.
"""

from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

__all__ = ["fused_allreduce_add_rms_quant_torch"]


def fused_allreduce_add_rms_quant_torch(
    input: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    quant_scale: torch.Tensor,
    quant_dtype: torch.dtype,
    group: Optional[ProcessGroup] = None,
    residual: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    """Pure torch reference: AllReduce + (optional) Add + RMSNorm + FP8 Quant.

    Args:
        input: Input tensor to all-reduce
        rms_weight: RMSNorm weight
        rms_eps: RMSNorm epsilon
        quant_scale: Quantization scale (unused for dynamic quant, kept for API)
        quant_dtype: Target quantization dtype (e.g., torch.float8_e4m3fn)
        group: Process group for all-reduce (defaults to WORLD)
        residual: Optional residual tensor for fused add

    Returns: (allreduce_out, rms_out, residual_out, quant_out, quant_scale_out)
             residual_out is None if residual is None
    """
    # Step 1: All-reduce using standard torch.distributed
    allreduce_out = input.clone()
    dist.all_reduce(allreduce_out, group=group)

    # Step 2: Optional residual add + RMSNorm (compute in float32)
    if residual is not None:
        residual_out = allreduce_out + residual
        rms_input = residual_out
    else:
        residual_out = None
        rms_input = allreduce_out

    # RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight
    rms_input_f32 = rms_input.to(torch.float32)
    variance = rms_input_f32.pow(2).mean(dim=-1, keepdim=True)
    rms_input_normed = rms_input_f32 * torch.rsqrt(variance + rms_eps)
    rms_out = (rms_input_normed * rms_weight.to(torch.float32)).to(
        input.dtype
    )

    # Step 3: Per-tensor FP8 quantization
    rms_out_f32 = rms_out.to(torch.float32)
    amax = rms_out_f32.abs().max().clamp(min=1e-12)
    fp8_max = torch.finfo(quant_dtype).max
    quant_scale_out = (amax / fp8_max).to(torch.float32).reshape(1)
    quant_out = (rms_out_f32 / quant_scale_out).clamp(
        -fp8_max, fp8_max
    ).to(quant_dtype)

    return allreduce_out, rms_out, residual_out, quant_out, quant_scale_out
