# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Reference implementation: NCCL AllReduce + RMSNorm + FP8 Quant + hipBLASLt GEMM.

Uses standard PyTorch ops (torch.distributed for allreduce,
torch._scaled_mm for GEMM). No iris or custom kernels.
Useful for correctness testing against fused iris variants.
"""

from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

__all__ = ["fused_allreduce_add_rms_quant_gemm_torch"]


def fused_allreduce_add_rms_quant_gemm_torch(
    input: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    quant_dtype: torch.dtype,
    group_name: str,
    gemm_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
    group: Optional[ProcessGroup] = None,
    residual: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """NCCL AllReduce + Add + RMSNorm + FP8 Quant + hipBLASLt GEMM.

    All FP8 is internal -- takes BF16 in, produces BF16 GEMM output.

    Returns (gemm_out, residual_out). residual_out is None when residual
    is None.
    """
    # Step 1: All-reduce using NCCL via torch.distributed
    allreduce_out = input.clone()
    dist.all_reduce(allreduce_out, group=group)

    # Step 2: Optional residual add + RMSNorm
    if residual is not None:
        residual_out = allreduce_out + residual
        rms_input = residual_out
    else:
        residual_out = None
        rms_input = allreduce_out

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

    # Step 4: hipBLASLt GEMM via torch._scaled_mm
    gemm_out = torch._scaled_mm(
        quant_out, gemm_weight, out_dtype=out_dtype,
        scale_a=quant_scale_out, scale_b=weight_scale, bias=bias,
    )

    return gemm_out, residual_out
