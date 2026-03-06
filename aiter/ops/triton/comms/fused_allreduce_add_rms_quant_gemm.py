# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fused AllReduce + RMSNorm + FP8 Quant + Scaled GEMM for ROCm.

This module provides the fused operation that combines:
1. All-reduce across tensor parallel GPUs
2. Optional residual addition
3. RMS normalization
4. FP8 per-row quantization
5. Scaled FP8 GEMM

All FP8 is internal to the fusion -- the op takes BF16 in and produces
BF16 GEMM output + updated residual.

Implementations:
1. "triton" - Iris AllReduce + inlined Triton GEMM. Single kernel launch.
2. "hipblaslt" - Iris AllReduce + external hipBLASLt GEMM. Two kernel launches.
3. "torch" - NCCL AllReduce + torch ops. Reference implementation.

Default: "triton"
"""

import logging
import os
from typing import Optional

import torch

__all__ = ["fused_allreduce_add_rms_quant_gemm"]

logger = logging.getLogger(__name__)

ALLREDUCE_IMPL = os.environ.get(
    "VLLM_ROCM_FUSED_ALLREDUCE", "triton"
)


def fused_allreduce_add_rms_quant_gemm(
    input: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    quant_dtype: torch.dtype,
    group_name: str,
    gemm_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
    residual: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Fused AllReduce + RMSNorm + FP8 Quant + Scaled GEMM.

    Returns (gemm_out, residual_out). residual_out is None when residual
    is None.
    """
    impl = ALLREDUCE_IMPL
    args = (
        input, rms_weight, rms_eps, quant_dtype, group_name,
        gemm_weight, weight_scale, out_dtype, residual, bias,
    )

    if impl == "triton":
        from .fused_allreduce_add_rms_quant_gemm_triton import (
            fused_allreduce_add_rms_quant_gemm_triton,
        )

        return fused_allreduce_add_rms_quant_gemm_triton(*args)

    elif impl == "hipblaslt":
        from .fused_allreduce_add_rms_quant_gemm_hipblaslt import (
            fused_allreduce_add_rms_quant_gemm_hipblaslt,
        )

        return fused_allreduce_add_rms_quant_gemm_hipblaslt(*args)

    elif impl == "torch":
        from .fused_allreduce_add_rms_quant_gemm_torch import (
            fused_allreduce_add_rms_quant_gemm_torch,
        )

        return fused_allreduce_add_rms_quant_gemm_torch(*args)

    else:
        raise ValueError(
            f"Unknown impl '{impl}', expected"
            f" 'triton', 'hipblaslt', or 'torch'"
        )
