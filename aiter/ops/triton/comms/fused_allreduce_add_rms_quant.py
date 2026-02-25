# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fused AllReduce + RMSNorm + Quantization for ROCm.

This module provides fused operations that combine:
1. All-reduce across tensor parallel GPUs
2. Optional residual addition
3. RMS normalization
4. FP8 per-tensor quantization

The fusion reduces memory bandwidth by avoiding intermediate writes.

Implementations:
1. "torch" - Pure torch reference implementation (see torch_allreduce.py)
2. "iris_oneshot" (default) - Iris fused single-kernel allreduce+rmsnorm+quant (see iris_oneshot_allreduce.py)
3. "iris_twoshot" - Iris two-shot reduce+broadcast allreduce+rmsnorm+quant (see iris_twoshot_allreduce.py)
"""

import logging
import os
from typing import Optional, Tuple

import torch

__all__ = ["fused_allreduce_add_rms_quant"]

logger = logging.getLogger(__name__)

ALLREDUCE_IMPL = os.environ.get("VLLM_ROCM_FUSED_ALLREDUCE", "iris_twoshot")


def fused_allreduce_add_rms_quant(
    input: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    quant_scale: torch.Tensor,
    quant_dtype: torch.dtype,
    group_name: str,
    residual: Optional[torch.Tensor] = None,
    impl: str = ALLREDUCE_IMPL,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    """Fused AllReduce + (optional) Add + RMSNorm + FP8 Per-Tensor Quant.

    Args:
        input: Input tensor to all-reduce
        rms_weight: RMSNorm weight
        rms_eps: RMSNorm epsilon
        quant_scale: Quantization scale (can be None for dynamic)
        quant_dtype: Target quantization dtype (e.g., torch.float8_e4m3fn)
        group_name: TP group name for all-reduce
        residual: Optional residual tensor for fused add
        impl: Implementation to use - "torch" (pure torch reference),
              "iris_oneshot" (Iris fused single-kernel, default),
              or "iris_twoshot" (Iris two-shot reduce+broadcast)

    Returns: (allreduce_out, rms_out, residual_out, quant_out, quant_scale_out)
             residual_out is None if residual is None
    """
    if impl == "iris_oneshot":
        from .iris_oneshot_allreduce import (
            fused_allreduce_add_rms_quant_iris_oneshot,
        )

        return fused_allreduce_add_rms_quant_iris_oneshot(
            input, rms_weight, rms_eps, quant_scale, quant_dtype, group_name,
            residual,
        )
    elif impl == "iris_twoshot":
        from .iris_twoshot_allreduce import (
            fused_allreduce_add_rms_quant_iris_twoshot,
        )

        return fused_allreduce_add_rms_quant_iris_twoshot(
            input, rms_weight, rms_eps, quant_scale, quant_dtype, group_name,
            residual,
        )
    elif impl == "torch":
        from .torch_allreduce import (
            fused_allreduce_add_rms_quant_torch,
        )

        return fused_allreduce_add_rms_quant_torch(
            input, rms_weight, rms_eps, quant_scale, quant_dtype,
            residual=residual,
        )
    else:
        raise ValueError(
            f"Unknown impl '{impl}', expected 'torch', 'iris_oneshot',"
            f" or 'iris_twoshot'"
        )
