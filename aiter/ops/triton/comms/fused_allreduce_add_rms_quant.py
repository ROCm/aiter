# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fused AllReduce + RMSNorm + FP8 Quant + Scaled GEMM for ROCm.

This module provides the fused operation that combines:
1. All-reduce across tensor parallel GPUs
2. Optional residual addition
3. RMS normalization
4. FP8 per-tensor quantization (delayed scaling)
5. Scaled FP8 GEMM

All FP8 is internal to the fusion -- the op takes BF16 in and produces
BF16 GEMM output + updated residual.

Implementations:
1. "torch" - Pure torch reference implementation (see torch_allreduce.py)
2. "iris_oneshot" - Iris fused single-kernel allreduce+rmsnorm+quant (see iris_oneshot_allreduce.py)
3. "iris_twoshot" - Iris two-shot reduce+broadcast allreduce+rmsnorm+quant (see iris_twoshot_allreduce.py)
4. "iris_twoshot_row" - Same as iris_twoshot but with per-row FP8 quant (see iris_twoshot_row_allreduce.py)
5. "iris_twoshot_delayed" (default) - Iris two-shot with delayed scaling + FP8 broadcast (see iris_twoshot_delayed_allreduce.py)
"""

import logging
import os
from typing import Optional

import torch

__all__ = ["fused_allreduce_add_rms_quant_gemm"]

logger = logging.getLogger(__name__)

ALLREDUCE_IMPL = os.environ.get(
    "VLLM_ROCM_FUSED_ALLREDUCE", "iris_twoshot_delayed"
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

    Dispatches to the selected implementation. Each implementation
    performs allreduce+rmsnorm+quant+gemm internally. All FP8 is
    internal -- takes BF16 in, produces BF16 GEMM output.

    Returns (gemm_out, residual_out). residual_out is None when residual
    is None.
    """
    impl = ALLREDUCE_IMPL
    args = (
        input, rms_weight, rms_eps, quant_dtype, group_name,
        gemm_weight, weight_scale, out_dtype, residual, bias,
    )

    if impl == "iris_oneshot":
        from .iris_oneshot_allreduce import (
            fused_allreduce_add_rms_quant_gemm_iris_oneshot,
        )

        return fused_allreduce_add_rms_quant_gemm_iris_oneshot(*args)

    elif impl == "iris_twoshot":
        from .iris_twoshot_allreduce import (
            fused_allreduce_add_rms_quant_gemm_iris_twoshot,
        )

        return fused_allreduce_add_rms_quant_gemm_iris_twoshot(*args)

    elif impl == "iris_twoshot_row":
        from .iris_twoshot_row_allreduce import (
            fused_allreduce_add_rms_row_quant_gemm_iris_twoshot,
        )

        return fused_allreduce_add_rms_row_quant_gemm_iris_twoshot(*args)

    elif impl == "iris_twoshot_delayed":
        from .iris_twoshot_delayed_allreduce import (
            fused_allreduce_add_rms_delayed_quant_gemm_iris_twoshot,
        )

        return fused_allreduce_add_rms_delayed_quant_gemm_iris_twoshot(*args)

    elif impl == "torch":
        from .torch_allreduce import (
            fused_allreduce_add_rms_quant_gemm_torch,
        )

        return fused_allreduce_add_rms_quant_gemm_torch(
            input, rms_weight, rms_eps, quant_dtype, group_name,
            gemm_weight, weight_scale, out_dtype,
            residual=residual, bias=bias,
        )

    else:
        raise ValueError(
            f"Unknown impl '{impl}', expected 'torch', 'iris_oneshot',"
            f" 'iris_twoshot', 'iris_twoshot_row', or 'iris_twoshot_delayed'"
        )
