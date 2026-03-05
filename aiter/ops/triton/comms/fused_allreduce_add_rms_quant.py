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
4. "iris_twoshot_row" - Same as iris_twoshot but with per-row FP8 quant + inlined Triton GEMM (see iris_twoshot_row_allreduce.py)
5. "iris_twoshot_delayed" - Iris two-shot with delayed scaling + FP8 broadcast (see iris_twoshot_delayed_allreduce.py)
6. "iris_twoshot_row_hipblaslt" - Same comm kernel as iris_twoshot_row but GEMM via torch._scaled_mm/hipBLASLt (see iris_twoshot_row_hipblaslt_allreduce.py)
7. "iris_twoshot_2d_hipblaslt" - 2D-tiled variant mimicking CCL two-shot structure, per-row FP8 quant + hipBLASLt GEMM (see iris_twoshot_2d_hipblaslt_allreduce.py)
8. "iris_partial_gemm" - Partial GEMM + allgather. No FP8 broadcast, no internal cross-rank barrier. Each rank GEMMs its own rows, then allgather assembles the full output via iris.load (see iris_partial_gemm_allreduce.py)

Default: "iris_twoshot_2d_hipblaslt" -- 2D-tiled variant, per-row FP8 quant with FP8 broadcast
(halved cross-rank traffic), vendor-tuned hipBLASLt GEMM.
"""

import atexit
import logging
import os
from collections import Counter
from typing import Optional

import torch

__all__ = ["fused_allreduce_add_rms_quant_gemm"]

logger = logging.getLogger(__name__)

ALLREDUCE_IMPL = os.environ.get(
    "VLLM_ROCM_FUSED_ALLREDUCE", "iris_twoshot_2d_hipblaslt"
)

# Track M values seen during execution for profiling/debugging.
_m_value_counts: Counter[int] = Counter()
def _log_m_summary() -> None:
    if not _m_value_counts:
        return
    total = sum(_m_value_counts.values())
    sorted_m = sorted(_m_value_counts.items())
    dist = ", ".join(f"M={m}: {c}" for m, c in sorted_m)
    logger.info(
        f"Fused allreduce M distribution ({total} calls): {dist}"
    )


atexit.register(_log_m_summary)


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
    _m_value_counts[input.shape[0]] += 1

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

    elif impl == "iris_twoshot_row_hipblaslt":
        from .iris_twoshot_row_hipblaslt_allreduce import (
            fused_allreduce_add_rms_row_quant_gemm_iris_twoshot_hipblaslt,
        )

        return fused_allreduce_add_rms_row_quant_gemm_iris_twoshot_hipblaslt(
            *args
        )

    elif impl == "iris_twoshot_2d_hipblaslt":
        from .iris_twoshot_2d_hipblaslt_allreduce import (
            fused_allreduce_add_rms_row_quant_gemm_iris_twoshot_2d_hipblaslt,
        )

        return fused_allreduce_add_rms_row_quant_gemm_iris_twoshot_2d_hipblaslt(
            *args
        )

    elif impl == "iris_twoshot_delayed":
        from .iris_twoshot_delayed_allreduce import (
            fused_allreduce_add_rms_delayed_quant_gemm_iris_twoshot,
        )

        return fused_allreduce_add_rms_delayed_quant_gemm_iris_twoshot(*args)

    elif impl == "iris_partial_gemm":
        from .iris_partial_gemm_allreduce import (
            fused_allreduce_add_rms_row_quant_gemm_iris_partial,
        )

        return fused_allreduce_add_rms_row_quant_gemm_iris_partial(*args)

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
            f" 'iris_twoshot', 'iris_twoshot_row',"
            f" 'iris_twoshot_row_hipblaslt', 'iris_twoshot_2d_hipblaslt',"
            f" 'iris_twoshot_delayed', or 'iris_partial_gemm'"
        )
