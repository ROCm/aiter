# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional

from torch import Tensor

from ..jit.core import compile_ops
from ..utility import dtypes


@compile_ops(
    "module_fused_qk_rmsnorm_group_quant", fc_name="fused_qk_rmsnorm_group_quant"
)
def _fused_qk_rmsnorm_group_quant_kernel(
    q_out_quantized: Optional[Tensor] = None,
    q_out_scale: Optional[Tensor] = None,
    q: Optional[Tensor] = None,
    q_weight: Optional[Tensor] = None,
    q_epsilon: float = 1e-6,
    q_out_unquantized: Optional[Tensor] = None,
    k_out: Optional[Tensor] = None,
    q_res_out: Optional[Tensor] = None,
    k: Optional[Tensor] = None,
    k_weight: Optional[Tensor] = None,
    k_epsilon: Optional[float] = None,
    q_residual: Optional[Tensor] = None,
    group_size: int = 128,
    transpose_scale: bool = False,
    gemma_norm: bool = False,
) -> None: ...


def fused_qk_rmsnorm_group_quant(
    q_out_quantized: Optional[Tensor] = None,
    q_out_scale: Optional[Tensor] = None,
    q: Optional[Tensor] = None,
    q_weight: Optional[Tensor] = None,
    q_epsilon: float = 1e-6,
    q_out_unquantized: Optional[Tensor] = None,
    k_out: Optional[Tensor] = None,
    q_res_out: Optional[Tensor] = None,
    k: Optional[Tensor] = None,
    k_weight: Optional[Tensor] = None,
    k_epsilon: Optional[float] = None,
    q_residual: Optional[Tensor] = None,
    group_size: int = 128,
    transpose_scale: bool = False,
    gemma_norm: bool = False,
) -> None:
    # No-quant mode: when q_out_scale is None we only do RMSNorm and write to q_out_unquantized.
    no_quant = q_out_scale is None
    if no_quant:
        if q_out_unquantized is None:
            raise ValueError(
                "fused_qk_rmsnorm_group_quant: q_out_unquantized must be provided "
                "when q_out_scale is None (no-quant mode)"
            )
    else:
        if q_out_quantized is None:
            raise ValueError(
                "fused_qk_rmsnorm_group_quant: q_out_quantized must be provided "
                "when q_out_scale is provided (quant mode)"
            )
        if q_out_quantized.dtype not in (dtypes.fp8, dtypes.fp4x2):
            raise ValueError(
                "fused_qk_rmsnorm_group_quant currently supports fp8/fp4x2 output quant only; "
                f"got {q_out_quantized.dtype}"
            )
        if q_out_quantized.dtype == dtypes.fp4x2:
            if transpose_scale:
                raise ValueError(
                    "fused_qk_rmsnorm_group_quant fp4x2 currently does not support transpose_scale=True"
                )
            n1 = q.size(1)
            if n1 % 2 != 0:
                raise ValueError(
                    f"q.size(1) must be even for fp4x2 packed output, got {n1}"
                )
            expected_packed = n1 // 2
            if q_out_quantized.size(1) != expected_packed:
                raise ValueError(
                    f"fp4x2 q_out_quantized.size(1) should be {expected_packed} "
                    f"(n1//2 packed), got {q_out_quantized.size(1)}"
                )

    _fused_qk_rmsnorm_group_quant_kernel(
        q_out_quantized,
        q_out_scale,
        q,
        q_weight,
        q_epsilon,
        q_out_unquantized,
        k_out,
        q_res_out,
        k,
        k_weight,
        k_epsilon,
        q_residual,
        group_size,
        transpose_scale,
        gemma_norm,
    )
