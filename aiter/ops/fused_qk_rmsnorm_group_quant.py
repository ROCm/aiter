# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional

from torch import Tensor

from ..jit.core import compile_ops
from ..utility import dtypes


@compile_ops(
    "module_fused_qk_rmsnorm_group_quant", fc_name="fused_qk_rmsnorm_group_quant"
)
def _fused_qk_rmsnorm_group_quant_hip_kernel(
    q_out_quantized: Tensor,
    q_out_scale: Tensor,
    q: Tensor,
    q_weight: Tensor,
    q_epsilon: float,
    q_out_unquantized: Optional[Tensor] = None,
    k_out: Optional[Tensor] = None,
    q_res_out: Optional[Tensor] = None,
    k: Optional[Tensor] = None,
    k_weight: Optional[Tensor] = None,
    k_epsilon: Optional[float] = None,
    q_residual: Optional[Tensor] = None,
    group_size: int = 128,
    transpose_scale: bool = False,
) -> None: ...


def fused_qk_rmsnorm_group_quant_hip(
    q_out_quantized: Tensor,
    q_out_scale: Tensor,
    q: Tensor,
    q_weight: Tensor,
    q_epsilon: float,
    q_out_unquantized: Optional[Tensor] = None,
    k_out: Optional[Tensor] = None,
    q_res_out: Optional[Tensor] = None,
    k: Optional[Tensor] = None,
    k_weight: Optional[Tensor] = None,
    k_epsilon: Optional[float] = None,
    q_residual: Optional[Tensor] = None,
    group_size: int = 128,
    transpose_scale: bool = False,
) -> None:
    if q_out_quantized.dtype not in (dtypes.fp8, dtypes.fp4x2):
        raise ValueError(
            "fused_qk_rmsnorm_group_quant_hip currently supports fp8/fp4x2 output quant only; "
            f"got {q_out_quantized.dtype}"
        )
    if q_out_quantized.dtype == dtypes.fp4x2 and transpose_scale:
        raise ValueError(
            "fused_qk_rmsnorm_group_quant_hip fp4x2 currently does not support transpose_scale=True"
        )

    _fused_qk_rmsnorm_group_quant_hip_kernel(
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
    )
