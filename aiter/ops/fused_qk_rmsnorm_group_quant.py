# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional

from torch import Tensor
from dataclasses import dataclass

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


@compile_ops(
    "module_fused_qk_rmsnorm_group_quant", fc_name="fused_qk_rmsnorm_per_token_quant"
)
def _fused_qk_rmsnorm_per_token_quant_kernel(
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
        if group_size <= 0:
            raise ValueError(
                "fused_qk_rmsnorm_group_quant requires group_size > 0; "
                "use fused_qk_rmsnorm_per_token_quant for per-token quant"
            )
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


def fused_qk_rmsnorm_per_token_quant(
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
    gemma_norm: bool = False,
) -> None:
    if q_out_quantized.dtype != dtypes.fp8:
        raise ValueError(
            "fused_qk_rmsnorm_per_token_quant currently supports fp8 output quant only; "
            f"got {q_out_quantized.dtype}"
        )
    if q_out_scale.dim() != 2 or q_out_scale.size(1) != 1:
        raise ValueError(
            "fused_qk_rmsnorm_per_token_quant expects q_out_scale with shape [m, 1]"
        )

    _fused_qk_rmsnorm_per_token_quant_kernel(
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
        gemma_norm,
    )


@dataclass
class QKRMSQuantArgs:
    group_quant: int = False
    per_token_quant: int = False
    group_size: int = 128
    transpose_scale: bool = False


def fused_qk_rmsnorm_quant(
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
    gemma_norm: bool = False,
    quant_args: Optional[QKRMSQuantArgs] = None,
) -> None:
    # Centralized interface
    if quant_args.group_quant:
        fused_qk_rmsnorm_group_quant(
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
            quant_args.group_size,
            quant_args.transpose_scale,
            gemma_norm,
        )
    elif quant_args.per_token_quant:
        fused_qk_rmsnorm_per_token_quant(
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
            gemma_norm,
        )
    else:
        raise RuntimeError("No quant args selected")
