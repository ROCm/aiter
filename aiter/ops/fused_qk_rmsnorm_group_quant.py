# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional

from torch import Tensor

from ..jit.core import compile_ops
from ..utility import dtypes


@compile_ops(
    "module_fused_qk_rmsnorm_group_quant", fc_name="fused_qk_rmsnorm_bf16"
)
def _fused_qk_rmsnorm_bf16_kernel(
    q_out: Tensor,
    k_out: Tensor,
    q: Tensor,
    q_weight: Tensor,
    q_epsilon: float,
    k: Tensor,
    k_weight: Tensor,
    k_epsilon: float,
) -> None: ...


def fused_qk_rmsnorm_bf16(
    q: Tensor,
    q_weight: Tensor,
    q_epsilon: float,
    k: Tensor,
    k_weight: Tensor,
    k_epsilon: float,
) -> tuple[Tensor, Tensor]:
    """Fused RMSNorm for Q and K latent projections in DeepSeek MLA.

    Replaces two sequential add_rmsnorm_quant_kernel launches with a single
    fused launch using blockIdx.y to process Q and K in parallel (small T)
    or in the same block sequentially (large T).

    Args:
        q:         [T, q_lora_rank] BF16/FP16 — Q latent after lora projection
        q_weight:  [q_lora_rank]    BF16/FP16 — q_a_layernorm weight
        q_epsilon: RMSNorm epsilon for Q
        k:         [T, kv_lora_rank] BF16/FP16 — KV latent (k_nope slice)
        k_weight:  [kv_lora_rank]    BF16/FP16 — kv_a_layernorm weight
        k_epsilon: RMSNorm epsilon for K

    Returns:
        (q_out, k_out): normalized tensors, same shape and dtype as inputs
    """
    q_out = q.new_empty(q.shape)
    k_out = k.new_empty(k.shape)
    _fused_qk_rmsnorm_bf16_kernel(q_out, k_out, q, q_weight, q_epsilon, k, k_weight, k_epsilon)
    return q_out, k_out


@compile_ops(
    "module_fused_qk_rmsnorm_group_quant", fc_name="fused_qk_rmsnorm_group_quant"
)
def _fused_qk_rmsnorm_group_quant_kernel(
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


def fused_qk_rmsnorm_group_quant(
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
    )
