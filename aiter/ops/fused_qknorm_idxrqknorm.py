# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional

from torch import Tensor

from ..jit.core import compile_ops


@compile_ops(
    "module_fused_qknorm_idxrqknorm",
    fc_name="fused_qknorm_idxrqknorm",
    develop=True,
)
def _fused_qknorm_idxrqknorm_hip(
    qkv: Tensor,
    q_norm_weight: Tensor,
    k_norm_weight: Tensor,
    cos_sin_cache: Tensor,
    positions: Tensor,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    index_q_norm_weight: Optional[Tensor],
    index_k_norm_weight: Optional[Tensor],
    num_index_heads: int,
    slot_mapping: Optional[Tensor],
    kv_cache: Optional[Tensor],
    index_cache: Optional[Tensor],
    block_size: int,
    q_out: Optional[Tensor],
    index_q_out: Optional[Tensor],
    index_slot_mapping: Optional[Tensor],
) -> None:
    pass


@compile_ops(
    "module_fused_qknorm_idxrqknorm",
    fc_name="fused_qknorm_idxrqknorm_fp8",
    develop=True,
)
def _fused_qknorm_idxrqknorm_fp8_hip(
    qkv: Tensor,
    q_norm_weight: Tensor,
    k_norm_weight: Tensor,
    cos_sin_cache: Tensor,
    positions: Tensor,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    index_q_norm_weight: Tensor,
    index_k_norm_weight: Tensor,
    num_index_heads: int,
    slot_mapping: Tensor,
    kv_cache: Tensor,
    index_cache: Tensor,
    block_size: int,
    q_out: Tensor,
    index_q_out: Tensor,
    index_slot_mapping: Tensor,
    kv_cache_dtype: str,
    k_scale: Tensor,
    v_scale: Tensor,
) -> None:
    pass


def fused_qknorm_idxrqknorm(
    qkv: Tensor,
    q_norm_weight: Tensor,
    k_norm_weight: Tensor,
    cos_sin_cache: Tensor,
    positions: Tensor,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    index_q_norm_weight: Optional[Tensor] = None,
    index_k_norm_weight: Optional[Tensor] = None,
    num_index_heads: int = 0,
    slot_mapping: Optional[Tensor] = None,
    kv_cache: Optional[Tensor] = None,
    index_cache: Optional[Tensor] = None,
    block_size: int = 0,
    q_out: Optional[Tensor] = None,
    index_q_out: Optional[Tensor] = None,
    index_slot_mapping: Optional[Tensor] = None,
    kv_cache_dtype: str = "auto",
    k_scale: Optional[Tensor] = None,
    v_scale: Optional[Tensor] = None,
) -> None:
    if (
        kv_cache is not None
        and isinstance(kv_cache_dtype, str)
        and kv_cache_dtype.startswith("fp8")
    ):
        if index_slot_mapping is None:
            index_slot_mapping = slot_mapping
        assert index_q_norm_weight is not None
        assert index_k_norm_weight is not None
        assert slot_mapping is not None
        assert index_cache is not None
        assert q_out is not None
        assert index_q_out is not None
        assert index_slot_mapping is not None
        assert k_scale is not None
        assert v_scale is not None
        _fused_qknorm_idxrqknorm_fp8_hip(
            qkv,
            q_norm_weight,
            k_norm_weight,
            cos_sin_cache,
            positions,
            num_heads,
            num_kv_heads,
            rotary_dim,
            eps,
            index_q_norm_weight,
            index_k_norm_weight,
            num_index_heads,
            slot_mapping,
            kv_cache,
            index_cache,
            block_size,
            q_out,
            index_q_out,
            index_slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
        return

    _fused_qknorm_idxrqknorm_hip(
        qkv,
        q_norm_weight,
        k_norm_weight,
        cos_sin_cache,
        positions,
        num_heads,
        num_kv_heads,
        rotary_dim,
        eps,
        index_q_norm_weight,
        index_k_norm_weight,
        num_index_heads,
        slot_mapping,
        kv_cache,
        index_cache,
        block_size,
        q_out,
        index_q_out,
        index_slot_mapping,
    )
