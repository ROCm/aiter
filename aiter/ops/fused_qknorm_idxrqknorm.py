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
    kv_cache_k: Optional[Tensor],
    kv_cache_v: Optional[Tensor],
    index_cache: Optional[Tensor],
    block_size: int,
    q_out: Optional[Tensor],
    index_q_out: Optional[Tensor],
    index_slot_mapping: Optional[Tensor],
    kv_cache_dtype: str = "auto",
    index_cache_dtype: str = "auto",
    k_scale: Optional[Tensor] = None,
    v_scale: Optional[Tensor] = None,
    asm_layout: bool = False,
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
    kv_cache_k: Optional[Tensor] = None,
    kv_cache_v: Optional[Tensor] = None,
    index_cache: Optional[Tensor] = None,
    block_size: int = 0,
    q_out: Optional[Tensor] = None,
    index_q_out: Optional[Tensor] = None,
    index_slot_mapping: Optional[Tensor] = None,
    kv_cache_dtype: str = "auto",
    index_cache_dtype: Optional[str] = None,
    k_scale: Optional[Tensor] = None,
    v_scale: Optional[Tensor] = None,
    asm_layout: bool = False,
) -> None:
    # The main K/V caches are always passed as separate kv_cache_k / kv_cache_v
    # tensors. asm_layout selects the in-cache addressing: page-16 SHUFFLE
    # (asm_layout=True) vs plain page-128 (asm_layout=False, where kv_cache_k /
    # kv_cache_v are typically the key/value slices of a fused
    # [num_blocks, 2, block_size, num_kv_heads, head_dim] cache).
    if index_cache_dtype is None:
        index_cache_dtype = (
            "fp8"
            if isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("fp8")
            else "auto"
        )

    use_fp8_kv_cache = (
        kv_cache_k is not None
        and isinstance(kv_cache_dtype, str)
        and kv_cache_dtype.startswith("fp8")
    )
    if use_fp8_kv_cache:
        if index_slot_mapping is None:
            index_slot_mapping = slot_mapping
        assert index_q_norm_weight is not None
        assert index_k_norm_weight is not None
        assert slot_mapping is not None
        assert kv_cache_v is not None
        assert index_cache is not None
        assert q_out is not None
        assert index_q_out is not None
        assert index_slot_mapping is not None
        assert k_scale is not None
        assert v_scale is not None

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
        kv_cache_k,
        kv_cache_v,
        index_cache,
        block_size,
        q_out,
        index_q_out,
        index_slot_mapping,
        kv_cache_dtype,
        index_cache_dtype,
        k_scale,
        v_scale,
        asm_layout,
    )
