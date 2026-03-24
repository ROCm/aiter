# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from torch import Tensor
from ..jit.core import compile_ops
from typing import Optional


@compile_ops("module_fused_qk_norm_rope_cache_quant_shuffle")
def fused_qk_norm_rope_cache_quant_shuffle(
    qkv: Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    qw: Tensor,
    kw: Tensor,
    cos_sin_cache: Tensor,
    is_neox_style: bool,
    pos_ids: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    slot_mapping: Tensor,
    kv_cache_dtype: str,
    k_scale: Tensor,
    v_scale: Tensor,
) -> None: ...


@compile_ops("module_fused_qk_norm_rope_cache_quant_shuffle", fc_name="fused_qk_rmsnorm")
def _fused_qk_rmsnorm_kernel(
    q: Tensor,
    q_weight: Tensor,
    q_eps: float,
    k: Tensor,
    k_weight: Tensor,
    k_eps: float,
) -> tuple[Tensor, Tensor]: ...


_FUSED_QK_FALLBACK_M = 16384


def fused_qk_rmsnorm(
    q: Tensor,
    q_weight: Tensor,
    q_eps: float,
    k: Tensor,
    k_weight: Tensor,
    k_eps: float,
) -> tuple[Tensor, Tensor]:
    m = q.size(0)
    q_n = q.size(1)
    k_n = k.size(1)
    if q_n != k_n and m >= _FUSED_QK_FALLBACK_M:
        from .rmsnorm import rmsnorm2d_fwd
        q_out = rmsnorm2d_fwd(q, q_weight, q_eps)
        k_out = rmsnorm2d_fwd(k, k_weight, k_eps)
        return q_out, k_out
    return _fused_qk_rmsnorm_kernel(q, q_weight, q_eps, k, k_weight, k_eps)


@compile_ops("module_fused_qk_norm_rope_cache_quant_shuffle")
def fused_qk_norm_rope_cache_block_quant_shuffle(
    qkv: Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    qw: Tensor,
    kw: Tensor,
    cos_sin_cache: Tensor,
    is_neox_style: bool,
    pos_ids: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    slot_mapping: Tensor,
    cu_q_len: Tensor,
    kv_cache_dtype: str,
    k_scale: Tensor,
    v_scale: Tensor,
    max_tokens_per_batch: int = 0,
) -> None: ...


@compile_ops("module_fused_qk_norm_rope_cache_quant_shuffle")
def fused_qk_norm_rope_cache_pts_quant_shuffle(
    qkv: Tensor,
    qw: Tensor,
    kw: Tensor,
    cos_sin: Tensor,
    positions: Tensor,
    num_tokens: int,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_size: int,
    is_neox_style: bool,
    eps: float,
    q_out: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    slot_mapping: Tensor,
    per_tensor_k_scale: Tensor,
    per_tensor_v_scale: Tensor,
    k_out: Optional[Tensor],
    v_out: Optional[Tensor],
    return_kv: bool,
    use_shuffle_layout: bool,
    block_size: int,
    x: int,
    rotary_dim: int = 0,
) -> None: ...


@compile_ops("module_fused_qk_norm_rope_cache_quant_shuffle")
def fused_qk_norm_rope_2way(
    q0: Tensor,
    k0: Tensor,
    q1: Tensor,
    k1: Tensor,
    w_q0: Tensor,
    w_k0: Tensor,
    w_q1: Tensor,
    w_k1: Tensor,
    cos_sin0: Tensor,
    cos_sin1: Tensor,
    batch_size: int,
    num_tokens0: int,
    num_tokens1: int,
    num_heads_q: int,
    num_heads_k: int,
    head_size: int,
    is_interleaved: bool,
    eps: float,
    out_q01: Tensor,
    out_k01: Tensor,
) -> None: ...
