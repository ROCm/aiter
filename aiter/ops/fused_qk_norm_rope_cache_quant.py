# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from ..jit.core import compile_ops
from ..utility.dtypes import get_dtype_fp8
from typing import Optional


@compile_ops(
    "module_fused_qk_norm_rope_cache_quant_shuffle",
    fc_name="fused_qk_norm_rope_cache_quant_shuffle",
    develop=True,
)
def _fused_qk_norm_rope_cache_quant_shuffle_hip(
    q: Tensor,
    k: Tensor,
    v: Tensor,
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
    v_cache: Tensor,  # 4D [B,Hv,D,page] or 5D shuffle [B,Hv,page//x,D,x], x=16//elem_size
    slot_mapping: Tensor,
    kv_cache_dtype: str,
    k_scale: Tensor,
    v_scale: Tensor,
) -> None: ...


def fused_qk_norm_rope_cache_quant_shuffle(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
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
) -> None:
    _fused_qk_norm_rope_cache_quant_shuffle_hip(
        q,
        k,
        v,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        qw,
        kw,
        cos_sin_cache,
        is_neox_style,
        pos_ids,
        k_cache,
        v_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )


@compile_ops(
    "module_fused_qk_norm_rope_cache_quant_shuffle",
    fc_name="fused_qk_rmsnorm",
    develop=True,
)
def _fused_qk_rmsnorm_kernel(
    q: Tensor,
    q_weight: Tensor,
    q_eps: float,
    k: Tensor,
    k_weight: Tensor,
    k_eps: float,
    q_out: Tensor,
    k_out: Tensor,
) -> None: ...


_FUSED_QK_FALLBACK_M = 16384


def _fused_qk_rmsnorm(
    q_out: Optional[Tensor],
    q: Tensor,
    q_weight: Tensor,
    q_eps: float,
    k_out: Optional[Tensor],
    k: Tensor,
    k_weight: Tensor,
    k_eps: float,
) -> tuple[Tensor, Tensor]:
    if q_out is None:
        q_out = torch.empty_like(q, dtype=q.dtype, device=q.device)
    if k_out is None:
        k_out = torch.empty_like(k, dtype=k.dtype, device=k.device)

    m = q.size(0)
    if m >= _FUSED_QK_FALLBACK_M:
        from .rmsnorm import rmsnorm

        rmsnorm(q_out, q, q_weight, q_eps)
        rmsnorm(k_out, k, k_weight, k_eps)
    else:
        _fused_qk_rmsnorm_kernel(q, q_weight, q_eps, k, k_weight, k_eps, q_out, k_out)
    return q_out, k_out


@compile_ops("module_fused_qk_norm_rope_cache_quant_shuffle", develop=True)
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


@compile_ops("module_fused_qk_norm_rope_cache_quant_shuffle", develop=True)
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


@compile_ops("module_fused_qk_norm_rope_cache_quant_shuffle", develop=True)
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


@compile_ops("module_fused_qk_norm_rope_cache_quant_shuffle", develop=True)
def fused_qk_norm_rope_1way(
    q: Tensor,
    k: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    cos_sin: Tensor,
    batch_size: int,
    num_tokens: int,
    num_heads_q: int,
    num_heads_k: int,
    head_size: int,
    is_interleaved: bool,
    eps: float,
    out_q: Tensor,
    out_k: Tensor,
) -> None:
    """Fused per-head RMSNorm + RoPE on q/k for the 1way (single-stream) layout.

    Dtype contract:
        q, k, w_q, w_k, out_q, out_k : torch.bfloat16 or torch.float16 (same dtype)
        cos_sin                      : torch.float32  (REQUIRED)

    cos_sin must be float32 to match the diffusers / qwen-image-edit reference
    (RoPE freqs are computed in fp32 there and the precision is consumed by the
    fp32 rope multiply). Passing bf16/fp16 cos_sin will raise inside the kernel.
    """
    ...


@compile_ops(
    "module_fused_qk_norm_rope_cache_quant_shuffle",
    fc_name="fused_qk_norm_rope_2way_fp8_perhead_quant",
    develop=True,
)
def _fused_qk_norm_rope_2way_fp8_perhead_quant_kernel(
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
    q_fp8: Tensor,
    k_fp8: Tensor,
    q_descale: Tensor,
    k_descale: Tensor,
    q_unquantized: Tensor,
    k_unquantized: Tensor,
) -> None: ...


def fused_qk_norm_rope_2way_fp8_perhead_quant(
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
    out_q01: Optional[Tensor] = None,
    out_k01: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Same as the pertensor variant, but with per-(batch, head) descales.

    Returns (q_fp8, k_fp8, q_descale, k_descale, q_bf16, k_bf16) where
    q_descale.shape == (batch_size, num_heads_q) and
    k_descale.shape == (batch_size, num_heads_k). These shapes match what
    CK FP8 flash attention accepts natively.
    """
    want_bf16 = out_q01 is not None or out_k01 is not None
    total_tokens = num_tokens0 + num_tokens1
    fp8_dtype = get_dtype_fp8()

    q_fp8 = torch.empty(
        (batch_size, total_tokens, num_heads_q, head_size),
        dtype=fp8_dtype,
        device=q0.device,
    )
    k_fp8 = torch.empty(
        (batch_size, total_tokens, num_heads_k, head_size),
        dtype=fp8_dtype,
        device=k0.device,
    )
    q_descale = torch.empty(
        (batch_size, num_heads_q), dtype=torch.float32, device=q0.device
    )
    k_descale = torch.empty(
        (batch_size, num_heads_k), dtype=torch.float32, device=k0.device
    )
    q_unquantized = (
        out_q01
        if out_q01 is not None
        else torch.empty(
            (batch_size, total_tokens, num_heads_q, head_size),
            dtype=q0.dtype,
            device=q0.device,
        )
    )
    k_unquantized = (
        out_k01
        if out_k01 is not None
        else torch.empty(
            (batch_size, total_tokens, num_heads_k, head_size),
            dtype=k0.dtype,
            device=k0.device,
        )
    )

    _fused_qk_norm_rope_2way_fp8_perhead_quant_kernel(
        q0,
        k0,
        q1,
        k1,
        w_q0,
        w_k0,
        w_q1,
        w_k1,
        cos_sin0,
        cos_sin1,
        batch_size,
        num_tokens0,
        num_tokens1,
        num_heads_q,
        num_heads_k,
        head_size,
        is_interleaved,
        eps,
        q_fp8,
        k_fp8,
        q_descale,
        k_descale,
        q_unquantized,
        k_unquantized,
    )

    if not want_bf16:
        q_unquantized = torch.empty(0, dtype=q0.dtype, device=q0.device)
        k_unquantized = torch.empty(0, dtype=k0.dtype, device=k0.device)

    return q_fp8, k_fp8, q_descale, k_descale, q_unquantized, k_unquantized


@compile_ops(
    "module_fused_qk_norm_rope_cache_quant_shuffle",
    fc_name="v_2way_per_head_fp8_quant",
    develop=True,
)
def _v_2way_per_head_fp8_quant_kernel(
    v0: Tensor,
    v1: Tensor,
    v_fp8: Tensor,
    v_descale: Tensor,
) -> None: ...


def v_2way_per_head_fp8_quant(v0: Tensor, v1: Tensor) -> tuple[Tensor, Tensor]:
    """Per-(batch, head) FP8 quant for concatenated [v0, v1] without bf16 cat."""
    batch_size = v0.size(0)
    num_heads = v0.size(2)
    head_size = v0.size(3)
    total_tokens = v0.size(1) + v1.size(1)
    fp8_dtype = get_dtype_fp8()
    v_fp8 = torch.empty(
        (batch_size, total_tokens, num_heads, head_size),
        dtype=fp8_dtype,
        device=v0.device,
    )
    v_descale = torch.empty(
        (batch_size, num_heads), dtype=torch.float32, device=v0.device
    )
    _v_2way_per_head_fp8_quant_kernel(v0, v1, v_fp8, v_descale)
    return v_fp8, v_descale
