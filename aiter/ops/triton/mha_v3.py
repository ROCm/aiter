# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations
from typing import Optional, Tuple, Union
import torch

from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_3


class _FlashAttnV3Func(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: float | None,
        causal: bool,
        qv: Optional[torch.Tensor],
        q_descale: Optional[torch.Tensor],
        k_descale: Optional[torch.Tensor],
        v_descale: Optional[torch.Tensor],
        window_size: Tuple[int, int],
        attention_chunk: int,
        softcap: float,
        num_splits: int,
        pack_gqa: Optional[bool],
        deterministic: bool,
        sm_margin: int,
    ):
        # Derive softmax scale if not provided (include qv width like Hopper v3)
        if softmax_scale is None:
            q_extra = qv.shape[-1] if qv is not None else 0
            softmax_scale = (q.shape[-1] + q_extra) ** (-0.5)

        # Fast validation of unsupported features
        if qv is not None:
            raise NotImplementedError("qv is not supported in AMD Triton v3 yet")
        if attention_chunk not in (0, 1):
            raise NotImplementedError("attention_chunk > 1 not supported (0 or 1 only)")
        if softcap != 0.0:
            raise NotImplementedError("softcap not implemented in AMD Triton v3")
        if num_splits != 1:
            raise NotImplementedError("num_splits != 1 not supported in AMD Triton v3")
        if pack_gqa is not None:
            raise NotImplementedError("pack_gqa not implemented in AMD Triton v3")
        if sm_margin != 0:
            raise NotImplementedError("sm_margin != 0 not supported in AMD Triton v3")

        out, softmax_lse = flash_attn_3.fwd(
            q,
            k,
            v,
            None,  # k_new
            None,  # v_new
            None,  # qv
            None,  # out tensor (allocate inside)
            None,  # cu_seqlens_q
            None,  # cu_seqlens_k
            None,  # cu_seqlens_k_new
            None,  # seqused_q
            None,  # seqused_k
            None,  # max_seqlen_q
            None,  # max_seqlen_k
            None,  # page_table
            None,  # kv_batch_idx
            None,  # leftpad_k
            None,  # rotary_cos
            None,  # rotary_sin
            None,  # seqlens_rotary
            q_descale,
            k_descale,
            v_descale,
            softmax_scale,
            causal,
            int(window_size[0]),
            int(window_size[1]),
            attention_chunk,
            softcap,
            False,  # rotary_interleaved
            None,  # scheduler_metadata
            num_splits,
            pack_gqa,
            sm_margin,
        )

        ctx.save_for_backward(
            q, k, v, out, softmax_lse, q_descale, k_descale, v_descale
        )
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.sm_margin = sm_margin
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, softmax_lse, q_descale, k_descale, v_descale = ctx.saved_tensors

        dq, dk, dv, _delta = flash_attn_3.bwd(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            None,  # dq
            None,  # dk
            None,  # dv
            None,  # cu_seqlens_q
            None,  # cu_seqlens_k
            None,  # seqused_q
            None,  # seqused_k
            None,  # max_seqlen_q
            None,  # max_seqlen_k
            ctx.softmax_scale,
            ctx.causal,
            int(ctx.window_size[0]),
            int(ctx.window_size[1]),
            ctx.softcap,
            ctx.deterministic,
            ctx.sm_margin,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        return (
            dq,  # q
            dk,  # k
            dv,  # v
            None,  # softmax_scale
            None,  # causal
            None,  # qv
            None,  # q_descale
            None,  # k_descale
            None,  # v_descale
            None,  # window_size
            None,  # attention_chunk
            None,  # softcap
            None,  # num_splits
            None,  # pack_gqa
            None,  # deterministic
            None,  # sm_margin
        )


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    qv: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    window_size: Tuple[int, int] = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    sm_margin: int = 0,
):
    """FlashAttention v3 entry point."""
    return _FlashAttnV3Func.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        qv,
        q_descale,
        k_descale,
        v_descale,
        window_size,
        attention_chunk,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        sm_margin,
    )


class _FlashAttnVarlenV3Func(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float | None,
        causal: bool,
        q_descale: torch.Tensor | None,
        k_descale: torch.Tensor | None,
        v_descale: torch.Tensor | None,
        window_size: tuple[int, int],
        attention_chunk: int,
        softcap: float,
        deterministic: bool,
        sm_margin: int,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        if attention_chunk != 0:
            raise NotImplementedError(
                "attention_chunk != 0 not supported in varlen v3 yet"
            )
        if softcap != 0.0:
            raise NotImplementedError("softcap not implemented in varlen v3 yet")
        if sm_margin != 0:
            raise NotImplementedError("sm_margin != 0 not supported in varlen v3 yet")

        out, softmax_lse = flash_attn_3.fwd(
            q,
            k,
            v,
            None,  # k_new
            None,  # v_new
            None,  # qv
            None,  # out tensor
            cu_seqlens_q,
            cu_seqlens_k,
            None,  # cu_seqlens_k_new
            None,  # seqused_q
            None,  # seqused_k
            max_seqlen_q,
            max_seqlen_k,
            None,  # page_table
            None,  # kv_batch_idx
            None,  # leftpad_k
            None,  # rotary_cos
            None,  # rotary_sin
            None,  # seqlens_rotary
            q_descale,
            k_descale,
            v_descale,
            softmax_scale,
            causal,
            int(window_size[0]),
            int(window_size[1]),
            attention_chunk,
            softcap,
            False,  # rotary_interleaved
            None,  # scheduler_metadata
            1,  # num_splits
            None,  # pack_gqa
            sm_margin,
        )

        ctx.save_for_backward(
            q, k, v, out, softmax_lse, q_descale, k_descale, v_descale
        )
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.sm_margin = sm_margin
        ctx.cu_seqlens_q = cu_seqlens_q
        ctx.cu_seqlens_k = cu_seqlens_k
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, softmax_lse, q_descale, k_descale, v_descale = ctx.saved_tensors

        dq, dk, dv, _delta = flash_attn_3.bwd(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            None,  # dq
            None,  # dk
            None,  # dv
            ctx.cu_seqlens_q,
            ctx.cu_seqlens_k,
            None,  # seqused_q
            None,  # seqused_k
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.softmax_scale,
            ctx.causal,
            int(ctx.window_size[0]),
            int(ctx.window_size[1]),
            ctx.softcap,
            ctx.deterministic,
            ctx.sm_margin,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        return (
            dq,
            dk,
            dv,
            None,  # cu_seqlens_q
            None,  # cu_seqlens_k
            None,  # max_seqlen_q
            None,  # max_seqlen_k
            None,  # softmax_scale
            None,  # causal
            None,  # q_descale
            None,  # k_descale
            None,  # v_descale
            None,  # window_size
            None,  # attention_chunk
            None,  # softcap
            None,  # deterministic
            None,  # sm_margin
        )


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    window_size: Tuple[int, int] = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    deterministic: bool = False,
    sm_margin: int = 0,
):
    """FlashAttention v3 varlen path."""
    return _FlashAttnVarlenV3Func.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        q_descale,
        k_descale,
        v_descale,
        window_size,
        attention_chunk,
        softcap,
        deterministic,
        sm_margin,
    )


def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    qv: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[Union[torch.Tensor, int]] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    num_splits: int = 0,
    pack_gqa: Optional[bool] = None,
    sm_margin: int = 0,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    return_softmax_lse: bool = False,
    page_table: Optional[torch.Tensor] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    rotary_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
):
    """
    Arguments mirror Hopper's `flash_attn_with_kvcache` with current backend limitations.
    Unsupported: backward, qv, softcap!=0, pack_gqa, sm_margin!=0, attention_chunk>1, num_splits>1,
    simultaneous varlen (cu_seqlens_q) + cache_seqlens tensor, and partial rotary inputs.
    """
    # Scale
    if softmax_scale is None:
        q_extra = qv.shape[-1] if qv is not None else 0
        softmax_scale = (q.shape[-1] + q_extra) ** (-0.5)

    # Feature guards
    if qv is not None:
        raise NotImplementedError("qv not supported in KV cache path yet")
    if softcap != 0.0:
        raise NotImplementedError("softcap not implemented in KV cache path")
    if pack_gqa is not None:
        raise NotImplementedError("pack_gqa not implemented in KV cache path")
    if sm_margin != 0:
        raise NotImplementedError("sm_margin != 0 not supported in KV cache path")
    if attention_chunk not in (0, 1):
        raise NotImplementedError("attention_chunk > 1 not supported (0 or 1 only)")
    if num_splits not in (0, 1):
        raise NotImplementedError("num_splits > 1 not supported in KV cache path")

    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )

    if cu_seqlens_q is not None and cache_seqlens is not None:
        raise NotImplementedError(
            "Varlen decode with cache_seqlens tensor not supported yet"
        )
    if (rotary_cos is None) ^ (rotary_sin is None):
        raise ValueError(
            "Both rotary_cos and rotary_sin must be provided together or neither"
        )
    if (
        (rotary_cos is not None)
        and rotary_seqlens is not None
        and cu_seqlens_q is None
        and cache_seqlens is None
    ):
        raise ValueError(
            "rotary_seqlens provided without cu_seqlens_q or cache_seqlens context"
        )

    kv_batch_idx = cache_batch_idx
    leftpad_k = cache_leftpad
    seqlens_rotary = rotary_seqlens

    out, softmax_lse = flash_attn_3.fwd(
        q,
        k_cache,
        v_cache,
        k,
        v,
        None,  # qv
        None,  # out allocate
        cu_seqlens_q,
        None,  # cu_seqlens_k
        cu_seqlens_k_new,
        None,  # seqused_q
        cache_seqlens if isinstance(cache_seqlens, torch.Tensor) else None,  # seqused_k
        max_seqlen_q,
        None,  # max_seqlen_k
        page_table,
        kv_batch_idx,
        leftpad_k,
        rotary_cos,
        rotary_sin,
        seqlens_rotary,
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        causal,
        int(window_size[0]),
        int(window_size[1]),
        attention_chunk,
        softcap,
        False,  # rotary_interleaved
        None,  # scheduler_metadata
        num_splits if num_splits != 0 else 1,
        pack_gqa,
        sm_margin,
    )
    return (out, softmax_lse) if return_softmax_lse else out
