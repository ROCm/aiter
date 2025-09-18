# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations
from typing import Optional, Tuple
import torch

from aiter.ops.triton.flash_attn_triton_amd import interface_fa_v3 as _fa3_amd


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

        requires_grad = any(t.requires_grad for t in (q, k, v)) and torch.is_grad_enabled()

        # Forward: basic path (no varlen / kv-cache parameters used here)
        out, softmax_lse = _fa3_amd.fwd(
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
            None,   # scheduler_metadata
            num_splits,
            pack_gqa,
            sm_margin,
        )

        if requires_grad:
            ctx.save_for_backward(q, k, v, out, softmax_lse)
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.deterministic = deterministic
            ctx.sm_margin = sm_margin
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv, _delta = _fa3_amd.bwd(
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
    softmax_scale: float | None = None,
    causal: bool = False,
    qv: torch.Tensor | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    window_size: tuple[int, int] = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: bool | None = None,
    deterministic: bool = False,
    sm_margin: int = 0,
):
    """FlashAttention v3 (Hopper-style) entry point using the AMD Triton backend.

    Returns only the attention output tensor (no LSE / probs) matching upstream FA3 public API.
    Unsupported features raise NotImplementedError until implemented in the backend.
    """
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
            raise NotImplementedError("attention_chunk != 0 not supported in varlen v3 yet")
        if softcap != 0.0:
            raise NotImplementedError("softcap not implemented in varlen v3 yet")
        if sm_margin != 0:
            raise NotImplementedError("sm_margin != 0 not supported in varlen v3 yet")

        requires_grad = any(t.requires_grad for t in (q, k, v)) and torch.is_grad_enabled()

        out, softmax_lse = _fa3_amd.fwd(
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
            None,   # scheduler_metadata
            1,      # num_splits
            None,   # pack_gqa
            sm_margin,
        )

        if requires_grad:
            ctx.save_for_backward(q, k, v, out, softmax_lse)
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
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv, _delta = _fa3_amd.bwd(
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
    softmax_scale: float | None = None,
    causal: bool = False,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    window_size: tuple[int, int] = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    deterministic: bool = False,
    sm_margin: int = 0,
):
    """FlashAttention v3 varlen path (AMD Triton backend).

    Returns only the attention output tensor. Unsupported features raise NotImplementedError.
    """
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
