# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import math
from itertools import pairwise

import torch
import triton

from aiter.ops.triton._triton_kernels.attention.broadcast_q_causal import (
    _broadcast_q_causal_dkdv,
    _broadcast_q_causal_dq,
    _broadcast_q_causal_forward,
)

_BLOCK_M = 128
_BLOCK_N = 64
_BATCH_PER_GROUP = 4
_SUPPORTED_HEAD_DIMS = (64, 128, 256)
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> int:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must have shape [tokens, heads, head_dim]")
    if q.shape[1:] != k.shape[1:] or q.shape[1:] != v.shape[1:]:
        raise ValueError("q, k, and v must have matching heads and head_dim")
    if q.dtype not in _SUPPORTED_DTYPES or q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must all use fp16 or bf16")
    if q.device != k.device or q.device != v.device or q.device != cu_seqlens_k.device:
        raise ValueError("q, k, v, and cu_seqlens_k must use the same device")
    if cu_seqlens_k.ndim != 1 or cu_seqlens_k.numel() < 1:
        raise ValueError("cu_seqlens_k must be a non-empty 1D tensor")
    if cu_seqlens_k.dtype not in (torch.int32, torch.int64):
        raise ValueError("cu_seqlens_k must use int32 or int64")
    if q.shape[2] not in _SUPPORTED_HEAD_DIMS:
        raise ValueError("head_dim must be one of 64, 128, or 256")
    return cu_seqlens_k.numel() - 1


def _validate_offsets(
    cu_seqlens_k: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> list[int]:
    offsets = cu_seqlens_k.detach().cpu().tolist()
    if offsets[0] != 0:
        raise ValueError("cu_seqlens_k[0] must be zero")
    if any(end < start for start, end in pairwise(offsets)):
        raise ValueError("cu_seqlens_k must be non-decreasing")
    if offsets[-1] > k.shape[0] or offsets[-1] > v.shape[0]:
        raise ValueError("cu_seqlens_k[-1] exceeds the packed K/V length")
    return [end - start for start, end in pairwise(offsets)]


def _launch_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
) -> None:
    batch_count = cu_seqlens_k.numel() - 1
    q_len, heads, head_dim = q.shape
    if q_len == 0 or batch_count == 0 or heads == 0:
        return
    grid = (batch_count, triton.cdiv(q_len, _BLOCK_M), heads)
    _broadcast_q_causal_forward[grid](
        q,
        k,
        v,
        cu_seqlens_k,
        out,
        lse,
        q_len,
        softmax_scale,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        *lse.stride(),
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        HEAD_DIM=head_dim,
        num_warps=4,
        num_stages=2,
        enable_fp_fusion=True,
    )


def _launch_dq(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    lse: torch.Tensor,
    out: torch.Tensor,
    grad_out: torch.Tensor,
    dq_partial: torch.Tensor,
    softmax_scale: float,
) -> None:
    batch_count = cu_seqlens_k.numel() - 1
    q_len, heads, head_dim = q.shape
    group_count = triton.cdiv(batch_count, _BATCH_PER_GROUP)
    if q_len == 0 or batch_count == 0 or heads == 0:
        return
    grid = (group_count, triton.cdiv(q_len, _BLOCK_M), heads)
    _broadcast_q_causal_dq[grid](
        q,
        k,
        v,
        cu_seqlens_k,
        lse,
        out,
        grad_out,
        dq_partial,
        q_len,
        batch_count,
        softmax_scale,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *grad_out.stride(),
        *lse.stride(),
        *out.stride(),
        *dq_partial.stride(),
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        HEAD_DIM=head_dim,
        BATCH_PER_GROUP=_BATCH_PER_GROUP,
        num_warps=4,
        num_stages=2,
        enable_fp_fusion=True,
    )


def _launch_dkdv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    lse: torch.Tensor,
    out: torch.Tensor,
    grad_out: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    softmax_scale: float,
    max_active_len: int,
) -> None:
    batch_count = cu_seqlens_k.numel() - 1
    q_len, heads, head_dim = q.shape
    if max_active_len == 0 or batch_count == 0 or heads == 0:
        return
    grid = (batch_count, triton.cdiv(max_active_len, _BLOCK_N), heads)
    _broadcast_q_causal_dkdv[grid](
        q,
        k,
        v,
        cu_seqlens_k,
        lse,
        out,
        grad_out,
        dk,
        dv,
        q_len,
        softmax_scale,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *grad_out.stride(),
        *lse.stride(),
        *out.stride(),
        *dk.stride(),
        *dv.stride(),
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        HEAD_DIM=head_dim,
        num_warps=4,
        num_stages=2,
        enable_fp_fusion=True,
    )


class _BroadcastQCausalVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_k,
        softmax_scale,
        return_lse,
        is_grad_enabled,
    ):
        batch_count = _validate_inputs(q, k, v, cu_seqlens_k)
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[2])
        q_len, heads, head_dim = q.shape
        out = torch.empty(
            (batch_count * q_len, heads, head_dim),
            device=q.device,
            dtype=q.dtype,
        )
        lse = torch.empty(
            (batch_count * q_len, heads),
            device=q.device,
            dtype=torch.float32,
        )
        _launch_forward(q, k, v, cu_seqlens_k, out, lse, softmax_scale)

        needs_grad = is_grad_enabled and any(
            tensor.requires_grad for tensor in (q, k, v)
        )
        if needs_grad:
            ctx.save_for_backward(q, k, v, cu_seqlens_k, out, lse)
            ctx.softmax_scale = softmax_scale
        if return_lse:
            ctx.mark_non_differentiable(lse)
            return out, lse
        return out

    @staticmethod
    def backward(ctx, grad_out, grad_lse=None):
        q, k, v, cu_seqlens_k, out, lse = ctx.saved_tensors
        lengths = _validate_offsets(cu_seqlens_k, k, v)
        batch_count = len(lengths)
        q_len, heads, head_dim = q.shape
        grad_out = grad_out.reshape((batch_count * q_len, heads, head_dim))
        grad_out = grad_out.to(device=q.device, dtype=q.dtype)

        needs_dq = ctx.needs_input_grad[0]
        needs_dkdv = ctx.needs_input_grad[1] or ctx.needs_input_grad[2]
        dq = None
        dk = None
        dv = None

        if needs_dq:
            group_count = triton.cdiv(batch_count, _BATCH_PER_GROUP)
            dq_partial = torch.empty(
                (group_count, q_len, heads, head_dim),
                device=q.device,
                dtype=torch.float32,
            )
            _launch_dq(
                q,
                k,
                v,
                cu_seqlens_k,
                lse,
                out,
                grad_out,
                dq_partial,
                ctx.softmax_scale,
            )
            dq = dq_partial.sum(dim=0).to(q.dtype)

        if needs_dkdv:
            dk = torch.zeros_like(k)
            dv = torch.zeros_like(v)
            _launch_dkdv(
                q,
                k,
                v,
                cu_seqlens_k,
                lse,
                out,
                grad_out,
                dk,
                dv,
                ctx.softmax_scale,
                max(lengths, default=0),
            )

        return dq, dk, dv, None, None, None, None


def broadcast_q_causal_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float | None = None,
    return_lse: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Compute causal attention with one query block shared by jagged histories.

    q has shape [q_len, heads, head_dim]. k and v have shape
    [total_kv, heads, head_dim] and are packed by cu_seqlens_k. Query row i is
    active only when i is less than min(q_len, history_len). An active query
    row attends local keys 0 through i. Inactive rows produce zero output and
    -inf LSE.

    Returns out when return_lse is false. Returns out and LSE when return_lse
    is true. LSE is non-differentiable. Backward is deterministic. dQ sums
    contributions from all sequences. dK and dV stay packed.

    The forward does not synchronize offsets to the host. Callers must supply
    non-decreasing offsets that start at zero and end within k and v. Kernel
    offset arithmetic is int32, so tensors must stay below 2**31 elements.
    """
    return _BroadcastQCausalVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_k,
        softmax_scale,
        return_lse,
        torch.is_grad_enabled(),
    )
