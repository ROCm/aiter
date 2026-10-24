# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import math
from dataclasses import dataclass

import torch
import triton

from aiter.ops.triton._triton_kernels.attention.mfalcon_attention import (
    _mfalcon_attention_kernel,
)
from aiter.ops.triton.utils._triton import arch_info


@dataclass(frozen=True)
class MFalconPlan:
    batch_size: int
    num_heads: int
    head_dim: int
    max_history_len: int
    max_num_candidates: int
    total_candidates: int
    block_m: int
    block_n: int
    seq_offsets: torch.Tensor
    num_candidates: torch.Tensor
    candidate_offsets: torch.Tensor
    q_shape: tuple[int, int, int]
    k_shape: tuple[int, int, int]
    v_shape: tuple[int, int, int]
    q_dtype: torch.dtype
    k_dtype: torch.dtype
    v_dtype: torch.dtype
    device: torch.device


def _default_tile_shape(head_dim: int) -> tuple[int, int]:
    if head_dim <= 64:
        return 64, 64
    if arch_info.get_arch() == "gfx950":
        return 16, 64
    return 32, 32


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_candidates: torch.Tensor,
    block_m: int,
    block_n: int,
) -> None:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must be three-dimensional")
    if q.shape != k.shape:
        raise ValueError("q and k must have the same shape")
    if v.shape[:2] != q.shape[:2] or v.shape[2] != q.shape[2]:
        raise ValueError("v must match q in token, head, and head-dim size")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("q must be float16 or bfloat16")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if not q.is_cuda or q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same CUDA device")
    if q.shape[1] <= 0:
        raise ValueError("num_heads must be positive")
    if q.shape[2] not in (16, 32, 64, 128, 256):
        raise ValueError("head_dim must be one of 16, 32, 64, 128, or 256")
    if seq_offsets.ndim != 1 or seq_offsets.numel() < 2:
        raise ValueError("seq_offsets must be one-dimensional with at least two values")
    if seq_offsets.dtype not in (torch.int32, torch.int64):
        raise ValueError("seq_offsets must be int32 or int64")
    if not seq_offsets.is_contiguous():
        raise ValueError("seq_offsets must be contiguous")
    if num_candidates.ndim != 1 or num_candidates.numel() != seq_offsets.numel() - 1:
        raise ValueError("num_candidates must have one value per sequence")
    if num_candidates.dtype not in (torch.int32, torch.int64):
        raise ValueError("num_candidates must be int32 or int64")
    if not num_candidates.is_contiguous():
        raise ValueError("num_candidates must be contiguous")
    if seq_offsets.device != q.device or num_candidates.device != q.device:
        raise ValueError("seq_offsets and num_candidates must match the q device")
    if seq_offsets[0].item() != 0:
        raise ValueError("seq_offsets must start with zero")
    if torch.any(seq_offsets[1:] < seq_offsets[:-1]).item():
        raise ValueError("seq_offsets must be non-decreasing")
    if seq_offsets[-1].item() != q.shape[0]:
        raise ValueError("seq_offsets must end at the total token count")
    if torch.any(num_candidates < 0).item():
        raise ValueError("num_candidates must be non-negative")
    seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
    if torch.any(num_candidates > seq_lengths).item():
        raise ValueError("num_candidates cannot exceed the sequence length")
    if block_m < 16 or block_m & (block_m - 1) != 0:
        raise ValueError("block_m must be a power of two and at least 16")
    if block_n < 16 or block_n & (block_n - 1) != 0:
        raise ValueError("block_n must be a power of two and at least 16")


def build_mfalcon_plan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_candidates: torch.Tensor,
    block_m: int | None = None,
    block_n: int | None = None,
) -> MFalconPlan:
    default_m, default_n = _default_tile_shape(q.shape[2])
    if block_m is None:
        block_m = default_m
    if block_n is None:
        block_n = default_n
    _validate_inputs(q, k, v, seq_offsets, num_candidates, block_m, block_n)

    seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
    history_lengths = seq_lengths - num_candidates
    candidate_offsets = torch.cat(
        [
            torch.zeros((1,), dtype=torch.int64, device=q.device),
            num_candidates.cumsum(dim=0, dtype=torch.int64),
        ]
    )
    return MFalconPlan(
        batch_size=num_candidates.numel(),
        num_heads=q.shape[1],
        head_dim=q.shape[2],
        max_history_len=history_lengths.max().item(),
        max_num_candidates=num_candidates.max().item(),
        total_candidates=num_candidates.sum().item(),
        block_m=block_m,
        block_n=block_n,
        seq_offsets=seq_offsets,
        num_candidates=num_candidates,
        candidate_offsets=candidate_offsets,
        q_shape=tuple(q.shape),
        k_shape=tuple(k.shape),
        v_shape=tuple(v.shape),
        q_dtype=q.dtype,
        k_dtype=k.dtype,
        v_dtype=v.dtype,
        device=q.device,
    )


def _validate_plan(
    plan: MFalconPlan,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_candidates: torch.Tensor,
    block_m: int | None,
    block_n: int | None,
) -> None:
    if plan.q_shape != tuple(q.shape) or plan.k_shape != tuple(k.shape):
        raise ValueError("plan tensor shapes do not match q and k")
    if plan.v_shape != tuple(v.shape):
        raise ValueError("plan tensor shape does not match v")
    if plan.q_dtype != q.dtype or plan.k_dtype != k.dtype or plan.v_dtype != v.dtype:
        raise ValueError("plan tensor dtypes do not match q, k, and v")
    if plan.device != q.device:
        raise ValueError("plan device does not match q")
    if plan.seq_offsets.data_ptr() != seq_offsets.data_ptr():
        raise ValueError("plan seq_offsets tensor was replaced")
    if plan.num_candidates.data_ptr() != num_candidates.data_ptr():
        raise ValueError("plan num_candidates tensor was replaced")
    if block_m is not None and block_m != plan.block_m:
        raise ValueError("block_m conflicts with the supplied plan")
    if block_n is not None and block_n != plan.block_n:
        raise ValueError("block_n conflicts with the supplied plan")


def mfalcon_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_candidates: torch.Tensor,
    plan: MFalconPlan | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run M-FALCON candidate attention on packed sequences.

    The input tensors contain history tokens followed by candidate tokens for
    each sequence. Only candidate outputs are returned. Each candidate attends
    to the complete history and to itself, but not to other candidates.

    Args:
        q: Query tensor with shape (total_tokens, num_heads, head_dim).
        k: Key tensor with the same shape and dtype as q.
        v: Value tensor with the same shape and dtype as q.
        seq_offsets: Cumulative offsets of each sequence. The first value is
            zero and the last value is total_tokens.
        num_candidates: Candidate count for each sequence.
        plan: Optional plan built by build_mfalcon_plan. Supply it to skip
            plan construction on repeated calls with the same inputs.
        block_m: Optional candidate tile size. It must be a power of two and
            at least 16.
        block_n: Optional history tile size. It must be a power of two and at
            least 16.
        return_lse: Return the log-sum-exp of the attention scores together
            with the output.

    Returns:
        The output tensor with shape (total_candidates, num_heads, head_dim).
        If return_lse is true, a tuple of the output tensor and the LSE tensor
        with shape (total_candidates, num_heads).

    Building a plan performs host synchronization to derive launch metadata.
    Reusing a plan is the intended hot path. A reused plan detects replacement
    of the ``seq_offsets`` and ``num_candidates`` tensors, but does not detect
    in-place mutation of their values.

    User-supplied tile sizes are not checked against architecture-specific LDS
    or register limits.
    """
    if plan is None:
        plan = build_mfalcon_plan(
            q, k, v, seq_offsets, num_candidates, block_m, block_n
        )
    else:
        _validate_plan(plan, q, k, v, seq_offsets, num_candidates, block_m, block_n)

    out = torch.empty(
        (plan.total_candidates, plan.num_heads, plan.head_dim),
        dtype=q.dtype,
        device=q.device,
    )
    lse = (
        torch.empty(
            (plan.total_candidates, plan.num_heads),
            dtype=torch.float32,
            device=q.device,
        )
        if return_lse
        else None
    )
    if plan.total_candidates == 0:
        return (out, lse) if return_lse else out

    grid = (
        triton.cdiv(plan.max_num_candidates, plan.block_m),
        plan.num_heads,
        plan.batch_size,
    )
    _mfalcon_attention_kernel[grid](
        q,
        k,
        v,
        out,
        lse,
        seq_offsets,
        num_candidates,
        plan.candidate_offsets,
        1.0 / math.sqrt(plan.head_dim),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        lse.stride(0) if lse is not None else 0,
        lse.stride(1) if lse is not None else 0,
        BLOCK_M=plan.block_m,
        BLOCK_N=plan.block_n,
        BLOCK_D=plan.head_dim,
        RETURN_LSE=return_lse,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        num_warps=4,
        num_stages=2,
    )
    return (out, lse) if return_lse else out
