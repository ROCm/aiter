# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sparse paged-decode attention over a unified KV pool with per-token paged
indices. See ``_triton_kernels/attention/pa_decode_sparse.py`` for the
kernels' caller contract.

This module exposes ``pa_decode_sparse`` — a 3D split-K + widened-BLOCK_H
+ pipelined-K-loop variant suitable for sparse decode (e.g. V4 top-k gather)
where each token's K range is an unordered subset of a unified KV pool.
"""

from typing import Optional

import torch
import triton

from aiter.ops.triton._triton_kernels.attention.pa_decode_sparse import (
    _pa_decode_sparse,
    _pa_decode_sparse_reduce,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def pa_decode_sparse(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    block_h: Optional[int] = None,
    kv_splits: Optional[int] = None,
) -> torch.Tensor:
    """Sparse paged-decode attention with split-K + widened BLOCK_H.

    Args:
        q: ``[N, H, D]`` decode queries, bf16/fp16.
        unified_kv: ``[total_pages, D]`` shared KV pool (page_size=1), same dtype as ``q``.
        kv_indices: ``[total_indices]`` int32 — per-token slot lists, flat.
            Per-token entries live in ``kv_indices[kv_indptr[t] : kv_indptr[t+1]]``.
            ``-1`` entries are skipped (sentinel for unused tail).
        kv_indptr: ``[N+1]`` int32 — true prefix sum.
        attn_sink: ``[H]`` per-head learnable softmax-denom bias (fp32).
        softmax_scale: scalar softmax scale.
        block_h: override ``BLOCK_H`` for the split kernel. Default picks
            ``next_pow2(min(H, 64))``, rounded up to the AMD MFMA min tile (16).
        kv_splits: override ``KV_SPLITS`` for the split-K grid axis. Default
            auto-infers to fill ~512 total CTAs while capping below the number
            of K-blocks, then rounds up to a power of 2.
        num_stages: software-pipeline depth of the K loop (default 2).

    Returns:
        ``[N, H, D]`` attention output, same dtype as ``q``.

    Optimizations targeted:
      (1) Wider ``BLOCK_H`` so all heads of a token are handled by one CTA →
          eliminates MLA-style KV re-fetch across head-block programs.
      (2) ``num_stages`` on the K loop pipelines KV gather behind the dot.
      (3) Split the K dimension across CTAs via a third grid axis →
          fixes grid undersubscription on long-context decode.
    """
    if not q.is_cuda:
        raise RuntimeError("pa_decode_sparse requires CUDA/HIP tensors")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(f"pa_decode_sparse expects fp16/bf16 q, got {q.dtype}")
    if unified_kv.dtype != q.dtype:
        raise RuntimeError(
            f"unified_kv dtype mismatch: kv={unified_kv.dtype}, q={q.dtype}"
        )

    T, H, D = q.shape
    _LOGGER.info(
        f"PA_DECODE_SPARSE T={T} H={H} D={D} " f"total_indices={kv_indices.shape[0]}"
    )

    out = torch.empty_like(q)
    kv_indices = kv_indices.to(torch.int32).contiguous()
    kv_indptr = kv_indptr.to(torch.int32).contiguous()

    if block_h is None:
        # Default: one CTA per token (kills the H/BLOCK_H KV duplication).
        # If H is too large to fit a single tile, halve until it does.
        block_h = triton.next_power_of_2(min(H, 64))
    else:
        block_h = triton.next_power_of_2(block_h)
    block_h = max(block_h, 16)  # AMD MFMA min tile

    n_head_blocks = triton.cdiv(H, block_h)
    h_padded = n_head_blocks * block_h
    block_d = triton.next_power_of_2(D)
    block_k = 16 if D >= 256 else 32
    num_stages = 2
    attn_num_warps = 4
    waves_per_eu = 0
    reduce_num_warps = 4

    # Infer KV_SPLITS from inputs when caller doesn't override.
    # Fill ~512 total CTAs (MI300X has 304 CUs) while never splitting K into
    # more pieces than there are K-blocks. Rounded up to a power of 2 so the
    # reduce kernel's tl.arange(0, KV_SPLITS) compiles; over-splitting past
    # max_kv_splits is handled by the kernel (empty splits early-return and
    # the reduce masks their stale partial-buffer slots).
    if kv_splits is None:
        max_kv_len = kv_indices.shape[0]
        max_num_wg = 512
        max_kv_splits = max(1, triton.cdiv(max_kv_len, block_k))
        kv_splits = max(1, max_num_wg // max(1, T * n_head_blocks))
        kv_splits = min(max_kv_splits, kv_splits)
        kv_splits = triton.next_power_of_2(kv_splits)

    m_partial = torch.empty(
        (T, kv_splits, h_padded), dtype=torch.float32, device=q.device
    )
    l_partial = torch.empty_like(m_partial)
    acc_partial = torch.empty(
        (T, kv_splits, h_padded, D), dtype=torch.float32, device=q.device
    )

    grid_attn = (T, n_head_blocks, kv_splits)
    _pa_decode_sparse[grid_attn](
        q,
        unified_kv,
        kv_indices,
        kv_indptr,
        m_partial,
        l_partial,
        acc_partial,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        unified_kv.stride(0),
        unified_kv.stride(1),
        m_partial.stride(0),
        m_partial.stride(1),
        m_partial.stride(2),
        l_partial.stride(0),
        l_partial.stride(1),
        l_partial.stride(2),
        acc_partial.stride(0),
        acc_partial.stride(1),
        acc_partial.stride(2),
        acc_partial.stride(3),
        H,
        D,
        kv_splits,
        float(softmax_scale),
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        num_warps=attn_num_warps,
        num_stages=num_stages,
        waves_per_eu=waves_per_eu,
    )

    # One reduce CTA per head. For small per-rank H (TP=8 → H ∈ {8, 16}) this
    # multiplies the reduce-side CTA count by H, replacing the previous single
    # under-occupied CTA per token with a small fan-out that hides launch
    # latency. tl.arange(0, 1) is a valid power-of-2 range.
    block_h_reduce = 1
    grid_reduce = (T, triton.cdiv(H, block_h_reduce))
    _pa_decode_sparse_reduce[grid_reduce](
        m_partial,
        l_partial,
        acc_partial,
        attn_sink,
        kv_indptr,
        out,
        m_partial.stride(0),
        m_partial.stride(1),
        m_partial.stride(2),
        l_partial.stride(0),
        l_partial.stride(1),
        l_partial.stride(2),
        acc_partial.stride(0),
        acc_partial.stride(1),
        acc_partial.stride(2),
        acc_partial.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        H,
        D,
        kv_splits,
        BLOCK_H=block_h_reduce,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        num_warps=reduce_num_warps,
    )
    return out
