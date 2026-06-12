# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Block-level mean-pooling primitives.
#
# These build the cheap O(nblock^2) block-attention proxy used by SpargeAttn /
# VFA block-sparse mask construction: each (B, H) sequence is split into blocks
# of ``block_size`` tokens, every block is reduced to its token mean, and the
# pooled representatives are scored against each other. ``get_pool_sim_triton_simmean``
# additionally flags blocks that are internally coherent enough to be summarized
# by their mean (the "meansim" predicate).

from __future__ import annotations

from typing import Optional, Tuple

import torch

from aiter.ops.triton._triton_kernels.pool import triton_bmm_pool_sim_simmean
from aiter.ops.triton.attention.utils import map_dims


def get_pool_sim_triton_simmean(
    x: torch.Tensor,
    block_size: int,
    simthreshd1: torch.Tensor,
    attention_scored_only: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Mean-pool each block and flag internally-similar blocks.

    Args:
        x: ``(B, H, N, D)`` tensor.
        block_size: number of tokens per block.
        simthreshd1: ``(H,)`` per-head similarity threshold.
        attention_scored_only: when ``True``, skip the intra-block similarity
            test entirely and return ``None`` for ``sim_blocks``.

    Steps:
        1. Pool (mean) within each block.
        2. Compute the mean pairwise cosine similarity within each block.
        3. Flag blocks whose mean self-similarity exceeds ``simthreshd1``.

    Note how the 3rd dimension ``N`` is reduced to ``nblock = N // block_size``;
    this keeps the downstream block-attention proxy at ``O(nblock^2)`` instead
    of the full ``O(N^2)``.

    Returns:
        pool: ``(B, H, nblock, D)`` tensor.
        sim_blocks: ``(B, H, nblock)`` bool tensor, or ``None`` when
            ``attention_scored_only`` is set.
    """
    x = x.contiguous()
    B, H, N, D = x.shape
    nblock = (N + block_size - 1) // block_size  # Number of blocks per feature map
    pool = torch.empty((B, H, nblock, D), device=x.device, dtype=x.dtype)
    if attention_scored_only:
        sim_blocks = None
        # The kernel needs a valid pointer; pass `pool` as an unused placeholder.
        sim_arg = pool
    else:
        sim_blocks = torch.empty((B, H, nblock), device=x.device, dtype=torch.bool)
        sim_arg = sim_blocks
    grid = (B, H, nblock)
    triton_bmm_pool_sim_simmean[grid](
        x, pool, sim_arg, simthreshd1, N=N, D=D, BS=block_size, SKIP_SIM=attention_scored_only
    )
    return pool, sim_blocks


def pool_blocks_mean(
    x: torch.Tensor,
    BLK: int,
    layout: str,
) -> torch.Tensor:
    """Mean-pooled block representatives.

    Returns fp32 ``[batch, nheads, num_blocks, head_dim]`` -- the per-block
    token mean of ``x``.
    """
    bshd_map = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]
    batch, seqlen, nheads, head_dim = map_dims(x.shape, bshd_map)
    num_blocks = (seqlen + BLK - 1) // BLK

    # Logical [batch, nheads, seqlen, head_dim] view regardless of layout.
    # Reduce each block directly with a fp32 accumulator (``dtype=torch.float32``)
    # so we never materialize a full-precision copy of the large sequence tensor,
    # and split off any ragged tail instead of padding the whole tensor -- both
    # avoid extra full-size allocations/copies that dominated this proxy.
    xv = x if layout == "bhsd" else x.permute(0, 2, 1, 3)
    n_full = seqlen // BLK
    full = n_full * BLK

    sums = (
        xv[:, :, :full, :]
        .reshape(batch, nheads, n_full, BLK, head_dim)
        .sum(dim=3, dtype=torch.float32)
    )

    rem = seqlen - full
    if rem == 0:
        return sums / BLK

    tail = xv[:, :, full:, :].sum(dim=2, dtype=torch.float32).unsqueeze(2)
    sums = torch.cat([sums, tail], dim=2)
    counts = torch.full((num_blocks,), float(BLK), device=x.device)
    counts[-1] = rem
    return sums / counts[None, None, :, None]


def compute_pooled_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    BLKQ: int,
    BLKK: int,
    layout: str,
) -> torch.Tensor:
    """SpargeAttn-style block-attention estimate used to rank candidate K blocks.

    Mean-pools Q and K into per-block representatives and forms the block-level
    score ``pooled_q @ pooled_k^T * head_dim**-0.5``.  Operates on the original
    (pre-quant) float Q/K, which keeps the estimate a genuine block-attention
    proxy.

    Returns fp32 ``[batch, nheads_q, num_q_blocks, num_k_blocks]``.
    """
    pooled_q = pool_blocks_mean(q, BLKQ, layout)
    pooled_k = pool_blocks_mean(k, BLKK, layout)

    nheads_q, nheads_k = pooled_q.shape[1], pooled_k.shape[1]
    if nheads_q != nheads_k:
        pooled_k = pooled_k.repeat_interleave(nheads_q // nheads_k, dim=1)

    head_dim = pooled_q.shape[-1]
    return torch.matmul(pooled_q, pooled_k.transpose(-1, -2)) * (head_dim ** -0.5)
