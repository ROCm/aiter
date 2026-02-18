# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations
from typing import Optional, Tuple
import torch


def block_attn_mask_to_ragged_lut(
    block_attn_mask: torch.Tensor,
    num_heads: Optional[int] = None,
    return_none_if_dense: bool = False,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Convert a dense block attention mask to a ragged look-up table of KV block
    indices per (batch, head, q_block). Used for block-sparse attention with no
    per-iteration branching in the kernel.

    block_attn_mask: Either (batch, num_q_blocks, num_kv_blocks) boolean for
        same mask for all heads, or (batch, num_heads, num_q_blocks, num_kv_blocks)
        for per-head masks. True = may attend, False = must not attend.
    num_heads: Required when block_attn_mask is 3D (number of Q heads). Ignored when 4D.
    return_none_if_dense: If True and the mask is all True (dense), return None so the
        caller can pass block_lut=None to fav3_sage_wrapper_func and use the dense path.
        Avoids building a very large LUT that can trigger munmap_chunk on MI300X/ROCm.
    Returns:
        kv_block_indices: 1D int32, concatenation of all KV block index lists.
        lut_start: 1D int32, length batch * num_heads * num_q_blocks. Index
            idx = batch_idx * (num_heads * num_q_blocks) + head_idx * num_q_blocks + q_block_idx.
        lut_count: 1D int32, same length as lut_start.
        When return_none_if_dense is True and the mask is all True, returns None instead.
    """
    device = block_attn_mask.device

    # 3D -> 4D: expand and fall through to 4D path
    if block_attn_mask.dim() == 3:
        if num_heads is None:
            raise ValueError("num_heads must be provided when block_attn_mask is 3D")
        batch, num_q_blocks, num_kv_blocks = block_attn_mask.shape
        if return_none_if_dense and block_attn_mask.all():
            return None
        block_attn_mask = block_attn_mask.unsqueeze(1).expand(
            batch, num_heads, num_q_blocks, num_kv_blocks
        )

    # 4D: (batch, num_heads, num_q_blocks, num_kv_blocks) — GPU vectorized path
    batch, num_heads, num_q_blocks, num_kv_blocks = block_attn_mask.shape
    if return_none_if_dense and block_attn_mask.all():
        return None

    counts = block_attn_mask.long().sum(dim=-1)
    lut_count = counts.reshape(-1).to(torch.int32)
    lut_start = (torch.cumsum(lut_count, dim=0) - lut_count).to(torch.int32)

    if lut_count.sum() == 0:
        kv_block_indices = torch.zeros(1, dtype=torch.int32, device=device)
        return kv_block_indices, lut_start, lut_count

    b_idx, h_idx, qb_idx, kb_idx = block_attn_mask.nonzero(as_tuple=True)
    linear = b_idx * (num_heads * num_q_blocks) + h_idx * num_q_blocks + qb_idx
    # Lexicographic sort (linear, kb_idx) so last K block is last per segment
    sort_key = linear * (num_kv_blocks + 1) + kb_idx
    sort_idx = sort_key.argsort(stable=True)
    kv_block_indices = kb_idx[sort_idx].to(torch.int32)

    return kv_block_indices, lut_start, lut_count
