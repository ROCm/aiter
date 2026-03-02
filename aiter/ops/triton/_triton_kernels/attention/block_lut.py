# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Triton kernel to build the block-sparse LUT (kv_block_indices) from a 4D block
attention mask without using nonzero or argsort.
"""

import torch
import triton
import triton.language as tl


@triton.jit()
def _block_attn_mask_to_lut_kernel(
    mask_ptr,
    lut_start_ptr,
    lut_count_ptr,
    kv_block_indices_ptr,
    stride_mask_b,
    stride_mask_h,
    stride_mask_qb,
    stride_mask_kb,
    num_heads,
    num_q_blocks,
    num_kv_blocks,
    BLOCK_KB: tl.constexpr,
):
    """
    Each program handles one (batch, head, q_block) row. It scans mask[b, h, qb, :]
    and writes the indices kb where the mask is True into the segment
    kv_block_indices[lut_start[linear_idx] : lut_start[linear_idx] + lut_count[linear_idx]].
    """
    linear_idx = tl.program_id(0)
    num_entries = num_heads * num_q_blocks
    # Decode (b, h, qb) from linear_idx = b * (num_heads * num_q_blocks) + h * num_q_blocks + qb
    b = linear_idx // num_entries
    remainder = linear_idx % num_entries
    h = remainder // num_q_blocks
    qb = remainder % num_q_blocks

    base = tl.load(lut_start_ptr + linear_idx)

    # Running write offset for this program's segment
    write_offset = 0

    for start_kb in range(0, num_kv_blocks, BLOCK_KB):
        kb_offs = start_kb + tl.arange(0, BLOCK_KB)
        in_bounds = kb_offs < num_kv_blocks

        # Row offset for (b, h, qb): mask[b, h, qb, start_kb : start_kb+BLOCK_KB]
        row_base = (
            b * stride_mask_b + h * stride_mask_h + qb * stride_mask_qb
        )
        mask_ptrs = mask_ptr + row_base + kb_offs * stride_mask_kb
        # Load mask chunk; bool loads as uint8/int8, non-zero = True
        mask_chunk = tl.load(mask_ptrs, mask=in_bounds, other=0)

        # Vectorized: mask_vals is 1 where we write, 0 otherwise
        mask_vals = (mask_chunk != 0).to(tl.int32)
        # Only count in-bounds positions
        mask_vals = tl.where(in_bounds, mask_vals, 0)
        cumsum = tl.cumsum(mask_vals, axis=0)
        # Store positions for this chunk: base + (cumsum - 1) where mask_vals
        store_offsets = base + write_offset + cumsum - 1
        chunk_kb = (start_kb + tl.arange(0, BLOCK_KB)).to(tl.int32)
        tl.store(
            kv_block_indices_ptr + store_offsets,
            chunk_kb,
            mask=mask_vals != 0,
        )
        write_offset = write_offset + tl.sum(mask_vals)

    # No return; kv_block_indices is written in place


def block_attn_mask_to_lut_kernel(
    block_attn_mask: torch.Tensor,
    lut_start: torch.Tensor,
    lut_count: torch.Tensor,
    kv_block_indices: torch.Tensor,
    BLOCK_KB: int = 128,
):
    """
    Launch the LUT-fill kernel. Caller must ensure block_attn_mask is 4D
    (batch, num_heads, num_q_blocks, num_kv_blocks) and kv_block_indices has
    length lut_count.sum().
    """
    batch, num_heads, num_q_blocks, num_kv_blocks = block_attn_mask.shape
    num_programs = batch * num_heads * num_q_blocks

    grid = (num_programs,)
    _block_attn_mask_to_lut_kernel[grid](
        block_attn_mask,
        lut_start,
        lut_count,
        kv_block_indices,
        stride_mask_b=block_attn_mask.stride(0),
        stride_mask_h=block_attn_mask.stride(1),
        stride_mask_qb=block_attn_mask.stride(2),
        stride_mask_kb=block_attn_mask.stride(3),
        num_heads=num_heads,
        num_q_blocks=num_q_blocks,
        num_kv_blocks=num_kv_blocks,
        BLOCK_KB=BLOCK_KB,
    )
