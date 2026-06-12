# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Generic Triton kernels for block-sparse attention mask construction:

  * :func:`block_attn_mask_to_lut_kernel` -- build the block-sparse LUT
    (kv_block_indices) from a 4D block attention mask without nonzero/argsort.
  * :func:`triton_fill_block_map_kernel` / :func:`triton_fill_causal_mask_kernel`
    -- SpargeAttn-style block-sparse mask construction.

Host wrappers live in ``aiter/ops/triton/attention/block_sparse.py``. The VFA
``m_init`` estimator kernel lives in
``aiter/ops/triton/_triton_kernels/attention/vfa.py``.
"""

import triton
import triton.language as tl


@triton.jit()
def block_attn_mask_to_lut_kernel(
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
        row_base = b * stride_mask_b + h * stride_mask_h + qb * stride_mask_qb
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


# ----------------------------------------------------------------------------
# SpargeAttn-style block-sparse mask construction kernels.
#
#   * ``triton_fill_block_map_kernel`` -- scatters the top-``num_to_select``
#     ranked K blocks per query block into a dense (B, H, Q, K) bool map.
#   * ``triton_fill_causal_mask_kernel`` -- materializes a block-level causal
#     mask accounting for a (possibly non-unit) query/key block-size ratio.
#
# The per-block mean-pooling proxy kernel (``triton_bmm_pool_sim_simmean``) that
# ranks candidate blocks lives in ``aiter/ops/triton/_triton_kernels/pool.py``
# (host wrappers in ``aiter/ops/triton/pool.py``).
#
# Reference: "SpargeAttn: Accurate Sparse Attention Accelerating Any Model
# Inference" (https://arxiv.org/abs/2502.18137).
# ----------------------------------------------------------------------------
@triton.jit
def triton_fill_block_map_kernel(
    final_map,
    num_to_select,
    sorted_indices,
    NK: tl.constexpr,
):
    b, h, q = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    B, H, Q = tl.num_programs(0), tl.num_programs(1), tl.num_programs(2)
    cur_num_to_select = tl.load(num_to_select + b * H * Q + h * Q + q)
    cur_sorted_idx_ptr = sorted_indices + b * H * Q * NK + h * Q * NK + q * NK
    cur_final_map_ptr = final_map + b * H * Q * NK + h * Q * NK + q * NK
    # Always select at least one block per query block.
    cur_num_to_select = (cur_num_to_select + 1) if cur_num_to_select == 0 else cur_num_to_select
    for i in range(cur_num_to_select):
        cur_idx = tl.load(cur_sorted_idx_ptr + i)
        tl.store(cur_final_map_ptr + cur_idx, 1)


@triton.jit
def triton_fill_causal_mask_kernel(mask, BqdivBk):
    q, k = tl.program_id(0), tl.program_id(1)
    Q, K = tl.num_programs(0), tl.num_programs(1)
    if k >= (q + 1) * BqdivBk:
        tl.store(mask + q * K + k, 0)
    else:
        tl.store(mask + q * K + k, 1)
