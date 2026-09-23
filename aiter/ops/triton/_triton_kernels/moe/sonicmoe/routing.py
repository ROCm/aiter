# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.moe.moe_routing.utils import keyed_add
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_col_partial_sum_repr = make_kernel_repr(
    "sonicmoe_col_partial_sum",
    ["E", "TOKENS_PER_TILE", "K_POW2", "K", "E_POW2"],
)
_general_col_partial_sum_repr = make_kernel_repr(
    "sonicmoe_general_col_partial_sum", ["E", "BLOCK_SIZE", "E_POW2"]
)
_general_metadata_stage2_repr = make_kernel_repr(
    "sonicmoe_general_metadata_stage2", ["BLOCK_SIZE"]
)
_token_offset_searchsorted_repr = make_kernel_repr(
    "sonicmoe_token_offset_searchsorted", ["BLOCK_SIZE", "N_ITERS"]
)


@triton.jit(repr=_col_partial_sum_repr)
def _compute_col_partial_sum_kernel(
    topk_indices_ptr,
    partial_sum_ptr,
    T,
    E: tl.constexpr,
    n_tiles,
    TOKENS_PER_TILE: tl.constexpr,
    K_POW2: tl.constexpr,  # next_power_of_2(K),
    K: tl.constexpr,  # actual number of experts per token
    E_POW2: tl.constexpr,  # next_power_of_2(E)
):
    # Each CTA builds one tile's per-expert histogram.
    tile_id = tl.program_id(0)

    for e_start in tl.static_range(0, E, E_POW2):
        e_offs = e_start + tl.arange(0, E_POW2)
        tl.store(
            partial_sum_ptr + e_offs * n_tiles + tile_id,
            tl.zeros([E_POW2], tl.int32),
            mask=e_offs < E,
        )

    tok_offs = tile_id * TOKENS_PER_TILE + tl.arange(0, TOKENS_PER_TILE)
    k_offs = tl.arange(0, K_POW2)
    tok_mask = tok_offs < T

    load_mask = tok_mask[:, None] & (k_offs[None, :] < K)
    safe_k = tl.minimum(k_offs, K - 1)  # avoid OOB when k_offs >= K
    expert_ids = tl.load(
        topk_indices_ptr + tok_offs[:, None] * K + safe_k[None, :],
        mask=load_mask,
        other=-1,
    )

    flat_experts = tl.reshape(expert_ids, [TOKENS_PER_TILE * K_POW2])
    flat_mask = tl.reshape(load_mask, [TOKENS_PER_TILE * K_POW2])
    safe_experts = tl.where(flat_mask, flat_experts, 0)

    tl.atomic_add(
        partial_sum_ptr + safe_experts * n_tiles + tile_id,
        tl.full([TOKENS_PER_TILE * K_POW2], 1, dtype=tl.int32),
        mask=flat_mask,
    )


@triton.jit(repr=_general_col_partial_sum_repr)
def _general_compute_col_partial_sum_kernel(
    selected_E_ptr,
    partial_sum_ptr,  # [E, n_tiles], column-major per tile
    TK,
    E: tl.constexpr,
    n_tiles,
    BLOCK_SIZE: tl.constexpr,
    E_POW2: tl.constexpr,
):
    tile_id = tl.program_id(0)

    for e_start in tl.static_range(0, E, E_POW2):
        e_offs = e_start + tl.arange(0, E_POW2)
        tl.store(
            partial_sum_ptr + e_offs * n_tiles + tile_id,
            tl.zeros([E_POW2], tl.int32),
            mask=e_offs < E,
        )

    offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < TK
    expert_ids = tl.load(selected_E_ptr + offs, mask=mask, other=-1)

    safe_experts = tl.where(mask, expert_ids, 0)
    tl.atomic_add(
        partial_sum_ptr + safe_experts * n_tiles + tile_id,
        tl.full([BLOCK_SIZE], 1, dtype=tl.int32),
        mask=mask,
    )


@triton.jit(repr=_general_metadata_stage2_repr)
def _general_metadata_compute_stage2(
    s_scatter_idx_ptr,
    s_reverse_scatter_idx_ptr,
    x_gather_idx_ptr,
    selected_E_ptr,
    sorted_selected_T_ptr,
    TK,
    partial_sum_ptr,  # [n_tiles, E] with strides (1, n_tiles)
    n_tiles,
    expert_offs_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE <= 32768)

    pid_m = tl.program_id(0)
    offs_local = tl.arange(0, BLOCK_SIZE)
    offs_global = pid_m * BLOCK_SIZE + offs_local
    mask = offs_global < TK

    expert = tl.load(selected_E_ptr + offs_global, mask=mask, other=-1).to(tl.uint32)

    # Pack expert and local offset into uint32 for a stable local sort.
    kv_pairs = tl.sort(((expert << 16) | offs_local).to(tl.uint32), 0)
    expert = kv_pairs >> 16
    mask = expert != 0xFFFF

    scan_input = (kv_pairs & 0xFFFF0000) | 0x00000001
    inclusive_run_lengths = tl.associative_scan(scan_input, 0, keyed_add)
    within_expert_rank = (inclusive_run_lengths - 1) & 0xFFFF

    s_reverse_scatter_val = tl.load(
        partial_sum_ptr + pid_m + expert * n_tiles, mask=mask
    )
    s_reverse_scatter_val += tl.load(expert_offs_ptr + expert, mask=mask)
    s_reverse_scatter_val += within_expert_rank

    presort_offs = kv_pairs & 0xFFFF
    entry_idx = pid_m * BLOCK_SIZE + presort_offs
    token_idx = tl.load(sorted_selected_T_ptr + entry_idx, mask=mask)

    tl.store(s_reverse_scatter_idx_ptr + entry_idx, s_reverse_scatter_val, mask=mask)
    tl.store(s_scatter_idx_ptr + s_reverse_scatter_val, entry_idx, mask=mask)
    tl.store(x_gather_idx_ptr + s_reverse_scatter_val, token_idx, mask=mask)


@triton.jit(repr=_token_offset_searchsorted_repr)
def _token_offset_searchsorted_kernel(
    sorted_T_ptr,  # [TK] int32, sorted ascending
    offset_ptr,  # [T+1] int32, output
    T,  # number of tokens
    TK,  # length of sorted_T
    BLOCK_SIZE: tl.constexpr,
    N_ITERS: tl.constexpr,  # ceil(log2(TK + 1)), controls binary search depth
):
    pid = tl.program_id(0)
    t_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = t_offs <= T  # T+1 total values: offset[0], ..., offset[T]

    t_vals = t_offs.to(tl.int32)

    # Find the first sorted token index greater than or equal to each query.
    lo = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    hi = tl.full([BLOCK_SIZE], TK, dtype=tl.int32)

    for _ in tl.static_range(0, N_ITERS):
        mid = (lo + hi) >> 1
        safe_mid = tl.where(mid < TK, mid, 0)
        val = tl.load(sorted_T_ptr + safe_mid, mask=mask & (TK > 0), other=T)
        go_right = (val < t_vals) & (mid < TK)
        lo = tl.where(go_right, mid + 1, lo)
        hi = tl.where(go_right, hi, mid)

    tl.store(offset_ptr + t_offs, lo, mask=mask)
