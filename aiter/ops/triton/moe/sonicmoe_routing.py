# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import math

import torch
import triton

from aiter.ops.triton._triton_kernels.moe.moe_routing.bitmatrix_sonicmoe import (
    _sonicmoe_bitmatrix_metadata_compute_stage1,
    _sonicmoe_bitmatrix_metadata_compute_stage2,
)
from aiter.ops.triton._triton_kernels.moe.moe_routing.routing_sonicmoe import (
    _sonicmoe_compute_col_partial_sum_kernel,
    _sonicmoe_general_compute_col_partial_sum_kernel,
    _sonicmoe_general_metadata_compute_stage2,
    _sonicmoe_token_offset_searchsorted_kernel,
)
from aiter.ops.triton.utils.sonicmoe_config_utils import get_sonicmoe_kernel_config


@torch.library.custom_op(
    "triton_kernels::TC_topk_router_metadata",
    mutates_args={
        "expert_frequency",
        "expert_frequency_offset",
        "x_gather_idx",
        "s_scatter_idx",
        "s_reverse_scatter_idx",
    },
)
def TC_topk_router_metadata_triton(
    topk_router_indices: torch.Tensor,
    E: int,
    expert_frequency: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    x_gather_idx: torch.Tensor,
    s_scatter_idx: torch.Tensor,
    s_reverse_scatter_idx: torch.Tensor,
) -> None:
    T, K = topk_router_indices.size()
    TK = T * K
    device = topk_router_indices.device
    E_POW2 = triton.next_power_of_2(E)
    K_POW2 = triton.next_power_of_2(K)
    config = get_sonicmoe_kernel_config("topk_routing")
    TOKENS_PER_BLOCK = config["ENTRIES_PER_TILE"] // K_POW2
    n_tiles = triton.cdiv(T, TOKENS_PER_BLOCK)

    # Transposed storage avoids cross-CTA histogram writes.
    col_partial_sum_trans = torch.empty(E, n_tiles, dtype=torch.int32, device=device)
    _sonicmoe_compute_col_partial_sum_kernel[(n_tiles,)](
        topk_router_indices,
        col_partial_sum_trans,
        T,
        E,
        n_tiles,
        TOKENS_PER_TILE=TOKENS_PER_BLOCK,
        K_POW2=K_POW2,
        K=K,
        E_POW2=E_POW2,
    )

    expert_frequency.copy_(col_partial_sum_trans.sum(dim=1, dtype=torch.int32))
    col_partial_sum = col_partial_sum_trans.T  # [n_tiles, E]

    _sonicmoe_bitmatrix_metadata_compute_stage1[(E + 2,)](
        expert_frequency,
        expert_frequency_offset,
        E,
        col_partial_sum,
        n_tiles,
        TK,
        BLOCK_M=config["PREFIX_BLOCK_M"],
        BLOCK_N=E_POW2,
    )

    _sonicmoe_bitmatrix_metadata_compute_stage2[(n_tiles,)](
        s_scatter_idx,
        s_reverse_scatter_idx,
        x_gather_idx,
        topk_router_indices,
        T,
        col_partial_sum,
        n_tiles,
        expert_frequency_offset[:E],
        K_POW2=K_POW2,
        TOKENS_PER_BLOCK=TOKENS_PER_BLOCK,
        K=K,
    )


@torch.library.custom_op(
    "triton_kernels::general_routing_router_metadata",
    mutates_args={
        "expert_frequency",
        "expert_frequency_offset",
        "x_gather_idx",
        "s_scatter_idx",
        "s_reverse_scatter_idx",
        "num_activated_expert_per_token_offset",
    },
)
def general_routing_router_metadata_triton(
    sorted_selected_T: torch.Tensor,
    selected_E: torch.Tensor,
    T: int,
    E: int,
    expert_frequency: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    x_gather_idx: torch.Tensor,
    s_scatter_idx: torch.Tensor,
    s_reverse_scatter_idx: torch.Tensor,
    num_activated_expert_per_token_offset: torch.Tensor,
) -> None:
    TK = selected_E.size(0)
    device = selected_E.device
    E_POW2 = triton.next_power_of_2(E)
    config = get_sonicmoe_kernel_config("general_routing")
    BLOCK_SIZE = config["BLOCK_SIZE"]
    n_tiles = triton.cdiv(TK, BLOCK_SIZE)

    col_partial_sum_trans = torch.empty(E, n_tiles, dtype=torch.int32, device=device)
    _sonicmoe_general_compute_col_partial_sum_kernel[(n_tiles,)](
        selected_E,
        col_partial_sum_trans,
        TK,
        E,
        n_tiles,
        BLOCK_SIZE=BLOCK_SIZE,
        E_POW2=E_POW2,
    )

    expert_frequency.copy_(col_partial_sum_trans.sum(dim=1, dtype=torch.int32))
    col_partial_sum = col_partial_sum_trans.T  # [n_tiles, E], strides (1, n_tiles)

    _sonicmoe_bitmatrix_metadata_compute_stage1[(E + 2,)](
        expert_frequency,
        expert_frequency_offset,
        E,
        col_partial_sum,
        n_tiles,
        TK,
        BLOCK_M=config["PREFIX_BLOCK_M"],
        BLOCK_N=E_POW2,
    )

    _sonicmoe_general_metadata_compute_stage2[(n_tiles,)](
        s_scatter_idx,
        s_reverse_scatter_idx,
        x_gather_idx,
        selected_E,
        sorted_selected_T,
        TK,
        col_partial_sum,
        n_tiles,
        expert_frequency_offset[:E],
        BLOCK_SIZE=BLOCK_SIZE,
    )

    N_ITERS = max(1, math.ceil(math.log2(TK + 1)))
    TOKEN_BLOCK = config["TOKEN_SEARCH_BLOCK"]
    n_token_blocks = triton.cdiv(T + 1, TOKEN_BLOCK)
    _sonicmoe_token_offset_searchsorted_kernel[(n_token_blocks,)](
        sorted_selected_T,
        num_activated_expert_per_token_offset,
        T,
        TK,
        BLOCK_SIZE=TOKEN_BLOCK,
        N_ITERS=N_ITERS,
    )
