import triton
import torch
import triton.language as tl
import pytest
from typing import Any, Dict, Optional, Tuple
import os
import json
import functools
import argparse
import sys

from aiter.ops.triton.moe_align_block_size import moe_align_block_size_triton

def _torch_moe_align_block_size(topk_ids: torch.Tensor, num_experts: int, top_k: int, block_size: int,
                          sorted_token_ids: torch.Tensor, expert_ids: torch.Tensor,
                          num_tokens_post_pad: torch.Tensor) -> None:
    M, top_k = topk_ids.shape

    expert_to_tokens = [[] for _ in range(num_experts)]
    # For each token, for each selected expert, we append (token_id, expert)
    for token_id in range(M):
        for j in range(top_k):
            e_id = topk_ids[token_id, j].item()
            expert_to_tokens[e_id].append(token_id * top_k + j)

    # Reorder tokens block by block, padding if needed
    reordered_token_ids = []
    reordered_expert_ids = []

    tot_num_tokens = topk_ids.numel()
    for e_id in range(num_experts):
        tokens_for_expert = expert_to_tokens[e_id]
        num_tokens = len(tokens_for_expert)

        n_blocks = ((num_tokens + block_size - 1) // block_size)
        # If not a multiple of block_size, pad up to the next multiple
        padded_size = n_blocks * block_size

        # Reorder all actual tokens for expert e_id
        reordered_token_ids.extend(tokens_for_expert)
        # reordered_expert_ids.extend([e_id]*num_tokens)
        reordered_expert_ids.extend([e_id] * n_blocks)

        # Pad with dummy token_id = -1 (or any sentinel), if needed
        if padded_size > num_tokens:
            pad_count = padded_size - num_tokens
            reordered_token_ids.extend([tot_num_tokens] * pad_count)

    token_length = len(reordered_token_ids)
    expert_length = len(reordered_expert_ids)

    sorted_token_ids[:token_length] = torch.tensor(reordered_token_ids, dtype=sorted_token_ids.dtype,
                                                   device=sorted_token_ids.device)
    expert_ids[:expert_length] = torch.tensor(reordered_expert_ids, dtype=expert_ids.dtype, device=expert_ids.device)

    # Fill remainder with -1 if these arrays are bigger than total_length
    if token_length < sorted_token_ids.numel():
        sorted_token_ids[token_length:] = tot_num_tokens

    num_tokens_post_pad.fill_(token_length)

def torch_moe_align_block_size(
    topk_ids: torch.Tensor, #[num_tkns, num_experts]
    num_experts: int,
    block_size: int,
):
    top_k = topk_ids.shape[1]
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded, ), dtype=torch.int32,
                             device=topk_ids.device)
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size) 
    expert_ids = torch.empty((max_num_m_blocks), dtype=torch.int32, device=topk_ids.device)
    sorted_ids.fill_(topk_ids.numel())
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    _torch_moe_align_block_size(topk_ids, num_experts, top_k, block_size, sorted_ids, expert_ids, num_tokens_post_pad)

    return sorted_ids, expert_ids, num_tokens_post_pad

def input_helper(M: int, E: int, top_k: int):
    values = torch.randn(M, E, dtype=torch.float16, device='cuda')

    softmax_vals = torch.softmax(values, dim=1)
    _, topk_ids = torch.topk(softmax_vals, k=top_k, dim=1)

    return topk_ids

def triton_moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded, ),
                                dtype=torch.int32,
                                device=topk_ids.device)
    sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty((max_num_m_blocks, ),
                                dtype=torch.int32,
                                device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1),
                                        dtype=torch.int32,
                                        device=topk_ids.device)
    
    moe_align_block_size_triton(topk_ids,
                                num_experts,
                                block_size,
                                sorted_ids,
                                expert_ids,
                                num_tokens_post_pad,
                            )

    return sorted_ids, expert_ids, num_tokens_post_pad

@pytest.mark.parametrize("M, E, top_k, block_size", 
    [(1, 2, 1, 16),
    (4, 2, 1, 16),
    (8, 2, 1, 16),
    (16, 2, 1, 16),
    (16, 8, 4, 16),
    (16, 8, 4, 32),
    (16, 8, 4, 128),
    (16, 48, 4, 64),
    (32, 224, 12, 128),
    ],
)
def test_correctness(M: int, E: int, top_k: int, block_size: int):
    topk_ids = input_helper(M, E, top_k)

    tri_sorted_ids, tri_expert_ids, tri_num_tokens_post_pad = triton_moe_align_block_size(topk_ids, block_size, E)
    torch_sorted_ids, torch_expert_ids, torch_num_tokens_post_pad = torch_moe_align_block_size(topk_ids, E, block_size)
    
    torch.eq(tri_sorted_ids, torch_sorted_ids)
    torch.eq(tri_num_tokens_post_pad, torch_num_tokens_post_pad)
    torch.eq(tri_expert_ids[:E], torch_expert_ids[:E])


