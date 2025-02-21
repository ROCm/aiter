import triton
import torch
import triton.language as tl
import pytest
from typing import Any, Dict, Optional
import os
import json
import functools
import argparse
import sys

from aiter.ops.triton.moe_op import moe_triton

def torch_moe(a, b, c, topk_ids, topk_weights, routed_weight, sorted_token_ids, expert_ids, num_tokens_post_padded,
              a_scale, b_scale, dtype, fp8):
    E, N, K = b.shape
    M, topk, _ = c.shape
    c = c.reshape(-1, c.shape[2])

    if fp8:
        a = a.to(dtype)

    for e in range(E):
        token_ids = (topk_ids == e).any(dim=-1)
        flat_topk_ids = topk_ids.view(-1)
        flat_token_ids = torch.arange(topk_ids.numel(), device=topk_ids.device)
        c_token_ids = flat_token_ids[flat_topk_ids == e]

        b_e = b[e]
        a_e = a[token_ids, :]

        if fp8:
            b_e = b_e.to(dtype)

        acc = torch.matmul(a_e, b_e.T)
        if routed_weight:
            acc = acc * topk_weights.view(-1)[c_token_ids].unsqueeze(-1)

        if fp8:
            acc = (acc * a_scale * b_scale[e]).to(dtype)

        c[c_token_ids, :] = acc

    c = c.reshape(M, topk, N)

    return c

def _moe_align_block_size(topk_ids: torch.Tensor, num_experts: int, top_k: int, block_size: int,
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

        # Pad with dummy token_id = topk_ids.numel()
        if padded_size > num_tokens:
            pad_count = padded_size - num_tokens
            reordered_token_ids.extend([topk_ids.numel()] * pad_count)

    token_length = len(reordered_token_ids)
    expert_length = len(reordered_expert_ids)

    sorted_token_ids[:token_length] = torch.tensor(reordered_token_ids, dtype=sorted_token_ids.dtype,
                                                   device=sorted_token_ids.device)
    expert_ids[:expert_length] = torch.tensor(reordered_expert_ids, dtype=expert_ids.dtype, device=expert_ids.device)

    # Fill remainder with topk_ids.numel() if these arrays are bigger than total_length
    if token_length < sorted_token_ids.numel():
        sorted_token_ids[token_length:] = topk_ids.numel() 
    if expert_length < expert_ids.numel():
        expert_ids[expert_length:] = topk_ids.numel()

    num_tokens_post_pad.fill_(token_length)


def moe_align_block_size(topk_ids: torch.Tensor, block_size: int,
                         num_experts: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding, ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]], block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts, with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible by block_size for proper block matrix operations.
    """
    top_k = topk_ids.shape[1]
    sorted_ids = torch.empty((topk_ids.numel() + num_experts * (block_size - 1), ), dtype=torch.int32,
                             device=topk_ids.device)
    expert_ids = torch.empty((topk_ids.numel() + num_experts, ), dtype=torch.int32, device=topk_ids.device)
    sorted_ids.fill_(topk_ids.numel())
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    _moe_align_block_size(topk_ids, num_experts, top_k, block_size, sorted_ids, expert_ids, num_tokens_post_pad)

    return sorted_ids, expert_ids, num_tokens_post_pad

def get_default_config() -> Dict[str, int]:
    config = {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}
    return config

def input_helper(M: int, N: int, K: int, top_k: int, E: int, routed_weight: bool, compute_type, fp8: bool):
    if fp8:
        a = torch.randn((M, K), dtype=compute_type, device='cuda')
        a = a.to(torch.float8_e4m3fnuz)
        b = torch.rand((E, N, K), dtype=compute_type, device='cuda')
        b = b.to(torch.float8_e4m3fnuz)
    else:
        b = torch.randn((E, N, K), dtype=compute_type, device='cuda')
        a = torch.randn((M, K), dtype=compute_type, device='cuda')
    c = torch.zeros((M, top_k, N), dtype=compute_type, device='cuda')

    if fp8:
        a_scale = torch.randn((1), dtype=torch.float32, device='cuda')
        b_scale = torch.randn((E), dtype=torch.float32, device='cuda')
    else:
        a_scale = None
        b_scale = None

    values = torch.randn(M, E, dtype=compute_type, device='cuda')

    softmax_vals = torch.softmax(values, dim=1)
    topk_weights, topk_ids = torch.topk(softmax_vals, k=top_k, dim=1)

    config = get_default_config()
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, config['BLOCK_SIZE_M'], E)

    if not routed_weight:
        return a, b, c, None, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded, config, a_scale, b_scale

    return a, b, c, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded, config, a_scale, b_scale


torch_to_tl_dtype = {torch.float16 : tl.float16, torch.bfloat16 : tl.bfloat16, torch.float32 : tl.float32}

@pytest.mark.parametrize("M, N, K, top_k, E", [(64, 14336, 4096, 2, 8), (16, 14336, 1, 2, 4), (4, 4, 8, 1, 2),
                                               (1, 14336, 128, 2, 4), (3, 14336, 128, 2, 4), (16, 14336, 128, 1, 4),
                                               (16, 14336, 128, 1, 1), (64, 7186, 128, 2, 8), (64, 3584, 128, 2, 8),
                                               (64, 1792, 128, 2, 8), (64, 64, 128, 2, 8), (1, 1024, 16384, 1, 2)])
@pytest.mark.parametrize('routed_weight', [True, False])
@pytest.mark.parametrize('fp8', [(False)]) #TODO add support for fp8
def test_correctness(M: int, N: int, K: int, top_k: int, E: int, routed_weight: bool, fp8: bool, compute_type=torch.bfloat16):
    #torch.manual_seed(20)
    a, b, triton_out, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded, config, a_scale, b_scale = input_helper(
        M, N, K, top_k, E, routed_weight=routed_weight, compute_type=compute_type, fp8=fp8)

    print(f"sorted_token_ids={sorted_token_ids}")
    print(f"expert_ids={expert_ids}")
    print(f"num_tokens_post_padded={num_tokens_post_padded}")
    moe_triton(a, b, triton_out, a_scale, b_scale, topk_weights, topk_ids, sorted_token_ids, expert_ids,
                       num_tokens_post_padded, routed_weight, top_k, config, torch_to_tl_dtype[compute_type], fp8, False)

    torch_out = torch.empty_like(triton_out)
    torch_out = torch_moe(a, b, torch_out, topk_ids, topk_weights, routed_weight, sorted_token_ids, expert_ids,
                        num_tokens_post_padded, a_scale, b_scale, compute_type, fp8)

    # Validate correctness
    torch.testing.assert_close(triton_out, torch_out, atol=1e-2, rtol=1e-2)