import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
from typing import Any, Callable, Dict, Optional, Tuple
from aiter.test_common import checkAllclose, perftest
from aiter.fused_moe_bf16_asm import moe_sorting_ck, fused_topk

BLOCK_SIZE_M = 32


@perftest(num_iters=3,num_warmup=0)
def test_moe_sorting_naive(topk_ids: torch.Tensor,
                      topk_weights: torch.Tensor,
                      num_experts: int,
                      expert_mask = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    block_size = BLOCK_SIZE_M
    
    device = topk_ids.device
    M, topk = topk_ids.shape
    topk = topk_ids.shape[1]
    max_num_tokens_padded = topk_ids.numel() + num_experts * block_size - topk
    max_num_m_blocks = int((max_num_tokens_padded+block_size-1)//block_size)
    init_val = topk << 24 | M
    sorted_ids = torch.full((max_num_tokens_padded, ), init_val,
                             dtype=torch.int32,
                             device=device)
    sorted_weights = torch.empty((max_num_tokens_padded, ),
                                 dtype=torch.float,
                                 device=device)
    sorted_expert_ids = torch.full((max_num_m_blocks, ), -1,
                                    dtype=torch.int32,
                                    device=device)
    num_tokens_post_pad = torch.empty((1),
                                      dtype=torch.int32,
                                      device=device)
    
    sorted_ids_begin = 0
    sorted_expert_ids_begin = 0
    skip_expert_num = 0
    for expertId in range(num_experts):
        if expert_mask != None and expert_mask[expertId] == 0:
            skip_expert_num += 1
            continue
        token_id, topk_id = torch.where(topk_ids == expertId)
        tokensNum = token_id.numel()
        sorted_expert_ids_num = (tokensNum + block_size - 1) // block_size
        tokensNumPad = sorted_expert_ids_num * block_size
        sorted_ids[sorted_ids_begin:sorted_ids_begin+tokensNum] = topk_id << 24 | token_id
        sorted_weights[sorted_ids_begin:sorted_ids_begin+tokensNum] = topk_weights[token_id, topk_id]
        sorted_ids_begin = sorted_ids_begin+tokensNumPad
        sorted_expert_ids[sorted_expert_ids_begin:sorted_expert_ids_begin+sorted_expert_ids_num] = expertId-skip_expert_num
        sorted_expert_ids_begin = sorted_expert_ids_begin+sorted_expert_ids_num

    num_tokens_post_pad[0] = sorted_ids_begin

    return sorted_ids, sorted_weights, sorted_expert_ids, num_tokens_post_pad


@perftest()
def test_moe_sorting_ck(topk_ids, topk_weights, num_experts, model_dim, moebuf_dtype, expert_mask = None):
    return moe_sorting_ck(topk_ids, topk_weights, num_experts, model_dim, moebuf_dtype, expert_mask=expert_mask)


def test_moe_sorting(dtype, token, model_dim, inter_dim, E, topk, has_expert_mask = False):
    dim = (token, model_dim, inter_dim)
    input = torch.randn((token, model_dim), dtype=dtype, device="cuda")
    score = torch.rand((token, E), device="cuda", dtype=dtype)

    topk_weights, topk_ids = fused_topk(input, score, topk, True)

    expert_mask = torch.randint(0, 2, (E,), dtype=topk_ids.dtype, device="cuda") if has_expert_mask else None

    (sorted_ids_a,
     sorted_weights_a,
     sorted_expert_ids_a,
     num_tokens_post_padded_a), avg_a = test_moe_sorting_naive(
        topk_ids, topk_weights, E, expert_mask)

    (sorted_ids_b,
     sorted_weights_b,
     sorted_expert_ids_b,
     num_tokens_post_padded_b,
     moe_buf), avg_b = test_moe_sorting_ck(topk_ids, topk_weights, E,
                                           model_dim, dtype, expert_mask)

    print(
        f"[perf] {token=}, {model_dim=}, {inter_dim=}, {E=}, {topk=}, dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}")
    checkAllclose(num_tokens_post_padded_a, num_tokens_post_padded_b, atol=0, msg='num_tokens_post_padded')
    mask = sorted_ids_a != (topk << 24 | token)
    num_tokens_post_pad = num_tokens_post_padded_a.item()
    checkAllclose(sorted_ids_a[:num_tokens_post_pad],
                  sorted_ids_b[:num_tokens_post_pad], msg='sorted_ids')
    checkAllclose(sorted_weights_a[mask],
                  sorted_weights_b[mask], msg='sorted_weights')
    
    expert_mask = sorted_expert_ids_a != -1
    checkAllclose(sorted_expert_ids_a[expert_mask],
                  sorted_expert_ids_b[expert_mask], msg='sorted_expert_ids')
    print(f"[passed~]")


print('test test_moe_sorting, no expert mask')
for dtype in [torch.bfloat16]:
    for m in [1, 7, 31, 64, 128, 256]:
        for E in [32, 256]:
            for top in [5, 8]:
                test_moe_sorting(dtype, m, 4096, 4096, E, top)

print('test test_moe_sorting, with expert mask')
for dtype in [torch.bfloat16]:
    for m in [1, 7, 31, 64, 128, 256]:
        for E in [32, 256]:
            for top in [5, 8]:
                test_moe_sorting(dtype, m, 4096, 4096, E, top, has_expert_mask=True)