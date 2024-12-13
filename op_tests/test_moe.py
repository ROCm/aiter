import torch
import torch.nn.functional as F
import triton.language as tl
import numpy as np
import sys
import os
from typing import Any, Callable, Dict, Optional, Tuple
from ater.test_common import checkAllclose, perftest
from ater.fused_moe_bf16_asm import asm_moe, torch_moe, moe_sorting_ck
from ater.fused_moe_gelu import fused_topk, moe_align_block_size, fused_experts
from ater.ops.shuffle import shuffle_weight
from test_smoothquant import pertoken_quant

BLOCK_SIZE_M = 32


@perftest()
def moe_sorting_vllm(topk_ids: torch.Tensor,
                     block_size: int,
                     num_experts: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    topk = topk_ids.shape[1]
    # max_num_tokens_padded = (
    #     topk_ids.numel() + num_experts * (block_size - 1)+block_size-1)//block_size*block_size
    max_num_tokens_padded = topk_ids.numel() + num_experts * block_size - topk
    # max_num_tokens_padded = int(
    #     (max_num_tokens_padded+block_size-1)//block_size*block_size)
    max_num_m_blocks = int((max_num_tokens_padded+block_size-1)//block_size)

    sorted_ids = torch.empty((max_num_tokens_padded, ),
                             dtype=torch.int32,
                             device=topk_ids.device)
    sorted_ids.fill_(topk_ids.shape[0]*topk)
    expert_ids = torch.empty((max_num_m_blocks, ),
                             dtype=torch.int32,
                             device=topk_ids.device)
    token_nums = torch.empty((max_num_m_blocks, ),
                             dtype=torch.int32,
                             device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1),
                                      dtype=torch.int32,
                                      device=topk_ids.device)
    ater.moe_align_block_size(topk_ids, num_experts, block_size, sorted_ids,
                              expert_ids, token_nums, num_tokens_post_pad)
    return sorted_ids, expert_ids, token_nums, num_tokens_post_pad


@perftest()
def moe_sorting_ck_test(topk_ids, topk_weights, num_experts, model_dim, moebuf_dtype):
    return moe_sorting_ck(topk_ids, topk_weights, num_experts, model_dim, moebuf_dtype)


def test_moe_sort(dtype, token, model_dim, inter_dim, E, topk):
    dim = (token, model_dim, inter_dim)
    input = torch.randn((token, model_dim), dtype=dtype, device="cuda")
    w1 = torch.randn((E, inter_dim, model_dim), dtype=dtype, device="cuda")
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype, device="cuda")
    score = torch.randn((token, E), device="cuda", dtype=dtype)

    topk_weights, topk_ids = fused_topk(input, score, topk, True)
    # print(f'{topk_weights=}')
    # print(f'{topk_ids=}')

    (sorted_ids_a,
     sorted_expert_ids_a,
     token_nums,
     num_tokens_post_padded_a), avg_a = moe_sorting_vllm(
        topk_ids, BLOCK_SIZE_M, E)
    sorted_ids_a = sorted_ids_a//topk

    (sorted_ids_b,
     sorted_weights_b,
     sorted_expert_ids_b,
     num_tokens_post_padded_b,
     moe_buf), avg_b = moe_sorting_ck_test(topk_ids, topk_weights, E,
                                           model_dim, dtype)
    # print(f'{num_tokens_post_padded_a=}')
    # print(f'{num_tokens_post_padded_b=}')
    # print(f'{sorted_ids_a.shape=}')
    # print(f'{sorted_ids_b.shape=}')
    # pad_a = (sorted_ids_a.shape[0]+BLOCK_SIZE_M -
    #          1)//BLOCK_SIZE_M*BLOCK_SIZE_M-sorted_ids_a.shape[0]
    # pad_b = (sorted_ids_b.shape[0]+BLOCK_SIZE_M -
    #          1)//BLOCK_SIZE_M*BLOCK_SIZE_M-sorted_ids_b.shape[0]
    # print(f'{F.pad(sorted_ids_a,(0,pad_a), "constant", 0).view(-1,BLOCK_SIZE_M)=}')
    # print(f'{F.pad(sorted_ids_b,(0,pad_b), "constant", 0).view(-1,BLOCK_SIZE_M)=}')
    # print(f'{sorted_expert_ids_a=}')
    # print(f'{sorted_expert_ids_b=}')
    # print(f'{moe_buf.max()=}')

    print(
        f"[perf] {token=}, {model_dim=}, {inter_dim=}, {E=}, {topk=}, dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}")
    if num_tokens_post_padded_a[0] != num_tokens_post_padded_b[0]:
        print("[F!!!]")
        return
    checkAllclose(num_tokens_post_padded_a, num_tokens_post_padded_b, atol=0)
    checkAllclose(sorted_ids_a[:num_tokens_post_padded_a[0]],
                  sorted_ids_b[:num_tokens_post_padded_b[0]])
    checkAllclose(sorted_expert_ids_a[:num_tokens_post_padded_a[0]//BLOCK_SIZE_M],
                  sorted_expert_ids_b[:num_tokens_post_padded_b[0]//BLOCK_SIZE_M])
    print(f"[passed~]")


# print('test test_moe_sort')
# for dtype in [torch.float16, torch.bfloat16][1:]:
#     for m in [1, 2, 4, 8, 16, 32, 64, 128, 256][3:]:
#         for dim in [4096, 8192, 16384, 32768, 65536][:-2]:
#             for hdim in [1024, 4096, 8192, 16384, 32768, 65536][:-2]:
#                 test_moe_sort(dtype, m, dim, hdim, 32, 5)


def permute_weight_a(x: torch.Tensor) -> torch.Tensor:
    # Hardcode BLOCK_K and BLOCK_N
    BK = 128
    BN = 128
    x_ = x
    x_ = x_.view(x.shape[0],
                 x.shape[1]//BN, BN//16, 16,
                 x.shape[2]//BK, BK//32, 4, 8)
    x_ = x_.permute(0, 1, 5, 2, 6, 4, 3, 7)
    x_ = x_.contiguous()
    x_ = x_.view(x.shape[0], x.shape[1], x.shape[2])
    return x_


@perftest()
def torch_moe_test(hidden_states, w1, w2, topk_weight, topk_ids,
                   # following for int8 quant
                   fc1_scale=None,  # [expert, inter_dim, 1]
                   fc2_scale=None,  # [expert, model_dim, 1]
                   fc1_smooth_scale=None,  # [expert, 1, model_dim]
                   fc2_smooth_scale=None,  # [expert, 1, inter_dim]
                   ):
    return torch_moe(hidden_states,
                     w1,
                     w2,
                     topk_weight,
                     topk_ids, fc1_scale, fc2_scale, fc1_smooth_scale, fc2_smooth_scale)


@perftest()
def asm_moe_test(hidden_states, w1, w2, topk_weight, topk_ids,
                 # following for int8 quant
                 fc1_scale=None,  # [expert, inter_dim, 1]
                 fc2_scale=None,  # [expert, model_dim, 1]
                 fc1_smooth_scale=None,  # [expert, 1, model_dim]
                 fc2_smooth_scale=None,  # [expert, 1, inter_dim]
                 ):
    return asm_moe(hidden_states,
                   w1,
                   w2,
                   topk_weight,
                   topk_ids, fc1_scale, fc2_scale, fc1_smooth_scale, fc2_smooth_scale)


@perftest()
def vllm_moe(hidden_states, w1, w2, topk_weight, topk_ids):
    return fused_experts(hidden_states,
                         w1,
                         w2,
                         topk_weight,
                         topk_ids,
                         inplace=False)


def test_fmoe(dtype, token, model_dim, inter_dim, E, topk, quant=False):
    input = torch.randn((token, model_dim), dtype=dtype, device="cuda")
    w1 = torch.randn((E, inter_dim, model_dim), dtype=dtype, device="cuda")
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype, device="cuda")
    score = torch.randn((token, E), device="cuda", dtype=dtype)
    topk_weights, topk_ids = fused_topk(input, score, topk, True)

    # ref implement
    # w1a = permute_weight_a(w1)
    # w2a = permute_weight_a(w2)
    w1a = w1
    w2a = w2
    avg_a = 1
    # ref1, avg_a = vllm_moe(input,
    #                        w1a,
    #                        w2a,
    #                        topk_weights,
    #                        topk_ids)
    # print(f'{ref1=}')

    if not quant:
        # ref2 implement
        ref2, avg_c = torch_moe_test(input,
                                     w1,
                                     w2,
                                     topk_weights,
                                     topk_ids)
        # print(f'{ref2=}')

        # b implement
        w1b = shuffle_weight(w1)
        w2b = shuffle_weight(w2)
        out_b, avg_b = asm_moe_test(input, w1b, w2b, topk_weights, topk_ids)
        # print(f'{out_b=}')
    else:
        w1, fc1_scale = pertoken_quant(w1, torch.float)
        w2, fc2_scale = pertoken_quant(w2, torch.float)

        sp1 = (E, inter_dim)
        sp2 = (E, model_dim)
        # [expert, 1, model_dim]
        fc1_smooth_scale = torch.randn(sp2, dtype=torch.float, device="cuda")
        # [expert, 1, inter_dim]
        fc2_smooth_scale = torch.randn(sp1, dtype=torch.float, device="cuda")

        # ref2 implement
        ref2, avg_c = torch_moe_test(input,
                                     w1,
                                     w2,
                                     topk_weights,
                                     topk_ids, fc1_scale, fc2_scale, fc1_smooth_scale, fc2_smooth_scale)
        print(f'{ref2=}')

        # b implement
        w1b = shuffle_weight(w1)
        w2b = shuffle_weight(w2)
        out_b, avg_b = asm_moe_test(input, w1b, w2b, topk_weights, topk_ids,
                                    fc1_scale, fc2_scale, fc1_smooth_scale, fc2_smooth_scale)
        print(f'{out_b=}')

    msg = f"[perf] {token=}, {model_dim=}, {inter_dim=}, {E=}, {topk=}, dtype: {dtype}, torch_avg: {avg_a:<8.2f} us, asm_avg: {avg_b:<8.2f} us,smtorch_k_avg: {avg_c:.2f} us, uplift: {avg_c/avg_b-1:.1%}"
    # checkAllclose(ref1, ref2, rtol=0.05, atol=20)
    checkAllclose(ref2, out_b, rtol=0.01, atol=100, msg=msg)


print('test test_fmoe 16 bit')
for dtype in [torch.float16, torch.bfloat16][1:]:
    for m in [1, 2, 4, 8, 16, 26, 32, 64, 128, 160, 192, 224, 256][-5:-4]:
        for dim in [4096, 8192, 16384, 32768][1:1+1]:
            for hdim in [1024, 2048, 3584, 4096, 8192, 16384, 32768][0:1]:
                test_fmoe(dtype, m, dim, hdim, 32, 5)
                # test_fmoe(dtype, m, dim, hdim, 32, 5, quant=True)
