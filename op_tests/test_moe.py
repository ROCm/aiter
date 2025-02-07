# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import triton.language as tl
import numpy as np
import sys
import os
from typing import Any, Callable, Dict, Optional, Tuple
from aiter.test_common import checkAllclose, perftest
from aiter.fused_moe_bf16_asm import asm_moe, torch_moe, moe_sorting_ck
from aiter.fused_moe_gelu import fused_topk, moe_align_block_size, fused_experts
from aiter.ops.shuffle import shuffle_weight
from aiter import pertoken_quant, ck_moe

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
    aiter.moe_align_block_size(topk_ids, num_experts, block_size, sorted_ids,
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


@perftest(num_warmup=1, num_iters=2)
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
                 a16=False,
                 ):

    return asm_moe(hidden_states,
                   w1,
                   w2,
                   topk_weight,
                   topk_ids, fc1_scale, fc2_scale, fc1_smooth_scale, fc2_smooth_scale, a16)


@perftest()
def ck_moe_test(hidden_states, w1, w2, topk_weight, topk_ids,
                # following for int8 quant
                fc1_scale=None,  # [expert, inter_dim, 1]
                fc2_scale=None,  # [expert, model_dim, 1]
                fc1_smooth_scale=None,  # [expert, 1, model_dim]
                fc2_smooth_scale=None,  # [expert, 1, inter_dim]
                ):
    return ck_moe(hidden_states,
                  w1,
                  w2,
                  topk_weight,
                  topk_ids,
                  fc1_scale, fc2_scale, fc1_smooth_scale, fc2_smooth_scale)


@perftest()
def vllm_moe(hidden_states, w1, w2, topk_weight, topk_ids):
    return fused_experts(hidden_states,
                         w1,
                         w2,
                         topk_weight,
                         topk_ids,
                         inplace=False)


quant_algo = [
    "No",  # g1u0/ck(g1ux) support
    "int8quant",  # g1u1 support
    "fp8quant",  # g1u1 support
    "int8smoothquant",  # g1u1/g1u0 support
    "fp8smoothquant",  # g1u1 support
]


def test_fmoe(dtype, token, model_dim, inter_dim, E, topk, quant='No', use_g1u1=False, shared_E=0):
    quantAlgoId = quant_algo.index(quant)
    if quantAlgoId not in [0, 3] and not use_g1u1:
        print("g1u0 only could test no quant and int8smoothquant")
        return

    quantstr = quant_algo[quantAlgoId]
    quant_dtype = torch.int8 if quantstr.startswith(
        'int8') else torch.float8_e4m3fnuz
    use_smooth = 'smooth' in quantstr

    input = torch.randn((token, model_dim), dtype=dtype, device="cuda")
    if use_g1u1:
        w1 = torch.randn((E+shared_E, inter_dim*2, model_dim),
                         dtype=dtype, device="cuda") / 10
    else:
        w1 = torch.randn((E+shared_E, inter_dim, model_dim),
                         dtype=dtype, device="cuda")
    w2 = torch.randn((E+shared_E, model_dim, inter_dim),
                     dtype=dtype, device="cuda")
    score = torch.randn((token, E), device="cuda", dtype=dtype)
    topk_weights, topk_ids = fused_topk(input, score, topk, True)

    if shared_E > 0:
        shared_E_score = 0.5
        s_topk_weights = torch.tensor([[shared_E_score, shared_E_score],] * token,
                                      dtype=torch.float32,
                                      device=input.device)
        topk_weights = torch.cat((topk_weights, s_topk_weights), dim=1)
        s_topk_ids = torch.tensor([[E, E+1],] * token,
                                  dtype=torch.int32,
                                  device=input.device)
        topk_ids = torch.cat((topk_ids, s_topk_ids), dim=1)

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

    if quantAlgoId == 0:
        # ref2 implement
        ref2, avg_c = torch_moe_test(input,
                                     w1,
                                     w2,
                                     topk_weights,
                                     topk_ids)

        # b implement
        w1b = shuffle_weight(w1)
        w2b = shuffle_weight(w2)

        if use_g1u1:
            out_b = ref2
            avg_b = 9999
            print("asm g1u1 only support quant/smoothquant Now")
        else:
            out_b, avg_b = asm_moe_test(
                input, w1b, w2b, topk_weights, topk_ids)

        # test ck moe
        out_ck, avg_ck = ck_moe_test(input, w1b, w2b, topk_weights, topk_ids,
                                     None, None,
                                     None, None)

        msg = f"[perf] {token=}, quant={quantstr}, {model_dim=}, {inter_dim=}, {E=}, {topk=}, dtype: {dtype}, torch_avg: {avg_c:<8.2f} us, asm_avg: {avg_b:.2f} us, ck_avg: {avg_ck:.2f} us, uplift: {avg_c/avg_b-1:.1%}"
        checkAllclose(ref2, out_b, rtol=0.01, atol=100, msg=msg)
        checkAllclose(ref2, out_ck, rtol=0.01, atol=100, msg="ck check")

    else:
        w1, fc1_scale = pertoken_quant(
            w1, torch.float, quant_dtype=quant_dtype)
        w2, fc2_scale = pertoken_quant(
            w2, torch.float, quant_dtype=quant_dtype)

        sp1 = (E+shared_E, inter_dim)
        sp2 = (E+shared_E, model_dim)

        if not use_smooth:
            fc1_smooth_scale = None
            fc2_smooth_scale = None
        else:
            # [expert, 1, model_dim]
            fc1_smooth_scale = torch.randn(
                sp2, dtype=torch.float, device="cuda")
            # [expert, 1, inter_dim]
            fc2_smooth_scale = torch.randn(
                sp1, dtype=torch.float, device="cuda")

        # ref2 implement
        ref2, avg_c = torch_moe_test(input, w1, w2, topk_weights, topk_ids,
                                     fc1_scale, fc2_scale,
                                     fc1_smooth_scale, fc2_smooth_scale)

        # b implement
        w1b = shuffle_weight(w1)
        w2b = shuffle_weight(w2)
        out_b, avg_b = asm_moe_test(input, w1b, w2b, topk_weights, topk_ids,
                                    fc1_scale, fc2_scale,
                                    fc1_smooth_scale, fc2_smooth_scale)

        def calculateTensorsSize(*args):
            num_btype = 0
            for el in args:
                if isinstance(el, torch.Tensor):
                    num_btype += el.element_size() * el.numel()
            return num_btype

        num_tb = calculateTensorsSize(input, input, w1b, w2b, topk_weights, topk_ids,
                                      fc1_scale, fc2_scale,
                                      fc1_smooth_scale, fc2_smooth_scale) / (1024*1024*1024*1024.0)
        bw = num_tb * 1e6 / avg_b
        print(f"[BW  ] {token=}, quant={quantstr}, {model_dim=}, {inter_dim=}, {E=}, {shared_E=}, {topk=}, dtype: {dtype}, asm_bandwidth: {bw:.2f}TB/s")

        if use_smooth and (inter_dim % 512 == 0 or
                           inter_dim % 320 == 0
                           ) and (
            (w1b.dtype == torch.float8_e4m3fnuz and inter_dim*2 == w1b.shape[1]) or
                (w1b.dtype == torch.int8 and inter_dim == w1b.shape[1])):
            out_b2, avg_b2 = asm_moe_test(input, w1b, w2b, topk_weights, topk_ids,
                                          fc1_scale, fc2_scale, fc1_smooth_scale, fc2_smooth_scale, a16=True)
            msg = f'[perf] a8w8 asm: {avg_b:.2f} vs a16w8 asm: {avg_b2:.2f} ......'
            checkAllclose(ref2, out_b2, atol=100, msg=msg)

        # # test ck moe, not support now
        # out_ck, avg_ck = ck_moe_test(input, w1b, w2b, topk_weights, topk_ids,
        #                              fc1_scale, fc2_scale,
        #                              fc1_smooth_scale, fc2_smooth_scale)

        msg = f"[perf] {use_g1u1=} {token=}, quant={quantstr}, {model_dim=}, {inter_dim=}, {E=}, {shared_E=}, {topk=}, dtype: {dtype}, torch_avg: {avg_c:<8.2f} us, asm_avg: {avg_b:.2f} us ...... uplift: {avg_c/avg_b-1:.1%}"
        checkAllclose(ref2, out_b, rtol=0.01, atol=100, msg=msg)
        # checkAllclose(ref2, avg_ck, rtol=0.01, atol=100)


# print('test test_fmoe 16 bit')
# print('\ng1u0 no quant')
# for dtype in [torch.float16, torch.bfloat16]:
#     for m in [128, 256]:
#         for dim in [4096, 8192]:
#             for hdim in [1024]:
#                 # test_fmoe(dtype, m, dim, hdim, 32, 5)
#                 test_fmoe(dtype, m, dim, hdim, 32, 5, quant='No')

# print('\ng1u1 no quant')
# for dtype in [torch.float16, torch.bfloat16]:
#     for m in [128, 256]:
#         for dim in [4096, 8192]:
#             for hdim in [1024]:
#                 # test_fmoe(dtype, m, dim, hdim, 32, 5)
#                 test_fmoe(dtype, m, dim, hdim, 32, 5,
#                           quant='No', use_g1u1=True)

# print('\ng1u1 int8quant')
# for dtype in [torch.bfloat16]:
#     for m in [128, 256]:
#         for dim in [4096, 8192]:
#             for hdim in [1024]:
#                 test_fmoe(dtype, m, dim, hdim, 32, 5,
#                           quant='int8quant', use_g1u1=True)

# print('\ng1u1 fp8quant')
# for dtype in [torch.bfloat16]:
#     for m in [128, 256]:
#         for dim in [4096, 8192]:
#             for hdim in [1024]:
#                 test_fmoe(dtype, m, dim, hdim, 32, 5,
#                           quant='fp8quant', use_g1u1=True)


# print('\ng1u0 int8smoothquant')
# for dtype in [torch.bfloat16]:
#     for m in [128]:
#         for dim in [4096, 6144,  8192]:
#             for hdim in [512, 1024]:
#                 test_fmoe(dtype, m, dim, hdim, 32, 5,
#                           quant='int8smoothquant', use_g1u1=False)

# print('\ng1u1 int8smoothquant not supported')
# for dtype in [torch.bfloat16]:
#     for m in [128]:
#         for dim in [4096, 6144,  8192]:
#             for hdim in [512, 1024, 1280]:
#                 test_fmoe(dtype, m, dim, hdim, 32, 5,
#                           quant='int8smoothquant', use_g1u1=True)

print('\ng1u1 fp8smoothquant')
for dtype in [torch.bfloat16]:
    for m in [128]:
        for dim in [4096, 6144,  8192]:
            for hdim in [512, 1024, 1280]:
                test_fmoe(dtype, m, dim, hdim, 32, 5,
                          quant='fp8smoothquant', use_g1u1=True)
