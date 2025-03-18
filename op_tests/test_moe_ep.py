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
from aiter.fused_moe_bf16_asm import asm_moe, torch_moe, moe_sorting_ck, fused_topk
from aiter.ops.shuffle import shuffle_weight
from aiter import pertoken_quant, ck_moe

BLOCK_SIZE_M = 32
MAX_TOKENS = 4096 * 4

@perftest(num_warmup=1, num_iters=2)
def torch_moe_test(hidden_states, w1, w2, topk_weight, topk_ids,
                   # following for int8 quant
                   fc1_scale=None,  # [expert, inter_dim, 1]
                   fc2_scale=None,  # [expert, model_dim, 1]
                   fc1_smooth_scale=None,  # [expert, 1, model_dim]
                   fc2_smooth_scale=None,  # [expert, 1, inter_dim]
                   expert_mask=None):
    return torch_moe(hidden_states,
                     w1,
                     w2,
                     topk_weight,
                     topk_ids, fc1_scale, fc2_scale, fc1_smooth_scale, fc2_smooth_scale, expert_mask)


@perftest(num_warmup=5, num_iters=20)
def asm_moe_test(hidden_states, w1, w2, topk_weight, topk_ids,
                 # following for int8 quant
                 fc1_scale=None,  # [expert, inter_dim, 1]
                 fc2_scale=None,  # [expert, model_dim, 1]
                 fc1_smooth_scale=None,  # [expert, 1, model_dim]
                 fc2_smooth_scale=None,  # [expert, 1, inter_dim]
                 a16=False,
                 expert_mask=None):

    return asm_moe(hidden_states,
                   w1,
                   w2,
                   topk_weight,
                   topk_ids, fc1_scale, fc2_scale, fc1_smooth_scale, fc2_smooth_scale, a16, expert_mask=expert_mask)


@perftest()
def ck_moe_test(hidden_states, w1, w2, topk_weight, topk_ids,
                # following for int8 quant
                fc1_scale=None,  # [expert, inter_dim, 1]
                fc2_scale=None,  # [expert, model_dim, 1]
                fc1_smooth_scale=None,  # [expert, 1, model_dim]
                fc2_smooth_scale=None,  # [expert, 1, inter_dim]
                expert_mask=None):
    return ck_moe(hidden_states,
                  w1,
                  w2,
                  topk_weight,
                  topk_ids,
                  fc1_scale, fc2_scale, fc1_smooth_scale, fc2_smooth_scale, expert_mask=expert_mask)


quant_algo = [
    "No",  # g1u0/ck(g1ux) support
    "int8quant",  # g1u1 support
    "fp8quant",  # g1u1 support
    "int8smoothquant",  # g1u1/g1u0 support
    "fp8smoothquant",  # g1u1 support
]


def test_fmoe_ep(dtype, token, model_dim, inter_dim, E, topk, quant='No', use_g1u1=False, shared_E=2, ep=8):
    # This gpu id in EP, this example use the last id
    ep_id = ep -1
    # total_expert = unshared_expert + shared_expert + fake_expert(only use this fake expert id to mask) 
    expert_mask = torch.randint(0, 2, (E+shared_E+1,), dtype=torch.int32, device="cuda")
    # The last expert 
    fake_expertid = expert_mask.numel() - 1
    # Ensure fake expert to be masked
    expert_mask[-1] = 0
    # Ensure shared expert not to be masked
    expert_mask[E:-1] = 1
    # # Get local expert Number in this gpu
    # local_E = torch.sum(expert_mask).item()

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
                         dtype=dtype, device="cuda") / 10
    w2 = torch.randn((E+shared_E, model_dim, inter_dim),
                     dtype=dtype, device="cuda") / 10
    score = torch.randn((token, E), device="cuda", dtype=dtype)

    # if shared_E > 0:
    shared_E_score = 0.1
    # init total_topk_ids, inference time you just need to fill ns_topk_ids in total_topk_ids
    total_topk_ids = torch.empty((MAX_TOKENS, topk + shared_E + 1), dtype=torch.int32, device=input.device)
    ns_topk_ids, s_topk_ids = total_topk_ids.split([topk, shared_E + 1], dim=1)
    shared_expert_ids = [E + i for i in range(shared_E + 1)]
    s_topk_ids_list = [[fake_expertid]* (shared_E + 1)] * MAX_TOKENS
    for i in range(ep_id, MAX_TOKENS, ep):
        s_topk_ids_list[i] = shared_expert_ids
    s_topk_ids[:] = torch.tensor(s_topk_ids_list, dtype=torch.int32, device=input.device)

    # init total_topk_weights, inference time you just need to fill ns_topk_weights in total_topk_weights
    total_topk_weights = torch.empty((MAX_TOKENS, topk + shared_E + 1), dtype=torch.float32, device=input.device)
    ns_topk_weights, s_topk_weights = total_topk_weights.split([topk, shared_E + 1], dim=1)
    s_topk_weights[:] = shared_E_score

    # inference time, use fused_topk to fill ns_topk_ids and ns_topk_weights
    fused_topk(input, score, topk, True, ns_topk_ids, ns_topk_weights)
    # inference time, topk_ids simply slices total_topk_ids into the number of input tokens, same for topk_weights
    topk_ids = total_topk_ids[:token]
    topk_weights = total_topk_weights[:token]

    # else:
    #     topk_ids, topk_weights = fused_topk(input, score, topk, True)

    if quantAlgoId == 0:
        # ref2 implement
        ref2, avg_c = torch_moe_test(input,
                                     w1,
                                     w2,
                                     topk_weights,
                                     topk_ids,
                                     expert_mask=expert_mask)

        # b implement
        w1b = shuffle_weight(w1)
        w2b = shuffle_weight(w2)

        if use_g1u1:
            out_b = ref2
            avg_b = 9999
            print("asm g1u1 only support quant/smoothquant Now")
        else:
            out_b, avg_b = asm_moe_test(
                input, w1b, w2b, topk_weights, topk_ids, expert_mask=expert_mask)

        # test ck moe
        out_ck, avg_ck = ck_moe_test(input, w1b, w2b, topk_weights, topk_ids,
                                     None, None,
                                     None, None, expert_mask)


        msg = f"[perf] {token=}, quant={quantstr}, {model_dim=}, {inter_dim=}, {E=}, {shared_E=}, {topk=}, {ep=}, dtype: {dtype}, torch_avg: {avg_c:<8.2f} us, asm_avg: {avg_b:.2f} us, ck_avg: {avg_ck:.2f} us, uplift: {avg_c/avg_b-1:.1%}"
        checkAllclose(ref2, out_b, rtol=0.01, atol=10, msg=msg)
        checkAllclose(ref2, out_ck, rtol=0.01, atol=10, msg="ck check")

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
                                     fc1_smooth_scale, fc2_smooth_scale, expert_mask)

        # b implement
        w1b = shuffle_weight(w1)
        w2b = shuffle_weight(w2)
        out_b, avg_b = asm_moe_test(input, w1b, w2b, topk_weights, topk_ids,
                                    fc1_scale, fc2_scale,
                                    fc1_smooth_scale, fc2_smooth_scale, expert_mask=expert_mask)

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
        print(f"[BW  ] {token=}, quant={quantstr}, {model_dim=}, {inter_dim=}, {E=}, {shared_E=}, {topk=}, {ep=}, {topk=}, dtype: {dtype}, asm_bandwidth: {bw:.2f}TB/s")

        if use_smooth and (inter_dim % 512 == 0 or
                           inter_dim % 320 == 0
                           ) and (
            (w1b.dtype == torch.float8_e4m3fnuz and inter_dim*2 == w1b.shape[1]) or
                (w1b.dtype == torch.int8 and inter_dim == w1b.shape[1])):
            out_b2, avg_b2 = asm_moe_test(input, w1b, w2b, topk_weights, topk_ids,
                                          fc1_scale, fc2_scale, fc1_smooth_scale, fc2_smooth_scale, a16=True, expert_mask=expert_mask)
            msg = f'[perf] a8w8 asm: {avg_b:.2f} vs a16w8 asm: {avg_b2:.2f} ......'
            checkAllclose(out_b, out_b2, atol=10, msg=msg)

        # # test ck moe, not support now
        # out_ck, avg_ck = ck_moe_test(input, w1b, w2b, topk_weights, topk_ids,
        #                              fc1_scale, fc2_scale,
        #                              fc1_smooth_scale, fc2_smooth_scale)

        msg = f"[perf] {use_g1u1=} {token=}, quant={quantstr}, {model_dim=}, {inter_dim=}, {E=}, {shared_E=}, {topk=}, {ep=}, {topk=}, dtype: {dtype}, torch_avg: {avg_c:<8.2f} us, asm_avg: {avg_b:.2f} us ...... uplift: {avg_c/avg_b-1:.1%}"
        checkAllclose(ref2, out_b, rtol=0.01, atol=10, msg=msg)
        # checkAllclose(ref2, avg_ck, rtol=0.01, atol=10)


print('test test_fmoe 16 bit')
print('\ng1u0 no quant')
for dtype in [torch.float16, torch.bfloat16]:
    for m in [7, 128, 256]:
        for dim in [4096, 8192]:
            for hdim in [1024]:
                for ep in [1, 2, 4, 8]:
                    test_fmoe_ep(dtype, m, dim, hdim, 32, 5, quant='No', shared_E=2, ep=ep)

print('\ng1u1 no quant')
for dtype in [torch.float16, torch.bfloat16]:
    for m in [7, 128, 256]:
        for dim in [4096, 8192]:
            for hdim in [1024]:
                for ep in [1, 2, 4, 8]:
                    test_fmoe_ep(dtype, m, dim, hdim, 32, 5,
                            quant='No', use_g1u1=True, shared_E=2, ep=ep)

print('\ng1u1 int8quant')
for dtype in [torch.bfloat16]:
    for m in [128, 256]:
        for dim in [4096, 8192]:
            for hdim in [1024]:
                for ep in [1, 2, 4, 8]:
                    test_fmoe_ep(dtype, m, dim, hdim, 32, 5,
                            quant='int8quant', use_g1u1=True, shared_E=2, ep=ep)

print('\ng1u1 fp8quant')
for dtype in [torch.bfloat16]:
    for m in [128, 256]:
        for dim in [4096, 8192]:
            for hdim in [1024]:
                for ep in [1, 2, 4, 8]:
                    test_fmoe_ep(dtype, m, dim, hdim, 32, 5,
                            quant='fp8quant', use_g1u1=True, shared_E=2, ep=ep)


print('\ng1u0 int8smoothquant')
for dtype in [torch.bfloat16]:
    for m in [128]:
        for dim in [4096, 6144,  8192]:
            for hdim in [512, 1024]:
                for ep in [1, 2, 4, 8]:
                    test_fmoe_ep(dtype, m, dim, hdim, 32, 5,
                            quant='int8smoothquant', use_g1u1=False, shared_E=2, ep=ep)

# print('\ng1u1 int8smoothquant not supported')
# for dtype in [torch.bfloat16]:
#     for m in [128]:
#         for dim in [4096, 6144,  8192]:
#             for hdim in [512, 1024, 1280]:
#                 for ep in [1, 2, 4, 8]:
#                     test_fmoe_ep(dtype, m, dim, hdim, 32, 5,
#                             quant='int8smoothquant', use_g1u1=True, shared_E=2, ep=ep)

print('\ng1u1 fp8smoothquant')
for dtype in [torch.bfloat16]:
    for m in [128]:
        for dim in [4096, 6144,  8192]:
            for hdim in [512, 1024, 1280]:
                for ep in [1, 2, 4, 8]:
                    test_fmoe_ep(dtype, m, dim, hdim, 32, 5,
                            quant='fp8smoothquant', use_g1u1=True, shared_E=2, ep=ep)

