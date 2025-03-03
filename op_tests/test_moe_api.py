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
from aiter.fused_moe_api import aiter_moe
from aiter.fused_moe_gelu import fused_topk, moe_align_block_size, fused_experts
from aiter.ops.shuffle import shuffle_weight
from aiter import pertoken_quant
from op_tests.int4_utils import *

BLOCK_SIZE_M = 32

@perftest(num_warmup=1, num_iters=2)
def torch_moe_test(hidden_states, w1, w2, topk_weight, topk_ids,
                   # following for int8 quant
                   fc1_scale=None,  # [expert, inter_dim, 1]
                   fc2_scale=None,  # [expert, model_dim, 1]
                   fc1_smooth_scale=None,  # [expert, 1, model_dim]
                   fc2_smooth_scale=None,  # [expert, 1, inter_dim]
                   activation = None
                   ):
    return torch_moe(hidden_states,
                     w1,
                     w2,
                     topk_weight,
                     topk_ids, fc1_scale, fc2_scale, fc1_smooth_scale, fc2_smooth_scale, activation=activation)


@perftest()
def aiter_moe_test(hidden_states, w1, w2, topk_weight, topk_ids,
                 # following for int8 quant
                 fc1_scale=None,  # [expert, inter_dim, 1]
                 fc2_scale=None,  # [expert, model_dim, 1]
                 fc1_smooth_scale=None,  # [expert, 1, model_dim]
                 fc2_smooth_scale=None,  # [expert, 1, inter_dim]
                 a16=False,
                 activation = None,
                 per_tensor_quant_scale=None
                 ):
    return aiter_moe(hidden_states,
                   w1,
                   w2,
                   topk_weight,
                   topk_ids, fc1_scale, fc2_scale, fc1_smooth_scale, fc2_smooth_scale, 
                   a16, activation, per_tensor_quant_scale)



quant_algo = [
    "No",  # g1u0/ck(g1ux) support
    "int8quant",  # g1u1 support
    "fp8quant",  # g1u1 support
    "int8smoothquant",  # g1u1/g1u0 support
    "fp8smoothquant",  # g1u1 support
    "wint4afp8quant", # g1u1 support
]


def test_fmoe(dtype, token, model_dim, inter_dim, E, topk, quant='No', use_g1u1=False, shared_E=0, activation=None):
    quantAlgoId = quant_algo.index(quant)
    if quantAlgoId not in [0, 3] and not use_g1u1:
        print("g1u0 only could test no quant and int8smoothquant")
        return

    quantstr = quant_algo[quantAlgoId]
    use_int4 = "wint4" in quantstr
    quant_dtype = torch.int8 if use_int4 or quantstr.startswith(
        'int8') else torch.float8_e4m3fnuz
    use_smooth = 'smooth' in quantstr
    input = torch.randn((token, model_dim), dtype=dtype, device="cuda")
    if use_g1u1:
        w1 = torch.randn((E+shared_E, inter_dim*2, model_dim),
                         dtype=dtype, device="cuda") / 10.0
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


        out_a, avg_a = aiter_moe_test(
            input, w1b, w2b, topk_weights, topk_ids, activation=activation)

        msg = f"[perf] {token=}, quant={quantstr}, {model_dim=}, {inter_dim=}, {E=}, {topk=}, dtype: {dtype}, torch_avg: {avg_c:<8.2f} us, aiter_avg: {avg_a:.2f} us, uplift: {avg_c/avg_a-1:.1%}"
        checkAllclose(ref2, out_a, rtol=0.01, atol=100, msg=msg)
    else:
        dtypeMax = 7 if use_int4 else None
        w1, fc1_scale = pertoken_quant(
            w1, torch.float, quant_dtype=quant_dtype, dtypeMax=dtypeMax)
        w2, fc2_scale = pertoken_quant(
            w2, torch.float, quant_dtype=quant_dtype, dtypeMax=dtypeMax)

        sp1 = (E+shared_E, inter_dim)
        sp2 = (E+shared_E, model_dim)

        if not use_smooth:
            fc1_smooth_scale = None
            fc2_smooth_scale = None
        else:
            if use_int4:
                #fixme @felix: hack here, int4 kernel need this buffer but not used, so ones.
                # [expert, 1, model_dim]
                fc1_smooth_scale = torch.ones(sp2, dtype=torch.float, device="cuda")
                # [expert, 1, inter_dim]
                fc2_smooth_scale = torch.ones(sp1, dtype=torch.float, device="cuda")
            else:
                # [expert, 1, model_dim]
                fc1_smooth_scale = torch.randn(sp2, dtype=torch.float, device="cuda")
                # [expert, 1, inter_dim]
                fc2_smooth_scale = torch.randn(sp1, dtype=torch.float, device="cuda")

        # ref2 implement
        ref2, avg_c = torch_moe_test(input, w1, w2, topk_weights, topk_ids,
                                     fc1_scale, fc2_scale,
                                     fc1_smooth_scale, fc2_smooth_scale, activation)

        # aiter implement
        if use_int4:
            w1 = rearrange_4bit_elements(convert_int8_to_uint32_int4(w1))
            w2 = rearrange_4bit_elements(convert_int8_to_uint32_int4(w2))
        w1b = shuffle_weight(w1)
        w2b = shuffle_weight(w2)
        out_a, avg_a = aiter_moe_test(input, w1b, w2b, topk_weights, topk_ids,
                                    fc1_scale, fc2_scale,
                                    fc1_smooth_scale, fc2_smooth_scale, activation=activation)

        def calculateTensorsSize(*args):
            num_btype = 0
            for el in args:
                if isinstance(el, torch.Tensor):
                    num_btype += el.element_size() * el.numel()
            return num_btype

        num_tb = calculateTensorsSize(input, input, w1b, w2b, topk_weights, topk_ids,
                                      fc1_scale, fc2_scale,
                                      fc1_smooth_scale, fc2_smooth_scale) / (1024*1024*1024*1024.0)
        bw = num_tb * 1e6 / avg_a
        print(f"[BW  ] {token=}, quant={quantstr}, {model_dim=}, {inter_dim=}, {E=}, {shared_E=}, {topk=}, dtype: {dtype}, aiter_bandwidth: {bw:.2f}TB/s")

        if use_smooth and (inter_dim % 512 == 0 or
                           inter_dim % 320 == 0
                           ) and (
            (w1b.dtype == torch.float8_e4m3fnuz and inter_dim*2 == w1b.shape[1]) or
                (w1b.dtype == torch.int8 and inter_dim == w1b.shape[1])):
            out_b2, avg_b2 = aiter_moe_test(input, w1b, w2b, topk_weights, topk_ids,
                                          fc1_scale, fc2_scale, fc1_smooth_scale, fc2_smooth_scale, a16=True, activation=activation)
            msg = f'[perf] a8w8 aiter: {avg_a:.2f} vs a16w8 aiter: {avg_b2:.2f} ......'
            checkAllclose(ref2, out_b2, atol=100, msg=msg)


        msg = f"[perf] {use_g1u1=} {token=}, quant={quantstr}, {model_dim=}, {inter_dim=}, {E=}, {shared_E=}, {topk=}, dtype: {dtype}, torch_avg: {avg_c:<8.2f} us, aiter_avg: {avg_a:.2f} us ...... uplift: {avg_c/avg_a-1:.1%}"
        checkAllclose(ref2, out_a, rtol=0.01, atol=100, msg=msg)


print('test test_fmoe 16 bit')
print('\ng1u0 no quant')
for dtype in [torch.float16, torch.bfloat16]:
    for m in [128, 256]:
        for dim in [4096, 8192]:
            for hdim in [1024]:
                # test_fmoe(dtype, m, dim, hdim, 32, 5)
                test_fmoe(dtype, m, dim, hdim, 32, 5, quant='No')

print('\ng1u1 no quant')
for dtype in [torch.float16, torch.bfloat16]:
    for m in [128, 256]:
        for dim in [4096, 8192]:
            for hdim in [1024]:
                # test_fmoe(dtype, m, dim, hdim, 32, 5)
                test_fmoe(dtype, m, dim, hdim, 32, 5,
                          quant='No', use_g1u1=True)

print('\ng1u0 no quant silu')
for dtype in [torch.float16, torch.bfloat16]:
    for m in [128, 256]:
        for dim in [4096, 8192]:
            for hdim in [1024]:
                # test_fmoe(dtype, m, dim, hdim, 32, 5)
                test_fmoe(dtype, m, dim, hdim, 32, 5, quant='No', activation="silu")

print('\ng1u1 no quant gelu')
for dtype in [torch.float16, torch.bfloat16]:
    for m in [128, 256]:
        for dim in [4096, 8192]:
            for hdim in [1024]:
                # test_fmoe(dtype, m, dim, hdim, 32, 5)
                test_fmoe(dtype, m, dim, hdim, 32, 5,
                          quant='No', use_g1u1=True, activation="gelu")

print('\ng1u1 int8quant')
for dtype in [torch.bfloat16]:
    for m in [128, 256]:
        for dim in [4096, 8192]:
            for hdim in [1024]:
                test_fmoe(dtype, m, dim, hdim, 32, 5,
                          quant='int8quant', use_g1u1=True)

print('\ng1u1 fp8quant')
for dtype in [torch.bfloat16]:
    for m in [128, 256]:
        for dim in [4096, 8192]:
            for hdim in [1024]:
                test_fmoe(dtype, m, dim, hdim, 32, 5,
                          quant='fp8quant', use_g1u1=True)


print('\ng1u0 int8smoothquant')
for dtype in [torch.bfloat16]:
    for m in [128]:
        for dim in [4096, 6144,  8192]:
            for hdim in [512, 1024]:
                test_fmoe(dtype, m, dim, hdim, 32, 5,
                          quant='int8smoothquant', use_g1u1=False)

print('\ng1u1 int8smoothquant')
for dtype in [torch.bfloat16]:
    for m in [128]:
        for dim in [4096, 6144,  8192]:
            for hdim in [512, 1024, 1280]:
                test_fmoe(dtype, m, dim, hdim, 32, 5,
                          quant='int8smoothquant', use_g1u1=True)

print('\ng1u1 fp8smoothquant')
for dtype in [torch.bfloat16]:
    for m in [128]:
        for dim in [4096, 6144,  8192]:
            for hdim in [512, 1024, 1280]:
                test_fmoe(dtype, m, dim, hdim, 32, 5,
                          quant='fp8smoothquant', use_g1u1=True)

print('\ng1u1 int4 silu')
for dtype in [torch.bfloat16]:
    for m in [32, 128]:
        # for dim in [1024]:
        for dim in [4096, 6144]:
            for hdim in [1024, 4096]:
                test_fmoe(dtype, m, dim, hdim, 8, 3,
                          quant='wint4afp8quant', use_g1u1=True, activation='silu')


print('\ng1u1 int4 gelu')
for dtype in [torch.bfloat16]:
    for m in [32, 128]:
        # for dim in [1024]:
        for dim in [4096, 6144]:
            for hdim in [1024, 4096]:
                test_fmoe(dtype, m, dim, hdim, 8, 3,
                          quant='wint4afp8quant', use_g1u1=True, activation='gelu')
