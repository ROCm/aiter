# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import triton.language as tl
import numpy as np
import sys
import os
from typing import Any, Callable, Dict, Optional, Tuple
import aiter
from aiter.test_common import checkAllclose, perftest
from aiter import pertoken_quant
from aiter.fused_moe_gelu import fused_topk
from aiter.fused_moe_bf16_asm import asm_moe, torch_moe, moe_sorting_ck, ck_moe_2stages,ck_moe_2stages_win4, get_block_size
from aiter.ops.shuffle import shuffle_weight
from op_tests.int4_utils import *
from aiter import ActivationType

@perftest(num_iters=2, num_warmup=1)
def torch_moe_stage1(hidden_states,
                     w1,  # E, inter_dim*2, model_dim
                     w2,  # E, model_dim, inter_dim
                     topk_weight, topk_ids,
                     dtype=torch.float16,
                     # following for quant
                     fc1_scale=None,  # [expert, inter_dim, 1]
                     w1_scale=None,  # [1]
                     a1_scale=None,  # [expert]]
                     block_size=32
                     ):
    ctype = torch.float  # compute type
    hidden_states = hidden_states.to(ctype)
    w1 = w1.to(ctype)

    B, D = hidden_states.shape
    E=w1.shape[0]
    topk = topk_weight.shape[1]
    N = w1.shape[1]
    num_experts, model_dim, inter_dim = w2.shape
    # hidden_states = hidden_states.view(
    #     B, -1, D).repeat(1, topk, 1)

    max_num_tokens_padded = topk_ids.numel() + num_experts * block_size - topk

    # gose to quant D_w8a8/w8a8
    if fc1_scale is not None:
        w1 = (w1.view(-1, D) * fc1_scale.view(-1, 1)).view(num_experts, -1, D)
    if a1_scale is not None and w1_scale is not None:
        hidden_states = hidden_states * a1_scale
        w1 = w1 * w1_scale.view(E, -1, 1)

    hidden_states = hidden_states.view(
        B, -1, D).repeat(1, topk, 1)
    out = torch.zeros(
        (B, topk, N),
        dtype=ctype,
        device=hidden_states.device,
    )
    for E_id in range(w1.shape[0]):
        mask = topk_ids == E_id
        if mask.sum():
            sub_tokens = hidden_states[mask]
            act_input = sub_tokens @ (w1[E_id].transpose(0, 1))
            out[mask] = act_input

    return out.to(dtype)


@perftest(num_iters=2, num_warmup=1)
def torch_moe_stage2(hidden_states,
                     w1,  # E, inter_dim*2, model_dim
                     w2,  # E, model_dim, inter_dim
                     topk_weights, topk_ids,
                     sorted_weights, sorted_ids,
                     sorted_expert_ids, num_valid_ids,
                     dtype=torch.float16,
                     w2_scale=None,  # [1]
                     a2_scale=None,  # [expert]]
                     block_size=32
                     ):
    ctype = torch.float  # compute type
    hidden_states = hidden_states.to(ctype)
    w2 = w2.to(ctype)

    E = w1.shape[0]
    token_num, topk = topk_ids.shape
    # M, _ = hidden_states.shape
    num_experts, model_dim, inter_dim = w2.shape
    max_num_m_blocks = sorted_expert_ids.shape[0]

    # gose to quant D_w8a8/w8a8
    if a2_scale is not None and w2_scale is not None:
        hidden_states = hidden_states * a2_scale
        w2 = w2 * w2_scale.view(E, -1, 1)

    out = torch.zeros(
        (token_num, topk, model_dim),
        dtype=ctype,
        device=hidden_states.device,
    )
    for E_id in range(w1.shape[0]):
        mask = topk_ids == E_id
        if mask.sum():
            sub_tokens = hidden_states[mask]
            act_input = sub_tokens @ (w2[E_id].transpose(0, 1))
            out[mask] = act_input     
    return (out * topk_weights.view(token_num, -1, 1)).sum(1).to(dtype)

@perftest()
def ck_moe_stage1(hidden_states,
                  w1,  # [E, inter_dim*2, model_dim]
                  w2,  # [E, model_dim, inter_dim]
                  sorted_token_ids,  # [max_num_tokens_padded]
                  sorted_expert_ids,  # [max_num_m_blocks]
                  num_valid_ids,  # [1]
                  w1_scale, a1_scale, dtype,
                  topk,
                  block_size=32
                  ):
    token_num = hidden_states.shape[0]

    D = w1.shape[1]
    num_experts, model_dim, inter_dim = w2.shape
    max_num_tokens_padded = sorted_token_ids.shape[0]
    # max_num_tokens_padded = sorted_expert_ids.shape[0]*block_size

    out = torch.zeros(
        (token_num, topk, D),
        dtype=dtype,
        device=hidden_states.device,
    )
    aiter.ck_moe_stage1(hidden_states, w1, w2, sorted_token_ids,
                        sorted_expert_ids, num_valid_ids, out, topk, w1_scale, a1_scale, block_size)
    return out


@perftest()
def ck_moe_stage2(hidden_states,
                  w1,  # [E, inter_dim*2, model_dim]
                  w2,  # [E, model_dim, inter_dim]
                  sorted_token_ids,  # [max_num_tokens_padded]
                  sorted_expert_ids,  # [max_num_m_blocks]
                  sorted_weights,  # [max_num_tokens_padded]
                  num_valid_ids,  # [1]
                  w2_scale, a2_scale, dtype,
                  topk,
                  block_size=32
                  ):
    token_num = hidden_states.shape[0]
    D = w2.shape[1]
    num_experts, model_dim, inter_dim = w2.shape
    max_num_tokens_padded = sorted_token_ids.shape[0]
    # max_num_tokens_padded = sorted_expert_ids.shape[0]*block_size

    out = torch.zeros(
        (token_num, D),
        dtype=dtype,
        device=hidden_states.device,
    )

    aiter.ck_moe_stage2(hidden_states, w1, w2, sorted_token_ids,
                        sorted_expert_ids, sorted_weights,
                        num_valid_ids, out, topk, w2_scale, a2_scale, block_size)
    return out


@perftest()
def ck_moe_fused_2stages(hidden_states,
                         # [expert(local_expert:EP), inter_dim(*2), dim] N,K
                         w1,
                         w2,  # [expert(local_expert:EP), dim, inter_dim]
                         topk_weight, topk_ids,
                         # following for int8 quant
                         # [expert(local_expert:EP), inter_dim, 1]
                         fc1_scale=None,
                         # [expert(local_expert:EP), model_dim, 1]
                         fc2_scale=None,
                         block_size=32,
                         a1_scale=None,
                         activation = ActivationType.Silu
                         ):
    return ck_moe_2stages_win4(hidden_states, w1, w2, topk_weight, topk_ids,
                          fc1_scale, fc2_scale, block_size=block_size, a1_scale=a1_scale, activation=activation)


def test_fmoe(dtype, token, model_dim, inter_dim, E, topk, quant='No', use_g1u1=False, shared_E=0, activation=ActivationType.Silu):
    input = torch.randn((token, model_dim), dtype=dtype, device="cuda") 
    if use_g1u1:
        w1 = torch.randn((E+shared_E, inter_dim*2, model_dim),
                         dtype=dtype, device="cuda")  / 10
    else:
        w1 = torch.randn((E+shared_E, inter_dim, model_dim),
                         dtype=dtype, device="cuda") 
    w2 = torch.randn((E+shared_E, model_dim, inter_dim),
                     dtype=dtype, device="cuda") 
    score = torch.randn((token, E), device="cuda", dtype=dtype)
    topk_weights, topk_ids = fused_topk(input, score, topk, True)

    E, model_dim, inter_dim = w2.shape
    M, topk = topk_ids.shape
    BLOCK_SIZE_M = get_block_size(M, topk, E)
    print("BLOCK_SIZE_M:",BLOCK_SIZE_M)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting_ck(topk_ids, topk_weights, E,
                                                                                           model_dim, dtype, BLOCK_SIZE_M)

    quant_dtype = torch.float8_e4m3fnuz
    quant_dtype_w = torch.int8
    if "perTensorQuant" in quant:
        w1_qt, w1_scale = aiter.pertoken_quant(w1.view(E, -1),
                                            quant_dtype=quant_dtype_w, dtypeMax=7)
        w2_qt, w2_scale = aiter.pertoken_quant(w2.view(E, -1),
                                            quant_dtype=quant_dtype_w, dtypeMax=7)
    else:
        w1_qt, w1_scale = aiter.pertoken_quant(w1,
                                            quant_dtype=quant_dtype_w, dtypeMax=7)
        w2_qt, w2_scale = aiter.pertoken_quant(w2,
                                            quant_dtype=quant_dtype_w, dtypeMax=7)
    
    ##for debug with CK
    #w2_scale = torch.ones((E+shared_E, 1),
    #                     dtype=quant_dtype_w, device="cuda") / 10
    
    
    if "perTensorQuant" in quant:
        a1_qt, a1_scale = aiter.per_tensor_quant(input,  quant_dtype=quant_dtype)
    else:
        a1_qt, a1_scale = aiter.pertoken_quant(input,  quant_dtype=quant_dtype)
        # a1_qt, a1_scale = aiter.per_token_dynamic_quant_fp8_hip(input)


    w1_qt = w1_qt.view(w1.shape)
    w2_qt = w2_qt.view(w2.shape)
    sp1 = (E+shared_E, inter_dim)
    sp2 = (E+shared_E, model_dim)
    # W int4 implement
    w1b = rearrange_4bit_elements(convert_int8_to_uint32_int4(shuffle_weight(w1_qt, (32, 32), use_int4=True)))
    w2b = rearrange_4bit_elements(convert_int8_to_uint32_int4(shuffle_weight(w2_qt, (32, 32), use_int4=True)))
    out1_ref, us_ref = torch_moe_stage1(a1_qt, w1_qt,
                                        w2_qt,
                                        topk_weights, topk_ids,
                                        dtype=dtype,
                                        fc1_scale=None,
                                        w1_scale=w1_scale,
                                        a1_scale=a1_scale,
                                        block_size=BLOCK_SIZE_M)

    if use_g1u1:
        gate, up = out1_ref.split([inter_dim, inter_dim], dim=-1)
        if activation== ActivationType.Silu:
            input2 = F.silu(gate) * up
        else:
            input2 = F.gelu(gate) * up
    else:
        if activation== ActivationType.Silu:
            input2 = F.silu(out1_ref)
        else:
            input2 = F.gelu(out1_ref)
    if "perTensorQuant" in quant:
        a2_qt, a2_scale = aiter.per_tensor_quant(input2,  quant_dtype=quant_dtype)
    else:
        a2_qt, a2_scale = aiter.pertoken_quant(input2.view(M, -1),  quant_dtype=quant_dtype)
        # a2_qt, a2_scale = aiter.per_token_dynamic_quant_fp8_hip(input2.view(M, -1))
        a2_qt = a2_qt.view(M, topk, -1)
    print(a2_qt.shape, a2_scale.shape)
    ##for debug with CK
    #a2_scale = torch.tensor(0.1, device='cuda:0')
    #sorted_weights = sorted_weights.fill_(0.1) 
    #topk_weights = topk_weights.fill_(0.1) 
    
    out2_ref, us_ref = torch_moe_stage2(a2_qt,
                                        w1_qt,  # E, inter_dim*2, model_dim
                                        w2_qt,  # E, model_dim, inter_dim
                                        topk_weights, topk_ids,
                                        sorted_weights, sorted_ids,
                                        sorted_expert_ids, num_valid_ids,
                                        dtype=dtype,
                                        # [expert, inter_dim, 1]
                                        w2_scale=w2_scale,
                                        a2_scale= a2_scale if "perTensorQuant" in quant else a2_scale.view(M, -1, 1).repeat(1, topk, 1),
                                        block_size=BLOCK_SIZE_M
                                        )
 

    out1_qt, us_s1 = ck_moe_stage1(a1_qt,
                                w1b,
                                w2b,
                                sorted_ids,
                                sorted_expert_ids,
                                num_valid_ids,
                                w1_scale, a1_scale,
                                dtype, topk, BLOCK_SIZE_M)
    
    #print("#######CK Stage 1##########")
    #print("CK GEMM1 OUT:",out1_qt)
    checkAllclose(out1_ref, out1_qt,
                  msg=f'ck_moe_stage1:{us_s1:.2f} us_s1, {token*model_dim*inter_dim*2*topk*2/us_s1/1000/1000:.2f} tflops......(quant:{quant_dtype})')
    if use_g1u1:
        gate, up = out1_qt.split([inter_dim, inter_dim], dim=-1)
        if activation== ActivationType.Silu:
            input2 = F.silu(gate) * up
        else:
            input2 = F.gelu(gate) * up
    else:
        if activation== ActivationType.Silu:
            input2 = F.silu(out1_ref)
        else:
            input2 = F.gelu(out1_ref)

    #print("#######CK Stage 2##########")      
    # a2_qt, a2_scale = aiter.per_tensor_quant(input2,  quant_dtype=quant_dtype)
    
    #print("aiter1 input2 for CK:",a2_qt[0,0])
    #print("aiter1 w2_qt for CK:",unpack_int4(w2b)[0])
    #print("aiter1 dtype for CK:",dtype)
    #print("aiter1 w2_scale for CK:",w2_scale[0])
    #print("aiter1 a2_scale for CK:",a2_scale)
    #print("aiter1 sorted_weights for CK:",sorted_weights)
    #print("#aiter BLOCK_SIZE_M for CK:",BLOCK_SIZE_M)
    #print("#aiter1 a2 for CK:",a2_qt.shape)
    #print("#aiter1 a2  for CK:",a2_qt)
    #print("#aiter1 w1b for CK:",w1b.shape)
    #print("#aiter1 w1b dtype for CK:",w1b.dtype)
    #print("#aiter1 w1 for CK:",w1.shape)
    #print("#aiter1 w2b for CK:",w2b.shape)
    #print("#aiter1 w2 for CK:",w2.shape)


    out2_qt, us_s2 = ck_moe_stage2(a2_qt,
                                w1b,
                                w2b,
                                sorted_ids,
                                sorted_expert_ids,
                                sorted_weights,
                                num_valid_ids,
                                w2_scale, a2_scale,
                                dtype, topk, BLOCK_SIZE_M)
    
    #print("CK GEMM2 OUT:",out2_qt)    
    checkAllclose(out2_ref, out2_qt,msg=f'ck_moe_stage2:{us_s2:.2f} us_s2, {token*model_dim*inter_dim*topk*2/us_s2/1000/1000:.2f} tflops......(quant:{quant_dtype})')

    #print("######Torch fused stage(FP16) VS  CK 2Stage out2_ref(INT4) ##########")
    #out_ref,us_fuse = torch_moe(input, w1, w2, topk_weights, topk_ids)
    #checkAllclose(out_ref, out2_ref,msg=f'Torch fused stage(FP16):{us_fuse:.2f} us, {token*model_dim*inter_dim*topk*2/us_fuse/1000/1000:.2f} tflops......(quant:{quant_dtype})')
    
    #print("######CK 2 stage merge run ##########")
    out_ck_qt, us = ck_moe_fused_2stages(input,
                                        w1b,
                                        w2b,
                                        topk_weights, topk_ids,
                                        w1_scale, w2_scale,
                                        block_size=BLOCK_SIZE_M,
                                        activation=activation
                                         )
    
    #print("CK 2 stage out merge:",out_ck_qt)
    checkAllclose(out2_ref, out_ck_qt,
                  msg=f'ck_moe_fused_2stages:{us:.2f} us, {token*model_dim*inter_dim*topk*2/us/1000/1000:.2f} tflops......(quant:{quant_dtype})', printNum=10000)




#    out_ck_nqt, us = ck_moe_fused_2stages(input,
#                                          shuffle_weight(w1, layout=(32, 32)),
#                                          shuffle_weight(w2, layout=(32, 32)),
#                                          topk_weights, topk_ids,
#                                          None, None,
#                                          #   block_size=BLOCK_SIZE_M
#                                          )
#    
#    checkAllclose(out_ref, out_ck_nqt,
#              msg=f'ck_moe_fused_2stages:{us:.2f} us, {token*model_dim*inter_dim*topk*2/us/1000/1000:.2f} tflops......(No quant)')

for dtype in [torch.float16]:
    for m in [1, 3, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096, 136000]:
        for dim in [6144]:
            for inter_dim in [4096]:
                expert, topk = 8, 2
                test_fmoe(dtype, m, dim, inter_dim, expert, topk,
                          quant='fp8_perTokenQuant', use_g1u1=True, activation=ActivationType.Silu)

for dtype in [torch.bfloat16]:
    for m in [64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096]:
        for dim in [6144]:
            for inter_dim in [4096]:
                expert, topk = 8, 2
                test_fmoe(dtype, m, dim, inter_dim, expert, topk,
                          quant='fp8_perTensorQuant', use_g1u1=True, activation=ActivationType.Gelu)