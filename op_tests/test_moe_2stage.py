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
from aiter.test_common import checkAllclose, perftest, benchmark
from op_tests.int4_utils import *

from aiter.fused_moe_bf16_asm import (
    fused_topk,
    asm_moe,
    torch_moe,
    moe_sorting_ck,
    ck_moe_2stages,
)
from aiter.ops.shuffle import shuffle_weight
from aiter import ActivationType

# torch.int4 = torch.uint32


@perftest(num_iters=3)
def torch_moe_stage1(
    hidden_states,
    w1,  # E, inter_dim*2, model_dim
    w2,  # E, model_dim, inter_dim
    topk_weight,
    topk_ids,
    dtype=torch.float16,
    # following for quant
    fc1_scale=None,  # [expert, inter_dim, 1]
    w1_scale=None,  # [1]
    a1_scale=None,  # [expert]]
    block_size=32,
):
    ctype = torch.float  # compute type
    hidden_states = hidden_states.to(ctype)
    w1 = w1.to(ctype)

    B, D = hidden_states.shape
    topk = topk_weight.shape[1]
    N = w1.shape[1]
    num_experts, model_dim, inter_dim = w2.shape

    max_num_tokens_padded = topk_ids.numel() + num_experts * block_size - topk

    # gose to quant D_w8a8/w8a8
    if fc1_scale is not None:
        w1 = (w1.view(-1, D) * fc1_scale.view(-1, 1)).view(num_experts, -1, D)
    if a1_scale is not None and w1_scale is not None:
        hidden_states = hidden_states * a1_scale
        w1 = w1 * w1_scale.view(w1_scale.shape[0], -1, 1)

    hidden_states = hidden_states.view(B, -1, D).repeat(1, topk, 1)

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


@perftest(num_iters=3)
def torch_moe_stage2(
    hidden_states,
    w1,  # E, inter_dim*2, model_dim
    w2,  # E, model_dim, inter_dim
    topk_weights,
    topk_ids,
    sorted_weights,
    sorted_ids,
    sorted_expert_ids,
    num_valid_ids,
    dtype=torch.float16,
    w2_scale=None,  # [1]
    a2_scale=None,  # [expert]]
    block_size=32,
):
    ctype = torch.float  # compute type
    hidden_states = hidden_states.to(ctype)
    w2 = w2.to(ctype)

    token_num, topk = topk_ids.shape
    # M, _ = hidden_states.shape
    num_experts, model_dim, inter_dim = w2.shape
    max_num_m_blocks = sorted_expert_ids.shape[0]
    hidden_states = hidden_states.view(token_num, topk, inter_dim)

    # gose to quant D_w8a8/w8a8
    if a2_scale is not None and w2_scale is not None:
        hidden_states = hidden_states * a2_scale.view(a2_scale.shape[0], -1, 1)
        w2 = w2 * w2_scale.view(num_experts, -1, 1)

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
def ck_moe_stage1(
    hidden_states,
    w1,  # [E, inter_dim*2, model_dim]
    w2,  # [E, model_dim, inter_dim]
    sorted_token_ids,  # [max_num_tokens_padded]
    sorted_expert_ids,  # [max_num_m_blocks]
    num_valid_ids,  # [1]
    w1_scale,
    a1_scale,
    dtype,
    topk,
    block_size=32,
):
    token_num = hidden_states.shape[0]
    D = w1.shape[1]
    # num_experts, model_dim, inter_dim = w2.shape
    max_num_tokens_padded = sorted_token_ids.shape[0]
    # max_num_tokens_padded = sorted_expert_ids.shape[0]*block_size

    out = torch.empty(
        (token_num, topk, D),
        dtype=dtype,
        device=hidden_states.device,
    )
    aiter.ck_moe_stage1(
        hidden_states,
        w1,
        w2,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        out,
        topk,
        w1_scale,
        a1_scale,
        block_size,
    )
    tmp = torch.empty(
        (token_num, topk, int(D / 2)), dtype=dtype, device=hidden_states.device
    )
    aiter.silu_and_mul(tmp, out)
    out = tmp
    return out


@perftest()
def ck_moe_stage2(
    hidden_states,
    w1,  # [E, inter_dim*2, model_dim]
    w2,  # [E, model_dim, inter_dim]
    sorted_token_ids,  # [max_num_tokens_padded]
    sorted_expert_ids,  # [max_num_m_blocks]
    sorted_weights,  # [max_num_tokens_padded]
    num_valid_ids,  # [1]
    w2_scale,
    a2_scale,
    dtype,
    topk,
    block_size=32,
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
    aiter.ck_moe_stage2(
        hidden_states,
        w1,
        w2,
        sorted_token_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        out,
        topk,
        w2_scale,
        a2_scale,
        block_size,
    )
    return out


@perftest()
def ck_moe_fused_2stages(
    hidden_states,
    # [expert(local_expert:EP), inter_dim(*2), dim] N,K
    w1,
    w2,  # [expert(local_expert:EP), dim, inter_dim]
    topk_weight,
    topk_ids,
    # following for int8 quant
    # [expert(local_expert:EP), inter_dim, 1]
    fc1_scale=None,
    # [expert(local_expert:EP), model_dim, 1]
    fc2_scale=None,
    block_size=32,
    a1_scale=None,
):
    return ck_moe_2stages(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        fc1_scale,
        fc2_scale,
        block_size=block_size,
        a1_scale=a1_scale,
    )


@perftest()
def asm_moe_stage1(
    hidden_states,
    w1,  # [E, inter_dim*2, model_dim]
    w2,  # [E, model_dim, inter_dim]
    sorted_token_ids,  # [max_num_tokens_padded]
    sorted_weights,
    sorted_expert_ids,  # [max_num_m_blocks]
    num_valid_ids,  # [1]
    w1_scale,
    a1_scale,
    dtype,
    topk,
    block_size=128,
):

    token_num = hidden_states.shape[0]
    D = w1.shape[1]
    D = int(D / 2)

    out = torch.empty(
        (token_num, topk, D),
        dtype=dtype,
        device=hidden_states.device,
    )

    aiter.moe_stage1_fp8_g1u1(
        hidden_states,
        w1,
        w2,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        out,
        "",
        block_size,
        a1_scale,
        w1_scale,
    )
    return out


@benchmark()
def test_fmoe(
    dtype,
    token,
    model_dim,
    inter_dim,
    E,
    topk,
    qType,
    AQDType,
    WQDType,
    BLOCK_SIZE_M,
    use_g1u1=False,
):
    torch_quant = aiter.get_torch_quant(qType)
    input = torch.randn((token, model_dim), dtype=dtype, device="cuda")
    if use_g1u1:
        w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype, device="cuda")
    else:
        w1 = torch.randn((E, inter_dim, model_dim), dtype=dtype, device="cuda")
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype, device="cuda")

    score = torch.randn((token, E), device="cuda", dtype=dtype)
    topk_weights, topk_ids = fused_topk(input, score, topk, True)

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = (
        moe_sorting_ck(topk_ids, topk_weights, E, model_dim, dtype, BLOCK_SIZE_M)
    )
    if qType == aiter.QuantType.per_Tensor:
        w1_qt, w1_scale = aiter.pertoken_quant(w1.view(E, -1), quant_dtype=WQDType)
        w2_qt, w2_scale = aiter.pertoken_quant(w2.view(E, -1), quant_dtype=WQDType)
    elif qType == aiter.QuantType.per_Token and WQDType == torch.int4:  # int4 w quant
        w1_qt, w1_scale = aiter.pertoken_quant(w1, quant_dtype=torch.int8, dtypeMax=7)
        w2_qt, w2_scale = aiter.pertoken_quant(w2, quant_dtype=torch.int8, dtypeMax=7)
    else:
        w1_qt, w1_scale = torch_quant(w1, quant_dtype=WQDType)
        w2_qt, w2_scale = torch_quant(w2, quant_dtype=WQDType)
    w1_qt = w1_qt_aiter = w1_qt.view(w1.shape)
    w2_qt = w2_qt_aiter = w2_qt.view(w2.shape)

    a1_qt, a1_scale = torch_quant(input, quant_dtype=AQDType)

    out1_ref, us_ref = torch_moe_stage1(
        a1_qt,
        w1_qt,
        w2_qt,
        topk_weights,
        topk_ids,
        dtype=dtype,
        fc1_scale=None,
        w1_scale=w1_scale,
        a1_scale=a1_scale,
        block_size=BLOCK_SIZE_M,
    )
    if use_g1u1:
        gate, up = out1_ref.split([inter_dim, inter_dim], dim=-1)
        out1_ref = F.silu(gate) * up
    else:
        out1_ref = F.gelu(out1_ref)

    # out_ref = torch_moe(input, w1, w2, topk_weights, topk_ids)
    # checkAllclose(out_ref, out2_ref, msg="[torch] 1_stage vs 2_stage")
    if WQDType == torch.int4:  # int4 w quant
        w1_qt_aiter = rearrange_4bit_elements(convert_int8_to_uint32_int4(w1_qt_aiter))
        w2_qt_aiter = rearrange_4bit_elements(convert_int8_to_uint32_int4(w2_qt_aiter))
    w1_qt_aiter = shuffle_weight(w1_qt_aiter, layout=(32, 32))
    w2_qt_aiter = shuffle_weight(w2_qt_aiter, layout=(32, 32))

    out1_ck, us = ck_moe_stage1(
        a1_qt,
        w1_qt_aiter,
        w2_qt_aiter,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        w1_scale,
        a1_scale,
        dtype,
        topk,
        BLOCK_SIZE_M,
    )

    checkAllclose(
        out1_ref,
        out1_ck,
        msg=f"[perf]  ck_moe_stage1:{us:.2f} us, {token*model_dim*inter_dim*topk*2/us/1000/1000:.2f} tflops......(quant:{AQDType})",
    )

    if WQDType != torch.int4:
        # asm int4 2 stage not support yet
        if qType == aiter.QuantType.per_Tensor:
            a1_scale = a1_scale.view(1).repeat(token)
            w1_scale = w1_scale.view(E, 1).repeat(1, w1.shape[-2])
        out1_asm, us = asm_moe_stage1(
            a1_qt,
            shuffle_weight(w1_qt, (16, 16)),
            shuffle_weight(w2_qt, (16, 16)),
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            w1_scale,
            a1_scale,
            dtype,
            topk,
            BLOCK_SIZE_M,
        )
        checkAllclose(
            out1_ref,
            out1_asm,
            msg=f"[perf] asm_moe_stage1:{us:.2f} us, {token*model_dim*inter_dim*topk*2/us/1000/1000:.2f} tflops......(quant:{AQDType})",
        )

    # ######################## stage 2 start ###########
    if qType == aiter.QuantType.per_Token:
        out1_ref = out1_ref.view(token, -1)
    a2_qt, a2_scale = torch_quant(out1_ref, quant_dtype=AQDType)
    out2_ref, us_ref = torch_moe_stage2(
        a2_qt,
        w1_qt,  # E, inter_dim*2, model_dim
        w2_qt,  # E, model_dim, inter_dim
        topk_weights,
        topk_ids,
        sorted_weights,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        dtype=dtype,
        # [expert, inter_dim, 1]
        w2_scale=w2_scale,
        a2_scale=a2_scale,
        block_size=BLOCK_SIZE_M,
    )

    if qType == aiter.QuantType.per_Token:
        out1_ck = out1_ck.view(token, -1)
    a2_qt, a2_scale = torch_quant(out1_ck, quant_dtype=AQDType)
    a2_qt = a2_qt.view(token, topk, -1)
    out2_ck, us = ck_moe_stage2(
        a2_qt,
        w1_qt_aiter,
        w2_qt_aiter,
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        w2_scale,
        a2_scale,
        dtype,
        topk,
        BLOCK_SIZE_M,
    )
    checkAllclose(
        out2_ref,
        out2_ck,
        msg=f"ck_moe_stage2:{us:.2f} us, {token*model_dim*inter_dim*topk*2/us/1000/1000:.2f} tflops......(quant:{AQDType})",
    )
    # # ######################## stage 2 end ###########


# per Token quant/a8w8
for dtype in [torch.bfloat16]:
    for m in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096][-1:]:
        for dim in [6144]:
            for inter_dim in [4096]:
                expert, topk = 8, 2
                test_fmoe(
                    dtype,
                    m,
                    dim,
                    inter_dim,
                    expert,
                    topk,
                    aiter.QuantType.per_Token,
                    torch.float8_e4m3fnuz,
                    torch.float8_e4m3fnuz,
                    BLOCK_SIZE_M=128,
                    use_g1u1=True,
                )

# per Tensor quant/a8w8
for dtype in [torch.bfloat16]:
    for m in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096]:
        for dim in [6144]:
            for inter_dim in [4096]:
                expert, topk = 8, 2
                test_fmoe(
                    dtype,
                    m,
                    dim,
                    inter_dim,
                    expert,
                    topk,
                    aiter.QuantType.per_Tensor,
                    torch.float8_e4m3fnuz,
                    torch.float8_e4m3fnuz,
                    BLOCK_SIZE_M=32,
                    use_g1u1=True,
                )
# per Tensor quant/a8w4
for dtype in [torch.bfloat16]:
    for m in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096]:
        for dim in [6144]:
            for inter_dim in [4096]:
                expert, topk = 8, 2
                test_fmoe(
                    dtype,
                    m,
                    dim,
                    inter_dim,
                    expert,
                    topk,
                    aiter.QuantType.per_Token,
                    torch.float8_e4m3fnuz,
                    torch.int4,
                    BLOCK_SIZE_M=32,
                    use_g1u1=True,
                )
