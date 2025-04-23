# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import math
import sys
import os
from typing import Any, Callable, Dict, Optional, Tuple
import itertools
import aiter
from aiter.test_common import (
    checkAllclose,
    perftest,
    benchmark,
    tensor_dump,
    run_perftest,
)
from aiter.int4_utils import *

from aiter.fused_moe_bf16_asm import get_block_size
from aiter.fused_moe import (
    fused_topk,
    get_inter_dim,
    torch_moe,
    moe_sorting,
    fused_moe,
    asm_stage1,
    ck_stage1,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.ops.shuffle import shuffle_weight
from aiter import ActivationType

torch.int4 = getattr(torch, "int4", torch.uint32)
torch.set_default_device("cuda")


@perftest()
def ck_moe_stage2(
    hidden_states,  # [M, topk, inter_dim]
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
        num_valid_ids,
        out,
        topk,
        w2_scale,
        a2_scale,
        block_size,
        sorted_weights,
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
    actType,
    qType,
    AQDType,
    WQDType,
    use_g1u1=False,
    doweight_stage1=False,
):
    torch_quant = aiter.get_torch_quant(qType)
    torch_act = aiter.get_torch_act(actType)
    input = torch.randn((token, model_dim), dtype=dtype) / math.sqrt(inter_dim)
    if use_g1u1:
        w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype)
    else:
        w1 = torch.randn((E, inter_dim, model_dim), dtype=dtype)
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype)

    score = torch.randn((token, E), dtype=dtype)
    topk_weights, topk_ids = fused_topk(input, score, topk, True)

    BLOCK_SIZE_M = 128
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, BLOCK_SIZE_M
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
    # w1_scale = w1_scale.fill_(1)
    # a1_scale = a1_scale.fill_(1)

    out1_ref, us_ref = run_perftest(
        torch_moe_stage1,
        a1_qt,
        w1_qt,
        w2_qt,
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=actType,
        quant_type=qType,
        a1_scale=a1_scale,
        w1_scale=w1_scale,
        num_iters=3,
        doweight=doweight_stage1,
    )

    if WQDType == torch.int4:  # int4 w quant
        w1_qt_aiter = rearrange_4bit_elements(convert_int8_to_uint32_int4(w1_qt_aiter))
        w2_qt_aiter = rearrange_4bit_elements(convert_int8_to_uint32_int4(w2_qt_aiter))
        w1_qt_aiter = shuffle_weight(w1_qt_aiter, layout=(32, 32))
        w2_qt_aiter = shuffle_weight(w2_qt_aiter, layout=(32, 32))
    else:
        w1_qt_aiter = shuffle_weight(w1_qt_aiter, layout=(16, 16))
        w2_qt_aiter = shuffle_weight(w2_qt_aiter, layout=(16, 16))

    # out1_ck = torch.empty((token, topk, inter_dim), dtype=dtype)
    # _, us = run_perftest(
    #     ck_stage1,
    #     a1_qt,
    #     w1_qt_aiter,
    #     w2_qt_aiter,
    #     sorted_ids,
    #     sorted_expert_ids,
    #     num_valid_ids,
    #     out1_ck,
    #     w1_scale=w1_scale,
    #     a1_scale=a1_scale,
    #     activation=actType,
    #     block_m=BLOCK_SIZE_M,
    #     sorted_weights=sorted_weights if doweight_stage1 else None
    # )

    # checkAllclose(
    #     out1_ref,
    #     out1_ck,
    #     msg=f"[perf]  ck_moe_stage1:{us:>8.2f} us, {token*model_dim*inter_dim*topk*2/us/1000/1000:>8.2f} tflops......(quant:{AQDType})",
    # )

    if WQDType != torch.int4:
        # asm int4 2 stage not support yet
        # if qType == aiter.QuantType.per_Tensor:
        #     # a1_scale = a1_scale.view(1).repeat(token)
        #     w1_scale = w1_scale.view(E, 1).repeat(1, w1.shape[-2])

        out1_asm = torch.empty((token, topk, inter_dim), dtype=dtype)
        _, us = run_perftest(
            asm_stage1,
            a1_qt,
            shuffle_weight(w1_qt, (16, 16)),
            shuffle_weight(w2_qt, (16, 16)),
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            out1_asm,
            kernelName="fmoe_stage1_bf16_pertokenFp8_doweight_g1u1_128x128_pf2",
            w1_scale=w1_scale,
            a1_scale=a1_scale,
            activation=actType,
            quant_type=qType,
            block_m=BLOCK_SIZE_M,
            sorted_weights=sorted_weights if doweight_stage1 else None
        )
        checkAllclose(
            out1_ref,
            out1_asm,
            msg=f"[perf] asm_moe_stage1:{us:>8.2f} us, {token*model_dim*inter_dim*topk*2/us/1000/1000:>8.2f} tflops......(quant:{AQDType})",
        )

    # ######################## stage 2 start ###########
    if qType == aiter.QuantType.per_Token:
        out1_ref = out1_ref.view(token, -1)
    a2_qt, a2_scale = torch_quant(out1_ref, quant_dtype=AQDType)
    # a2_scale = a2_scale.fill_(1)
    out2_ref, us_ref = run_perftest(
        torch_moe_stage2,
        a2_qt,
        w1_qt,  # E, inter_dim*2, model_dim
        w2_qt,  # E, model_dim, inter_dim
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=qType,
        w2_scale=w2_scale,
        a2_scale=a2_scale,
        num_iters=3,
        doweight=not doweight_stage1,
    )
    # # out_ref = torch_moe(
    # #     input,
    # #     w1_qt,
    # #     w2_qt,
    # #     topk_weights,
    # #     topk_ids,
    # #     fc1_scale=w1_scale,
    # #     fc2_scale=w2_scale,
    # # )
    # # checkAllclose(out_ref, out2_ref, msg="[torch] 1_stage vs 2_stage")

    # if qType == aiter.QuantType.per_Token:
    #     out1_ck = out1_ck.view(token, -1)
    # a2_qt, a2_scale = torch_quant(out1_ck, quant_dtype=AQDType)
    # a2_qt = a2_qt.view(token, topk, -1)
    # if qType == aiter.QuantType.No:
    #     a2_scale = torch.tensor(1.0, dtype=torch.float, device=a2_qt.device)
    # out2_ck, us = ck_moe_stage2(
    #     a2_qt,
    #     w1_qt_aiter,
    #     w2_qt_aiter,
    #     sorted_ids,
    #     sorted_expert_ids,
    #     sorted_weights,
    #     num_valid_ids,
    #     w2_scale,
    #     a2_scale,
    #     dtype,
    #     topk,
    #     BLOCK_SIZE_M,
    # )
    # checkAllclose(
    #     out2_ref,
    #     out2_ck,
    #     msg=f"ck_moe_stage2:{us:>8.2f} us, {token*model_dim*inter_dim*topk*2/us/1000/1000:>8.2f} tflops......(quant:{AQDType})",
    # )
    # ######################## stage 2 end ###########

    # ######################## fused 2 stage #########

    out2_aiter, us_fuse = run_perftest(
        fused_moe,
        input,
        w1_qt_aiter,
        w2_qt_aiter,
        topk_weights,
        topk_ids,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        quant_type=qType,
        activation=actType,
        doweight_stage1=doweight_stage1,
    )

    err = checkAllclose(
        out2_ref,
        out2_aiter,
        msg=f"aiter_all_stages:{us_fuse:>8.2f} us......",
    )
    return {"us": us_fuse, "err": err}


list_dtype = [torch.bfloat16, torch.float16][:1]
list_dim = [(6144, 4096)]
list_tokenNum = [
    1,
    3,
    5,
    16,
    32,
    64,
    128,
    256,
    1024,
    4096,
    163840,
]
list_quant = [
    # (aiter.QuantType.No, None, None),  # a16w16
    (aiter.QuantType.per_Tensor, torch.float8_e4m3fnuz, torch.float8_e4m3fnuz),  # a8w8
    # (aiter.QuantType.per_Token, torch.float8_e4m3fnuz, torch.float8_e4m3fnuz),  # a8w8
    # (aiter.QuantType.per_Token, torch.float8_e4m3fnuz, torch.int4),  # a8w4
]
list_act = [aiter.ActivationType.Silu, aiter.ActivationType.Gelu][:1]
list_doweight_stage1 = [True]
expert, topk = 8, 2

import pandas as pd

for (
    dtype,
    act_type,
    (quant_type, aq_dtype, wq_dtype),
    (model_dim, inter_dim),
    doweight_stage1
) in itertools.product(list_dtype, list_act, list_quant, list_dim, list_doweight_stage1):
    df = []
    for m in list_tokenNum:
        ret = test_fmoe(
            dtype,
            m,
            model_dim,
            inter_dim,
            expert,
            topk,
            act_type,
            quant_type,
            aq_dtype,
            wq_dtype,
            use_g1u1=True,
            doweight_stage1=doweight_stage1,
        )
        df.append(ret)
    df = pd.DataFrame(df)
    aiter.logger.info(f"summary:\n{df}")


# # per Tensor quant/a8w4
# for dtype in [torch.bfloat16]:
#     for m in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096][-1:]:
#         for dim in [6144]:
#             for inter_dim in [4096]:
#                 expert, topk = 8, 2
#                 test_fmoe(
#                     dtype,
#                     m,
#                     dim,
#                     inter_dim,
#                     expert,
#                     topk,
#                     aiter.ActivationType.Silu,
#                     aiter.QuantType.per_Token,
#                     torch.float8_e4m3fnuz,
#                     torch.int4,
#                     use_g1u1=True,
#                 )
