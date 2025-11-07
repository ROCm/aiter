# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import itertools
import aiter
from aiter import dtypes
from aiter.test_common import checkAllclose, benchmark, run_perftest
from aiter.int4_utils import *
from aiter.utility import fp4_utils
from aiter.jit.utils.chip_info import get_gfx
import argparse
import pandas as pd
import numpy as np

from aiter.fused_moe import (
    fused_topk,
    moe_sorting,
    fused_moe,
    torch_moe_stage1,
    torch_moe_stage2,
    get_block_size_M,
)


from aiter.ops.shuffle import (
    shuffle_weight,
    shuffle_scale_a16w4,
    shuffle_weight_a16w4,
    shuffle_weight_NK,
)
from aiter import ActivationType

torch.int4 = getattr(torch, "int4", torch.uint32)
torch.set_default_device("cuda")


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
    Activation=ActivationType.Gelu,
    quant_type=aiter.QuantType.No,
    sorted_weights=None,  # [max_num_tokens_padded]
):
    token_num = hidden_states.shape[0]
    D = w2.shape[-1]
    # max_num_tokens_padded = sorted_expert_ids.shape[0]*block_size

    if w1.dtype is torch.uint32:
        D = D * 8

    out = torch.empty((token_num, topk, D), dtype=dtype)

    aiter.ck_moe_stage1_fwd(
        hidden_states,
        w1,
        w2,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        out,
        topk,
        "",
        w1_scale,
        a1_scale,
        block_size,
        sorted_weights,
        quant_type,
        Activation,
    )

    return out


def ck_moe_stage2(
    hidden_states,
    w1,  # [E, inter_dim*2, model_dim]
    w2,  # [E, model_dim, inter_dim]
    sorted_token_ids,  # [max_num_tokens_padded]
    sorted_expert_ids,  # [max_num_m_blocks]
    num_valid_ids,  # [1]
    w2_scale,
    a2_scale,
    dtype,
    topk,
    block_size=32,
    Activation=ActivationType.Gelu,
    quant_type=aiter.QuantType.No,
    sorted_weights=None,  # [max_num_tokens_padded]
):
    token_num = hidden_states.shape[0]
    D = w2.shape[1]
    # max_num_tokens_padded = sorted_expert_ids.shape[0]*block_size

    out = torch.zeros(
        (token_num, D),
        dtype=dtype,
        device=hidden_states.device,
    )
    aiter.ck_moe_stage2_fwd(
        hidden_states,
        w1,
        w2,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        out,
        topk,
        "",
        w2_scale,
        a2_scale,
        block_size,
        sorted_weights,
        quant_type,
        Activation,
    )
    return out


def cktile_moe_stage1(
    hidden_states,
    w1,  # [E, inter_dim*2, model_dim]
    w2,  # [E, model_dim, inter_dim]
    sorted_token_ids,  # [max_num_tokens_padded]
    sorted_expert_ids,  # [max_num_m_blocks]
    num_valid_ids,  # [1]
    w1_scale,
    a1_scale,
    exp_bias1,
    dtype,
    topk,
    n_pad_zeros=0,
    k_pad_zeros=0,
    block_size=32,
    Activation=ActivationType.Silu,
    quant_type=aiter.QuantType.No,
    sorted_weights=None,  # [max_num_tokens_padded]
):
    token_num = hidden_states.shape[0]
    _, n1, k1 = w1.shape
    _, k2, n2 = w2.shape
    D = n2 if k2 == k1 else n2 * 2  # bit4 format
    # max_num_tokens_padded = sorted_expert_ids.shape[0]*block_size

    if w1.dtype is torch.uint32:
        D = D * 8
    out = torch.empty((token_num, topk, D), dtype=dtype)
    # print("Run cktile_moe_stage1: M=%d, N(N*2)=%d, K=%d, topk=%d, expert=%d"%(token_num, w1.shape[1], hidden_states.shape[1], topk, w1.shape[0]))
    aiter.moe_cktile2stages_gemm1(
        hidden_states,
        w1,
        out,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        n_pad_zeros,
        k_pad_zeros,
        sorted_weights,
        a1_scale,
        w1_scale,
        exp_bias1,
        block_size,
    )
    return out


def cktile_moe_stage2(
    hidden_states,
    w1,  # [E, inter_dim*2, model_dim]
    w2,  # [E, model_dim, inter_dim]
    sorted_token_ids,  # [max_num_tokens_padded]
    sorted_expert_ids,  # [max_num_m_blocks]
    num_valid_ids,  # [1]
    w2_scale,
    a2_scale,
    exp_bias2,
    dtype,
    topk,
    n_pad_zeros=0,
    k_pad_zeros=0,
    block_size=32,
    Activation=ActivationType.Silu,
    quant_type=aiter.QuantType.No,
    sorted_weights=None,  # [max_num_tokens_padded]
    zeros_out=False,
):
    token_num = hidden_states.shape[0]
    D = w2.shape[1]
    # max_num_tokens_padded = sorted_expert_ids.shape[0]*block_size

    out = torch.empty(
        (token_num, D),
        dtype=dtype,
        device=hidden_states.device,
    )
    if zeros_out:
        out.fill_(0)
    # print("Run cktile_moe_stage2: M=%d, N=%d, K=%d, topk=%d, expert=%d"%(hidden_states.shape[0]*hidden_states.shape[1], w2.shape[1], hidden_states.shape[2], topk, w2.shape[0]))
    aiter.moe_cktile2stages_gemm2(
        hidden_states,
        w2,
        out,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        n_pad_zeros,
        k_pad_zeros,
        sorted_weights,
        a2_scale,
        w2_scale,
        exp_bias2,
        block_size,
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
    hidden_pad=0,
    intermediate_pad=0,
):
    if get_gfx() not in ["gfx950"] and qType == aiter.QuantType.per_1x32:
        return
    torch_quant = aiter.get_torch_quant(qType)
    torch_act = aiter.get_torch_act(actType)
    input = torch.randn((token, model_dim), dtype=dtype)
    if use_g1u1:
        w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype)
        if hidden_pad != 0 and intermediate_pad != 0:
            w1[:, :, -hidden_pad:] = 0
            w1[:, -intermediate_pad:, :] = 0
            w1[:, inter_dim - intermediate_pad : inter_dim, :] = 0
        exp_bias1 = torch.clamp(torch.randn((E, inter_dim * 2), dtype=dtype), -1.0, 1.0)
    else:
        w1 = torch.randn((E, inter_dim, model_dim), dtype=dtype)
        exp_bias1 = torch.clamp(torch.randn((E * inter_dim), dtype=dtype), -1.0, 1.0)
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype)
    if hidden_pad != 0 and intermediate_pad != 0:
        w2[:, :, -intermediate_pad:] = 0
        w2[:, -hidden_pad:, :] = 0
    exp_bias2 = torch.clamp(torch.randn((E, model_dim), dtype=dtype), -1.0, 1.0)
    score = torch.randn((token, E), dtype=dtype)
    topk_weights, topk_ids = fused_topk(input, score, topk, True)
    # sequence
    # topk_ids_list = [[((i * topk) + j)% E for j in range(topk)] for i in range(token)]
    # topk_ids = torch.tensor(topk_ids_list, device=topk_ids.device, dtype=topk_ids.dtype)

    M, _ = topk_ids.shape

    BLOCK_SIZE_M = get_block_size_M(M, topk, E, inter_dim)
    if (
        qType == aiter.QuantType.per_1x32
        and (AQDType in [dtypes.bf16, dtypes.fp16])
        and (WQDType == dtypes.fp4x2)
    ):  # a16w4
        BLOCK_SIZE_M = 32 if M > 1024 else 16
    if qType == aiter.QuantType.per_128x128:
        BLOCK_SIZE_M = 64 if M > 64 else 16
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, BLOCK_SIZE_M
    )

    if qType == aiter.QuantType.per_Tensor:
        w1_qt, w1_scale = aiter.pertoken_quant(w1.view(E, -1), quant_dtype=WQDType)
        w2_qt, w2_scale = aiter.pertoken_quant(w2.view(E, -1), quant_dtype=WQDType)
        w1_qt = w1_qt.view(w1.shape)
        w2_qt = w2_qt.view(w2.shape)
    elif qType == aiter.QuantType.per_Token and WQDType == torch.int4:  # int4 w quant
        w1_qt, w1_scale = aiter.pertoken_quant(w1, quant_dtype=dtypes.i8, dtypeMax=7)
        w2_qt, w2_scale = aiter.pertoken_quant(w2, quant_dtype=dtypes.i8, dtypeMax=7)
    elif qType == aiter.QuantType.per_128x128:

        def weight_per_128x128_quant(weight, quant_dtype):
            E, dim1, dim2 = weight.shape
            weight_blocks = weight.view(
                E, dim1 // 128, 128, dim2 // 128, 128
            )  # [E, num_blocks_dim1, 128, num_blocks_dim2, 128]
            weight_blocks = weight_blocks.permute(
                0, 1, 3, 2, 4
            ).contiguous()  # [E, num_blocks_dim1, num_blocks_dim2, 128, 128]
            weight_blocks = weight_blocks.view(
                E, -1, 128 * 128
            )  # [E, num_blocks, 128*128]
            weight_qt, weight_scale = aiter.pertoken_quant(
                weight_blocks, quant_dtype=quant_dtype
            )
            weight_qt = weight_qt.view(
                E, dim1 // 128, dim2 // 128, 128, 128
            )  # [E, num_blocks_dim1, num_blocks_dim2, 128, 128]
            weight_qt = weight_qt.permute(
                0, 1, 3, 2, 4
            ).contiguous()  # [E, num_blocks_dim1, 128, num_blocks_dim2, 128]
            weight_qt = weight_qt.view(E, dim1, dim2)  # [E, dim1, dim2]
            weight_scale = weight_scale.view(
                E, dim1 // 128, dim2 // 128
            )  # [E, num_blocks_dim1, num_blocks_dim2]
            return weight_qt, weight_scale

        w1_qt, w1_scale = weight_per_128x128_quant(w1, quant_dtype=WQDType)
        w2_qt, w2_scale = weight_per_128x128_quant(w2, quant_dtype=WQDType)
    else:
        w1_qt, w1_scale = torch_quant(w1, quant_dtype=WQDType)
        w2_qt, w2_scale = torch_quant(w2, quant_dtype=WQDType)

    if qType != aiter.QuantType.per_1x32:
        w1_qt = w1_qt_aiter = w1_qt.view(w1.shape)
        w2_qt = w2_qt_aiter = w2_qt.view(w2.shape)
    else:
        w1_qt = w1_qt_aiter = w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2)
        w2_qt = w2_qt_aiter = w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2)

    # Quant-ing a
    if qType == aiter.QuantType.per_128x128:
        a1_qt, a1_scale = aiter.pertoken_quant(
            input.view(token, -1, 128), quant_dtype=AQDType
        )
        a1_qt = a1_qt.view(token, model_dim)
        a1_scale = a1_scale.squeeze(-1)
    elif (
        qType == aiter.QuantType.per_1x32
        and (AQDType in [dtypes.bf16, dtypes.fp16])
        and WQDType == dtypes.fp4x2
    ):  # a16w4
        a1_qt = input.to(AQDType)
        a1_scale = None
    else:
        a1_qt, a1_scale = torch_quant(input, quant_dtype=AQDType)

    # bias dtype convert
    if (
        qType == aiter.QuantType.per_1x32
        and (AQDType in [dtypes.bf16, dtypes.fp16])
        and (WQDType == dtypes.fp4x2)
    ):  # a16w4
        exp_bias1_aiter = exp_bias1.to(dtypes.fp32)
        exp_bias2_aiter = exp_bias2.to(dtypes.fp32)
    else:
        exp_bias1_aiter = exp_bias1 = None
        exp_bias2_aiter = exp_bias2 = None

    # pre-shuffle
    w1_scale_aiter = w1_scale
    w2_scale_aiter = w2_scale
    if WQDType == torch.int4:  # int4 w quant
        w1_qt_aiter = rearrange_4bit_elements(
            convert_int8_to_uint32_int4(
                shuffle_weight(w1_qt_aiter, (16, 16), use_int4=True)
            )
        )
        w2_qt_aiter = rearrange_4bit_elements(
            convert_int8_to_uint32_int4(
                shuffle_weight(w2_qt_aiter, (16, 16), use_int4=True)
            )
        )
        w1_scale_aiter = fp4_utils.e8m0_shuffle(w1_scale)
        w2_scale_aiter = fp4_utils.e8m0_shuffle(w2_scale)
    elif (
        qType == aiter.QuantType.per_1x32
        and (AQDType in [dtypes.bf16, dtypes.fp16])
        and (WQDType == dtypes.fp4x2)
    ):  # a16w4
        w1_qt_aiter = shuffle_weight_a16w4(w1_qt_aiter, 16, True)
        w1_scale_aiter = shuffle_scale_a16w4(w1_scale, E, True)
        w2_qt_aiter = shuffle_weight_a16w4(w2_qt_aiter, 16, False)
        w2_scale_aiter = shuffle_scale_a16w4(w2_scale, E, False)
    # elif WQDType != dtypes.fp4x2 and (get_gfx() in ["gfx950"]):
    #     inst_K = 128 // w1_qt_aiter.element_size()
    #     w1_qt_aiter = shuffle_weight_NK(w1_qt_aiter, 16, inst_K)
    #     w2_qt_aiter = shuffle_weight_NK(w2_qt_aiter, 16, inst_K)
    elif WQDType != dtypes.fp4x2:
        w1_qt_aiter = shuffle_weight(w1_qt_aiter, layout=(16, 16))
        w2_qt_aiter = shuffle_weight(w2_qt_aiter, layout=(16, 16))
        w1_scale_aiter = fp4_utils.e8m0_shuffle(w1_scale)
        w2_scale_aiter = fp4_utils.e8m0_shuffle(w2_scale)

    # # ######################## stage 1 start ###########
    out1_ref = torch_moe_stage1(
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
        w1_bias=exp_bias1,
        doweight=doweight_stage1,
    )

    # # ######################## ck stage 1 start ###########
    # if WQDType == dtypes.fp4x2 or AQDType == dtypes.fp4x2:
    #     out1_ck = torch.zeros((token, topk, inter_dim), dtype=dtype)
    # else:
    #     out1_ck = torch.empty((token, topk, inter_dim), dtype=dtype)
    # if (
    #     qType == aiter.QuantType.per_1x32
    #     and (AQDType in [dtypes.bf16, dtypes.fp16])
    #     and (WQDType == dtypes.fp4x2)
    # ):  # a16w4:
    #     npad0 = intermediate_pad // 64 * 64
    #     kpad0 = hidden_pad // 128 * 128
    #     out1_ck, us1 = run_perftest(
    #         cktile_moe_stage1,
    #         a1_qt,
    #         w1_qt_aiter,
    #         w2_qt_aiter,
    #         sorted_ids,
    #         sorted_expert_ids,
    #         num_valid_ids,
    #         w1_scale_aiter,
    #         a1_scale,
    #         exp_bias1_aiter,
    #         dtype,
    #         topk,
    #         npad0 * 2,
    #         kpad0,
    #         BLOCK_SIZE_M,
    #         actType,
    #         quant_type=qType,
    #         sorted_weights=sorted_weights if doweight_stage1 else None,
    #         # needTrace=True,
    #         # num_iters=2,
    #         # num_warmup=0,
    #     )
    # else:
    #     out1_ck, us1 = run_perftest(
    #         ck_moe_stage1,
    #         a1_qt,
    #         w1_qt_aiter,
    #         w2_qt_aiter,
    #         sorted_ids,
    #         sorted_expert_ids,
    #         num_valid_ids,
    #         w1_scale,
    #         a1_scale,
    #         dtype,
    #         topk,
    #         BLOCK_SIZE_M,
    #         actType,
    #         quant_type=qType,
    #         sorted_weights=sorted_weights if doweight_stage1 else None,
    #         needTrace=True,
    #     )

    # checkAllclose(
    #     out1_ref,
    #     out1_ck,
    #     msg=f"[perf]  ck_moe_stage1:{us1:>8.2f} us, {token*model_dim*inter_dim*2*topk*2/us1/1000/1000:>8.2f} tflops......(quant:{AQDType})",
    # )

    # diff = torch.abs(out1_ref - out1_ck)
    # max_value= diff.max()
    # multi_index = np.unravel_index(torch.argmax(diff).item(), diff.shape)
    # print("max_diff", max_value.item(), ",ref=", out1_ref[multi_index].item(), ",ck=", out1_ck[multi_index].item())
    # ######################## stage 1 end ###########

    # if WQDType != torch.int4:
    #     # asm int4 2 stage not support yet
    #     if qType == aiter.QuantType.per_Tensor:
    #         a1_scale = a1_scale.view(1).repeat(token)
    #         w1_scale = w1_scale.view(E, 1).repeat(1, w1.shape[-2])

    #     out1_asm = torch.empty((token, topk, inter_dim), dtype=dtype)
    #     _, us = run_perftest(
    #         asm_stage1,
    #         a1_qt,
    #         shuffle_weight(w1_qt, (16, 16)),
    #         shuffle_weight(w2_qt, (16, 16)),
    #         sorted_ids,
    #         sorted_expert_ids,
    #         num_valid_ids,
    #         out1_asm,
    #         topk,
    #         kernelName="fmoe_stage1_bf16_pertokenFp8_g1u1_128x128_pf2",
    #         w1_scale=w1_scale,
    #         a1_scale=a1_scale,
    #         activation=actType,
    #         quant_type=qType,
    #         block_m=BLOCK_SIZE_M,
    #     )
    #     checkAllclose(
    #         out1_ref,
    #         out1_asm,
    #         msg=f"[perf] asm_moe_stage1:{us:>8.2f} us, {token*model_dim*inter_dim*topk*2/us/1000/1000:>8.2f} tflops......(quant:{AQDType})",
    #     )

    # ######################## stage 2 start ###########
    if qType == aiter.QuantType.per_128x128:
        a2_qt, a2_scale = aiter.pertoken_quant(
            out1_ref.view(token, -1, 128), quant_dtype=AQDType
        )
        a2_scale = a2_scale.view(token, topk, -1)
    elif (
        qType == aiter.QuantType.per_1x32
        and (AQDType in [dtypes.bf16, dtypes.fp16])
        and (WQDType == dtypes.fp4x2)
    ):  # a16w4
        a2_qt = out1_ref
        a2_scale = None
    else:
        a2_qt, a2_scale = torch_quant(out1_ref, quant_dtype=AQDType)
    a2_qt = a2_qt.view(token, topk, -1)

    out2_ref = torch_moe_stage2(
        a2_qt,
        w1_qt,  # E, inter_dim*2, model_dim
        w2_qt,  # E, model_dim, inter_dim
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=qType,
        w2_scale=w2_scale,
        a2_scale=a2_scale,
        w2_bias=exp_bias2,
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

    # out2_ck = torch.empty((token, model_dim), dtype=dtype)
    # if (
    #     qType == aiter.QuantType.per_1x32
    #     and (AQDType in [dtypes.bf16, dtypes.fp16])
    #     and (WQDType == dtypes.fp4x2)
    # ):  # a16w4
    #     npad0 = hidden_pad // 64 * 64
    #     kpad0 = intermediate_pad // 128 * 128
    #     _, us2 = run_perftest(
    #         cktile_moe_stage2,
    #         a2_qt,
    #         w1_qt_aiter,
    #         w2_qt_aiter,
    #         sorted_ids,
    #         sorted_expert_ids,
    #         num_valid_ids,
    #         w2_scale_aiter,
    #         a2_scale,
    #         exp_bias2_aiter,
    #         dtype,
    #         topk,
    #         npad0,
    #         kpad0,
    #         BLOCK_SIZE_M,
    #         actType,
    #         quant_type,
    #         sorted_weights if not doweight_stage1 else None,
    #         # needTrace=True,
    #         # num_iters=2,
    #         # num_warmup=0,
    #     )
    #     out2_ck = cktile_moe_stage2(
    #         a2_qt,
    #         w1_qt_aiter,
    #         w2_qt_aiter,
    #         sorted_ids,
    #         sorted_expert_ids,
    #         num_valid_ids,
    #         w2_scale_aiter,
    #         a2_scale,
    #         exp_bias2_aiter,
    #         dtype,
    #         topk,
    #         npad0,
    #         kpad0,
    #         BLOCK_SIZE_M,
    #         actType,
    #         quant_type,
    #         sorted_weights if not doweight_stage1 else None,
    #         True,
    #     )
    # else:
    #     out2_ck, us2 = run_perftest(
    #         ck_moe_stage2,
    #         a2_qt,
    #         w1_qt_aiter,
    #         w2_qt_aiter,
    #         sorted_ids,
    #         sorted_expert_ids,
    #         num_valid_ids,
    #         w2_scale,
    #         a2_scale,
    #         dtype,
    #         topk,
    #         BLOCK_SIZE_M,
    #         actType,
    #         quant_type,
    #         sorted_weights if not doweight_stage1 else None,
    #     )

    # checkAllclose(
    #     out2_ref,
    #     out2_ck,
    #     msg=f"[perf]  ck_moe_stage2:{us2:>8.2f} us, {token*model_dim*inter_dim*topk*2/us2/1000/1000:>8.2f} tflops......(quant:{AQDType})",
    # )

    # diff = torch.abs(out2_ref - out2_ck)
    # max_value= diff.max()
    # multi_index = np.unravel_index(torch.argmax(diff).item(), diff.shape)
    # print("max_diff", max_value.item(), ",ref=", out2_ref[multi_index].item(), ",ck=", out2_ck[multi_index].item())
    # ######################## stage 2 end ###########
    us1 = 0
    out2_ck, us2 = run_perftest(
        fused_moe,
        input,
        w1_qt_aiter,
        w2_qt_aiter,
        topk_weights,
        topk_ids,
        w1_scale=w1_scale_aiter,
        w2_scale=w2_scale_aiter,
        quant_type=qType,
        activation=actType,
        doweight_stage1=doweight_stage1,
        intermediate_pad=intermediate_pad,
        hidden_pad=hidden_pad,
        bias1=exp_bias1_aiter,
        bias2=exp_bias2_aiter,
    )
    err = checkAllclose(
        out2_ref,
        out2_ck,
        msg=f"ck_moe_2stages:{us2:>8.2f} us, {token*model_dim*inter_dim*3*topk*2/us2/1000/1000:>8.2f} tflops......(quant:{AQDType})",
    )

    # return {"gemm1(us)": us1, "gemm2(us)": us2}
    # def calc_diff(x: torch.Tensor, y: torch.Tensor):
    #     x, y = x.double(), y.double()
    #     denominator = (x * x + y * y).sum()
    #     sim = 2 * (x * y).sum() / denominator
    #     return 1 - sim

    # logits_diff = calc_diff(out2_ref, out2_ck)
    # assert logits_diff < 1e-3

    # return {"gemm1(us)": us1, "gemm2(us)": us2, "err": err}
    return {"us": us2, "err": err}


l_dtype = ["bf16", "fp16"][:1]
# l_dim = [(6144, 4096)]
l_dim = [(7168, 256)]
# l_dim = [(3072, 3072)]
l_tokenNum = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    1024,
    2048,
    3072,
    4096,
    8192,
    163840,
]
l_act = [aiter.ActivationType.Silu, aiter.ActivationType.Gelu]
l_quant = [
    (aiter.QuantType.No, None, None), # a16w16
    (aiter.QuantType.per_Tensor, dtypes.fp8, dtypes.fp8),  # a8w8
    (aiter.QuantType.per_Token, dtypes.fp8, dtypes.fp8),  # a8w8
    (aiter.QuantType.per_Token, dtypes.fp8, torch.int4),  # a8w4
    # (aiter.QuantType.per_1x32, dtypes.fp4x2, dtypes.fp4x2),  # a4w4
    (aiter.QuantType.per_128x128, dtypes.fp8, dtypes.fp8),  # a8w8
    (aiter.QuantType.per_1x32, dtypes.bf16, dtypes.fp4x2),  # a16w4
]
l_doweight_stage1 = [False, True][:1]
l_hidden_intermediate_pad = [(0, 0), (65, 65), (129, 191)]


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=l_dtype,
    nargs="?",
    const=None,
    default=None,
    help="""Data type.
    e.g.: -d bf16""",
)

parser.add_argument(
    "-dim",
    type=dtypes.str2tuple,
    nargs="?",
    const=None,
    default=None,
    help="""Model dimension.
    e.g.: -dim 6144,4096""",
)

parser.add_argument(
    "-t",
    "--tokenNum",
    type=int,
    nargs="?",
    const=None,
    default=None,
    help="""Number of tokens.
    e.g.: -t 1024""",
)

parser.add_argument(
    "-q",
    "--quant",
    type=int,
    choices=range(len(l_quant)),
    help="""select quantization type:
    0 : aiter.QuantType.No, None, None),  # a16w16
    1: aiter.QuantType.per_Tensor, dtypes.fp8, dtypes.fp8  # a8w8
    2: aiter.QuantType.per_Token, dtypes.fp8, dtypes.fp8  # a8w8
    3: aiter.QuantType.per_Token, dtypes.fp8, torch.int4  # a8w4
    4: aiter.QuantType.per_1x32, dtypes.fp4x2, dtypes.fp4x2  # a4w4
    5: aiter.QuantType.per_128x128, dtypes.fp8, dtypes.fp8,  # a8w8""",
)

parser.add_argument(
    "-a",
    "--act",
    type=str,
    choices=["silu", "gelu"],
    default=None,
    help="""Select activation type.
    e.g.: -a silu""",
)

parser.add_argument(
    "-s",
    "--doweight_stage1",
    type=dtypes.str2bool,
    nargs="?",
    const=None,
    default=None,
    help="""Whether to do weight in stage 1. Default is [False, True].
    -s f    # False.
    -s t    # True.""",
)

parser.add_argument(
    "-e",
    "--expert",
    type=int,
    default=256,
    help="""Number of experts.
    e.g.: -e 8""",
)

parser.add_argument(
    "-k",
    "--topk",
    type=int,
    default=8,
    help="""Number of top experts.
    e.g.: -k 2""",
)

args = parser.parse_args()
if args.dtype is None:
    l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
else:
    l_dtype = [dtypes.d_dtypes[args.dtype]]

if args.dim is not None:
    l_dim = [args.dim]

if args.tokenNum is not None:
    l_tokenNum = [args.tokenNum]

l_quant = [l_quant[args.quant]] if args.quant is not None else l_quant

if args.act is not None:
    l_act = [getattr(aiter.ActivationType, args.act.capitalize())]

if args.doweight_stage1 is not None:
    l_doweight_stage1 = [args.doweight_stage1]

df = []
for (
    dtype,
    (quant_type, aq_dtype, wq_dtype),
    (model_dim, inter_dim),
    doweight_stage1,
) in itertools.product(l_dtype, l_quant, l_dim, l_doweight_stage1):
    if (quant_type, aq_dtype, wq_dtype) == (
        aiter.QuantType.per_1x32,
        dtypes.bf16,
        dtypes.fp4x2,
    ):
        for hidden_pad, intermediate_pad in l_hidden_intermediate_pad:
             for m in l_tokenNum:
                ret = test_fmoe(
                    dtype,
                    m,
                    model_dim,
                    inter_dim,
                    args.expert,
                    args.topk,
                    aiter.ActivationType.Swiglu,
                    quant_type,
                    aq_dtype,
                    wq_dtype,
                    use_g1u1=True,
                    doweight_stage1=doweight_stage1,
                    hidden_pad=hidden_pad,
                    intermediate_pad=intermediate_pad,
                )
                df.append(ret)
    else:
        for act_type in l_act:
            for m in l_tokenNum:
                ret = test_fmoe(
                    dtype,
                    m,
                    model_dim,
                    inter_dim,
                    args.expert,
                    args.topk,
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
