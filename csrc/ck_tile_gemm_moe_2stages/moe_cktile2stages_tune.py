# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import torch
import os
import aiter
import pandas as pd
from aiter import dtypes
from aiter.test_common import perftest
from aiter.ops.shuffle import shuffle_weight
import argparse
from aiter.utility.mp_tuner import mp_tuner
from moe_cktile2stages_common import get_gemm1_kernels_list, get_gemm2_kernels_list

import itertools
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
    torch_moe_stage1,
    torch_moe_stage2,
)

from aiter.ops.shuffle import shuffle_weight, shuffle_weight_NK
from aiter import ActivationType
torch.set_default_device("cuda")

#padding
npad0 = 192
kpad0 = 128

def shuffle_mxfp4_weight(src: torch.Tensor, NLane: int, gate_up: bool) -> torch.Tensor:
    """
    src: shape [experts_cnt, N, K_pk], where K_pk = K // 2
    Returns: shuffled tensor of shape [experts_cnt, N0*2, K0, KLane, NLane, KPack]
    """
    # print("gemm shape:", src.shape)
    experts_cnt, N, K_pk = src.shape
    if gate_up:
        N = N // 2
    KPack = 16
    KLane = 64 // NLane #4
    N0 = N // NLane
    K0 = K_pk // (KLane * KPack)
    assert KLane * KPack * K0 == K_pk, f"K({K_pk}) is not a divisble of 64."
    assert NLane * N0 == N, f"N({K_pk}) is not a divisble of 16."
    if (gate_up):
        src_reshaped = src.view(experts_cnt, 2, N0, NLane, K0, KLane, KPack)  # [E,2, N0, NLane ,K0, KLane, KPack]
        src_reshaped = src_reshaped.permute(0, 2, 1, 4, 5, 3, 6).contiguous()  # [E, N0, 2, K0, KLane, NLane, KPack]
        interleaved = src_reshaped.view(*src.shape)
    else:
        src_reshaped = src.view(experts_cnt, N0, NLane, K0, KLane, KPack)
        interleaved = src_reshaped.permute(0, 1, 3, 4, 2, 5).contiguous().view(*src.shape)
    # print("interleaved shape:", interleaved.shape)
    return interleaved.contiguous()

def shuffle_mxfp4_scale(src: torch.Tensor, experts_cnt: int, gate_up: bool) -> torch.Tensor:
    n_experts, k_ = src.shape
    n_ = n_experts // experts_cnt
    # MXFP4 constants
    K_Pack = 2
    N_Pack = 2
    N_Lane = 16
    K_Lane = 64 // N_Lane  # 4

    # Basic dimensions
    K1 = k_ // K_Pack // K_Lane  # k_ // 8
    N1 = n_ // N_Lane // N_Pack        # n_ // 32
    real_k =32 * k_ * K_Pack * K_Lane # 1x32 quant
    assert K1 * K_Pack * K_Lane == k_, f"K {k_*32} must be divisible of 256"
    # print("src shape", src.shape)
    # Reshape based on moe_kind
    if gate_up:
        # Reshape to: [E, N_Pack, N1, N_Lane, K1, K_Pack, K_Lane]
        shfl_scale = src.view(experts_cnt, N_Pack, N1, N_Lane, K1, K_Pack, K_Lane)
        # Permute to: [E, N1, K1, K_Lane, N_Lane, K_Pack, N_Pack]
        shfl_scale = shfl_scale.permute(0, 2, 4, 6, 3, 5, 1).contiguous()
    else:
        # Reshape to: [E, K1, K_Pack, K_Lane, N1, N_Pack, N_Lane]
        shfl_scale = src.view(experts_cnt, N1, N_Pack, N_Lane, K1, K_Pack, K_Lane)
        # Permute to: [E, N1, K1, K_Lane, N_Lane, K_Pack, N_Pack]
        shfl_scale = shfl_scale.permute(0, 1, 4, 6, 3, 5, 2).contiguous()
    # print("shf_scale shape:", shfl_scale.shape)
    return shfl_scale.view(*src.shape).contiguous()


def get_untuned_gemm_list(untuned_gemm_file):
    assert os.path.exists(
        untuned_gemm_file
    ), f"Not exist untuned file: {untuned_gemm_file}"
    untunedf = pd.read_csv(untuned_gemm_file)
    filtered_df = untunedf.drop_duplicates().reset_index(drop=True)
    return filtered_df


def get_tuned_gemm_list(tuned_gemm_file):
    if os.path.exists(tuned_gemm_file):
        tunedf = pd.read_csv(tuned_gemm_file)
    else:
        tunedf = pd.DataFrame(
            columns=["cu_num", "M", "N", "K", "expert", "topk", "blockM", "kernelId_gemm1", "us_gemm1", "kernelName_gemm1" , "kernelId_gemm2", "us_gemm2", "kernelName_gemm2"]
        )
    return tunedf

'''need to un-padd stage1 result'''
def torch_moe_stage1_tune(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    a1_scale,
    w1_scale,
    w1_bias,
    dtype,
    activation,
    quant_type,
    doweight,
):
    out = torch_moe_stage1(
                            hidden_states,
                            w1,
                            w2,
                            topk_weight,
                            topk_ids,
                            dtype,
                            activation,
                            quant_type,
                            a1_scale,
                            w1_scale,
                            w1_bias,
                            doweight,
                           )
    return out[:,:-npad0]

def torch_moe_stage2_tune(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    w2_scale,
    a2_scale,
    w2_bias, 
    dtype,
    quant_type,
    doweight,
):
    return torch_moe_stage2(
                            hidden_states,
                            w1,
                            w2,
                            topk_weights,
                            topk_ids,
                            dtype,
                            quant_type,
                            w2_scale,
                            a2_scale,
                            w2_bias, 
                            doweight,
                           )

def cktile_moe_stage1_tune(
    hidden_states,
    w1,  # [E, inter_dim*2, model_dim]
    w2,  # [E, model_dim, inter_dim]
    sorted_token_ids,  # [max_num_tokens_padded]
    sorted_expert_ids,  # [max_num_m_blocks]
    num_valid_ids,  # [1]
    w1_scale,
    a1_scale,
    exp_bias1,
    sorted_weights,  # [max_num_tokens_padded]
    dtype,
    topk,
    n_pad_zeros = 0,
    k_pad_zeros = 0,
    block_size=32,
    Activation=ActivationType.Silu,
    quant_type=aiter.QuantType.No,
    kernel_id = -1,
):
    token_num = hidden_states.shape[0]
    _, n1, k1 = w1.shape
    _, k2, n2 = w2.shape
    D = n2 if k2 == k1 else n2*2 #bit4 format
    # max_num_tokens_padded = sorted_expert_ids.shape[0]*block_size

    if w1.dtype is torch.uint32:
        D = D * 8
    out = torch.empty((token_num, topk, D), dtype=dtype)
    # print("Run cktile_moe_stage1: M=%d, N(N*2)=%d, K=%d, topk=%d, expert=%d"%(token_num, w1.shape[1], hidden_states.shape[1], topk, w1.shape[0]))
    aiter.moe_cktile2stages_gemm1_tune(
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
        kernel_id,
    )
    return out[:,:-npad0]

def cktile_moe_stage2_tune(
    hidden_states,
    w1,  # [E, inter_dim*2, model_dim]
    w2,  # [E, model_dim, inter_dim]
    sorted_token_ids,  # [max_num_tokens_padded]
    sorted_expert_ids,  # [max_num_m_blocks]
    num_valid_ids,  # [1]
    w2_scale,
    a2_scale,
    exp_bias2,
    sorted_weights,  # [max_num_tokens_padded]
    dtype,
    topk,
    n_pad_zeros = 0,
    k_pad_zeros = 0,
    block_size=32,
    Activation=ActivationType.Silu,
    quant_type=aiter.QuantType.No,
    zeros_out = False,
    kernel_id = -1,
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
    aiter.moe_cktile2stages_gemm2_tune(
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
        kernel_id,
    )
    return out


def generate_data(
    dtype,
    token,
    model_dim,
    inter_dim,
    BLOCK_SIZE_M,
    E,
    topk,
    qType,
    AQDType,
    WQDType,
    use_g1u1,
    do_weight_stage1,
    seed, 
    device="cuda",
):
    if get_gfx() not in ["gfx950"] and qType == aiter.QuantType.per_1x32:
        return
    torch_quant = aiter.get_torch_quant(qType)
    input = torch.randn((token, model_dim), dtype=dtype, device=device)
    input2 = torch.randn((token, topk, inter_dim), dtype=dtype, device=device) / 10
    if use_g1u1:
        w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype, device=device)
        w1[:,:,-kpad0:] = 0
        w1[:,-npad0:,:] = 0
        w1[:,inter_dim-npad0:inter_dim,:] = 0
        exp_bias1 = torch.clamp(torch.randn((E, inter_dim * 2), dtype=dtype, device=device), -1.0, 1.0)
    else:
        w1 = torch.randn((E, inter_dim, model_dim), dtype=dtype, device=device)
        exp_bias1 = torch.clamp(torch.randn((E * inter_dim), dtype=dtype, device=device), -1.0, 1.0)
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype, device=device)
    w2[:,:,-kpad0:] = 0
    w2[:,-npad0:,:] = 0
    exp_bias2 = torch.clamp(torch.randn((E, model_dim), dtype=dtype, device=device), -1.0, 1.0)
    score = torch.randn((token, E), dtype=dtype, device=device)
    topk_weights, topk_ids = fused_topk(input, score, topk, True)

    M, _ = topk_ids.shape

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, BLOCK_SIZE_M
    )
    
    #Quant-ing w
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

    #Re-shape and pack-k
    if qType != aiter.QuantType.per_1x32:
        w1_qt = w1_qt_aiter = w1_qt.view(w1.shape)
        w2_qt = w2_qt_aiter = w2_qt.view(w2.shape)
    else:
        w1_qt = w1_qt_aiter = w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2)
        w2_qt = w2_qt_aiter = w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2)

    #Quant-ing a
    if qType == aiter.QuantType.per_128x128:
        a1_qt, a1_scale = aiter.pertoken_quant(
            input.view(token, -1, 128), quant_dtype=AQDType
        )
        a1_qt = a1_qt.view(token, model_dim)
        a1_scale = a1_scale.squeeze(-1)
    elif qType == aiter.QuantType.per_1x32 and (AQDType in [dtypes.bf16, dtypes.fp16]): #a16w4
        a1_qt = input.to(AQDType)
        a1_scale = None
    else:
        a1_qt, a1_scale = torch_quant(input, quant_dtype=AQDType)

    #bias dtype convert
    if qType == aiter.QuantType.per_1x32 and (AQDType in [dtypes.bf16, dtypes.fp16]) and (WQDType == dtypes.fp4x2): #a16w4
        exp_bias1_aiter = exp_bias1.to(dtypes.fp32)
        exp_bias2_aiter = exp_bias2.to(dtypes.fp32)
    else:
        exp_bias1_aiter = exp_bias1 = None
        exp_bias2_aiter = exp_bias2 = None

    #pre-shuffle
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
    elif qType == aiter.QuantType.per_1x32 and (AQDType in [dtypes.bf16, dtypes.fp16]) and (WQDType == dtypes.fp4x2): #a16w4
        w1_qt_aiter = shuffle_mxfp4_weight(w1_qt_aiter, 16, True)
        w1_scale_aiter = shuffle_mxfp4_scale(w1_scale, E, True)
        w2_qt_aiter = shuffle_mxfp4_weight(w2_qt_aiter, 16, False)
        w2_scale_aiter = shuffle_mxfp4_scale(w2_scale, E, False)
    elif WQDType != dtypes.fp4x2 and (get_gfx() in ["gfx950"]):
        inst_K = 128 // w1_qt_aiter.element_size()
        w1_qt_aiter = shuffle_weight_NK(w1_qt_aiter, 16, inst_K)
        w2_qt_aiter = shuffle_weight_NK(w2_qt_aiter, 16, inst_K)
    elif WQDType != dtypes.fp4x2:
        w1_qt_aiter = shuffle_weight(w1_qt_aiter, layout=(16, 16))
        w2_qt_aiter = shuffle_weight(w2_qt_aiter, layout=(16, 16))
        

    # ######################## stage 2 ###########
    if qType == aiter.QuantType.per_128x128:
        a2_qt, a2_scale = aiter.pertoken_quant(
            input2.view(token, -1, 128), quant_dtype=AQDType
        )
        a2_scale = a2_scale.view(token, topk, -1)
    if qType == aiter.QuantType.per_1x32 and (AQDType in [dtypes.bf16, dtypes.fp16]):
        a2_qt = input2.to(AQDType)
        a2_scale = None
    else:
        a2_qt, a2_scale = torch_quant(input2, quant_dtype=AQDType)
    a2_qt = a2_qt.view(token, topk, -1)

    #for mp_runner
    sorted_weights_gemm1 = sorted_weights if do_weight_stage1 else None
    sorted_weights_gemm2 = sorted_weights if not do_weight_stage1 else None

    return (a1_qt, a1_scale, a2_qt, a2_scale, 
            w1_qt, w1_scale, exp_bias1,  w2_qt, w2_scale, exp_bias2, topk_weights, topk_ids,
            w1_qt_aiter, w1_scale_aiter, exp_bias1_aiter, w2_qt_aiter, w2_scale_aiter, exp_bias2_aiter, 
             sorted_ids, sorted_expert_ids, num_valid_ids, sorted_weights_gemm1, sorted_weights_gemm2)

def get_kernel_list(AQDType, WQDType, quant_type, activation, mul_routed_weight_stage):
    type_dict = {
        dtypes.fp8: "fp8",
        dtypes.fp4x2: "fp4",
        dtypes.bf16: "bf16",
        dtypes.fp16: "fp16",
    }
    quant_dict = {
        aiter.QuantType.No: "no",
        aiter.QuantType.per_Tensor: "per_tensor",
        aiter.QuantType.per_Token: "per_token",
        aiter.QuantType.per_1x32: "1x32",
        aiter.QuantType.per_1x128:  "1x128",
    }

    a_dtype = type_dict[AQDType]
    b_dtype = type_dict[WQDType]

    quant_type_str = quant_dict[quant_type]
    _, gemm1_kernel_list = get_gemm1_kernels_list(
            a_dtype,
            b_dtype,
            quant_type_str,
            "silu",
            mul_routed_weight_stage == 1,
            True,
        )
    tag, gemm2_kernel_list = get_gemm2_kernels_list(
            a_dtype,
            b_dtype,
            quant_type_str,
            "",
            mul_routed_weight_stage == 2,
            True,
        )
    return gemm1_kernel_list, gemm2_kernel_list

data_txt_idx_list = [n.strip() for n in 
                    "a1_qt, a1_scale, a2_qt, a2_scale, "
                    "w1_qt, w1_scale, exp_bias1,  w2_qt, w2_scale, exp_bias2, topk_weights, topk_ids,"
                    "w1_qt_aiter, w1_scale_aiter, exp_bias1_aiter, w2_qt_aiter, w2_scale_aiter, exp_bias2_aiter, "
                    "sorted_ids, sorted_expert_ids, num_valid_ids, sorted_weights_gemm1, sorted_weights_gemm2".split(",")]
data_txt_idx_dict = {data_txt_idx_list[i] : i for i in range(len(data_txt_idx_list))}

def transform_txt_idx(txt_list):
    return([data_txt_idx_dict[i.strip()] for i in txt_list])

def tune_gemm_list(
    untunedf,
    tunedf,
    # expert, 
    # topk, 
    a_type,
    b_type,
    c_type,
    quant_type,
    act_type,
    mul_routed_weight_stage,
    issorted=False,
    mp_num=1,
    shape_grouped=False,
    forced=False,
):
    gpu = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(gpu)
    cu_num = device_properties.multi_processor_count
    gemm1_kernel_list, gemm2_kernel_list = get_kernel_list(a_type, b_type, quant_type, act_type, mul_routed_weight_stage)
    task_gemm1 = []
    tasks_data_gemm1 = []  # [(kernel_nums, datas)]
    task_gemm2 = []
    tasks_data_gemm2 = []  # [(kernel_nums, datas)]
    doweight_stage1 = mul_routed_weight_stage == 1
    use_g1u1 = True
    gemm1_ck_idx = transform_txt_idx("a1_qt,w1_qt_aiter,w2_qt_aiter,sorted_ids,sorted_expert_ids,num_valid_ids,w1_scale_aiter,a1_scale,exp_bias1_aiter,sorted_weights_gemm1".split(","))
    gemm1_ref_idx = transform_txt_idx("a1_qt,w1_qt,w2_qt,topk_weights,topk_ids,a1_scale,w1_scale,exp_bias1".split(","))
    gemm2_ck_idx = transform_txt_idx("a2_qt,w1_qt_aiter,w2_qt_aiter,sorted_ids,sorted_expert_ids,num_valid_ids,w2_scale_aiter,a2_scale,exp_bias2_aiter,sorted_weights_gemm2".split(","))
    gemm2_ref_idx = transform_txt_idx("a2_qt,w1_qt,w2_qt,topk_weights,topk_ids,w2_scale,a2_scale,exp_bias2".split(","))
    seed = 100
    for i in range(len(untunedf)):
        M = untunedf.loc[i, "M"]
        N = untunedf.loc[i, "N"]
        K = untunedf.loc[i, "K"]
        expert = untunedf.loc[i, "expert"]
        topk  = untunedf.loc[i, "topk"]
        blockM = int(untunedf.loc[i, "blockM"])
        if (
            tunedf[
                (tunedf["M"] == M)
                & (tunedf["N"] == N)
                & (tunedf["K"] == K)
                & (tunedf["K"] == K)
                & (tunedf["expert"] == expert)
                & (tunedf["topk"]  == topk)
                & (tunedf["cu_num"] == cu_num)
            ].empty
            or forced
        ):
            seed += 1
            gemm1_blockM_dict = {key: value for key, value in gemm1_kernel_list.items() if (value.MPerBlock == blockM)}
            gemm2_blockM_dict = {key: value for key, value in gemm2_kernel_list.items() if (value.MPerBlock == blockM)}
            total_kernel_nums = 0
            #gemm1
            for id, _ in gemm1_blockM_dict.items():
                info = ((cu_num, M, N, K, expert, topk, blockM), id)
                task_gemm1.append(
                    (
                        info,
                        generate_data,
                        (c_type, M, K, N, blockM, expert, topk, quant_type, a_type, b_type, use_g1u1, doweight_stage1, seed),
                        cktile_moe_stage1_tune,
                        (
                            gemm1_ck_idx,
                            c_type,
                            topk,
                            npad0 * 2,
                            kpad0,
                            blockM,
                            act_type,
                            quant_type,
                            id,
                        ),
                        {},
                        torch_moe_stage1_tune,
                        (
                            gemm1_ref_idx,
                            c_type,
                            act_type,
                            quant_type,
                            doweight_stage1,
                        ),
                        {},
                        None,
                        1e-2,
                        0.1,
                    )
                )
                total_kernel_nums = total_kernel_nums + 1

            tasks_data_gemm1.append((total_kernel_nums, ()))
            total_kernel_nums = 0
            #gemm2
            for id, _ in gemm2_blockM_dict.items():
                info = ((cu_num, M, N, K, expert, topk, blockM), id)
                task_gemm2.append(
                    (
                        info,
                        generate_data,
                        (c_type, M, K, N, blockM, expert, topk, quant_type, a_type, b_type, use_g1u1, doweight_stage1, seed),
                        cktile_moe_stage2_tune,
                        (
                            gemm2_ck_idx,
                            c_type,
                            topk,
                            npad0,
                            kpad0,
                            blockM,
                            act_type,
                            quant_type,
                            True,
                            id,
                        ),
                        {},
                        torch_moe_stage2_tune,
                        (
                            gemm2_ref_idx,
                            c_type,
                            quant_type,
                            not doweight_stage1,
                        ),
                        {},
                        None,
                        5e-2,
                        0.1,
                    )
                )
                total_kernel_nums = total_kernel_nums + 1
            tasks_data_gemm2.append((total_kernel_nums, ()))
        if (len(tasks_data_gemm2) != len(tasks_data_gemm2)):
                raise Exception(f"Num of task_set for tunning gemm1 and gemm2 are not equal. {len(tasks_data_gemm2)} vs. {len(tasks_data_gemm2)}")
        if task_gemm1 or task_gemm2:
            ret_gemm1 = mp_tuner(task_gemm1, tasks_data_gemm1, mp_num, False, shape_grouped)
            ret_gemm2 = mp_tuner(task_gemm2, tasks_data_gemm2, mp_num, False, shape_grouped)
            # print("ret_gemm1:\n", ret_gemm1)
            # print("ret_gemm2:\n", ret_gemm2)
            if (len(ret_gemm1) != len(ret_gemm2)):
                raise Exception(f"Num of result for tunning gemm1 and gemm2 are not equal. {len(ret_gemm1)} vs. {len(ret_gemm2)}")
            for i in range(len(ret_gemm1)):
                el_gemm1 = ret_gemm1[i]
                el_gemm2 = ret_gemm2[i]
                info_gemm1, time_gemm1, err_ratio_gemm1 = el_gemm1
                info_gemm2, time_gemm2, err_ratio_gemm2 = el_gemm2
                (cu_num, M, N, K, expert, topk, blockM), kernelId_gemm1 = info_gemm1
                kernelId_gemm2 = info_gemm2[-1]
                kernelName_gemm1 = "None" if kernelId_gemm1 == -1 else gemm1_kernel_list[kernelId_gemm1].name
                kernelName_gemm2 = "None" if kernelId_gemm2 == -1 else gemm2_kernel_list[kernelId_gemm2].name
                print(
                    f"Tuning result for M:{M}, N:{N}, K:{K}, expert{expert}, topk{topk}, BlockM:{blockM}, cu_num:{cu_num} is "
                    f"Gemm1: {kernelId_gemm1} {kernelName_gemm1}, {time_gemm1}us. "
                    f"Gemm2: {kernelId_gemm2} {kernelName_gemm2}, {time_gemm2}us."
                )
                temp = pd.DataFrame(
                    {
                        "M": [M],
                        "N": [N],
                        "K": [K],
                        "expert": [expert],
                        "topk": [topk],
                        "blockM": [blockM],
                        "cu_num": [cu_num],
                        "kernelId_gemm1": [kernelId_gemm1],
                        "us_gemm1": [time_gemm1],
                        "kernelName_gemm1": [kernelName_gemm1],
                        "kernelId_gemm2": [kernelId_gemm2],
                        "us_gemm2": [time_gemm2],
                        "kernelName_gemm2": [kernelName_gemm2],
                    }
                )
                tunedf = pd.concat([tunedf, temp], ignore_index=True).drop_duplicates(
                    subset=["M", "N", "K", "expert", "topk", "blockM", "cu_num"], keep="last"
                )
        else:
            print(f"M:{M}, N:{N}, K{K}, expert{expert}, topk{topk}, BlockM{blockM} is in tuned gemm, skip!!!")
        print()
        print()
    issorted = True
    if issorted:
        tunedf = tunedf.sort_values(by=["cu_num","blockM", "expert", "topk","M", "N", "K"])
    print("Totall tuning result:")
    print(tunedf)
    return tunedf


if __name__ == "__main__":
    l_quant = [
    (aiter.QuantType.No, None, None),  # a16w16
    (aiter.QuantType.per_Tensor, dtypes.fp8, dtypes.fp8),  # a8w8
    (aiter.QuantType.per_Token, dtypes.fp8, dtypes.fp8),  # a8w8
    (aiter.QuantType.per_Token, dtypes.fp8, torch.int4),  # a8w4
    (aiter.QuantType.per_1x32, dtypes.fp4x2, dtypes.fp4x2),  # a4w4
    (aiter.QuantType.per_128x128, dtypes.fp8, dtypes.fp8),  # a8w8
    (aiter.QuantType.per_1x32, dtypes.bf16, dtypes.fp4x2),  # a16w4
]

    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for CKTILE moe a16w4",
    )

    parser.add_argument(
        "-i",
        "--untune_file",
        # default="aiter/configs/cktile_moe_2stage_a16w4_untuned.csv",
        default="aiter/configs/test_untuned.csv",
        required=False,
        help="input",
    )

    parser.add_argument(
        "--mp",
        type=int,
        default=1, #torch.cuda.device_count(),
        help="Tuning on multiple GPUs using multiple processes",
    )

    parser.add_argument(
        "-o",
        "--tune_file",
        # default="aiter/configs/cktile_moe_2stage_a16w4_tuned.csv",
        default="aiter/configs/test_tuned.csv",
        required=False,
        help="output: tuning result store this file",
    )

    # parser.add_argument(
    #     "-k", "--splitK", action="store_true", required=False, help="Use splitK kernels"
    # )

    parser.add_argument(
        "--sort",
        action="store_true",
        required=False,
        help="Arranged according to the M N K size",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        required=False,
        help="force to tune all kernels, even if they are already tuned",
    )
    
    # parser.add_argument(
    #     "-a",
    #     "--a_dtype",
    #     nargs="*",
    #     required=False,
    #     type=str,
    #     choices=["f8", "i8", "f16", "b16"],
    #     help="select input dtype",
    # )

    # parser.add_argument(
    #     "-b",
    #     "--b_dtype",
    #     nargs="*",
    #     required=False,
    #     type=str,
    #     choices=["f8", "i8", "f16", "b16", "i4"],
    #     help="select weight dtype",
    # )

    parser.add_argument(
        "-c",
        "--c_dtype",
        default="bf16",
        required=False,
        type=str,
        choices=["fp16", "bf16"],
        help="select out dtype",
    )

    # parser.add_argument(
    #     "-q",
    #     "--quant_type",
    #     default="per_tensor",
    #     required=False,
    #     type=str,
    #     choices=[
    #         "per_tensor",
    #         "per_token",
    #         "1x32",
    #         "128x128",
    #         "no",
    #     ],
    #     help="select quant_type",
    # )

    parser.add_argument(
    "-q",
    "--quant",
    type=int,
    choices=range(len(l_quant)),
    default=6,
    help="""select quantization type:
    0 : aiter.QuantType.No, None, None),  # a16w16
    1: aiter.QuantType.per_Tensor, dtypes.fp8, dtypes.fp8  # a8w8
    2: aiter.QuantType.per_Token, dtypes.fp8, dtypes.fp8  # a8w8
    3: aiter.QuantType.per_Token, dtypes.fp8, torch.int4  # a8w4
    4: aiter.QuantType.per_1x32, dtypes.fp4x2, dtypes.fp4x2  # a4w4
    5: aiter.QuantType.per_128x128, dtypes.fp8, dtypes.fp8  # a8w8
    6: aiter.QuantType.per_1x32, dtypes.bf16, dtypes.fp4x2  # a16w4""",
    )

    parser.add_argument(
        "-act",
        "--activation",
        default="silu",
        required=False,
        type=str,
        choices=["silu", "gelu"],
        help="select activation",
    )

    parser.add_argument(
        "-m",
        "--mul_routed_weight_stage",
        default=2,
        required=False,
        type=int,
        choices=[1, 2],
        help="select quant_type",
    )

    args = parser.parse_args()

    quant_type, aq_type, bq_type = l_quant[args.quant]
    act_type = getattr(aiter.ActivationType, args.activation.capitalize())
    c_type = dtypes.d_dtypes[args.c_dtype] 

    untunedf = get_untuned_gemm_list(args.untune_file)
    tunedf = get_tuned_gemm_list(args.tune_file)
    tunedf = tune_gemm_list(
        untunedf, tunedf, 
        aq_type,
        bq_type,
        c_type,
        quant_type,
        act_type,
        args.mul_routed_weight_stage,
        args.sort, args.mp, False, args.force
    )
    tunedf.to_csv(args.tune_file, index=False)