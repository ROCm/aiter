# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
import pandas as pd
import argparse
import time
import os
import sys
from aiter import QuantType
from aiter.jit.core import get_asm_dir, AITER_CSRC_DIR
from aiter.fused_moe import (
    fused_topk,
    moe_sorting,
    asm_stage1,
    torch_moe_stage1,
    torch_moe_stage2,
    fused_moe_1stage_dict,
    torch_moe,
)
from aiter import ck_moe_stage1_fwd, ck_moe_stage2_fwd, dtype2str_dict
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.mp_tuner import mp_tuner
from aiter.int4_utils import (
    rearrange_4bit_elements,
    convert_int8_to_uint32_int4,
)
from aiter import dtypes
from aiter import ActivationType as ActivationType
from aiter.jit.utils.chip_info import get_gfx
import functools
from aiter.utility import fp4_utils
import torch.nn.functional as F
from einops import rearrange


sys.path.insert(0, f"{AITER_CSRC_DIR}/ck_gemm_moe_2stages_codegen/")
from gemm_moe_ck2stages_common import get_gemm1_kernels_list, get_gemm2_kernels_list


torch.set_default_device("cuda")
torch.int4 = getattr(torch, "int4", torch.uint32)


def get_kernels_dict(file, key="tile_m"):
    if not os.path.exists(file):
        print(f"ASM kernel list file not exist: {file}")
        return {}
    df = pd.read_csv(file)
    kernel_dict = df.groupby(key)["knl_name"].apply(list).to_dict()
    return kernel_dict


def weight_quant(
    weight,
    qType,
    quant_dtype,
):
    E, dim1, dim2 = weight.shape
    if qType == aiter.QuantType.per_Tensor and quant_dtype != torch.int4:
        weight_qt, weight_scale = aiter.pertoken_quant(
            weight.view(E, -1), quant_dtype=quant_dtype
        )
    elif qType == QuantType.per_1x128:
        weight_qt = (
            weight.view(E, dim1 // 128, 128, dim2 // 128, 128)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(E, -1, 128 * 128)
        )
        weight_qt, weight_scale = aiter.pertoken_quant(
            weight_qt, quant_dtype=quant_dtype
        )
        weight_qt = weight_qt.view(E, -1)
        weight_qt = (
            weight_qt.view(E, dim1 // 128, dim2 // 128, 128, 128)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(E, dim1, dim2)
        )
    elif (
        qType == aiter.QuantType.per_Tensor and quant_dtype == torch.int4
    ):  # int4 w quant
        weight_qt, weight_scale = aiter.pertoken_quant(
            weight.view(E, -1), quant_dtype=dtypes.i8, dtypeMax=7
        )
    elif (
        qType == aiter.QuantType.per_Token and quant_dtype == torch.int4
    ):  # int4 w quant
        weight_qt, weight_scale = aiter.pertoken_quant(
            weight, quant_dtype=dtypes.i8, dtypeMax=7
        )
    else:
        torch_quant = aiter.get_torch_quant(qType)
        weight_qt, weight_scale = torch_quant(weight, quant_dtype=quant_dtype)
    return weight_qt, weight_scale


def ck_moe_stage1_fwd_out(
    a1_qt,
    w1_qt_shffle_ck,
    w2_qt_shffle_ck,
    sorted_ids,
    sorted_expert_ids,
    sorted_weights,
    num_valid_ids,
    w1_scale,
    a1_scale,
    dtype,
    topk,
    kernelName,
    blockM,
    q_type,
    act_type,
):
    inter_dim = w1_qt_shffle_ck.shape[1] // 2
    token_num = a1_qt.shape[0]

    out = torch.empty(
        (token_num, topk, inter_dim),
        dtype=dtype,
        device=a1_qt.device,
    )
    out = ck_moe_stage1_fwd(
        a1_qt,
        w1_qt_shffle_ck,
        w2_qt_shffle_ck,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        out,
        topk,
        kernelName,
        w1_scale,
        a1_scale,
        blockM,
        sorted_weights,
        q_type,
        act_type,
    )
    if q_type == QuantType.per_1x128:
        quant_func = aiter.get_hip_quant(q_type)
        a2, a2_scale = quant_func(
            out,
            quant_dtype=a1_qt.dtype,
        )
        out = a2
    return out


def ck_moe_stage2_fwd_out(
    a2_qt,
    w1_qt_shffle_ck,
    w2_qt_shffle_ck,
    sorted_ids,
    sorted_expert_ids,
    sorted_weights,
    num_valid_ids,
    w2_scale,
    a2_scale,
    dtype,
    topk,
    kernelName,
    blockM,
    q_type,
    act_type,
):
    model_dim = w2_qt_shffle_ck.shape[1]
    token_num = a2_qt.shape[0]

    out = torch.zeros(
        (token_num, model_dim),
        dtype=dtype,
        device=a2_qt.device,
    )
    return ck_moe_stage2_fwd(
        a2_qt,
        w1_qt_shffle_ck,
        w2_qt_shffle_ck,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        out,
        topk,
        kernelName,
        w2_scale,
        a2_scale,
        blockM,
        sorted_weights,
        q_type,
        act_type,
    )


def run_asm_stage1(
    input,
    w1,
    w2,
    sorted_ids,
    sorted_expert_ids,
    sorted_weights,
    num_valid_ids,
    out,
    a1_scale,
    w1_scale,
    topk,
    block_m,
    kernelName,
    ksplit,
    activation,
    quant_type,
    doweight_stage1,
):
    if not doweight_stage1:
        sorted_weights = None
    asm_stage1(
        input,
        w1,
        w2,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        out,
        topk,
        block_m,
        kernelName,
        ksplit,
        activation,
        quant_type,
        a1_scale,
        w1_scale,
        sorted_weights,
    )
    return out


# do weight at stage1
def run_1stage_fmoe_g1u1_tkw1(
    hidden_states,
    a1,
    w1,
    w2,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    a1_scale,
    w1_scale,
    w2_scale,
    fc2_smooth_scale=None,
    quant_type=QuantType.No,
    isG1U1=False,
    activation=ActivationType.Silu,
    kernel_name="",
    topk=2,
    dtype=dtypes.bf16,
):
    moe_buf = torch.zeros(
        (a1.shape[0], a1.shape[1]),
        dtype=dtype,
        device="cuda",
    )
    aiter.fmoe_g1u1_tkw1(
        moe_buf,
        a1,
        w1,
        w2,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        a1_scale,
        w1_scale,
        w2_scale,
        kernel_name,
        fc2_smooth_scale=fc2_smooth_scale,
        activation=activation,
    )
    return moe_buf


def run_1stage_fmoe_fp8_blockscale_g1u1(
    hidden_states,
    a1,
    w1,
    w2,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    a1_scale,
    w1_scale,
    w2_scale,
    fc2_smooth_scale=None,
    quant_type=QuantType.No,
    isG1U1=False,
    activation=ActivationType.Silu,
    kernel_name="",
    topk=2,
    dtype=dtypes.bf16,
):
    moe_buf = torch.zeros(
        (a1.shape[0], a1.shape[1]),
        dtype=dtype,
        device="cuda",
    )
    aiter.fmoe_fp8_blockscale_g1u1(
        moe_buf,
        a1,
        w1,
        w2,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        a1_scale,
        w1_scale,
        w2_scale,
        kernel_name,
        fc2_smooth_scale=fc2_smooth_scale,
        activation=activation,
    )
    return moe_buf


def run_1stage_fmoe_g1u1(
    hidden_states,
    a1,
    w1,
    w2,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    a1_scale,
    w1_scale,
    w2_scale,
    fc2_smooth_scale=None,
    quant_type=QuantType.No,
    isG1U1=False,
    activation=ActivationType.Silu,
    kernel_name="",
    topk=2,
    dtype=dtypes.bf16,
):
    moe_buf = torch.zeros(
        (a1.shape[0], a1.shape[1]),
        dtype=dtype,
        device="cuda",
    )
    aiter.fmoe_g1u1(
        moe_buf,
        a1,
        w1,
        w2,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        a1_scale,
        w1_scale,
        w2_scale,
        kernel_name,
        fc2_smooth_scale=fc2_smooth_scale,
        activation=activation,
    )
    return moe_buf


def get_1stage_fmoe_func(quant_type, q_dtype_a, activation, isG1U1, doweight_stage1):
    fmoe_func = None
    if quant_type == QuantType.No and activation == ActivationType.Silu and not isG1U1:
        print("not support No Quant Silu G1U0 1 stage tuning!")
    else:
        if quant_type == QuantType.per_1x128:
            fmoe_func = run_1stage_fmoe_fp8_blockscale_g1u1
        elif (q_dtype_a == dtypes.fp8) & doweight_stage1:
            fmoe_func = run_1stage_fmoe_g1u1_tkw1
        elif isG1U1:
            fmoe_func = run_1stage_fmoe_g1u1

    return fmoe_func


def generate_data(
    token,
    model_dim,
    inter_dim,
    expert,
    topk,
    dtype,
    q_dtype_a,
    q_dtype_w,
    q_type,
    use_g1u1,
    blockM,
    device="cuda",
):
    torch.manual_seed(0)
    input = torch.randn((token, model_dim), dtype=dtype) / 10
    if use_g1u1:
        w1 = torch.randn((expert, inter_dim * 2, model_dim), dtype=dtype) / 10
    else:
        w1 = torch.randn((expert, inter_dim, model_dim), dtype=dtype) / 10
    w2 = torch.randn((expert, model_dim, inter_dim), dtype=dtype)
    w1_qt, w1_scale = weight_quant(w1, q_type, quant_dtype=q_dtype_w)
    w2_qt, w2_scale = weight_quant(w2, q_type, quant_dtype=q_dtype_w)
    w1_qt = w1_qt.view(w1.shape)
    w2_qt = w2_qt.view(w2.shape)

    score = torch.randn((token, expert), dtype=dtype)
    topk_weights, topk_ids = fused_topk(input, score, topk, True)
    if q_type == QuantType.per_1x128:
        a1_qt, a1_scale = aiter.pertoken_quant(
            input.view(token, -1, 128), quant_dtype=q_dtype_a
        )
        a1_qt = a1_qt.view(token, model_dim)
        a1_scale = a1_scale.squeeze(-1)
    else:
        torch_quant = aiter.get_torch_quant(q_type)
        a1_qt, a1_scale = torch_quant(input, quant_dtype=q_dtype_a)
    del w1, w2, score

    w1_qt_shffle = shuffle_weight(w1_qt, (16, 16))
    w2_qt_shffle = shuffle_weight(w2_qt, (16, 16))
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids, topk_weights, expert, model_dim, dtype, blockM
    )
    return (
        input,
        a1_qt,
        w1_qt,
        w2_qt,
        w1_qt_shffle,
        w2_qt_shffle,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk_ids,
        topk_weights,
        moe_buf,
        a1_scale,
        w1_scale,
        w2_scale,
    )


def generate_asm_stage1(
    token,
    model_dim,
    inter_dim,
    expert,
    topk,
    act_type,
    dtype,
    q_dtype_a,
    q_dtype_w,
    q_type,
    use_g1u1,
    doweight_stage1,
    blockM,
    device="cuda",
):
    (
        input,
        a1_qt,
        w1_qt,
        w2_qt,
        w1_qt_shffle,
        w2_qt_shffle,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk_ids,
        topk_weights,
        moe_buf,
        a1_scale,
        w1_scale,
        w2_scale,
    ) = generate_data(
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        dtype,
        q_dtype_a,
        q_dtype_w,
        q_type,
        use_g1u1,
        blockM,
        device,
    )
    if q_type == QuantType.per_1x128:
        ratio = a1_scale.element_size() // a1_qt.element_size()
        out1 = torch.zeros(
            (token + (token * ratio + 127) // 128, topk, inter_dim),
            dtype=a1_qt.dtype,
        )
    else:
        out1 = torch.empty(
            (token, topk, inter_dim),
            dtype=dtype,
        )
    a1_scale_t = a1_scale
    if q_type == QuantType.per_1x128:
        a1_scale_t = a1_scale.t().contiguous()
    return (
        a1_qt,
        w1_qt_shffle,
        w2_qt_shffle,
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        out1,
        a1_scale_t,
        w1_scale,
        topk_weights,
        topk_ids,
        w1_qt,
        w2_qt,
        a1_scale,
    )


def generate_data_2stages(
    token,
    model_dim,
    inter_dim,
    expert,
    topk,
    act_type,
    dtype,
    q_dtype_a,
    q_dtype_w,
    q_type,
    use_g1u1,
    doweight_stage1,
    blockM,
    stage=1,
    device="cuda",
):
    (
        input,
        a1_qt,
        w1_qt,
        w2_qt,
        w1_qt_shffle,
        w2_qt_shffle,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk_ids,
        topk_weights,
        moe_buf,
        a1_scale,
        w1_scale,
        w2_scale,
    ) = generate_data(
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        dtype,
        q_dtype_a,
        q_dtype_w,
        q_type,
        use_g1u1,
        blockM,
        device,
    )
    if q_dtype_w == torch.int4:
        w1_qt_shffle_ck = rearrange_4bit_elements(
            convert_int8_to_uint32_int4(shuffle_weight(w1_qt, (16, 16), use_int4=True))
        )
        w2_qt_shffle_ck = rearrange_4bit_elements(
            convert_int8_to_uint32_int4(shuffle_weight(w2_qt, (16, 16), use_int4=True))
        )
    else:
        w1_qt_shffle_ck = w1_qt_shffle
        w2_qt_shffle_ck = w2_qt_shffle
    if stage == 1:
        if not doweight_stage1:
            sorted_weights = None
        return (
            a1_qt,
            w1_qt_shffle_ck,
            w2_qt_shffle_ck,
            a1_scale,
            w1_scale,
            sorted_ids,
            sorted_expert_ids,
            sorted_weights,
            num_valid_ids,
            moe_buf,
            w1_qt,
            w2_qt,
            topk_weights,
            topk_ids,
        )
    elif stage == 2:
        ref1 = run_torch_moe_stage1(
            a1_qt,
            w1_qt,
            w2_qt,
            topk_weights,
            topk_ids,
            a1_scale=a1_scale,
            w1_scale=w1_scale,
            dtype=dtype,
            activation=act_type,
            quant_type=q_type,
            doweight_stage1=doweight_stage1,
            topk=topk,
        )
        if q_type == QuantType.per_1x128:
            ref1, ref_scale = aiter.pertoken_quant(
                ref1.view(ref1.shape[0], -1, 128), quant_dtype=q_dtype_a
            )
            ref1 = ref1.view(ref1.shape[0], topk, -1)
            ref_scale = ref_scale.view(token, -1)
            a2_qt = ref1
            a2_scale = ref_scale
        else:
            torch_quant = aiter.get_torch_quant(q_type)
            a2_qt, a2_scale = torch_quant(ref1, quant_dtype=q_dtype_a)
        a2_qt = a2_qt.view(token, topk, -1)
        if doweight_stage1:
            sorted_weights = None
        return (
            a2_qt,
            w1_qt_shffle_ck,
            w2_qt_shffle_ck,
            a2_scale,
            w2_scale,
            sorted_ids,
            sorted_expert_ids,
            sorted_weights,
            num_valid_ids,
            moe_buf,
            w1_qt,
            w2_qt,
            topk_weights,
            topk_ids,
        )


def generate_data_1stage(
    token,
    model_dim,
    inter_dim,
    expert,
    topk,
    act_type,
    dtype,
    q_dtype_a,
    q_dtype_w,
    q_type,
    use_g1u1,
    blockM=32,
    device="cuda",
):
    (
        input,
        a1_qt,
        w1_qt,
        w2_qt,
        w1_qt_shffle,
        w2_qt_shffle,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk_ids,
        topk_weights,
        moe_buf,
        a1_scale,
        w1_scale,
        w2_scale,
    ) = generate_data(
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        dtype,
        q_dtype_a,
        q_dtype_w,
        q_type,
        use_g1u1,
        blockM,
        device,
    )
    a1_scale_t = a1_scale
    if q_type == QuantType.per_1x128:
        a1_scale_t = a1_scale.t().contiguous()
    ##smooth scale
    # [expert, 1, model_dim]
    fc1_smooth_scale = torch.randn(
        (expert, 1, model_dim), dtype=dtypes.fp32, device=device
    )
    # [expert, 1, inter_dim]
    fc2_smooth_scale = torch.randn(
        (expert, 1, inter_dim), dtype=dtypes.fp32, device=device
    )
    fc1_smooth_scale = None
    fc2_smooth_scale = None
    if q_type == QuantType.per_1x32:
        a1_scale = fp4_utils.moe_mxfp4_sort(
            a1_scale,
            sorted_ids,
            num_valid_ids,
            token,
            blockM,
        )
        w1_scale = w1_scale.view(expert, -1)
        w2_scale = w2_scale.view(expert, -1)

    return (
        input,  # 0
        a1_qt,  # 1
        w1_qt_shffle,  # 2
        w2_qt_shffle,  # 3
        sorted_ids,  # 4
        sorted_weights,  # 5
        sorted_expert_ids,  # 6
        num_valid_ids,  # 7
        moe_buf,  # 8
        a1_scale,  # 9
        w1_scale,  # 10
        w2_scale,  # 11
        w1_qt,  # 12
        w2_qt,  # 13
        topk_weights,  # 14
        topk_ids,  # 15
        fc1_smooth_scale,  # 16
        fc2_smooth_scale,  # 17
        a1_scale_t,
    )


def run_torch_moe_stage1(
    a1_qt,
    w1_qt,
    w2_qt,
    topk_weights,
    topk_ids,
    a1_scale,
    w1_scale,
    dtype,
    activation,
    quant_type,
    doweight_stage1,
    topk,
):
    ref1 = torch_moe_stage1(
        a1_qt,
        w1_qt,
        w2_qt,
        topk_weights,
        topk_ids,
        activation=activation,
        quant_type=quant_type,
        dtype=dtype,
        a1_scale=a1_scale,
        w1_scale=w1_scale,
        doweight=doweight_stage1,
    )
    if quant_type == QuantType.per_1x128:
        ref1, ref_scale = aiter.pertoken_quant(
            ref1.view(ref1.shape[0], -1, 128), quant_dtype=a1_qt.dtype
        )
        ref1 = ref1.view(ref1.shape[0], topk, -1)
    return ref1


def run_torch_moe_stage2(
    a2_qt,
    w1_qt,
    w2_qt,
    topk_weights,
    topk_ids,
    a2_scale,
    w2_scale,
    dtype,
    quant_type,
    doweight_stage1,
):
    return torch_moe_stage2(
        a2_qt,
        w1_qt,
        w2_qt,
        topk_weights,
        topk_ids,
        dtype,
        quant_type,
        a2_scale=a2_scale,
        w2_scale=w2_scale,
        doweight=not doweight_stage1,
    )


def run_torch_moe_stage1_ref(
    a1_qt,
    w1_qt,
    w2_qt,
    topk_weights,
    topk_ids,
    a1_scale,
    w1_scale,
    dtype,
    activation,
    quant_type,
    doweight_stage1,
    topk,
):
    ref1 = run_torch_moe_stage1(
        a1_qt,
        w1_qt,
        w2_qt,
        topk_weights,
        topk_ids,
        activation=activation,
        quant_type=quant_type,
        dtype=dtype,
        a1_scale=a1_scale,
        w1_scale=w1_scale,
        doweight_stage1=doweight_stage1,
        topk=topk,
    )
    token = a1_qt.shape[0]
    inter_dim = w2_qt.shape[-1]
    if quant_type == QuantType.per_1x128:
        ref1, ref_scale = aiter.pertoken_quant(
            ref1.view(ref1.shape[0], -1, 128), quant_dtype=a1_qt.dtype
        )
        ref1 = ref1.view(ref1.shape[0], topk, -1)
        ref_scale = ref_scale.view(token, -1)
        a2_qt = ref1
        a2_qt = a2_qt.view(token, topk, -1)
        a2_scale = ref_scale
        ratio = a1_scale.element_size() // a1_qt.element_size()
        out1 = torch.zeros(
            (token + (token * ratio + 127) // 128, topk, inter_dim),
            dtype=a1_qt.dtype,
        )
        ref1_asm = torch.zeros_like(out1)
        ref1_asm[:token] = a2_qt
        ref1_asm[token:, ...].view(-1)[: token * topk * inter_dim * ratio // 128] = (
            a2_scale.view(a1_qt.dtype).view(-1)
        )
        return ref1_asm

    else:
        out1 = torch.empty(
            (token, topk, inter_dim),
            dtype=dtype,
        )
        return ref1


## 1 stage ref
def torch_moe_test(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    # following for int8 quant
    fc1_scale=None,  # [expert, inter_dim, 1]
    fc2_scale=None,  # [expert, model_dim, 1]
    fc1_smooth_scale=None,  # [expert, 1, model_dim]
    fc2_smooth_scale=None,  # [expert, 1, inter_dim]
    activation=ActivationType.Silu,
    doweight_stage1=False,
    q_type_a=dtypes.fp8,
):
    if doweight_stage1 & (q_type_a == dtypes.fp8):
        return torch_moe_tkw1(
            hidden_states,
            w1,
            w2,
            topk_weight,
            topk_ids,
            fc1_scale,
            fc2_scale,
            fc1_smooth_scale,
            fc2_smooth_scale,
            None,
            activation,
        )
    return torch_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        fc1_scale,
        fc2_scale,
        fc1_smooth_scale,
        fc2_smooth_scale,
        None,
        activation,
    )


def torch_moe_tkw1(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    # following for int8 quant
    fc1_scale=None,  # [expert(local_expert:EP), inter_dim, 1]
    fc2_scale=None,  # [expert(local_expert:EP), model_dim, 1]
    fc1_smooth_scale=None,  # [expert(local_expert:EP), 1, model_dim]
    fc2_smooth_scale=None,  # [expert(local_expert:EP), 1, inter_dim]
    expert_mask=None,
    activation=ActivationType.Silu,
):
    computeType = dtypes.fp32
    dtype = hidden_states.dtype
    hidden_states = hidden_states.to(computeType)
    w1 = w1.to(computeType)
    w2 = w2.to(computeType)
    B, D = hidden_states.shape
    topk = topk_weight.shape[1]
    if expert_mask is not None:
        local_expert_hash = expert_mask.cumsum(0, dtype=dtypes.i32) - 1
        local_expert_hash[expert_mask == 0] = -1
        topk_ids = local_expert_hash[topk_ids]

    hidden_states = hidden_states.view(B, -1, D).repeat(1, topk, 1)
    out = torch.zeros(
        (B, topk, D),
        dtype=computeType,
        device=hidden_states.device,
    )

    inter_dim = w2.shape[2]
    if w2.shape[2] * 2 == w1.shape[1]:
        # g1u1(w1 include gate and up)
        moeType = "g1u1"
    else:
        # g1u0(w1 only include gate)
        moeType = "g1u0"

    if fc1_scale is not None:
        # gose to quant D_w8a8/w8a8
        expert = w1.shape[0]
        w2D = w2.shape[-1]
        w1 = (w1.view(-1, D) * fc1_scale.view(-1, 1)).view(expert, -1, D)
        w2 = (w2.view(-1, w2D) * fc2_scale.view(-1, 1)).view(expert, -1, w2D)

    if fc1_smooth_scale is not None:
        expert = fc1_smooth_scale.shape[0]
        fc1_smooth_scale = fc1_smooth_scale.view(expert, -1)
        fc2_smooth_scale = fc2_smooth_scale.view(expert, -1)

    for E_id in range(w1.shape[0]):
        mask = topk_ids == E_id
        if mask.sum():
            sub_tokens = hidden_states[mask]
            if fc1_smooth_scale is not None:
                sub_tokens = sub_tokens * (fc1_smooth_scale[E_id])

            act_input = sub_tokens @ (w1[E_id].transpose(0, 1))
            if moeType == "g1u1":
                gate, up = act_input.split([inter_dim, inter_dim], dim=-1)
                gate = gate * (topk_weight.view(B, -1, 1)[mask])
                up = up * (topk_weight.view(B, -1, 1)[mask])
                if activation == ActivationType.Gelu:
                    act_out = F.gelu(gate) * up
                else:
                    act_out = F.silu(gate) * up
            else:
                if activation == ActivationType.Gelu:
                    act_out = F.gelu(act_input)
                else:
                    act_out = F.silu(act_input)
            if fc2_smooth_scale is not None:
                act_out = act_out * (fc2_smooth_scale[E_id])
            act_out, act_out_scale = aiter.pertoken_quant(
                act_out, quant_dtype=dtypes.fp8, dtypeMax=None
            )
            out[mask] = (
                act_out.to(computeType)
                @ (w2[E_id].transpose(0, 1))
                * act_out_scale.view(-1, 1)
            )

    return out.sum(dim=1).to(dtype)


def torch_moe_blockscale(
    hidden_states,
    w1,  # [expert, inter_dim*2, model_dim]
    w2,  # [expert, model_dim, inter_dim]
    topk_weight,
    topk_ids,
    # following for quant
    a_scale=None,
    # [expert, inter_dim/blk_m, model_dim/blk_k]
    fc1_scale=None,
    # [expert, model_dim/blk_m, inter_dim/blk_k]
    fc2_scale=None,
    expert_mask=None,
    scale_blks=(128, 128),
    dtype=dtypes.bf16,
):
    computeType = dtypes.fp32
    hidden_states = hidden_states.to(computeType)
    w1 = w1.to(computeType)
    w2 = w2.to(computeType)
    token_num, topk = topk_ids.shape
    expert, model_dim, inter_dim = w2.shape
    B, D = hidden_states.shape
    topk = topk_weight.shape[1]
    if expert_mask is not None:
        local_expert_hash = expert_mask.cumsum(0, dtype=dtypes.i32) - 1
        local_expert_hash[expert_mask == 0] = -1
        topk_ids = local_expert_hash[topk_ids]

    blk_n, blk_k = scale_blks
    if a_scale is not None:
        hidden_states = hidden_states.view(token_num, -1, blk_k) * a_scale.unsqueeze(-1)
        hidden_states = hidden_states.view(token_num, -1)

    hidden_states = hidden_states.view(token_num, 1, model_dim).repeat(1, topk, 1)
    out = torch.zeros(
        (B, topk, D),
        dtype=computeType,
        device=hidden_states.device,
    )
    if w2.shape[2] * 2 == w1.shape[1]:
        moeType = "g1u1"
    else:
        moeType = "g1u0"

    nblk_n = inter_dim // blk_n
    nblk_k = model_dim // blk_k
    if fc1_scale is not None:
        # gose to quant D_w8a8/w8a8
        # blk_n, blk_k = scale_blks
        # expert, nblk_n, nblk_k = fc1_scale.shape
        fc1_scale = rearrange(
            fc1_scale.view(-1, 1)
            .repeat(1, blk_n * blk_k)
            .view(expert, -1, nblk_k, blk_n, blk_k),
            "e num_blk_n num_blk_k blk_n blk_k -> e (num_blk_n blk_n) (num_blk_k blk_k)",
        )
        fc2_scale = rearrange(
            fc2_scale.view(-1, 1)
            .repeat(1, blk_n * blk_k)
            .view(expert, nblk_k, nblk_n, blk_k, blk_n),
            "e num_blk_n num_blk_k blk_n blk_k -> e (num_blk_n blk_n) (num_blk_k blk_k)",
        )
        w1 = w1 * fc1_scale
        w2 = w2 * fc2_scale

    for E_id in range(w1.shape[0]):
        mask = topk_ids == E_id
        if mask.sum():
            sub_tokens = hidden_states[mask]
            act_input = sub_tokens @ (w1[E_id].transpose(0, 1))
            if moeType == "g1u1":
                gate, up = act_input.split([inter_dim, inter_dim], dim=-1)
                act_out = F.silu(gate) * up
            else:
                act_out = F.gelu(act_input)
            out[mask] = act_out @ (w2[E_id].transpose(0, 1))

    return (out * topk_weight.view(B, -1, 1)).sum(dim=1).to(dtype)


def go(
    untunedf,
    tunedf,
    mp_num=1,
):
    startTS = time.perf_counter()
    # blockMs = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160]
    blockMs = [32, 64, 128]

    args = [
        "cu_num",
        "token",
        "model_dim",
        "inter_dim",
        "expert",
        "topk",
        "act_type",
        "dtype",
        "q_dtype_a",
        "q_dtype_w",
        "q_type",
        "use_g1u1",
        "doweight_stage1",
    ]
    print(untunedf[args])
    prorfiles = []
    bests = []
    tasks = []
    tasks_ck = []
    task_1stage = []
    in_data = []
    for line in untunedf[args].values:
        (
            cu_num,
            token,
            model_dim,
            inter_dim,
            expert,
            topk,
            act_type,
            dtype,
            q_dtype_a,
            q_dtype_w,
            q_type,
            use_g1u1,
            doweight_stage1,
        ) = line
        info = line
        dtype = eval(dtype)
        q_dtype_a = eval(q_dtype_a)
        q_dtype_w = eval(q_dtype_w)
        q_type = eval(q_type)
        q_type = QuantType.per_1x128 if q_type == QuantType.per_128x128 else q_type
        print("\nStart tuning", line)
        if not use_g1u1:
            print("no moe solution(g1u0) can tune for ", line)
            continue
        act_type = eval(act_type)

        kernels_list_csv = f"{get_asm_dir()}/fmoe_2stages/fmoe_stage1_bf16_pertoken{{quantDtype}}{{extraInfo}}_g1u1.csv"

        extraInfo = ""
        if q_type == QuantType.per_1x128:
            extraInfo += "_blockscale"
        if doweight_stage1:
            extraInfo += "_doweight"

        if q_dtype_a == dtypes.i8:
            quantDtype = "Int8"
        elif q_dtype_a == dtypes.fp8:
            quantDtype = "Fp8"
        else:
            quantDtype = ""

        asm_kernels = get_kernels_dict(
            kernels_list_csv.format(quantDtype=quantDtype, extraInfo=extraInfo)
        )

        _, ck_stage1_kernels = get_gemm1_kernels_list(
            dtype2str_dict[q_dtype_a],
            dtype2str_dict[q_dtype_w],
            False,
            str(q_type).split(".")[-1].lower(),
            str(act_type).split(".")[-1].lower(),
            doweight_stage1,
        )
        _, ck_stage2_kernels = get_gemm2_kernels_list(
            dtype2str_dict[q_dtype_a],
            dtype2str_dict[q_dtype_w],
            False,
            str(q_type).split(".")[-1].lower(),
            not doweight_stage1,
        )
        for blockM in blockMs:
            if use_g1u1 and q_dtype_w != torch.int4:
                for el in asm_kernels.get(blockM, []):
                    tasks.append(
                        (
                            (info, "stage1", el, blockM),  # tag
                            generate_asm_stage1,
                            (
                                token,
                                model_dim,
                                inter_dim,
                                expert,
                                topk,
                                act_type,
                                dtype,
                                q_dtype_a,
                                q_dtype_w,
                                q_type,
                                use_g1u1,
                                doweight_stage1,
                                blockM,
                            ),
                            run_asm_stage1,  # func
                            (
                                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                topk,
                                blockM,
                                el,
                                0,
                                act_type,
                                q_type,
                                doweight_stage1,
                            ),
                            {},
                            run_torch_moe_stage1_ref,
                            (
                                [0, 12, 13, 10, 11, 14, 9],
                                dtype,
                                act_type,
                                q_type,
                                doweight_stage1,
                                topk,
                            ),
                            {},
                            (None),
                            0.01,
                            0.01,
                            True,
                        )
                    )

            if blockM in [32, 64, 128] and use_g1u1:
                for kernel in ck_stage1_kernels.values():
                    if kernel.MPerBlock != blockM:
                        continue
                    tasks_ck.append(
                        (
                            (info, "stage1", kernel.name, blockM),  # tag
                            generate_data_2stages,
                            (
                                token,
                                model_dim,
                                inter_dim,
                                expert,
                                topk,
                                act_type,
                                dtype,
                                q_dtype_a,
                                q_dtype_w,
                                q_type,
                                use_g1u1,
                                doweight_stage1,
                                blockM,
                                1,
                            ),
                            ck_moe_stage1_fwd_out,  # func
                            (
                                [0, 1, 2, 5, 6, 7, 8, 4, 3],
                                dtype,
                                topk,
                                kernel.name,
                                blockM,
                                q_type,
                                act_type,
                            ),
                            {},
                            run_torch_moe_stage1,
                            (
                                [0, 10, 11, 12, 13, 3, 4],
                                dtype,
                                act_type,
                                q_type,
                                doweight_stage1,
                                topk,
                            ),
                            {},
                            (None),
                            0.01,
                            0.01,
                            True,
                        )
                    )

                for kernel in ck_stage2_kernels.values():
                    if kernel.MPerBlock != blockM:
                        continue
                    tasks_ck.append(
                        (
                            (info, "stage2", kernel.name, blockM),  # tag
                            generate_data_2stages,
                            (
                                token,
                                model_dim,
                                inter_dim,
                                expert,
                                topk,
                                act_type,
                                dtype,
                                q_dtype_a,
                                q_dtype_w,
                                q_type,
                                use_g1u1,
                                doweight_stage1,
                                blockM,
                                2,
                            ),
                            ck_moe_stage2_fwd_out,  # func
                            (
                                [0, 1, 2, 5, 6, 7, 8, 4, 3],
                                dtype,
                                topk,
                                kernel.name,
                                blockM,
                                q_type,
                                act_type,
                            ),
                            {},
                            run_torch_moe_stage2,
                            (
                                [0, 10, 11, 12, 13, 3, 4],
                                dtype,
                                q_type,
                                doweight_stage1,
                            ),
                            {},
                            (None),
                            0.01,
                            0.01,
                            True,
                        )
                    )

        ## asm moe 1 stage tuning
        chip_name = get_gfx()
        run_1stage_kernels_all = fused_moe_1stage_dict[chip_name]
        # print(run_1stage_kernels_all)
        key = (act_type, q_type, dtype, q_dtype_a, q_dtype_w, use_g1u1)
        acti_dir = ""
        if act_type == ActivationType.Silu:
            acti_dir = "silu"
        elif act_type == ActivationType.Gelu:
            acti_dir = "gelu"
        up = 1 if use_g1u1 else 0
        extraInfo_1stage = ""
        if doweight_stage1:
            extraInfo_1stage = "_tkw1"
            if q_dtype_a == dtypes.fp8:
                quantDtype = "Int8"  ## tmp solution, need to be updated
        if q_type == QuantType.No:
            quantDtype_1stage = "noquant"
        elif q_type == QuantType.per_1x128:
            quantDtype_1stage = "blockscale" + quantDtype
        else:
            quantDtype_1stage = "pertoken" + quantDtype

        kernels_list_csv_1stage = f"{get_asm_dir()}/fmoe/{acti_dir}/fmoe_bf16_{{quantDtype_1stage}}_g1u{up}_{acti_dir}{{extraInfo_1stage}}.csv"
        asm_kernels_1stage = {}
        if (
            q_type != QuantType.No
            and q_type != QuantType.per_1x32
            and q_type != QuantType.per_Tensor
            and q_dtype_w != torch.int4
        ):
            asm_kernels_1stage = get_kernels_dict(
                kernels_list_csv_1stage.format(
                    quantDtype_1stage=quantDtype_1stage,
                    extraInfo_1stage=extraInfo_1stage,
                ),
                key=["subGU_m", "subGU_n", "smf"],
            )
        fmoe_func = get_1stage_fmoe_func(
            q_type, q_dtype_a, act_type, use_g1u1, doweight_stage1
        )
        for tile_m, tile_n, smf in asm_kernels_1stage.keys():
            if inter_dim % tile_n != 0 or smf != 0:
                continue

            for el in asm_kernels_1stage.get((tile_m, tile_n, 0), []):
                task_1stage.append(
                    (
                        (info, "asm_1stage", el, tile_m),
                        generate_data_1stage,
                        (
                            token,
                            model_dim,
                            inter_dim,
                            expert,
                            topk,
                            act_type,
                            dtype,
                            q_dtype_a,
                            q_dtype_w,
                            q_type,
                            use_g1u1,
                            tile_m,
                        ),
                        fmoe_func,
                        (
                            [0, 1, 2, 3, 4, 5, 6, 7, 18, 10, 11, 17],
                            q_type,
                            use_g1u1,
                            act_type,
                            el,
                            topk,
                            dtype,
                        ),
                        {},
                        (
                            torch_moe_blockscale
                            if q_type == QuantType.per_1x128
                            else torch_moe_test
                        ),
                        (
                            ([1, 12, 13, 14, 15, 9, 10, 11], None, (128, 128), dtype)
                            if q_type == QuantType.per_1x128
                            else (
                                [0, 12, 13, 14, 15, 10, 11, 16, 17],
                                act_type,
                                doweight_stage1,
                                q_dtype_a,
                            )
                        ),
                        {},
                        (None),
                        0.01,
                        1,
                        True,
                    )
                )
        if tasks is None and tasks_ck is None and task_1stage is None:
            print("no moe solution can tune for ", line)
            continue
        print(
            f"tasks is {len(tasks)}, tasks_ck is {len(tasks_ck)}, task_1stage is {len(task_1stage)}"
        )
    in_data.append((len(tasks) + len(tasks_ck) + len(task_1stage), ()))
    rets = []
    if len(tasks) + len(tasks_ck) + len(task_1stage) > 0:
        ### shape_grouped should be False as multiple stages
        rets = mp_tuner(tasks + tasks_ck + task_1stage, in_data, mp_num, True, False)
    if not rets:
        print("no shape to tune or no solution found")
        return None, None
    profileDF = []
    from collections import defaultdict

    ##group results by info[0](key)
    grouped_rets = defaultdict(list)
    for info, us, max_err_ratio in rets:
        grouped_rets[tuple(info[0])].append((info[1:], us, max_err_ratio))
    grouped_results = grouped_rets.items()
    for key, rets in grouped_results:
        (
            cu_num,
            token,
            model_dim,
            inter_dim,
            expert,
            topk,
            act_type,
            dtype,
            q_dtype_a,
            q_dtype_w,
            q_type,
            use_g1u1,
            doweight_stage1,
        ) = key
        profileDF = []
        for (stage, kernelName, block_m), us, err in rets:
            if us == float("inf"):
                continue
            profileDF.append(
                [
                    stage,
                    cu_num,
                    token,
                    model_dim,
                    inter_dim,
                    expert,
                    topk,
                    act_type,
                    dtype,
                    q_dtype_a,
                    q_dtype_w if q_dtype_w != torch.int4 else "torch.int4",
                    q_type,
                    use_g1u1,
                    doweight_stage1,
                    block_m,
                    0,
                    us,
                    kernelName,
                    f"{err:.1%}",
                ]
            )
        profileDF = pd.DataFrame(
            profileDF,
            columns=["stage"] + args + ["block_m", "ksplit", "us", "kernelName", "err"],
        )
        prorfiles.append(profileDF)
        profileDF = profileDF.sort_values("us").drop_duplicates(
            ["stage", "block_m"], keep="first"
        )
        stage1_profileDF = profileDF[profileDF["stage"] == "stage1"].drop(
            columns=["stage"], axis=1
        )
        stage1_profileDF = stage1_profileDF.rename(
            columns={"kernelName": "kernelName1", "err": "err1", "us": "us1"}
        )
        stage2_profileDF = profileDF[profileDF["stage"] == "stage2"].drop(
            columns=["stage", "ksplit"], axis=1
        )
        stage2_profileDF = stage2_profileDF.rename(
            columns={"kernelName": "kernelName2", "err": "err2", "us": "us2"}
        )
        asm_1stage_profileDF = profileDF[profileDF["stage"] == "asm_1stage"].drop(
            columns=["stage"], axis=1
        )
        asm_1stage_profileDF = asm_1stage_profileDF.rename(
            columns={"kernelName": "kernelName1", "err": "err1", "us": "us1"}
        )
        empty_1stage_profileDF = pd.DataFrame(index=asm_1stage_profileDF.index)

        empty_1stage_profileDF["kernelName2"] = None
        empty_1stage_profileDF["err2"] = 0
        empty_1stage_profileDF["us2"] = 0
        asm_1stage_profileDF = pd.concat(
            [asm_1stage_profileDF, empty_1stage_profileDF], axis=1
        )
        asm_1stage_profileDF["run_1stage"] = 1
        profileDF = pd.merge(
            stage1_profileDF,
            stage2_profileDF,
            on=[
                "cu_num",
                "token",
                "model_dim",
                "inter_dim",
                "expert",
                "topk",
                "act_type",
                "dtype",
                "q_dtype_a",
                "q_dtype_w",
                "q_type",
                "use_g1u1",
                "doweight_stage1",
                "block_m",
            ],
            how="inner",
        )
        profileDF["run_1stage"] = 0
        profileDF = pd.concat([profileDF, asm_1stage_profileDF], axis=0)
        if profileDF.shape[0] == 0:
            print("no moe solution can pass for ", key)
            continue
        profileDF["total_us"] = profileDF["us1"] + profileDF["us2"]
        best_one = profileDF.loc[profileDF["total_us"].idxmin()]
        bests.append(best_one)
        print(f"finish tuning, cost {time.perf_counter()-startTS:.8f}s")
    if len(prorfiles) > 0:
        return pd.concat(prorfiles), pd.concat(bests, axis=1).T
    else:
        return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--untune_file",
        default="aiter/configs/untuned_fmoe.csv",
        required=False,
        help="input",
    )

    parser.add_argument(
        "-o",
        "--tune_file",
        default="aiter/configs/tuned_fmoe.csv",
        required=False,
        help="output: tuning result store this file",
    )
    parser.add_argument(
        "-o2",
        "--profile_file",
        default="aiter/configs/profile_fmoe.csv",
        required=False,
        help="output: tuning result store this file",
    )

    parser.add_argument(
        "--sort",
        action="store_true",
        required=False,
        help="Arranged according to the B M N K size",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        required=False,
        help="All the kernels are tuned, if not, only kernels that are not in the tuned_fmoe.csv are tuned",
    )

    parser.add_argument(
        "--last",
        action="store_true",
        required=False,
        help="Only last kernel is tuned, if not, only kernels that are not in the tuned_fmoe.csv are tuned",
    )
    parser.add_argument(
        "--mp",
        type=int,
        default=torch.cuda.device_count(),
        help="Tuning on multiple GPUs using multiple processes",
    )

    args = parser.parse_args()
    untunedf = pd.read_csv(args.untune_file)
    untunedf = untunedf.drop_duplicates(keep="last")

    if not args.all or args.last:
        if os.path.exists(args.tune_file):
            old_tunedf = pd.read_csv(args.tune_file)
        else:
            old_tunedf = pd.DataFrame(
                columns=[
                    "cu_num",
                    "token",
                    "model_dim",
                    "inter_dim",
                    "expert",
                    "topk",
                    "act_type",
                    "dtype",
                    "q_dtype_a",
                    "q_dtype_w",
                    "q_type",
                    "use_g1u1",
                    "doweight_stage1",
                    "block_m",
                    "ksplit",
                    "us1",
                    "kernelName1",
                    "err1",
                    "us2",
                    "kernelName2",
                    "err2",
                    "total_us",
                    "run_1stage",
                ]
            )
    else:
        old_tunedf = None

    if args.last:
        untunedf = untunedf.iloc[-1:]

    elif old_tunedf is not None and not args.all:
        gpu = torch.cuda.current_device()
        device_properties = torch.cuda.get_device_properties(gpu)
        cu_num = device_properties.multi_processor_count
        untunedf["cu_num"] = cu_num
        untunedf_cols = untunedf.columns
        mask = untunedf.apply(tuple, axis=1).isin(
            old_tunedf[untunedf_cols].apply(tuple, axis=1)
        )
        untunedf = untunedf[~mask]

    tunedf = None
    # tunedf = pd.read_csv(args.tune_file)
    profiles, tunedf = go(untunedf, tunedf, args.mp)
    if old_tunedf is not None and tunedf is not None:
        tunedf = pd.concat([old_tunedf, tunedf], axis=0)
    if tunedf is not None:
        tunedf = tunedf.astype(str).drop_duplicates(
            subset=[
                "cu_num",
                "token",
                "model_dim",
                "inter_dim",
                "expert",
                "topk",
                "act_type",
                "dtype",
                "q_dtype_a",
                "q_dtype_w",
                "q_type",
                "use_g1u1",
                "doweight_stage1",
            ],
            keep="last",
        )
        tunedf.to_csv(args.tune_file, index=False)
    if profiles is not None:
        profiles.to_csv(args.profile_file, index=False)
