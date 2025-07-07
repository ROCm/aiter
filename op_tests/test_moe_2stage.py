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

from aiter.fused_moe import (
    fused_topk,
    moe_sorting,
    fused_moe,
    torch_moe_stage1,
    torch_moe_stage2,
    get_block_size_M,
)


from aiter.ops.shuffle import shuffle_weight
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
    if get_gfx() not in ["gfx950"] and qType == aiter.QuantType.per_1x32:
        return
    torch_quant = aiter.get_torch_quant(qType)
    torch_act = aiter.get_torch_act(actType)
    input = torch.randn((token, model_dim), dtype=dtype)
    if use_g1u1:
        w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype)
    else:
        w1 = torch.randn((E, inter_dim, model_dim), dtype=dtype)
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype)

    score = torch.randn((token, E), dtype=dtype)
    topk_weights, topk_ids = fused_topk(input, score, topk, True)

    M, _ = topk_ids.shape

    BLOCK_SIZE_M = get_block_size_M(M, topk, E, inter_dim)
    if qType == aiter.QuantType.per_128x128:
        BLOCK_SIZE_M = 64
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

    w1_qt_aiter = w1_qt
    w2_qt_aiter = w2_qt

    if qType == aiter.QuantType.per_128x128:
        a1_qt, a1_scale = aiter.pertoken_quant(
            input.view(token, -1, 128), quant_dtype=AQDType
        )
        a1_qt = a1_qt.view(token, model_dim)
        a1_scale = a1_scale.squeeze(-1)
    else:
        a1_qt, a1_scale = torch_quant(input, quant_dtype=AQDType)
    # w1_scale = w1_scale.fill_(1)
    # a1_scale = a1_scale.fill_(1)

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
        doweight=doweight_stage1,
    )

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
    else:
        w1_qt_aiter = shuffle_weight(w1_qt_aiter, layout=(16, 16))
        w2_qt_aiter = shuffle_weight(w2_qt_aiter, layout=(16, 16))

    # # ######################## ck stage 1 start ###########
    # # a1_qt, a1_scale = torch_quant(input, quant_dtype=AQDType)
    # # out1_ck = torch.empty((token, topk, inter_dim), dtype=dtype)
    # out1_ck, us = run_perftest(
    #     ck_moe_stage1,
    #     a1_qt,
    #     w1_qt_aiter,
    #     w2_qt_aiter,
    #     sorted_ids,
    #     sorted_expert_ids,
    #     num_valid_ids,
    #     w1_scale,
    #     a1_scale,
    #     dtype,
    #     topk,
    #     BLOCK_SIZE_M,
    #     actType,
    #     quant_type=qType,
    #     sorted_weights=sorted_weights if doweight_stage1 else None,
    #     needTrace=True,
    # )

    # checkAllclose(
    #     out1_ref,
    #     out1_ck,
    #     msg=f"[perf]  ck_moe_stage1:{us:>8.2f} us, {token*model_dim*inter_dim*2*topk*2/us/1000/1000:>8.2f} tflops......(quant:{AQDType})",
    # )
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

    # out2_ck, us = run_perftest(
    #     ck_moe_stage2,
    #     a2_qt,
    #     w1_qt_aiter,
    #     w2_qt_aiter,
    #     sorted_ids,
    #     sorted_expert_ids,
    #     num_valid_ids,
    #     w2_scale,
    #     a2_scale,
    #     dtype,
    #     topk,
    #     BLOCK_SIZE_M,
    #     actType,
    #     quant_type,
    #     sorted_weights if not doweight_stage1 else None,
    # )

    # checkAllclose(
    #     out2_ref,
    #     out2_ck,
    #     msg=f"[perf]  ck_moe_stage2:{us:>8.2f} us, {token*model_dim*inter_dim*topk*2/us/1000/1000:>8.2f} tflops......(quant:{AQDType})",
    # )
    # ######################## stage 2 end ###########

    # # ######################## fused 2 stage #########
    # out2_ck, us = run_perftest(
    #     ck_moe_2stages,
    #     input,
    #     w1_qt_aiter,
    #     w2_qt_aiter,
    #     topk_weights,
    #     topk_ids,
    #     quant_type=qType,
    #     fc1_scale=w1_scale,  # [expert(local_expert:EP), inter_dim, 1]
    #     fc2_scale=w2_scale,  # [expert(local_expert:EP), model_dim, 1]
    #     block_size=BLOCK_SIZE_M,
    #     activation=actType,
    #     doweight_stage1=doweight_stage1,
    # )
    # checkAllclose(
    #     out2_ref,
    #     out2_ck,
    #     msg=f"ck_moe_2stages:{us:>8.2f} us, {token*model_dim*inter_dim*3*topk*2/us/1000/1000:>8.2f} tflops......(quant:{AQDType})",
    # )

    if dtype == dtypes.bf16:
        out2_aiter, us_fuse = run_perftest(
            fused_moe,
            input,
            w1_qt_aiter,
            w2_qt_aiter,
            topk_weights,
            topk_ids,
            w1_scale=fp4_utils.e8m0_shuffle(
                w1_scale
            ),  # e8m0_shuffle will do nothing if it's a fp32
            w2_scale=fp4_utils.e8m0_shuffle(w2_scale),
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


l_dtype = ["bf16", "fp16"]
l_dim = [(6144, 4096)]
l_tokenNum = [
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
l_quant = [
    (aiter.QuantType.No, None, None),  # a16w16
    (aiter.QuantType.per_Tensor, dtypes.fp8, dtypes.fp8),  # a8w8
    (aiter.QuantType.per_Token, dtypes.fp8, dtypes.fp8),  # a8w8
    (aiter.QuantType.per_Token, dtypes.fp8, torch.int4),  # a8w4
    (aiter.QuantType.per_1x32, dtypes.fp4x2, dtypes.fp4x2),  # a4w4
    # (aiter.QuantType.per_128x128, dtypes.fp8, dtypes.fp8),  # a8w8 TODO add test
]
l_act = [aiter.ActivationType.Silu, aiter.ActivationType.Gelu]
l_doweight_stage1 = [False, True]

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
    # (aiter.QuantType.per_128x128, dtypes.fp8, dtypes.fp8),  # a8w8 TODO add test""",
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
    default=8,
    help="""Number of experts.
    e.g.: -e 8""",
)

parser.add_argument(
    "-k",
    "--topk",
    type=int,
    default=2,
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

for (
    dtype,
    act_type,
    (quant_type, aq_dtype, wq_dtype),
    (model_dim, inter_dim),
    doweight_stage1,
) in itertools.product(l_dtype, l_act, l_quant, l_dim, l_doweight_stage1):
    df = []
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
