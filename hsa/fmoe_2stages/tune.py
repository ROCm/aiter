# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import aiter
import pandas as pd
import argparse
import time
from aiter import ActivationType, QuantType
from aiter.fused_moe import (
    fused_topk,
    moe_sorting,
    asm_stage1,
    ck_stage1,
    torch_moe_stage1,
)
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.mp_tuner import mp_tuner
from aiter.test_common import checkAllclose
from aiter import QuantType
from aiter.int4_utils import *

torch.set_default_device("cuda")
torch.int4 = torch.uint32

def go(
    untunedf,
    tunedf,
):
    startTS = time.perf_counter()
    blockMs = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160]
    asm_kernels = {
        16: [
            "fmoe_stage1_bf16_pertokenFp8_g1u1_16x128_4tg_pf2",
            "fmoe_stage1_bf16_pertokenFp8_g1u1_16x128_4tg_pf3",
            "fmoe_stage1_bf16_pertokenFp8_g1u1_16x256_2tg_pf3",
            "fmoe_stage1_bf16_pertokenFp8_g1u1_16x256_3tg_pf2",
            "fmoe_stage1_bf16_pertokenFp8_g1u1_16x512_2tg_pf2",
            "fmoe_stage1_bf16_pertokenFp8_g1u1_16x512_pf3",
        ],
        32: [
            "fmoe_stage1_bf16_pertokenFp8_g1u1_32x128_3tg_pf2",
            "fmoe_stage1_bf16_pertokenFp8_g1u1_32x128_3tg_pf3",
            "fmoe_stage1_bf16_pertokenFp8_g1u1_32x256_2tg_pf2",
            "fmoe_stage1_bf16_pertokenFp8_g1u1_32x256_2tg_pf3",
            "fmoe_stage1_bf16_pertokenFp8_g1u1_32x512_pf2",
            "fmoe_stage1_bf16_pertokenFp8_g1u1_32x512_pf3",
        ],
        48: [
            "fmoe_stage1_bf16_pertokenFp8_g1u1_48x128_2tg_pf2",
            "fmoe_stage1_bf16_pertokenFp8_g1u1_48x128_2tg_pf3",
            "fmoe_stage1_bf16_pertokenFp8_g1u1_48x256_pf2",
            "fmoe_stage1_bf16_pertokenFp8_g1u1_48x256_pf3",
            "fmoe_stage1_bf16_pertokenFp8_g1u1_48x512_pf2",
            "fmoe_stage1_bf16_pertokenFp8_g1u1_48x512_pf3",
        ],
        64: [
            "fmoe_stage1_bf16_pertokenFp8_g1u1_64x128_2tg_pf2",
            "fmoe_stage1_bf16_pertokenFp8_g1u1_64x128_2tg_pf3",
            "fmoe_stage1_bf16_pertokenFp8_g1u1_64x256_pf2",
            "fmoe_stage1_bf16_pertokenFp8_g1u1_64x256_pf3",
        ],
        80: [
            "fmoe_stage1_bf16_pertokenFp8_g1u1_80x128_2tg_pf2",
            "fmoe_stage1_bf16_pertokenFp8_g1u1_80x128_pf3",
            "fmoe_stage1_bf16_pertokenFp8_g1u1_80x256_pf2",
            "fmoe_stage1_bf16_pertokenFp8_g1u1_80x256_pf3",
        ],
        96: [
            "fmoe_stage1_bf16_pertokenFp8_g1u1_96x128_pf2",
            "fmoe_stage1_bf16_pertokenFp8_g1u1_96x128_pf3",
        ],
        112: [
            "fmoe_stage1_bf16_pertokenFp8_g1u1_112x128_pf2",
            "fmoe_stage1_bf16_pertokenFp8_g1u1_112x128_pf3",
        ],
        128: [
            "fmoe_stage1_bf16_pertokenFp8_g1u1_128x128_pf2",
        ],
        144: [
            "fmoe_stage1_bf16_pertokenFp8_g1u1_144x128_pf2",
            "fmoe_stage1_bf16_pertokenFp8_g1u1_144x128_pf3",
        ],
        160: [
            "fmoe_stage1_bf16_pertokenFp8_g1u1_160x128_pf2",
            "fmoe_stage1_bf16_pertokenFp8_g1u1_160x128_pf3",
        ],
    }
    args = [
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
    ]
    print(untunedf[args])
    prorfiles = []
    bests = []
    for line in untunedf[args].values:
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
        ) = line
        dtype = eval(dtype)
        q_dtype_a = eval(q_dtype_a)
        q_dtype_w = eval(q_dtype_w)
        if q_dtype_a == torch.int8:
            print(f'no moe solution for ', line)
            continue
        q_type = eval(q_type)
        act_type = eval(act_type)
        torch_quant = aiter.get_torch_quant(q_type)
        input = torch.randn((token, model_dim), dtype=dtype)
        if q_dtype_w == torch.int4:
            if use_g1u1:
                w1 = torch.randn((expert, inter_dim * 2, model_dim), dtype=dtype) / 10
            else:
                w1 = torch.randn((expert, inter_dim, model_dim), dtype=dtype)
            w2 = torch.randn((expert, model_dim, inter_dim), dtype=dtype)
            w1_qt, w1_scale = torch_quant(w1, quant_dtype=torch.int8, dtypeMax=7)
            w2_qt, w2_scale = torch_quant(w2, quant_dtype=torch.int8, dtypeMax=7)
        else:
            if use_g1u1:
                w1 = torch.randn((expert, inter_dim * 2, model_dim), dtype=dtype) / 10
            else:
                w1 = torch.randn((expert, inter_dim, model_dim), dtype=dtype)
            w2 = torch.randn((expert, model_dim, inter_dim), dtype=dtype)
            w1_qt, w1_scale = torch_quant(w1, quant_dtype=q_dtype_w)
            w2_qt, w2_scale = torch_quant(w2, quant_dtype=q_dtype_w)
        w1_qt = w1_qt.view(w1.shape)
        w2_qt = w2_qt.view(w2.shape)
        score = torch.randn((token, expert), dtype=dtype)
        topk_weights, topk_ids = fused_topk(input, score, topk, True)
        a1_qt, a1_scale = torch_quant(input, quant_dtype=q_dtype_a)

        ref = torch_moe_stage1(
            a1_qt,
            w1_qt,
            w2_qt,
            topk_weights,
            topk_ids,
            activation=act_type,
            dtype=dtype,
            a1_scale=a1_scale,
            w1_scale=w1_scale,
        )

        tasks = []
        tasks_ck = []
        for blockM in blockMs:
            sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = (
                moe_sorting(topk_ids, topk_weights, expert, model_dim, dtype, blockM)
            )
            out = torch.empty(
                (token, topk, inter_dim),
                dtype=dtype,
            )
            if use_g1u1 and dtype == torch.bfloat16 and \
                act_type == ActivationType.Silu and q_dtype_w == torch.float8_e4m3fnuz:
                for el in asm_kernels.get(blockM, []):
                    tasks.append(
                        (
                            (el, blockM),  # tag
                            asm_stage1,  # func
                            (
                                a1_qt,
                                shuffle_weight(w1_qt, (16, 16)),
                                shuffle_weight(w2_qt, (16, 16)),
                                sorted_ids,
                                sorted_expert_ids,
                                num_valid_ids,
                                out,
                                blockM,
                                el,
                                0,
                                act_type,
                                a1_scale,
                                w1_scale,
                            ),
                        )
                    )

            if blockM in [32, 64, 128]:
                if q_dtype_w == torch.int4:
                    w1_qt_shffle = rearrange_4bit_elements(convert_int8_to_uint32_int4(shuffle_weight(w1_qt, (32, 32), use_int4=True)))
                else:
                    w1_qt_shffle = shuffle_weight(w1_qt, layout=(32, 32))
                
                tasks_ck.append(
                    (
                        (f"ck_{blockM}", blockM),  # tag
                        ck_stage1,  # func
                        (
                            a1_qt,
                            w1_qt_shffle,
                            w2_qt,
                            sorted_ids,
                            sorted_expert_ids,
                            num_valid_ids,
                            out,
                            blockM,
                            act_type,
                            a1_scale,
                            w1_scale,
                        ),
                    )
                )
        if tasks is None and tasks_ck is None:
            print(f'no moe solution for ', line)
        rets = mp_tuner(tasks + tasks_ck)

        profileDF = []
        for (tag, block_m), us, _ in rets:
            err = checkAllclose(
                ref.to("cpu"), _, msg=f"[{tag:<50}]: {us:.2f}us ......      "
            )
            profileDF.append(
                [
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
                    block_m,
                    0,
                    us,
                    tag,
                    f"{err:.1%}",
                ]
            )
        profileDF = pd.DataFrame(
            profileDF, columns=args + ["block_m", "ksplit", "us", "tag", "err"]
        )
        best_one = profileDF.loc[profileDF["us"].idxmin()]
        prorfiles.append(profileDF)
        bests.append(best_one)
    print(f"finish tuning, cost {time.perf_counter()-startTS:.8f}s")
    return pd.concat(prorfiles), pd.concat(bests, axis=1).T


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

    args = parser.parse_args()
    untunedf = pd.read_csv(args.untune_file)
    tunedf = None
    # tunedf = pd.read_csv(args.tune_file)
    profiles, tunedf = go(untunedf, tunedf)
    tunedf.to_csv(args.tune_file, index=False)
    profiles.to_csv(args.profile_file, index=False)
