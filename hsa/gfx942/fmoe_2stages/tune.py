# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import aiter
import pandas as pd
import argparse
import time
import os
from aiter import ActivationType, QuantType
from aiter.jit.core import get_asm_dir
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
from aiter import dtypes

torch.set_default_device("cuda")
torch.int4 = torch.uint32


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
    elif qType == QuantType.per_128x128:
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


def go(
    untunedf,
    tunedf,
):
    startTS = time.perf_counter()
    # blockMs = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160]
    blockMs = [32, 64, 128]

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
        "doweight_stage1"
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
            doweight_stage1
        ) = line
        dtype = eval(dtype)
        q_dtype_a = eval(q_dtype_a)
        q_dtype_w = eval(q_dtype_w)
        q_type = eval(q_type)
        print("\nStart tuning", line)
        # if q_dtype_a == dtypes.i8 and q_type == QuantType.per_Tensor:
        #     print(f'no moe solution for ', line)
        #     continue
        act_type = eval(act_type)
        input = torch.randn((token, model_dim), dtype=dtype)
        if use_g1u1:
            w1 = torch.randn((expert, inter_dim * 2, model_dim), dtype=dtype) / 10
        else:
            w1 = torch.randn((expert, inter_dim, model_dim), dtype=dtype)
        w2 = torch.randn((expert, model_dim, inter_dim), dtype=dtype)
        w1_qt, w1_scale = weight_quant(w1, q_type, quant_dtype=q_dtype_w)
        w2_qt, w2_scale = weight_quant(w2, q_type, quant_dtype=q_dtype_w)
        w1_qt = w1_qt.view(w1.shape)
        w2_qt = w2_qt.view(w2.shape)
        score = torch.randn((token, expert), dtype=dtype)
        topk_weights, topk_ids = fused_topk(input, score, topk, True)
        if q_type == QuantType.per_128x128:
            a1_qt, a1_scale = aiter.pertoken_quant(
                input.view(token, -1, 128), quant_dtype=q_dtype_a
            )
            a1_qt = a1_qt.view(token, model_dim)
            a1_scale = a1_scale.squeeze(-1)
        else:
            torch_quant = aiter.get_torch_quant(q_type)
            a1_qt, a1_scale = torch_quant(input, quant_dtype=q_dtype_a)
        del input, w1, w2, score

        ref = torch_moe_stage1(
            a1_qt,
            w1_qt,
            w2_qt,
            topk_weights,
            topk_ids,
            activation=act_type,
            quant_type=q_type,
            dtype=dtype,
            a1_scale=a1_scale,
            w1_scale=w1_scale,
            doweight=doweight_stage1
        )
        if q_type == QuantType.per_128x128:
            ref, ref_scale = aiter.pertoken_quant(
                ref.view(ref.shape[0], -1, 128), quant_dtype=q_dtype_a
            )
            ref = ref.view(ref.shape[0], topk, -1).to(dtypes.fp32)
            ref_scale = ref_scale.view(token, -1)

        tasks = []
        tasks_ck = []

        kernels_list_csv = f"{get_asm_dir()}/fmoe_2stages/fmoe_stage1_bf16_pertoken{{quantDtype}}{{extraInfo}}_g1u1.csv"

        def get_kernels_dict(file):
            if not os.path.exists(file):
                print(f"ASM kernel list file not exist: {file}")
                return {}
            df = pd.read_csv(file)
            kernel_dict = df.groupby("tile_m")["knl_name"].apply(list).to_dict()
            return kernel_dict

        extraInfo = ""
        if q_type == QuantType.per_128x128:
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

        for blockM in blockMs:
            sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = (
                moe_sorting(topk_ids, topk_weights, expert, model_dim, dtype, blockM)
            )
            if q_type != QuantType.per_128x128:
                out = torch.empty(
                    (token, topk, inter_dim),
                    dtype=dtype,
                )
            else:
                ratio = a1_scale.element_size() // a1_qt.element_size()
                out = torch.empty(
                    (token + (token * ratio + 127) // 128, topk, inter_dim),
                    dtype=q_dtype_a,
                )

            if use_g1u1 and q_dtype_w != torch.int4:
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
                                out.view(dtypes.bf16),
                                blockM,
                                el,
                                0,
                                act_type,
                                q_type,
                                (
                                    a1_scale.t().contiguous()
                                    if q_type == QuantType.per_128x128
                                    else a1_scale
                                ),
                                w1_scale,
                                sorted_weights if doweight_stage1 else None,
                            ),
                        )
                    )

            if blockM in [32, 64, 128] and q_type != QuantType.per_128x128:
                if q_dtype_w == torch.int4:
                    w1_qt_shffle = rearrange_4bit_elements(
                        convert_int8_to_uint32_int4(
                            shuffle_weight(w1_qt, (32, 32), use_int4=True)
                        )
                    )
                else:
                    w1_qt_shffle = shuffle_weight(w1_qt, layout=(16, 16))

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
                            sorted_weights if doweight_stage1 else None,
                        ),
                    )
                )
        if tasks is None and tasks_ck is None:
            print(f"no moe solution for ", line)
            continue
        rets = mp_tuner(tasks + tasks_ck)

        profileDF = []
        for (tag, block_m), us, _ in rets:
            if us == float("inf"):
                continue
            if q_type == QuantType.per_128x128:
                scale = (
                    _[token:, ...]
                    .view(-1)[: (token * topk * inter_dim * 4 // 128)]
                    .view(dtypes.fp32)
                    .view(token, -1)
                )
                _ = _[:token, :, :].to(dtypes.fp32)
            err = checkAllclose(
                ref.to("cpu"), _.to(ref.dtype), msg=f"[{tag:<50}]: {us:.2f}us ......      "
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
                    q_dtype_w if q_dtype_w != torch.int4 else "torch.int4",
                    q_type,
                    use_g1u1,
                    doweight_stage1,
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

    args = parser.parse_args()
    untunedf = pd.read_csv(args.untune_file)
    untunedf = untunedf.drop_duplicates(keep="last")
    
    if not args.all or args.last:
        if os.path.exists(args.tune_file):
            old_tunedf = pd.read_csv(args.tune_file)
        else:
            old_tunedf = None
    else:
        old_tunedf = None

    if args.last:
        untunedf = untunedf.iloc[-1:]
    elif old_tunedf is not None and not args.all:
        untunedf_cols = untunedf.columns
        mask = untunedf.apply(tuple, axis=1).isin(
            old_tunedf[untunedf_cols].apply(tuple, axis=1)
        )
        untunedf = untunedf[~mask]
    
    tunedf = None
    # tunedf = pd.read_csv(args.tune_file)
    profiles, tunedf = go(untunedf, tunedf)
    if old_tunedf is not None and tunedf is not None:
        tunedf = pd.concat([old_tunedf, tunedf], axis=0)
    if tunedf is not None:
        tunedf = tunedf.astype(str).drop_duplicates(subset=["token","model_dim","inter_dim","expert","topk","act_type","dtype",
                                                "q_dtype_a","q_dtype_w","q_type","use_g1u1","doweight_stage1"], keep="last")
        tunedf.to_csv(args.tune_file, index=False)
    if profiles is not None:
        profiles.to_csv(args.profile_file, index=False)
