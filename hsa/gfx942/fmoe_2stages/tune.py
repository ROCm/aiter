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

sys.path.insert(0, f"{AITER_CSRC_DIR}/ck_gemm_moe_2stages_codegen/")
from gemm_moe_ck2stages_common import get_gemm1_kernels_list, get_gemm2_kernels_list

torch.set_default_device("cuda")
torch.int4 = getattr(torch, "int4", torch.uint32)


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
    num_valid_ids,
    dtype,
    topk,
    kernelName,
    w1_scale,
    a1_scale,
    blockM,
    sorted_weights,
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
    num_valid_ids,
    dtype,
    topk,
    kernelName,
    w2_scale,
    a2_scale,
    blockM,
    sorted_weights,
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
        "doweight_stage1",
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
            doweight_stage1,
        ) = line
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
        del input, w1, w2, score

        ref1 = torch_moe_stage1(
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
            doweight=doweight_stage1,
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
        ref2 = torch_moe_stage2(
            a2_qt,
            w1_qt,
            w2_qt,
            topk_weights,
            topk_ids,
            quant_type=q_type,
            dtype=dtype,
            a2_scale=a2_scale,
            w2_scale=w2_scale,
            doweight=not doweight_stage1,
        )

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

        w1_qt_shffle = shuffle_weight(w1_qt, (16, 16))
        w2_qt_shffle = shuffle_weight(w2_qt, (16, 16))

        for blockM in blockMs:
            sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = (
                moe_sorting(topk_ids, topk_weights, expert, model_dim, dtype, blockM)
            )
            del moe_buf
            if q_type != QuantType.per_1x128:
                out1 = torch.empty(
                    (token, topk, inter_dim),
                    dtype=dtype,
                )
            else:
                ratio = a1_scale.element_size() // a1_qt.element_size()
                out1 = torch.empty(
                    (token + (token * ratio + 127) // 128, topk, inter_dim),
                    dtype=q_dtype_a,
                )

            if use_g1u1 and q_dtype_w != torch.int4:
                for el in asm_kernels.get(blockM, []):
                    tasks.append(
                        (
                            ("stage1", el, blockM),  # tag
                            asm_stage1,  # func
                            (
                                a1_qt,
                                w1_qt_shffle,
                                w2_qt_shffle,
                                sorted_ids,
                                sorted_expert_ids,
                                num_valid_ids,
                                out1.view(dtypes.bf16),
                                topk,
                                blockM,
                                el,
                                0,
                                act_type,
                                q_type,
                                (
                                    a1_scale.t().contiguous()
                                    if q_type == QuantType.per_1x128
                                    else a1_scale
                                ),
                                w1_scale,
                                sorted_weights if doweight_stage1 else None,
                            ),
                            {},
                            None,
                            (),
                            {},
                            (ref1),
                            0.01,
                            0.01,
                            True,
                        )
                    )

            if blockM in [32, 64, 128]:
                if q_dtype_w == torch.int4:
                    w1_qt_shffle_ck = rearrange_4bit_elements(
                        convert_int8_to_uint32_int4(
                            shuffle_weight(w1_qt, (16, 16), use_int4=True)
                        )
                    )
                    w2_qt_shffle_ck = rearrange_4bit_elements(
                        convert_int8_to_uint32_int4(
                            shuffle_weight(w2_qt, (16, 16), use_int4=True)
                        )
                    )
                else:
                    w1_qt_shffle_ck = w1_qt_shffle
                    w2_qt_shffle_ck = w2_qt_shffle

                for kernel in ck_stage1_kernels.values():
                    if kernel.MPerBlock != blockM:
                        continue
                    tasks_ck.append(
                        (
                            ("stage1", kernel.name, blockM),  # tag
                            ck_moe_stage1_fwd_out,  # func
                            (
                                a1_qt,
                                w1_qt_shffle_ck,
                                w2_qt_shffle_ck,
                                sorted_ids,
                                sorted_expert_ids,
                                num_valid_ids,
                                dtype,
                                topk,
                                kernel.name,
                                w1_scale,
                                a1_scale,
                                blockM,
                                sorted_weights if doweight_stage1 else None,
                                q_type,
                                act_type,
                            ),
                            {},
                            None,
                            (),
                            {},
                            (ref1),
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
                            ("stage2", kernel.name, blockM),  # tag
                            ck_moe_stage2_fwd_out,  # func
                            (
                                a2_qt,
                                w1_qt_shffle_ck,
                                w2_qt_shffle_ck,
                                sorted_ids,
                                sorted_expert_ids,
                                num_valid_ids,
                                dtype,
                                topk,
                                kernel.name,
                                w2_scale,
                                a2_scale,
                                blockM,
                                sorted_weights if not doweight_stage1 else None,
                                q_type,
                                act_type,
                            ),
                            {},
                            None,
                            (),
                            {},
                            (ref2),
                            0.01,
                            0.01,
                            True,
                        )
                    )

        if tasks is None and tasks_ck is None:
            print("no moe solution can tune for ", line)
            continue
        print(f"tasks is {len(tasks)}, tasks_ck is {len(tasks_ck)}")
        in_data = [(len(tasks) + len(tasks_ck), ())]
        rets = mp_tuner(tasks + tasks_ck, in_data, 1, True)

        profileDF = []
        for (stage, kernelName, block_m), us, err in rets:
            if us == float("inf"):
                continue
            profileDF.append(
                [
                    stage,
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
        profileDF = pd.merge(
            stage1_profileDF,
            stage2_profileDF,
            on=[
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
        if profileDF.shape[0] == 0:
            print("no moe solution can pass for ", line)
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
        tunedf = tunedf.astype(str).drop_duplicates(
            subset=[
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
