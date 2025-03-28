# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import aiter
import pandas as pd
import argparse
from aiter.fused_moe_bf16_asm import (
    fused_topk,
    moe_sorting_ck,
)
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.mp_tuner import mp_tuner
from aiter.test_common import checkAllclose
from aiter import QuantType

torch.set_default_device("cuda")


def asm_stage1(
    a1_qt,
    w1_qt,
    w2_qt,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    out,
    kernelName,
    blockM,
    a1_scale,
    w1_scale,
):
    aiter.moe_stage1_fp8_g1u1(
        a1_qt,
        w1_qt,
        w2_qt,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        out,
        kernelName,
        blockM,
        a1_scale,
        w1_scale,
    )
    return out


def ck_stage1(
    input,
    w1,
    w2,
    sorted_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,
    topk,
    w1_scale,
    a1_scale,
    blockM,
    token,
    dtype,
):
    tmp = torch.empty(
        (token, topk, w1.shape[1]),
        dtype=dtype,
        device=input.device,
    )
    aiter.ck_moe_stage1(
        input,
        w1,
        w2,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        tmp,
        topk,
        w1_scale,
        a1_scale,
        blockM,
    )
    aiter.silu_and_mul(out, tmp)
    return out


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
        w1 = w1 * w1_scale.view(num_experts, -1, 1)

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


def go(
    untunedf,
    tunedf,
):
    blockMs = [16, 32, 48, 64, 128, 160]
    asm_kernels = {
        16: [
            "fmoe_stage1_bf16_pertokenFp8_g1u1_16x64",
            "fmoe_stage1_bf16_pertokenFp8_g1u1_16x64_pf2",
            "fmoe_stage1_bf16_pertokenFp8_g1u1_16x512_pf2",
        ],
        32: [
            "fmoe_stage1_bf16_pertokenFp8_g1u1_32x64",
            "fmoe_stage1_bf16_pertokenFp8_g1u1_32x128",
            "fmoe_stage1_bf16_pertokenFp8_g1u1_32x128_2tg_pf2",
            "fmoe_stage1_bf16_pertokenFp8_g1u1_32x128_2tg_pf3",
            "fmoe_stage1_bf16_pertokenFp8_g1u1_32x128_3tg_pf2",
            "fmoe_stage1_bf16_pertokenFp8_g1u1_32x512_pf2",
        ],
        48: ["fmoe_stage1_bf16_pertokenFp8_g1u1_48x128"],
        128: [
            "fmoe_stage1_bf16_pertokenFp8_g1u1_128x128",
            "fmoe_stage1_bf16_pertokenFp8_g1u1_128x128_pf2",
        ],
        160: ["fmoe_stage1_bf16_pertokenFp8_g1u1_160x128_pf2"],
    }
    args = [
        "token",
        "model_dim",
        "inter_dim",
        "expert",
        "topk",
        "dtype",
        "q_dtype",
        "q_type",
        "use_g1u1",
    ]
    print(untunedf[args])
    prorfiles = []
    bests = []
    for line in untunedf[args].values:
        token, model_dim, inter_dim, expert, topk, dtype, q_dtype, q_type, use_g1u1 = (
            line
        )
        dtype = eval(dtype)
        q_dtype = eval(q_dtype)
        q_type = eval(q_type)
        torch_quant = aiter.get_torch_quant(q_type)
        input = torch.randn((token, model_dim), dtype=dtype)
        if use_g1u1:
            w1 = torch.randn((expert, inter_dim * 2, model_dim), dtype=dtype) / 10
        else:
            w1 = torch.randn((expert, inter_dim, model_dim), dtype=dtype)
        w2 = torch.randn((expert, model_dim, inter_dim), dtype=dtype)

        score = torch.randn((token, expert), dtype=dtype)
        topk_weights, topk_ids = fused_topk(input, score, topk, True)
        w1_qt, w1_scale = torch_quant(w1, quant_dtype=q_dtype)
        w2_qt, w2_scale = torch_quant(w2, quant_dtype=q_dtype)
        w1_qt = w1_qt.view(w1.shape)
        w2_qt = w2_qt.view(w2.shape)
        a1_qt, a1_scale = torch_quant(input, quant_dtype=q_dtype)

        out1_ref = torch_moe_stage1(
            a1_qt,
            w1_qt,
            w2_qt,
            topk_weights,
            topk_ids,
            dtype=dtype,
            fc1_scale=None,
            w1_scale=w1_scale,
            a1_scale=a1_scale,
        )
        gate, up = out1_ref.split([inter_dim, inter_dim], dim=-1)
        ref = F.silu(gate) * up

        tasks = []
        tasks_ck = []
        for blockM in blockMs:
            sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = (
                moe_sorting_ck(topk_ids, topk_weights, expert, model_dim, dtype, blockM)
            )
            out = torch.empty(
                (token, topk, inter_dim),
                dtype=dtype,
            )
            for el in asm_kernels.get(blockM, []):
                tasks.append(
                    (
                        el,  # tag
                        asm_stage1,  # func
                        (
                            a1_qt,
                            shuffle_weight(w1_qt, (16, 16)),
                            shuffle_weight(w2_qt, (16, 16)),
                            sorted_ids,
                            sorted_weights,
                            sorted_expert_ids,
                            num_valid_ids,
                            out,
                            el,
                            blockM,
                            a1_scale,
                            w1_scale,
                        ),
                    )
                )

            if blockM in [32, 64, 128]:
                tasks_ck.append(
                    (
                        f"ck_{blockM}",  # tag
                        ck_stage1,  # func
                        (
                            a1_qt,
                            shuffle_weight(w1_qt, layout=(32, 32)),
                            w2_qt,
                            sorted_ids,
                            sorted_expert_ids,
                            num_valid_ids,
                            out,
                            topk,
                            w1_scale,
                            a1_scale,
                            blockM,
                            token,
                            dtype,
                        ),
                    )
                )
        rets = mp_tuner(tasks + tasks_ck)

        profileDF = []
        for tag, us, _ in rets:
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
                    dtype,
                    q_dtype,
                    q_type,
                    use_g1u1,
                    us,
                    tag,
                    f"{err:.1%}",
                ]
            )
        profileDF = pd.DataFrame(profileDF, columns=args + ["us", "tag", "err"])
        best_one = profileDF.loc[profileDF["us"].idxmin()]
        prorfiles.append(profileDF)
        bests.append(best_one)
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
