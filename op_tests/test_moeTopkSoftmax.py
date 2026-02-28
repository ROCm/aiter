# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter.test_common import (
    checkAllclose,
    benchmark,
    run_perftest,
    perftest,
)
from aiter import dtypes, get_gfx
import pandas as pd
import argparse

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)


@perftest(num_iters=2, num_warmup=1)
def test_nofuse(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    gating_output = torch.nn.functional.softmax(
        gating_output.float(),
        dim=-1,
    )
    topk_weights, topk_ids = gating_output.topk(
        k=topk,
        dim=-1,
        largest=True,
        sorted=True,
    )

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids.to(dtypes.i32)


@perftest()
def test_fuse(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    # hidden_states = torch.empty(gating_output.shape, dtype=dtypes.fp32, device=gating_output.device)
    # from aiter.fused_moe import fused_topk
    # return fused_topk(hidden_states, gating_output, topk, renormalize)

    M, expert = gating_output.shape
    topk_weights = torch.empty_strided(
        (M, topk), (topk + 10, 1), dtype=dtypes.fp32, device=gating_output.device
    )
    topk_ids = torch.empty_strided(
        (M, topk), (topk + 10, 1), dtype=dtypes.i32, device=gating_output.device
    )
    token_expert_indicies = torch.empty_strided(
        (M, topk), (topk + 10, 1), dtype=dtypes.i32, device=gating_output.device
    )
    aiter.topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indicies,
        gating_output,
        renormalize,
    )
    return topk_weights, topk_ids


@perftest()
def test_asm(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    M, expert = gating_output.shape
    topk_weights = torch.empty_strided(
        (M, topk), (topk + 10, 1), dtype=dtypes.fp32, device=gating_output.device
    )
    topk_ids = torch.empty_strided(
        (M, topk), (topk + 10, 1), dtype=dtypes.i32, device=gating_output.device
    )
    token_expert_indicies = torch.empty_strided(
        (M, topk), (topk + 10, 1), dtype=dtypes.i32, device=gating_output.device
    )
    aiter.topk_softmax_asm(
        topk_weights,
        topk_ids,
        token_expert_indicies,
        gating_output,
        renormalize,
    )
    del token_expert_indicies  # Not used. Will be used in the future.
    return topk_weights, topk_ids


@benchmark()
def test_topk_softmax(dtype, token, E, topk, renormalize=True):
    gating_output = torch.randn((token, E), dtype=dtype, device="cuda")

    (topk_weights_a, topk_ids_a), avg_a = test_nofuse(gating_output, topk, renormalize)
    id_ref, _ref = torch.sort(topk_ids_a)
    w_ref = topk_weights_a.gather(1, _ref)

    func_dict = {"hip": test_fuse, "asm": test_asm}
    ret = {}
    for tag, func in func_dict.items():
        if tag == "asm" and not (
            (E, topk) in [(128, 4), (128, 6), (128, 8), (256, 6), (256, 8), (384, 8)]
            and dtype in [dtypes.bf16, dtypes.fp32]
        ):
            continue
        (topk_weights, topk_ids), us = func(gating_output, topk, renormalize)
        topk_ids = topk_ids.to(dtypes.i32)
        id, _ref = torch.sort(topk_ids)
        weight = topk_weights.gather(1, _ref)
        ret[f"{tag} err"] = checkAllclose(w_ref, weight, msg=f"{tag} topk_weights")
        checkAllclose(id_ref, id, msg=f"{tag} topk_ids")
        ret[f"{tag} us"] = us
    return ret


# this function test a value/index pair, like the output of a topk function
# w.r.t a target dim
def check_topk_softmax_allclose(
    ref_val,
    ref_idx,
    tar_val,
    tar_idx,
    scores,
    bias,
    target_dim=-1,  # last dim by default
    target_dim_len=-1,  # the dim could be larger than ref/tar val dim length. if -1, then same size as
    sort_before_compare=True,  # this is useful when we don't care about the absolute position of the val/idx
    rtol=1e-2,
    atol=1e-2,
    tol_err_ratio=0.05,
    msg="",
    printNum=8,
    printLog=True,
):
    from aiter import logger

    # first let's sort the index in case
    if sort_before_compare:
        # NOTE: need add bias before sorting
        _, _r_sorted_idx = torch.sort(
            ref_val
            + bias.repeat(ref_val.shape[0], 1).gather(-1, ref_idx.to(dtype=torch.int64))
        )
        _, _t_sorted_idx = torch.sort(
            tar_val
            + bias.repeat(ref_val.shape[0], 1).gather(-1, tar_idx.to(dtype=torch.int64))
        )
        r_val = ref_val.gather(target_dim, _r_sorted_idx)
        t_val = tar_val.gather(target_dim, _t_sorted_idx)
        r_idx = ref_idx.gather(target_dim, _r_sorted_idx)
        t_idx = tar_idx.gather(target_dim, _t_sorted_idx)
    else:
        r_val = ref_val
        t_val = tar_val
        r_idx = ref_idx
        t_idx = tar_idx

    if target_dim_len < 0:
        target_dim_len = ref_val.shape[target_dim]

    assert target_dim_len >= ref_val.shape[target_dim]

    original_shape = list(ref_val.shape)
    original_shape[target_dim] = target_dim_len

    is_close_v = torch.isclose(r_val, t_val, rtol=rtol, atol=atol)
    is_close_i = torch.isclose(r_idx, t_idx)  # use high resolution for index

    scores_for_choice = scores.view(original_shape)
    if bias != None:
        scores_for_choice = scores_for_choice + bias.unsqueeze(0)

    if is_close_v.all():
        if printLog:
            logger.info(
                f"{msg}[check_topk_softmax_allclose/value {atol=} {rtol=} \033[32mpassed~\033[0m]"
            )

        if is_close_i.all():
            if printLog:
                logger.info(
                    f"{msg}[check_topk_softmax_allclose/index \033[32mpassed~\033[0m]"
                )
            return 0
        else:
            # this case there must be some duplicate value, and due to compare order, index maybe different
            mask = ~(is_close_i)
            val_mask = torch.zeros(original_shape, dtype=torch.bool)
            mismatch_r = scores_for_choice.gather(-1, r_idx.to(dtype=torch.int64))[mask]
            mismatch_t = scores_for_choice.gather(-1, t_idx.to(dtype=torch.int64))[mask]

            # if index mismatch, the the index pointed value must be the same
            # below we are checking such case
            is_close_dup_i = torch.isclose(mismatch_r, mismatch_t, rtol=rtol, atol=atol)

            if not is_close_dup_i.all():
                # this check should contain same index mask bool tensor, otherwise something wrong
                num = mask.sum()
                printNum = min(printNum, num)
                percent = (num / r_val.numel()).item()
                logger.info(
                    f"""{msg}[check_topk_softmax_allclose/index \033[32mfailed~\033[0m]"""
                )
                for i_row in range(r_idx.shape[0]):
                    for i_col in range(r_idx.shape[1]):
                        if r_idx[i_row, i_col] != t_idx[i_row, i_col]:
                            sr = scores_for_choice[i_row, r_idx[i_row, i_col]]
                            st = scores_for_choice[i_row, t_idx[i_row, i_col]]
                            is_close_ = torch.isclose(sr, st, rtol=rtol, atol=atol)
                            logger.info(
                                f"{msg} [{i_row}x{i_col}], r:{r_idx[i_row, i_col]}->{sr}, t:{t_idx[i_row, i_col]}->{st}"
                            )
                return 1

            else:
                if printLog:
                    logger.info(
                        f"{msg}[check_topk_softmax_allclose/index(duplicated) \033[32mpassed~\033[0m]"
                    )
                return 0

    else:
        mask = ~is_close_v
        num = mask.sum()
        printNum = min(printNum, num)
        percent = (num / r_val.numel()).item()
        if not printLog:
            return percent
        r_msked = r_val[mask]
        t_msked = t_val[mask]
        delta = (r_msked - t_msked).abs()
        if percent > tol_err_ratio:
            logger.info(
                f"""{msg}[check_topk_softmax_allclose.value {atol=} {rtol=} \033[31mfailed!\033[0m]
    ref  : {r_msked[:printNum]}
    tar  : {t_msked[:printNum]}
    delta:
           {delta[:printNum]}"""
            )
        return percent


@aiter.test_common.benchmark()
def test_biased_grouped_topk(
    token, expert, group, topk, topk_group, need_renorm, dtype, scale_factor=1.0
):
    ret = {}
    gating_output = torch.randn((token, expert), dtype=dtype)
    correction_bias = torch.randn((expert,), dtype=dtype)

    (w_ref, id_ref, score_ref), us_ref = run_perftest(
        aiter.biased_grouped_topk_torch,
        gating_output,
        correction_bias,
        topk,
        need_renorm,
        group,
        topk_group,
        True,  # return score
        num_iters=2,
        num_warmup=1,
    )
    w_ref = w_ref * scale_factor
    w_aiter = torch.empty_strided((token, topk), (topk + 10, 1), dtype=dtypes.fp32)
    id_aiter = torch.empty_strided((token, topk), (topk + 10, 1), dtype=dtypes.i32)
    _, us_aiter = run_perftest(
        aiter.biased_grouped_topk_hip,
        gating_output,
        correction_bias,
        w_aiter,
        id_aiter,
        group,
        topk_group,
        need_renorm,
        scale_factor,
    )

    # use a special function to check result. The HIP topk may using sort algorithm
    # ... which will make the result order unpredictable
    err = check_topk_softmax_allclose(
        w_ref,
        id_ref,
        w_aiter,
        id_aiter,
        score_ref,
        correction_bias,
        target_dim_len=expert,
        msg=f"[golden vs aiter]:{us_ref:>8.2f} us vs {us_aiter:>8.2f} us......",
    )
    id_ref, _ref = torch.sort(id_ref)
    id_aiter, _aiter = torch.sort(id_aiter)
    w_ref = w_ref.gather(1, _ref)
    w_aiter = w_aiter.gather(1, _aiter)
    # print(f'  {id_ref=}')
    # print(f'{id_aiter=}')
    # print(f'  {w_ref=}')
    # print(f'{w_aiter=}')
    # err = checkAllclose(w_ref, w_aiter, msg="topk_weights [golden vs aiter]")
    # checkAllclose(
    #     id_ref,
    #     id_aiter,
    #     msg=f"topk_ids     [golden vs aiter]:{us_ref:>8.2f} us vs {us_aiter:>8.2f} us......",
    # )
    ret["us_aiter"] = us_aiter
    ret["err_aiter"] = err
    # return {"err": err, "us": us_aiter}

    w_sglang = torch.empty_strided((token, topk), (topk + 10, 1), dtype=dtypes.fp32)
    id_sglang = torch.empty_strided((token, topk), (topk + 10, 1), dtype=dtypes.i32)
    _, us_sglang = run_perftest(
        aiter.moe_fused_gate,
        gating_output,
        correction_bias,
        w_sglang,
        id_sglang,
        group,
        topk_group,
        topk,
        0,
        scale_factor,
    )

    w_sglang = _[0]
    id_sglang = _[1]

    id_sglang, _sglang = torch.sort(id_sglang)
    w_sglang = w_sglang.gather(1, _sglang)
    ret["us_sglang"] = us_sglang

    # print(f"{w_ref=}")
    # print(f"{w_sglang=}")
    # print(f"{id_ref=}")
    # print(f"{id_sglang=}")

    err = checkAllclose(w_ref, w_sglang, msg="topk_weights [golden vs sglang]")
    checkAllclose(
        id_ref,
        id_sglang,
        msg=f"topk_ids     [aiter vs sglang]:{us_aiter:>8.2f} us vs {us_sglang:>8.2f} us......",
    )
    ret["err_sglang"] = err
    return ret


@benchmark()
def test_grouped_topk(
    token,
    expert,
    group,
    topk,
    topk_group,
    need_renorm,
    dtype,
    scale_factor=1.0,
    scoring_func="softmax",
):
    gating_output = torch.randn((token, expert), dtype=dtype)

    (w_ref, id_ref), us_ref = run_perftest(
        aiter.grouped_topk_torch,
        gating_output,
        topk,
        need_renorm,
        group,
        topk_group,
        scoring_func,
        num_iters=2,
        num_warmup=1,
    )
    w_ref = w_ref * scale_factor
    w_aiter = torch.empty_strided((token, topk), (topk + 10, 1), dtype=dtypes.fp32)
    id_aiter = torch.empty_strided((token, topk), (topk + 10, 1), dtype=dtypes.i32)
    is_softmax = True if scoring_func == "softmax" else False
    _, us_aiter = run_perftest(
        aiter.grouped_topk,
        gating_output,
        w_aiter,
        id_aiter,
        group,
        topk_group,
        need_renorm,
        is_softmax,
        scale_factor,
    )
    id_ref, _ref = torch.sort(id_ref)
    id_aiter, _aiter = torch.sort(id_aiter)
    err = checkAllclose(
        w_ref.gather(1, _ref),
        w_aiter.gather(1, _aiter),
        msg="topk_weights [golden vs aiter]",
    )
    checkAllclose(
        id_ref,
        id_aiter,
        msg=f"topk_ids     [golden vs aiter]:{us_ref:>8.2f} us vs {us_aiter:>8.2f} us......",
    )

    return {"err": err, "us": us_aiter}


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    choices=[dtypes.d_dtypes["fp32"], dtypes.d_dtypes["bf16"]],
    nargs="*",
    metavar="{fp32, bf16}",
    default=[dtypes.d_dtypes["fp32"], dtypes.d_dtypes["bf16"]],
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-e",
    "--expert",
    type=int,
    nargs="*",
    default=[128, 256],
    help="""Number of experts.
    e.g.: -e 64""",
)
parser.add_argument(
    "-t",
    "--token",
    type=int,
    # choices=l_token,
    nargs="*",
    default=[
        1,
        2,
        5,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        10000,
        16384,
        65536,
        163840,
    ],
    help="""Number of tokens.
    e.g.: -t 64""",
)
parser.add_argument(
    "-k",
    type=int,
    default=8,
    help="""Number of topk.
    e.g.: -k 8""",
)

args = parser.parse_args()

df = []
for dtype in args.dtype:
    for e in args.expert:
        for m in args.token:
            ret = test_topk_softmax(dtype, m, e, args.k)
            df.append(ret)
df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("moeTopkSoftmax summary (markdown):\n%s", df_md)

df = []
for token in args.token:
    # DeepSeek-R1
    topk = 8
    group = 8
    topk_group = 4
    expert = 256
    dtype = dtypes.bf16
    need_renorm = True
    ret = test_biased_grouped_topk(
        token, expert, group, topk, topk_group, need_renorm, dtype
    )
    df.append(ret)
df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("moeTopkSoftmax_biased_grouped_topk summary (markdown):\n%s", df_md)

df = []
for token in args.token:
    for scoring_func in ["softmax", "sigmoid"]:
        # DeepSeek-R1
        topk = 8
        group = 8
        topk_group = 4
        expert = 256
        dtype = dtypes.bf16
        need_renorm = True
        ret = test_grouped_topk(
            token,
            expert,
            group,
            topk,
            topk_group,
            need_renorm,
            dtype,
            scale_factor=1.5,
            scoring_func=scoring_func,
        )
        df.append(ret)
df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("moeTopkSoftmax_grouped_topk summary (markdown):\n%s", df_md)
