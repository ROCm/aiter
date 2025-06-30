# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter.test_common import (
    checkAllclose,
    benchmark,
    run_perftest,
    perftest,
)
from aiter import dtypes
import pandas as pd
import argparse

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)


@perftest()
def test_nofuse(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    M, _ = hidden_states.shape

    topk_weights = torch.empty(M, topk, dtype=dtypes.fp32, device=hidden_states.device)
    topk_ids = torch.empty(M, topk, dtype=dtypes.i32, device=hidden_states.device)
    token_expert_indicies = torch.empty(
        M, topk, dtype=dtypes.i32, device=hidden_states.device
    )

    aiter.topk_softmax(
        topk_weights, topk_ids, token_expert_indicies, gating_output.float(), False
    )
    del token_expert_indicies  # Not used. Will be used in the future.

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids


@perftest()
def test_fuse(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    from aiter.fused_moe import fused_topk

    return fused_topk(hidden_states, gating_output, topk, renormalize)


@benchmark()
def test_topk_softmax(dtype, token, E, topk):
    hidden_states = torch.randn((token, 1), dtype=dtype, device="cuda")
    gating_output = torch.randn((m, E), dtype=dtype, device="cuda")

    (topk_weights_a, topk_ids_a), avg_a = test_nofuse(
        hidden_states, gating_output, topk, True
    )
    (topk_weights_b, topk_ids_b), avg_b = test_fuse(
        hidden_states, gating_output, topk, True
    )
    err = checkAllclose(topk_weights_a, topk_weights_b, atol=0.03)
    checkAllclose(topk_ids_a, topk_ids_b, atol=0, msg="topk_ids")
    return {"err": err, "us": avg_b}


@aiter.test_common.benchmark()
def test_biased_grouped_topk(
    token, expert, group, topk, topk_group, need_renorm, dtype, scale_factor=1.0
):
    gating_output = torch.randn((token, expert), dtype=dtype)
    correction_bias = torch.randn((expert,), dtype=dtype)

    (w_ref, id_ref), us_ref = run_perftest(
        aiter.biased_grouped_topk_torch,
        gating_output,
        correction_bias,
        topk,
        need_renorm,
        group,
        topk_group,
        num_iters=2,
        num_warmup=1,
    )
    w_ref = w_ref * scale_factor
    w_aiter = torch.empty_strided((token, topk), (topk + 10, 1), dtype=dtypes.fp32)
    id_aiter = torch.empty_strided((token, topk), (topk + 10, 1), dtype=dtypes.i32)
    _, us_aiter = run_perftest(
        aiter.biased_grouped_topk,
        gating_output,
        correction_bias,
        w_aiter,
        id_aiter,
        group,
        topk_group,
        need_renorm,
        scale_factor,
    )
    id_ref, _ref = torch.sort(id_ref)
    id_aiter, _aiter = torch.sort(id_aiter)
    w_ref = w_ref.gather(1, _ref)
    w_aiter = w_aiter.gather(1, _aiter)
    # print(f'  {id_ref=}')
    # print(f'{id_aiter=}')
    # print(f'  {w_ref=}')
    # print(f'{w_aiter=}')
    err = checkAllclose(w_ref, w_aiter, msg="topk_weights [golden vs aiter]")
    checkAllclose(
        id_ref,
        id_aiter,
        msg=f"topk_ids     [golden vs aiter]:{us_ref:>8.2f} us vs {us_aiter:>8.2f} us......",
    )
    return {"err": err, "us": us_aiter}


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
    _, us_aiter = run_perftest(
        aiter.grouped_topk,
        gating_output,
        w_aiter,
        id_aiter,
        group,
        topk_group,
        need_renorm,
        scoring_func,
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


l_dtype = ["bf16", "fp16"]
l_expert = [64, 256]
l_m = [1, 8, 16, 32, 64, 128, 256, 65536, 163840]
l_token = [1, 2, 5, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 10000, 16384]

parser = argparse.ArgumentParser(description="config input of test")
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=l_dtype,
    nargs="?",
    const=None,
    default=None,
    help="data type",
)
parser.add_argument(
    "-e",
    "--expert",
    type=int,
    choices=l_expert,
    nargs="?",
    const=None,
    default=None,
    help="number of experts",
)
parser.add_argument(
    "-m",
    type=int,
    default=None,
)
parser.add_argument(
    "-t",
    "--token",
    type=int,
    choices=l_token,
    nargs="?",
    const=None,
    default=None,
    help="number of tokens",
)

args = parser.parse_args()
if args.dtype is None:
    l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
else:
    l_dtype = [dtypes.d_dtypes[args.dtype]]
if args.expert is not None:
    l_expert = [args.expert]
if args.m is not None:
    l_m = [args.m]
if args.token is not None:
    l_token = [args.token]

df = []
for dtype in l_dtype:
    for e in l_expert:
        for m in l_m:
            ret = test_topk_softmax(dtype, m, e, 5)
            df.append(ret)
df = pd.DataFrame(df)
aiter.logger.info(f"summary:\n{df}")

df = []
for token in l_token:
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
aiter.logger.info(f"summary:\n{df}")

df = []
for token in l_token:
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
aiter.logger.info(f"summary:\n{df}")
