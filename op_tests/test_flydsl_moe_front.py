# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.flydsl.moe_front_gfx1250 import (
    flydsl_moe_router_topk,
    router_gemm_split_k,
    router_topk_config,
)
from aiter.ops.topk import topk_gating
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.tuned_gemm import tgemm

torch.set_default_device("cuda")
E, TOPK = 128, 4


def run_unfused(x, w, bias, nvr):
    tw = torch.empty(x.shape[0], TOPK, dtype=torch.float32)
    ti = torch.empty(x.shape[0], TOPK, dtype=torch.int32)
    logits = tgemm.mm(x, w, None, otype=torch.bfloat16)
    topk_gating(
        tw, ti, logits, bias, True, 1.0, score_func="sigmoid", num_valid_rows=nvr
    )
    return tw, ti


@benchmark()
def test_moe_router_topk(m, k, case):
    x = torch.randn(m, k).to(torch.bfloat16)
    w = (torch.randn(E, k) * 0.02).to(torch.bfloat16)
    bias = torch.randn(E) * 0.05
    nvr = torch.tensor([m - 2], dtype=torch.int32) if case == "pad_rows" else None
    if case == "ties":  # equal experts across lanes and inside one lane
        for a, b in ((5, 77), (3, 35), (64, 96), (0, 127)):
            w[b], bias[b] = w[a], bias[a]
    elif case == "saturate":  # sigmoid == 1.0 on many experts
        x, bias = (x.float() * 60).to(torch.bfloat16), bias.zero_()
    split_k = router_gemm_split_k(m, E, k)
    (ref_w, ref_i), us_ref = run_perftest(run_unfused, x, w, bias, nvr)
    fused = (flydsl_moe_router_topk, x, w, bias, TOPK, split_k)
    (out_w, out_i), us = run_perftest(*fused, num_valid_rows=nvr)
    err = max(
        checkAllclose(ref_w.view(torch.int32), out_w.view(torch.int32), 0, 0),
        checkAllclose(ref_i, out_i, 0, 0, msg="ids"),
    )
    assert err == 0, f"fused router differs from tgemm.mm + topk_gating ({case})"
    return {"split_k": split_k, "unfused us": us_ref, "fused us": us, "err": err}


def check_tuned_rows():
    """Each row of this GPU resolves: a fused row to its config (a router GEMM
    retune can kill one), an unfused row to None (tgemm.mm + topk_gating)."""
    df = pd.read_csv(AITER_CONFIGS.AITER_CONFIG_MOE_ROUTER_TOPK_FILE)
    df = df[(df.gfx == get_gfx_runtime()) & (df.cu_num == get_cu_num())]
    for r in df.itertuples():
        cfg = router_topk_config(r.M, r.N, r.K, r.topk)
        want = {"split_k": r.split_k, "prefetch": r.prefetch} if r.fused else None
        assert cfg == want, f"row {r} resolves to {cfg}"


def main():
    if get_gfx_runtime() != "gfx1250":
        aiter.logger.warning("flydsl_moe_router_topk is gfx1250-only; skipping")
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", type=int, nargs="*", default=[1, 16, 33, 256])
    ap.add_argument("-k", type=int, nargs="*", default=[6144])
    ap.add_argument(
        "--case", nargs="*", default=["rand", "ties", "saturate", "pad_rows"]
    )
    args = ap.parse_args()
    check_tuned_rows()
    df = [
        test_moe_router_topk(m, k, case)
        for m, k, case in itertools.product(args.m, args.k, args.case)
        if router_gemm_split_k(m, E, k) is not None
    ]
    assert df, "no shape runs on the FlyDSL router GEMM"
    aiter.logger.info("moe_router_topk:\n%s", pd.DataFrame(df).to_markdown(index=False))


if __name__ == "__main__":
    main()
