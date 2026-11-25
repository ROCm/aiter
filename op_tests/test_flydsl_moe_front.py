# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import contextlib
import itertools

import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl import moe_front_gfx1250
from aiter.ops.flydsl.moe_front_gfx1250 import (
    flydsl_moe_router_topk,
    router_quant_rows_shapes,
    router_split_k,
)
from aiter.ops.topk import topk_gating
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.tuned_gemm import tgemm

torch.set_default_device("cuda")
E, TOPK = 128, 4
BF16 = str(torch.bfloat16)


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
    split_k = router_split_k(m, E, k, BF16, BF16)
    (ref_w, ref_i), us_ref = run_perftest(run_unfused, x, w, bias, nvr)
    fused = (flydsl_moe_router_topk, x, w, bias, TOPK, split_k)
    (out_w, out_i), us = run_perftest(*fused, num_valid_rows=nvr)
    err = max(
        checkAllclose(ref_w.view(torch.int32), out_w.view(torch.int32), 0, 0),
        checkAllclose(ref_i, out_i, 0, 0, msg="ids"),
    )
    assert err == 0, f"fused router differs from tgemm.mm + topk_gating ({case})"
    return {"split_k": split_k, "unfused us": us_ref, "fused us": us, "err": err}


@contextlib.contextmanager
def split_k8_kernel():
    """Route the call to the split-K kernel the 8-row launch replaces."""
    takes = moe_front_gfx1250._router8_takes
    moe_front_gfx1250._router8_takes = lambda *a: False
    try:
        yield
    finally:
        moe_front_gfx1250._router8_takes = takes


@benchmark()
def test_router8_rows(k):
    """The 8-row launch vs the split-K 8 kernel: weights, ids and MXFP4 rows byte-identical."""
    m = 8
    x = torch.randn(m, k).to(torch.bfloat16)
    w = (torch.randn(E, k) * 0.02).to(torch.bfloat16)
    bias = torch.randn(E) * 0.05
    split_k = router_split_k(m, E, k, BF16, BF16)
    assert split_k == 8 and moe_front_gfx1250._router8_takes(
        m, k, 8, TOPK, True, 1, False
    )
    shapes = router_quant_rows_shapes(m, k)[:2]

    def run():
        q = tuple(torch.zeros(s, dtype=torch.uint8) for s in shapes)
        return run_perftest(
            flydsl_moe_router_topk, x, w, bias, TOPK, split_k, quant_rows=q
        ) + (q,)

    with split_k8_kernel():
        (ref_w, ref_i), us_ref, ref_q = run()
    (out_w, out_i), us, out_q = run()
    assert torch.equal(ref_w.view(torch.int32), out_w.view(torch.int32)), "weights"
    assert torch.equal(ref_i, out_i), "ids"
    assert all(torch.equal(a, b) for a, b in zip(ref_q, out_q)), "MXFP4 rows"
    return {"split-K 8 us": us_ref, "8-row us": us, "err": 0}


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
    df = [
        test_moe_router_topk(m, k, case)
        for m, k, case in itertools.product(args.m, args.k, args.case)
        if router_split_k(m, E, k, BF16, BF16) is not None
    ]
    assert df, "no shape runs on the FlyDSL router GEMM"
    df += [test_router8_rows(k) for k in args.k if k % 256 == 0]
    aiter.logger.info("moe_router_topk:\n%s", pd.DataFrame(df).to_markdown(index=False))


if __name__ == "__main__":
    main()
