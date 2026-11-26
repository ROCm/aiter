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
    flydsl_moe_router_topk_supported,
    router_gemm_split_k,
)
from aiter.ops.topk import topk_gating
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.tuned_gemm import tgemm

torch.set_default_device("cuda")
E = 128


def run_unfused(x, w, topk, bias, nvr):
    tw = torch.empty(x.shape[0], topk, dtype=torch.float32)
    ti = torch.empty(x.shape[0], topk, dtype=torch.int32)
    logits = tgemm.mm(x, w, None, otype=torch.bfloat16)
    topk_gating(
        tw, ti, logits, bias, True, 1.0, score_func="sigmoid", num_valid_rows=nvr
    )
    return tw, ti


@benchmark()
def test_moe_router_topk(m, k, topk, case):
    x = torch.randn(m, k).to(torch.bfloat16)
    w = (torch.randn(E, k) * 0.02).to(torch.bfloat16)
    bias = torch.randn(E) * 0.05
    nvr = torch.tensor([m - 2], dtype=torch.int32) if case == "pad_rows" else None
    if case == "ties":  # equal experts across lanes and inside one lane
        for a, b in ((5, 77), (3, 35), (64, 96), (0, 127)):
            w[b], bias[b] = w[a], bias[a]
    elif case == "saturate":  # sigmoid == 1.0 on many experts
        x, bias = (x.float() * 60).to(torch.bfloat16), bias.zero_()
    # The router GEMM's own split_k: the fused launch then gives its bytes.
    split_k = router_gemm_split_k(m, E, k)
    (ref_w, ref_i), us_ref = run_perftest(run_unfused, x, w, topk, bias, nvr)
    (out_w, out_i), us = run_perftest(
        flydsl_moe_router_topk, x, w, topk, bias, num_valid_rows=nvr, split_k=split_k
    )
    err = max(
        checkAllclose(ref_w.view(torch.int32), out_w.view(torch.int32), 0, 0),
        checkAllclose(ref_i, out_i, 0, 0, msg="ids"),
    )
    assert err == 0, f"fused router differs from tgemm.mm + topk_gating ({case})"
    return {"split_k": split_k, "unfused us": us_ref, "fused us": us, "err": err}


def check_supported():
    """Every tuned row of this GPU: supported where it is fused (a router GEMM
    retune can kill one), not where it is not. Unimplemented arguments are
    unsupported, and the op raises on them."""
    df = pd.read_csv(AITER_CONFIGS.AITER_CONFIG_MOE_ROUTER_TOPK_FILE)
    df = df[(df.gfx == get_gfx_runtime()) & (df.cu_num == get_cu_num())]
    for r in df.itertuples():
        x = torch.empty(r.M, r.K, dtype=torch.bfloat16)
        w = torch.empty(r.N, r.K, dtype=torch.bfloat16)
        ok = flydsl_moe_router_topk_supported(x, w, r.topk, torch.empty(r.N))
        assert ok == bool(r.fused), f"row {r}: supported {ok}"
    x = torch.empty(16, 6144, dtype=torch.bfloat16)
    w = torch.empty(E, 6144, dtype=torch.bfloat16)
    for args in ((x, w, 4, None, True, 1.0, "softmax"), (x, w[:96], 4)):
        assert not flydsl_moe_router_topk_supported(*args)
        try:
            flydsl_moe_router_topk(*args)
        except ValueError:
            continue
        raise AssertionError(f"no ValueError for {args[2:]}")


def main():
    if get_gfx_runtime() != "gfx1250":
        aiter.logger.warning("flydsl_moe_router_topk is gfx1250-only; skipping")
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", type=int, nargs="*", default=[1, 16, 33, 256])
    ap.add_argument("-k", type=int, nargs="*", default=[6144])
    ap.add_argument("--topk", type=int, nargs="*", default=[1, 4, 8, 32])
    ap.add_argument(
        "--case", nargs="*", default=["rand", "ties", "saturate", "pad_rows"]
    )
    args = ap.parse_args()
    check_supported()
    df = [
        test_moe_router_topk(m, k, topk, case)
        for m, k, topk, case in itertools.product(args.m, args.k, args.topk, args.case)
        if router_gemm_split_k(m, E, k) is not None
    ]
    assert df, "no shape runs on the FlyDSL router GEMM"
    aiter.logger.info("moe_router_topk:\n%s", pd.DataFrame(df).to_markdown(index=False))


if __name__ == "__main__":
    main()
