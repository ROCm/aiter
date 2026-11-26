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
)
from aiter.ops.topk import topk_gating
from aiter.test_common import benchmark, run_perftest
from aiter.tuned_gemm import tgemm

torch.set_default_device("cuda")
E = 128
# Selection values (score + bias) closer than this may rank either way: the
# kernels round the logits to bf16 (0.5 ulp moves a sigmoid score <= 9e-4).
NEAR_TIE = 2e-3


def make_inputs(m, n, k, case):
    x = torch.randn(m, k).to(torch.bfloat16)
    w = (torch.randn(n, k) * 0.02).to(torch.bfloat16)
    bias = torch.randn(n) * 0.05
    nvr = torch.tensor([m - 2], dtype=torch.int32) if case == "pad_rows" else None
    if case == "ties":  # equal experts across lanes and inside one lane
        for a, b in ((5, 77), (3, 35), (64, 96), (0, 127)):
            w[b], bias[b] = w[a], bias[a]
    elif case == "saturate":  # sigmoid == 1.0 on many experts
        x, bias = (x.float() * 60).to(torch.bfloat16), bias.zero_()
    return x, w, bias, nvr


def run_torch(x, w, topk, bias, nvr):
    """fp32 logits, biased sigmoid top-k, renormalized; and per row the
    smallest gap between neighbors of the top-(k+1) selection values."""
    scores = torch.sigmoid(x.float() @ w.float().t())
    top = (scores + bias).topk(topk + 1, dim=1)
    ids = top.indices[:, :topk]
    weights = scores.gather(1, ids)
    weights = weights / weights.sum(1, keepdim=True)
    gap = (top.values[:, :-1] - top.values[:, 1:]).min(1).values
    if nvr is not None:  # padding rows: weight 0, id -1
        pad = torch.arange(x.shape[0]) >= nvr
        weights[pad], ids[pad], gap[pad] = 0, -1, float("inf")
    return weights, ids.int(), gap


def check_routes(ref, out_w, out_i):
    """Rows that differ from ``run_torch``: other ids outside a near tie, or
    the same ids with other weights."""
    ref_w, ref_i, gap = ref
    same = (ref_i == out_i).all(1)
    close = torch.isclose(ref_w, out_w, rtol=1e-2, atol=1e-3).all(1)
    return int(((~same & (gap >= NEAR_TIE)) | (same & ~close)).sum())


def run_unfused(x, w, topk, bias, nvr):
    tw = torch.empty(x.shape[0], topk, dtype=torch.float32)
    ti = torch.empty(x.shape[0], topk, dtype=torch.int32)
    logits = tgemm.mm(x, w, None, otype=torch.bfloat16)
    topk_gating(
        tw, ti, logits, bias, True, 1.0, score_func="sigmoid", num_valid_rows=nvr
    )
    return tw, ti


@benchmark()
def test_moe_router_topk(m, k, topk, split_k, case):
    x, w, bias, nvr = make_inputs(m, E, k, case)
    ref = run_torch(x, w, topk, bias, nvr)
    candidates = {
        "unfused": (run_unfused, x, w, topk, bias, nvr),
        "fused": (flydsl_moe_router_topk, x, w, topk, bias),
    }
    kwargs = {"fused": {"num_valid_rows": nvr, "split_k": split_k}}
    ret = {}
    for name, (fn, *args) in candidates.items():
        (out_w, out_i), us = run_perftest(fn, *args, **kwargs.get(name, {}))
        ret[f"{name} us"], ret[f"{name} err"] = us, check_routes(ref, out_w, out_i)
    ret["near-tie rows"] = int((ref[2] < NEAR_TIE).sum())
    assert ret["fused err"] == ret["unfused err"] == 0, f"routes differ: {ret}"
    return ret


def check_supported():
    """Every tuned row of this GPU: supported where it is fused, not where it is
    not. Unimplemented arguments are unsupported, and the op raises on them."""
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
    ap.add_argument("-m", type=int, nargs="*", default=[1, 33, 256])
    ap.add_argument("-k", type=int, nargs="*", default=[6144])
    ap.add_argument("--topk", type=int, nargs="*", default=[1, 4, 32])
    ap.add_argument("--split_k", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    ap.add_argument(
        "--case", nargs="*", default=["rand", "ties", "saturate", "pad_rows"]
    )
    args = ap.parse_args()
    check_supported()
    df = [
        test_moe_router_topk(*shape)
        for shape in itertools.product(
            args.m, args.k, args.topk, args.split_k, args.case
        )
    ]
    aiter.logger.info("moe_router_topk:\n%s", pd.DataFrame(df).to_markdown(index=False))


if __name__ == "__main__":
    main()
