# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Bytes and timing of the MiniMax-M3.1 TP1 ragged-tile MoE GEMM2
(aiter.ops.m3_tp1_prefill.moe_g2_p6).

The reference is aiter's v2 "reduce" GEMM2 writing per-route rows through
``_flydsl_v2_stage2_wrapper(per_slot_out=...)``, on the deployed shape (128
experts, top-4, H 6144, I 3072, sort block 128) and the same inputs: aiter's
moe_sorting of a routing, random MXFP4 ``a2`` codes and e8m0 scales in GEMM1's
output layout (any GEMM1 output is such bytes), random MXFP4 W2 (shuffled for
aiter, ``relayout_w2`` for this kernel). Random FP4 codes make the fp32 sums
order-sensitive (reversing the K order changes about 5e-5 of the route values),
so equal bytes also mean the same accumulation order; ``wide`` spreads the ``a2``
scales over +-24 binades on top, ``same4`` routes every token to experts 0..3,
``half`` leaves experts 64.. idle and ``one`` puts expert 5 in every top-4.
Every route row and its moe_reduction must be equal, byte for byte; outputs start
as NaN.

    python3 op_tests/test_m3_tp1_moe_g2.py
    python3 op_tests/test_m3_tp1_moe_g2.py --quick
"""

import argparse
import sys

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import _flydsl_v2_stage2_wrapper, moe_sorting
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.moe_kernels import _run_moe_reduction
from aiter.ops.m3_tp1_prefill import moe_g2_p6, relayout_w2
from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight
from aiter.test_common import benchmark, run_perftest

torch.set_default_device("cuda")

E, H, I, TOPK, SBM = 128, 6144, 3072, 4, 128
KN2 = "flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x256_reduce_persist_sbm128"


def make_w2(seed):
    """Random MXFP4 W2: aiter's shuffled copy and the relayout_w2 one."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    w2_q = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, generator=g)
    w2_s = torch.randint(118, 127, (E, H, I // 32), dtype=torch.uint8, generator=g)
    w2 = moe_shuffle_weight(
        w2_q.view(dtypes.fp4x2), experts_cnt=E, is_guinterleave=False, gate_up=False
    )
    s2 = moe_shuffle_scale(
        w2_s.view(-1, I // 32), E, is_guinterleave=False, gate_up=False
    )
    b, bs = relayout_w2(w2_q, w2_s)
    return w2.view(torch.uint8), s2, b, bs


def make_inputs(m, seed, routing, scales):
    g = torch.Generator(device="cuda").manual_seed(seed * 7919 + m)
    logits = torch.randn(m, E, generator=g)
    if routing == "skew":
        logits += 1.5 * torch.randn(E, generator=g)
    elif routing == "same4":  # the longest expert runs
        logits[:, :TOPK] += 100.0
    elif routing == "half":
        logits[:, E // 2 :] = -1e9
    elif routing == "one":  # one expert with every token
        logits[:, 5] += 10.0
    tw, tid = torch.topk(torch.softmax(logits, -1), TOPK, dim=-1)
    sorted_ids, sorted_w, sorted_e, nvalid, _ = moe_sorting(
        tid.to(torch.int32), tw.float(), E, H, torch.bfloat16, block_size=SBM
    )
    rows = sorted_ids.numel()
    a2 = torch.randint(0, 256, (rows, I // 2), dtype=torch.uint8, generator=g)
    lo, hi = (118, 131) if scales == "narrow" else (103, 152)
    a2_scale = torch.randint(
        lo, hi, ((rows + 31) // 32 * 32, I // 32), dtype=torch.uint8, generator=g
    )
    return a2, a2_scale, sorted_ids, sorted_w, sorted_e, nvalid


def run_v2(inp, m, out=None):
    a2, a2_scale, sorted_ids, sorted_w, sorted_e, nvalid = inp
    w2, s2, _, _ = W2
    if out is None:
        out = torch.full((m, TOPK, H), float("nan"), dtype=torch.bfloat16)
    _flydsl_v2_stage2_wrapper(
        inter_states=a2,
        w1=None,
        w2=w2,
        sorted_token_ids=sorted_ids,
        sorted_expert_ids=sorted_e,
        num_valid_ids=nvalid,
        out=torch.empty((m, H), dtype=torch.bfloat16),
        topk=TOPK,
        kernelName=KN2,
        model_dim=H,
        inter_dim=I,
        num_experts=E,
        w2_scale=s2,
        a2_scale=a2_scale,
        sorted_weights=sorted_w,
        bias2=None,
        block_m=SBM,
        per_slot_out=out,
    )
    return out


def run_g2(inp, m):
    a2, a2_scale, sorted_ids, sorted_w, sorted_e, nvalid = inp
    _, _, b, bs = W2
    out = torch.full((m, TOPK, H), float("nan"), dtype=torch.bfloat16)
    moe_g2_p6(a2, a2_scale, b, bs, sorted_ids, sorted_w, sorted_e, nvalid, out, m, TOPK)
    return out


def ndiff(a, b):
    return int((a.view(torch.uint16) != b.view(torch.uint16)).sum())


def reduce(rows, m):
    out = torch.empty((m, H), dtype=torch.bfloat16)
    _run_moe_reduction(rows, out, m, TOPK, H)
    return out


@benchmark()
def test_moe_g2(m, seed, routing="random", scales="narrow"):
    inp = make_inputs(m, seed, routing, scales)
    ref = run_v2(inp, m)
    got = run_g2(inp, m)
    again = run_g2(inp, m)
    d = {
        "diff values": ndiff(got, ref),
        "unwritten": int(torch.isnan(got).sum()),
        "rerun diff": ndiff(again, got),
        "reduce diff": ndiff(reduce(got, m), reduce(ref, m)),
    }
    assert int(torch.isnan(ref).sum()) == 0, "the reference left route rows unwritten"
    assert all(v == 0 for v in d.values()), f"m={m} seed={seed} {routing} {scales}: {d}"
    return {**d, "sort blocks": int(inp[5][0]) // SBM}


@benchmark()
def perf_moe_g2(m):
    inp = make_inputs(m, 0, "random", "narrow")
    a2, a2_scale, sorted_ids, sorted_w, sorted_e, nvalid = inp
    _, _, b, bs = W2
    out = torch.empty((m, TOPK, H), dtype=torch.bfloat16)
    _, us = run_perftest(
        moe_g2_p6,
        a2,
        a2_scale,
        b,
        bs,
        sorted_ids,
        sorted_w,
        sorted_e,
        nvalid,
        out,
        m,
        TOPK,
        num_iters=20,
        num_warmup=3,
    )
    _, us_v2 = run_perftest(
        run_v2, inp, m, torch.empty_like(out), num_iters=20, num_warmup=3
    )
    tflop = 2.0 * m * TOPK * H * I / 1e12
    return {
        "us (with tile list)": us,
        "v2 reduce GEMM2 us": us_v2,
        "speedup": us_v2 / us,
        "PF": tflop / us * 1e3,
    }


def expect_error(fn, *args, match):
    try:
        fn(*args)
        torch.cuda.synchronize()
    except RuntimeError as e:
        assert match in str(e), f"unexpected error: {e}"
        return
    raise AssertionError(f"{fn.__name__} accepted invalid arguments ({match})")


def test_arg_checks():
    m = 64
    inp = make_inputs(m, 0, "random", "narrow")
    a2, a2_scale, sorted_ids, sorted_w, sorted_e, nvalid = inp
    _, _, b, bs = W2
    out = torch.empty((m, TOPK, H), dtype=torch.bfloat16)
    big = torch.zeros(4097 * SBM, dtype=torch.int32)
    expect_error(
        moe_g2_p6,
        a2,
        a2_scale,
        b,
        bs,
        big,
        sorted_w,
        sorted_e,
        nvalid,
        out,
        m,
        TOPK,
        match="at most 4096 sort blocks",
    )
    out_h = torch.empty((m, TOPK, 4096), dtype=torch.bfloat16)
    expect_error(
        moe_g2_p6,
        a2,
        a2_scale,
        b,
        bs,
        sorted_ids,
        sorted_w,
        sorted_e,
        nvalid,
        out_h,
        m,
        TOPK,
        match="out must be",
    )
    a2_i = torch.empty((a2.shape[0], 2048), dtype=torch.uint8)
    expect_error(
        moe_g2_p6,
        a2_i,
        a2_scale,
        b,
        bs,
        sorted_ids,
        sorted_w,
        sorted_e,
        nvalid,
        out,
        m,
        TOPK,
        match="a2 must be",
    )
    aiter.logger.info("m3_tp1 moe_g2 argument checks passed")


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Test the MiniMax-M3.1 TP1 ragged-tile MoE GEMM2.",
)
parser.add_argument(
    "--only", nargs="*", choices=["numerics", "args", "perf"], default=None
)
parser.add_argument("--quick", action="store_true", help="fewer and smaller cases")
args = parser.parse_args()

if get_gfx_runtime() != "gfx950":
    aiter.logger.info("m3_tp1 moe_g2 targets gfx950; skipping")
    sys.exit(0)


def wanted(part):
    return args.only is None or part in args.only


W2 = make_w2(0)  # module-level: @benchmark would log a tensor argument
tables = {}
if wanted("args"):
    test_arg_checks()
if wanted("numerics"):
    rows = []
    sizes = [1000, 8192] if args.quick else [1000, 4096, 5000, 8192, 16384, 32768]
    for m in sizes:
        for seed in [0] if args.quick else [0, 1, 2]:
            rows.append(test_moe_g2(m, seed))
    for seed in (0, 1):
        rows.append(test_moe_g2(8192, seed, routing="skew"))
    for m in [8192] if args.quick else [8192, 16384]:
        rows.append(test_moe_g2(m, 0, scales="wide"))
    for m in [1, 129] if args.quick else [1, 2, 17, 37, 129, 255, 513]:
        rows.append(test_moe_g2(m, 0))
    for m in (1, 300, 8192):
        rows.append(test_moe_g2(m, 0, routing="same4"))
    rows.append(test_moe_g2(4096, 0, routing="half"))
    rows.append(test_moe_g2(8192, 0, routing="one"))
    tables["moe_g2_p6 vs v2 reduce GEMM2 (per-route rows)"] = pd.DataFrame(rows)
if wanted("perf"):
    rows = [perf_moe_g2(m) for m in (4096, 8192, 16384, 32768)]
    tables["moe_g2_p6 timing"] = pd.DataFrame(rows)
for title, df in tables.items():
    aiter.logger.info(f"{title}:\n{df.to_markdown(index=False)}")
