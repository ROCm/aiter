# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""m3_moe_sort_quant_fwd (MiniMax-M3.1 FHMoE decode: opus sort + moe_buf zero + MXFP4/MXFP8 quant
in one launch) against the production pair it replaces: moe_sorting -> fused_dynamic_mx_quant_moe_sort.

test_sort_quant: byte equality of num_valid_ids, sorted ids / weights / expert ids (valid range),
  payload (real rows), every real route row's 192 scale bytes, and moe_buf zero; both paths timed.
  ``nvalid < n`` = graph pad rows (num_local_tokens), the normal case of padded decode batches.
test_pad_poison: the whole FHMoE call (fused_moe, stage 1 + 2, deterministic reduce mode) with the pad
  rows of x set to NaN / Inf / huge garbage; the real rows must equal, bit for bit, the production
  path on clean pad rows, and the atomic (shipping) mode must keep the real rows finite.
"""

import argparse
import itertools
import os
import sys

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter import fused_moe as fm
from aiter.fused_moe import moe_sorting
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.quant import fused_dynamic_mx_quant_moe_sort
from aiter.test_common import benchmark, run_perftest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_moe_fhmoe_m3 as fh  # the M3.1 FHMoE layer, routing and fused_moe call)

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]
H, E, K, KS, BM, NG = 6144, 128, 4, 5, 32, 192
QDT = {"a4": dtypes.fp4x2, "a8": dtypes.fp8}


def make_inputs(n, nvalid, data, seed=0):
    """x and the FHMoE top-5 (sigmoid + bias top-4 x2 renorm, shared column [128, 1.0]) as ATOM routes it."""
    g = torch.Generator(device="cuda").manual_seed(seed + 31 * n)
    x = torch.randn(n, H, generator=g) * (torch.rand(n, 1, generator=g) * 4 + 0.1)
    x[:, ::97] *= 20  # outliers, so the group scales vary
    logits = torch.randn(n, E, generator=g) * 2
    bias = torch.randn(E, generator=g) * 0.05
    if data == "ties":  # exact ties of the biased key over expert pairs
        logits[:, 1::2], bias[1::2] = logits[:, 0::2], bias[0::2]
    w = torch.empty(n, KS, dtype=dtypes.fp32)
    ids = torch.empty(n, KS, dtype=dtypes.i32)
    w[:, K], ids[:, K] = 1.0, E
    nvr = torch.tensor([nvalid], dtype=dtypes.i32)
    aiter.topk_gating(
        w[:, :K],
        ids[:, :K],
        logits.to(dtypes.bf16),
        bias,
        True,
        2.0,
        score_func="sigmoid",
        num_valid_rows=nvr,
    )
    return x.to(dtypes.bf16), w, ids, nvr


def run_prod(x, w, ids, nvr, act):
    sid, sw, se, nv, buf = moe_sorting(
        ids, w, E + 1, H, dtypes.bf16, block_size=BM, num_local_tokens=nvr
    )[:5]
    xq, xs = fused_dynamic_mx_quant_moe_sort(
        x,
        sorted_ids=sid,
        num_valid_ids=nv,
        token_num=x.shape[0],
        topk=KS,
        block_size=BM,
        quant_dtype=QDT[act],
        num_rows=nvr,
        sorted_weights=sw,
        num_experts_upper_bound=E + 1,
    )
    return sid, sw, se, nv, buf, xq, xs


def run_m3(x, w, ids, nvr, act):
    buf = torch.full((x.shape[0], H), 7.0, dtype=dtypes.bf16)  # must come back zero
    return aiter.m3_moe_sort_quant(x, ids, w, QDT[act], nvr, buf)


def shuffle_idx(r, c):  # aiter::mx_scale_shuffle_idx(NG, r, c)
    return (
        (r // 32 * NG) * 32
        + (c // 8) * 256
        + (c % 4) * 64
        + (r % 16) * 4
        + (c % 8) // 4 * 2
        + (r % 32) // 16
    )


def mismatches(ref, out, T):
    """Fields of out that differ (bytes) from the production ref on everything production defines."""
    u8 = lambda t: t.contiguous().view(torch.uint8)
    bad = []
    if not torch.equal(ref[3], out[3]):
        return [f"num_valid {ref[3].tolist()} vs {out[3].tolist()}"]
    V = int(ref[3][0])
    for name, i, m in (
        ("sorted_ids", 0, V),
        ("sorted_w", 1, V),
        ("sorted_eids", 2, V // BM),
    ):
        if not torch.equal(u8(ref[i][:m]), u8(out[i][:m])):
            bad.append(name)
    if not torch.equal(u8(ref[5][:T]), u8(out[5][:T])):
        bad.append("payload")
    sid = ref[0][:V]
    rows = torch.nonzero(((sid & 0xFFFFFF) < T) & ((sid >> 24) < KS)).flatten()
    idx = shuffle_idx(rows[:, None], torch.arange(NG)[None, :])
    if not torch.equal(u8(ref[6]).view(-1)[idx], u8(out[6]).view(-1)[idx]):
        bad.append("scale")
    if bool((out[4] != 0).any()):
        bad.append("moe_buf not zero")
    return bad


@benchmark()
def test_sort_quant(n, nvalid, act, data):
    x, w, ids, nvr = make_inputs(n, nvalid, data)
    ref = run_prod(x, w, ids, nvr, act)
    nbytes = (
        x.numel() * 2 + n * H * 2 + n * H // (2 if act == "a4" else 1)
    )  # x in, moe_buf zero, payload out
    ret = {"gfx": get_gfx()}
    for name, fn in (("prod", run_prod), ("m3", run_m3)):
        out, us = run_perftest(fn, x, w, ids, nvr, act)
        bad = mismatches(ref, out, nvalid)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = 0.0  # data movement only
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = len(bad)
        if bad:
            aiter.logger.error(
                "m3_moe_sort_quant %s n=%d nvalid=%d %s %s: %s",
                name,
                n,
                nvalid,
                act,
                data,
                bad,
            )
    return ret


def _fh(L, x, w, ids, act, nvr, disabled):
    fm._fused_decode_sort_quant_disabled = (
        disabled  # True = the production sort + QS path
    )
    try:
        return fm.fused_moe(
            x,
            L.w1,
            L.w2,
            w,
            ids,
            activation=aiter.ActivationType.Swiglu,
            quant_type=aiter.QuantType.per_1x32,
            w1_scale=L.w1s,
            w2_scale=L.w2s,
            swiglu_limit=fh.LIMIT,
            gate_mode="separated",
            quant_dtype_a=QDT[act],
            dtype=dtypes.bf16,
            shared_w1=L.sw1,
            shared_w2=L.sw2,
            shared_w1_scale=L.ss1,
            shared_w2_scale=L.ss2,
            shared_expert_id=E,
            num_local_tokens=nvr,
        )
    finally:
        fm._fused_decode_sort_quant_disabled = False


@benchmark()
def test_pad_poison(n, nvalid, act, poison):
    _, L = fh._layers(act)
    x, w, ids, nvr = make_inputs(n, nvalid, "randn", seed=5)
    xp = x.clone()
    xp[nvalid:] = {"nan": float("nan"), "inf": float("inf"), "huge": 3.0e38}[poison]
    with fh.reduce_mode(True):
        base = _fh(L, x, w, ids, act, nvr, True)
        new = _fh(L, xp, w, ids, act, nvr, False)
        base_p = _fh(L, xp, w, ids, act, nvr, True)
    with fh.reduce_mode(False):
        atomic = _fh(L, xp, w, ids, act, nvr, False)
    torch.cuda.synchronize()
    eq = lambda a, b: torch.equal(
        a[:nvalid].view(torch.int16), b[:nvalid].view(torch.int16)
    )
    return {
        "gfx": get_gfx(),
        "real rows == prod (clean pads)": eq(new, base),
        "real rows == prod (same poison)": eq(new, base_p),
        "atomic real rows finite": bool(torch.isfinite(atomic[:nvalid]).all()),
        "m3 err": int(
            not (
                eq(new, base)
                and eq(new, base_p)
                and torch.isfinite(atomic[:nvalid]).all()
            )
        ),
    }


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "m3_moe_sort_quant_fwd unsupported on %s; skipping", get_gfx()
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    parser.add_argument(
        "-n",
        "--tokens",
        type=int,
        nargs="*",
        default=[1, 2, 3, 4, 5, 8, 12, 13, 16, 24, 31, 32, 48, 64, 100, 128, 200, 256],
    )
    parser.add_argument(
        "--pads",
        type=int,
        nargs="*",
        default=[0, 3],
        help="pad rows (n - num_local_tokens)",
    )
    parser.add_argument(
        "-a", "--act", nargs="*", default=["a8", "a4"], choices=["a8", "a4"]
    )
    parser.add_argument("--data", nargs="*", default=["randn", "ties"])
    parser.add_argument("--poison", nargs="*", default=["nan", "inf", "huge"])
    parser.add_argument(
        "--poison_n", type=int, nargs="*", default=[4, 8, 16, 32, 64, 128]
    )
    args = parser.parse_args()
    rows = [
        test_sort_quant(n, max(1, n - p), a, d)
        for n, p, a, d in itertools.product(
            args.tokens, sorted(set(args.pads)), args.act, args.data
        )
        if p == 0 or n > p
    ]
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "m3_moe_sort_quant_fwd vs moe_sorting + QS (markdown):\n%s",
        df.to_markdown(index=False),
    )
    # padded graph batches: spec decode / off-ladder concurrency (5 -> 8, 12 -> 16, 24 -> 32, 96 -> 128 ...)
    pads = [(n, nv) for n in args.poison_n for nv in (n * 5 // 8, n - 1) if 0 < nv < n]
    df = pd.DataFrame(
        [
            test_pad_poison(n, nv, a, p)
            for (n, nv), a, p in itertools.product(pads, args.act, args.poison)
        ]
    )
    aiter.logger.info(
        "m3_moe_sort_quant_fwd pad-poison through FHMoE (markdown):\n%s",
        df.to_markdown(index=False),
    )
    failed = int(sum(r["m3 err"] for r in rows) + df["m3 err"].sum())
    aiter.logger.info(
        "m3_moe_sort_quant_fwd: %s", "PASS" if failed == 0 else f"FAIL ({failed})"
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
