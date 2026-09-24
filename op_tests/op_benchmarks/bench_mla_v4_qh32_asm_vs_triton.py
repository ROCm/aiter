# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 qh32 MLA v4 decode: ASM vs Triton, one command.

For every (batch, kv) shape, split = --split-total // batch (clamped to >= 1)
unless --split-kv is given. Each shape is timed as:
  * asm    : aiter.mla.mla_decode_fwd_v4_nm with its default dispatch (asm_path
             column: fused for 2 <= split <= 16, else nosplit / 2stage)
  * triton : pa_decode_sparse (total, plus stage1 / reduce split)
ASM shapes/inputs/timing come from op_tests/test_mla_v4_kargpreld.py, Triton
from op_tests/triton_tests/attention/test_mla_v4_triton.py; both use
run_perftest with the same iters / warmup and default arg rotation. ASM runs
fp8 KV (+ bf16 rope), Triton runs bf16 KV.

Usage:
  ENABLE_CK=0 python op_tests/op_benchmarks/bench_mla_v4_qh32_asm_vs_triton.py
  ... -b 16 64 256 -c 136 384 1024 --split-total 256 --iters 500 --repeat 3
  ... --split-kv 4 8          # fixed split(s) for every batch instead
  ... --csv /tmp/cmp.csv      # also dump the table
"""

import argparse
import os
import statistics
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_OP_TESTS = os.path.dirname(_HERE)
# op_benchmarks/triton/ would shadow the real `triton` package.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
sys.path.insert(0, _OP_TESTS)
sys.path.insert(0, os.path.join(_OP_TESTS, "triton_tests", "attention"))

import test_mla_v4_kargpreld as asm_t
import test_mla_v4_triton as tri_t

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.mla import MLA_V4_FUSED_MAX_SPLITS
from aiter.test_common import run_perftest

GQA = 32
D = 512


def _asm_us(batch, kv, split, sink, iters, warmup):
    asm_t._PERF["num_iters"] = iters
    asm_t._PERF["num_warmup"] = warmup
    r = asm_t.test_mla_v4_nm(
        batch=batch,
        kv_seq_lens=kv,
        q_seq_logical=1,
        num_kv_splits=split,
        gqa_ratio=GQA,
        attn_sink=sink,
    )
    return r["v4_nm us"], r["v4_nm err"]


def _triton_us(batch, kv, split, iters, warmup):
    q, ukv, idx, indptr, sink, scale = tri_t._make_inputs(
        batch, GQA, D, kv, batch * kv, variable_len=False
    )
    args = (q, ukv, idx, indptr, sink, scale)
    kw = {
        "has_invalid": False,
        "kv_splits": split,
        "num_iters": iters,
        "num_warmup": warmup,
    }
    _, tot = run_perftest(tri_t.pa_decode_sparse, *args, skip_reduce=False, **kw)
    s1 = tot
    if split > 1:
        _, s1 = run_perftest(tri_t.pa_decode_sparse, *args, skip_reduce=True, **kw)
    return tot, s1


def _asm_path(split):
    if split == 1:
        return "nosplit"
    if os.environ.get("AITER_MLA_V4_FUSED", "1") == "0":
        return "2stage"
    return "fused" if split <= MLA_V4_FUSED_MAX_SPLITS else "2stage"


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    p.add_argument("-b", "--batch", type=int, nargs="+", default=[16, 64, 256])
    p.add_argument("-c", "--kv", type=int, nargs="+", default=[136, 384, 1024])
    p.add_argument(
        "--split-total",
        type=int,
        default=256,
        help="split = split_total // batch (>= 1). Ignored with --split-kv.",
    )
    p.add_argument("--split-kv", type=int, nargs="+", default=None)
    p.add_argument("--attn-sink", type=int, choices=[0, 1], default=1)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument(
        "--repeat", type=int, default=1, help="Runs per case; median is reported."
    )
    p.add_argument("--csv", default=None, help="Also write the table to this csv.")
    args = p.parse_args()

    if get_gfx() != "gfx1250":
        print(f"qh32 v4 asm decode needs gfx1250, got {get_gfx()}")
        return 1

    cases = []
    for b in args.batch:
        splits = args.split_kv or [max(1, args.split_total // b)]
        for kv in args.kv:
            cases += [(b, kv, s) for s in splits]

    med = statistics.median
    rows = []
    for b, kv, s in cases:
        path = _asm_path(s)
        au, tt, t1, errs = [], [], [], []
        for _ in range(args.repeat):
            us, err = _asm_us(b, kv, s, bool(args.attn_sink), args.iters, args.warmup)
            au.append(us)
            errs.append(err)
            tot, s1 = _triton_us(b, kv, s, args.iters, args.warmup)
            tt.append(tot)
            t1.append(s1)
        asm = med(au)
        tri = med(tt)
        tri_s1 = med(t1)
        rows.append(
            {
                "batch": b,
                "kv": kv,
                "split": s,
                "asm_path": path,
                "asm_us": round(asm, 2),
                "asm_err": max(errs),
                "triton_us": round(tri, 2),
                "triton_s1_us": round(tri_s1, 2),
                "triton_s2_us": round(max(0.0, tri - tri_s1), 2),
                "triton/asm": round(tri / asm, 2),
            }
        )
        print(
            f"[done] b={b} kv={kv} split={s} asm({path})={asm:.2f}us "
            f"triton={tri:.2f}us",
            flush=True,
        )

    df = pd.DataFrame(rows)
    note = (
        f"qh32 decode, attn_sink={bool(args.attn_sink)}, iters={args.iters}, "
        f"warmup={args.warmup}, repeat={args.repeat} (median). asm_us = aiter "
        f"default dispatch (asm_path). triton/asm > 1 means asm is faster."
    )
    print("\n" + note + "\n")
    print(df.to_markdown(index=False))
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\ncsv -> {args.csv}")
    return 0


if __name__ == "__main__":
    aiter.logger.setLevel("WARNING")
    sys.exit(main())
