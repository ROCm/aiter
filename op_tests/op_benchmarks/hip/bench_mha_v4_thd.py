#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Device time and TFLOPS of mha_v4 over THD operands (whole documents delimited
by cu_seqlens), hd256 BF16 on gfx950, causal and non-causal.

The time is the call's kernels, the LSE merge of a split included. Correctness
is op_tests/test_mha_v4.py's. A pack is its documents' lengths:

    python bench_mha_v4_thd.py --heads 24,4 6,1 --packs 4096 300,4096
    python bench_mha_v4_thd.py --mask causal --kv-splits 1 auto
"""

import argparse
import itertools

import pandas as pd
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.mha_v4 import AttentionFormat, mha_v4
from aiter.test_common import run_perftest

HEAD_DIM = 256
BF16 = (AttentionFormat.BF16,) * 3


def _pairs(text):
    return tuple(int(part) for part in text.split(","))


def _kv_splits(text):
    return text if text == "auto" else int(text)


def time_pack(heads_q, heads_kv, seqlens, causal, kv_splits):
    torch.manual_seed(0)
    total = sum(seqlens)
    q = torch.randn((total, heads_q, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((total, heads_kv, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    cu_seqlens = torch.tensor(
        [0, *itertools.accumulate(seqlens)], device="cuda", dtype=torch.int32
    )
    out = torch.empty_like(q)
    _, us = run_perftest(
        mha_v4,
        q,
        k,
        v,
        *BF16,
        out=out,
        kv_splits=kv_splits,
        cu_seqlens=cu_seqlens,
        max_seqlen=max(seqlens),
        causal=causal,
        num_iters=10,
        num_warmup=2,
    )
    keys = [(length + 1) / 2 if causal else length for length in seqlens]
    flops = sum(
        4.0 * heads_q * HEAD_DIM * length * walked
        for length, walked in zip(seqlens, keys)
    )
    return {
        "heads": f"{heads_q}/{heads_kv}",
        "docs": len(seqlens),
        "tokens": total,
        "longest": max(seqlens),
        "mask": "causal" if causal else "none",
        "kv_splits": kv_splits,
        "us": us,
        "TFLOPS": flops / us / 1e6,
    }


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--heads",
        type=_pairs,
        nargs="*",
        default=[(24, 4), (12, 2), (6, 1)],
        help="Hq,Hkv pairs",
    )
    parser.add_argument(
        "--packs",
        type=_pairs,
        nargs="*",
        default=[
            (1024,),
            (2048,),
            (4096,),
            (8192,),
            (300, 4096),
            (1, 31, 33, 255, 256, 257, 1000, 3000, 8192),
        ],
        help="packs, each the comma-separated lengths of its documents",
    )
    parser.add_argument("--mask", choices=["causal", "none", "both"], default="both")
    parser.add_argument(
        "--kv-splits",
        type=_kv_splits,
        nargs="*",
        default=["auto"],
        help='KV slices per work-group, each a positive int or "auto"',
    )
    args = parser.parse_args()
    if get_gfx() != "gfx950":
        print(f"skip: mha_v4 has no THD kernel for {get_gfx()}")
        return

    masks = {"causal": [True], "none": [False], "both": [True, False]}[args.mask]
    rows = [
        time_pack(*heads, pack, causal, kv_splits)
        for causal, heads, pack, kv_splits in itertools.product(
            masks, args.heads, args.packs, args.kv_splits
        )
    ]
    print(pd.DataFrame(rows).to_string(index=False, float_format="{:.4g}".format))


if __name__ == "__main__":
    main()
