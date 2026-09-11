#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Launch one native MXFP8 MoE stage repeatedly for separate ATT/PMC capture."""

import argparse
import csv
import functools
import os

import torch
from tune_mxfp8_8wave import fm, inputs

from aiter.ops.flydsl.mxfp8_moe_8wave import stage1, stage2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--tokens", type=int, default=32768)
    p.add_argument("--inter-dim", type=int, default=384)
    p.add_argument("--stage", type=int, choices=(1, 2), required=True)
    p.add_argument(
        "--run-only-stages",
        action="store_true",
        help="Require AOT cache for the native stage kernels",
    )
    args = p.parse_args()
    with open(args.csv) as f:
        row = next(
            r
            for r in csv.DictReader(f)
            if int(r["token"]) == args.tokens and int(r["inter_dim"]) == args.inter_dim
        )
    (x, w1, w2, weights, ids), kwargs = inputs(row, 42)
    packed, sw, eids, valid, _ = fm.moe_sorting(
        ids,
        weights,
        w1.shape[0],
        x.shape[-1],
        x.dtype,
        block_size=256,
        accumulate=False,
    )
    if args.stage == 1:
        call = functools.partial(
            stage1,
            x,
            w1,
            w2,
            packed,
            eids,
            valid,
            None,
            ids.shape[-1],
            block_m=256,
            kernelName=row["kernelName1"],
            w1_scale=kwargs["w1_scale"],
        )
    else:
        rows = (packed.numel() + 255) // 256 * 256
        kp = (args.inter_dim + 255) // 256 * 256
        aq = (
            torch.randn(rows, kp, device=x.device)
            .to(torch.float8_e4m3fn)
            .view(torch.int8)
        )
        sa = torch.full((rows, kp // 32), 127, device=x.device, dtype=torch.uint8)
        out = torch.empty_like(x)
        call = functools.partial(
            stage2,
            aq,
            w1,
            w2,
            packed,
            eids,
            valid,
            out,
            ids.shape[-1],
            block_m=256,
            kernelName=row["kernelName2"],
            w2_scale=kwargs["w2_scale"],
            a2_scale=sa,
            sorted_weights=sw,
        )
    if args.run_only_stages:
        os.environ["FLYDSL_RUNTIME_RUN_ONLY"] = "1"
    for _ in range(5):
        call()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
