#!/usr/bin/env python3
"""Repeat one production MXFP8 GEMM for rocprofv3 counter collection."""

import argparse

import torch

import aiter
from aiter import dtypes
from aiter.ops.shuffle import shuffle_mxfp8fp4_scale, shuffle_weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--loops", type=int, default=100)
    args = parser.parse_args()
    m, n, k = args.m, args.n, args.k
    torch.manual_seed(131000 + m + n + k)
    x = torch.randint(0, 255, (m, k), dtype=torch.uint8, device="cuda").view(dtypes.fp8)
    weight = torch.randint(
        0, 255, (n, k), dtype=torch.uint8, device="cuda"
    ).view(dtypes.fp8)
    xs = torch.randint(
        124, 131, (m, k // 32), dtype=torch.uint8, device="cuda"
    ).view(dtypes.fp8_e8m0)
    ws = torch.randint(
        124, 131, (n, k // 32), dtype=torch.uint8, device="cuda"
    ).view(dtypes.fp8_e8m0)
    weight = shuffle_weight(weight, layout=(16, 16))
    ws = shuffle_mxfp8fp4_scale(ws)
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")

    def run():
        return aiter.gemm_a8w8_blockscale_bpreshuffle(
            x, weight, xs, ws, dtypes.bf16, out=out
        )

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    for _ in range(args.loops):
        run()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
