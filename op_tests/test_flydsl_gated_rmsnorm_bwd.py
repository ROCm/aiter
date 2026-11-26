# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gated RMSNorm backward for the Kimi-K3 KDA output norm: o [T, H, 128]."""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.gated_rmsnorm_bwd import (
    gated_rmsnorm_bwd,
    gated_rmsnorm_bwd_torch,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]
EPS = 1e-6


def run_torch(o, gp, w, dy):
    return gated_rmsnorm_bwd_torch(o, gp, w, dy, EPS)


@benchmark()
def test_gated_rmsnorm_bwd(T, H, D, dtype):
    o = torch.randn(T, H, D, dtype=dtype)
    gp = torch.randn(T, H, D, dtype=dtype)
    w = (1.0 + 0.1 * torch.randn(D)).to(dtype)
    dy = torch.randn(T, H, D, dtype=dtype)
    ref = run_torch(o, gp, w, dy)

    candidates = {"flydsl": gated_rmsnorm_bwd}
    # read o, gp, dy; write fp32 do, dgp
    nbytes = 3 * o.numel() * o.element_size() + 2 * o.numel() * 4
    flops = 12 * o.numel()
    ret = {}
    for name, fn in candidates.items():
        out, us = run_perftest(fn, o, gp, w, dy, EPS)
        err = 0.0
        for got, exp, what, tol in zip(
            out, ref, ("do", "dg", "dw"), (1e-3, 1e-3, 1e-2)
        ):
            e = checkAllclose(
                exp.to(dtypes.fp32),
                got.to(dtypes.fp32),
                rtol=tol,
                atol=tol,
                msg=f"{name} {what}",
            )
            err = max(err, e)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
        assert err < 0.05, f"{name} mismatch vs torch reference"
    return ret


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d", "--dtype", type=dtypes.str2Dtype, nargs="*", default=[dtypes.bf16]
    )
    parser.add_argument(
        "-t", "--tokens", type=int, nargs="*", default=[1000, 1001, 2048]
    )
    parser.add_argument("--heads", type=int, nargs="*", default=[96, 48])
    parser.add_argument("--head_dim", type=int, nargs="*", default=[128])
    args = parser.parse_args()

    gfx = get_gfx_runtime()
    if gfx not in SUPPORTED_GFX:
        aiter.logger.info("skip: %s not in %s", gfx, SUPPORTED_GFX)
        return
    rows = [
        test_gated_rmsnorm_bwd(T, H, D, dt)
        for dt, T, H, D in itertools.product(
            args.dtype, args.tokens, args.heads, args.head_dim
        )
    ]
    aiter.logger.info(
        "gated_rmsnorm_bwd summary:\n%s", pd.DataFrame(rows).to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
