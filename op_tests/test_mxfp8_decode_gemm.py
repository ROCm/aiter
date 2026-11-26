# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.mxfp8_decode_gemm_gfx1250 import (
    mxfp8_decode_gemm,
    mxfp8_decode_gemm_supported,
)
from aiter.ops.shuffle import shuffle_mxfp8_bpreshuffle_scale, shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")


def e8m0(shape, lo, hi):
    return torch.randint(lo, hi, shape, dtype=torch.int32).to(torch.uint8)


def run_torch(codes, scales):  # fp64 dequantized operand
    s = torch.exp2(scales.double() - 127.0)
    return (codes.double().view(codes.shape[0], -1, 32) * s[..., None]).flatten(1)


@benchmark()
def test_mxfp8_decode_gemm(m, n, k):
    x = torch.randn(m, k).clamp(-8, 8).to(torch.float8_e4m3fn)
    w = (torch.randn(n, k) * 0.5).clamp(-8, 8).to(torch.float8_e4m3fn)
    xs, ws = e8m0((m, k // 32), 120, 127), e8m0((n, k // 32), 118, 126)
    ref = run_torch(x, xs) @ run_torch(w, ws).T
    xs, ws = xs.view(torch.float8_e8m0fnu), ws.view(torch.float8_e8m0fnu)
    w_sh = shuffle_weight(w, layout=(16, 16))
    ws_sh = shuffle_mxfp8_bpreshuffle_scale(ws)
    gemm = aiter.gemm_a8w8_blockscale
    prod, prod_us = run_perftest(gemm, x, w_sh, xs, ws_sh, isBpreshuffled=True)
    out = torch.full((m + 7, n), -3.0, dtype=torch.bfloat16)
    y, us = run_perftest(mxfp8_decode_gemm, x, xs, w_sh, ws_sh, out=out[:m])
    assert torch.all(out[m:] == -3.0), "rows past M written"
    err, err_prod = ((t.double() - ref).abs().max().item() for t in (y, prod))
    assert err <= 2 * err_prod + 1e-6, (err, err_prod)
    flips = checkAllclose(prod.float(), y.float(), rtol=0, atol=0, tol_err_ratio=5e-3)
    assert flips <= 5e-3, f"{flips:.2%} of bf16 outputs differ from the tuned GEMM"
    return {"tuned us": prod_us, "decode us": us, "max err": err, "flips": flips}


def main():
    if get_gfx_runtime() != "gfx1250":
        aiter.logger.warning("mxfp8_decode_gemm is gfx1250-only; skipping")
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", type=int, nargs="*", default=[1, 8, 16, 24, 32])
    ap.add_argument(
        "-nk", type=dtypes.str2tuple, nargs="*", default=[(6144, 2048), (2304, 6144)]
    )
    args = ap.parse_args()
    shapes = [
        (m, n, k)
        for n, k in args.nk
        for m in args.m
        if mxfp8_decode_gemm_supported(m, n, k)
    ]
    assert shapes, "no shape is served by mxfp8_decode_gemm"
    df = pd.DataFrame([test_mxfp8_decode_gemm(*mnk) for mnk in shapes])
    aiter.logger.info("mxfp8_decode_gemm:\n%s", df.to_markdown(index=False))


if __name__ == "__main__":
    main()
