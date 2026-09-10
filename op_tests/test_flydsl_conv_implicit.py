#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness for the FlyDSL implicit-GEMM convolution.

The entry point dispatches 1D/2D/3D off the filter rank, so every rank is
covered here along with the keyword surface (stride, padding incl. "same",
dilation, groups, bias, layouts, split-K).

Usage::

    python op_tests/test_flydsl_conv_implicit.py
"""

import argparse

import torch
import torch.nn.functional as F

from aiter import dtypes
from aiter.ops.flydsl import flydsl_conv_implicit
from aiter.test_common import benchmark, checkAllclose, run_perftest

TOL = {"rtol": 2e-2, "atol": 2e-2}


def _ref(x, w, bias, rank, **kw):
    fn = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[rank]
    out = fn(x.float(), w.float(), None if bias is None else bias.float(), **kw)
    return out.to(x.dtype)


@benchmark()
def test_conv_implicit(case, rank, xshape, wshape, dtype, kw, ref_kw=None, bias=False):
    torch.manual_seed(0)
    x = torch.randn(xshape, device="cuda", dtype=dtype)
    w = torch.randn(wshape, device="cuda", dtype=dtype)
    b = torch.randn(wshape[0], device="cuda", dtype=dtype) if bias else None

    ref = _ref(x, w, b, rank, **(kw if ref_kw is None else ref_kw))
    out, us = run_perftest(flydsl_conv_implicit, x, w, b, **kw, num_rotate_args=1)
    err = checkAllclose(ref, out, msg=f"{case}: ", **TOL)
    return {"case": case, "dtype": str(dtype), "us": us, "err": err}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--dtype", default="bf16", choices=["bf16", "fp16"])
    args = p.parse_args()
    dtype = dtypes.bf16 if args.dtype == "bf16" else dtypes.fp16

    # 3D
    x3, w3 = (1, 32, 4, 16, 16), (48, 32, 3, 3, 3)
    test_conv_implicit("3d_3x3x3_pad1", 3, x3, w3, dtype, {"padding": 1})
    test_conv_implicit("3d_bias", 3, x3, w3, dtype, {"padding": 1}, bias=True)
    test_conv_implicit("3d_stride2", 3, x3, w3, dtype, {"stride": 2, "padding": 1})
    test_conv_implicit("3d_dilation2", 3, x3, w3, dtype, {"padding": 2, "dilation": 2})
    test_conv_implicit("3d_same", 3, x3, w3, dtype, {"padding": "same"})
    test_conv_implicit(
        "3d_groups4",
        3,
        (1, 32, 4, 16, 16),
        (48, 8, 3, 3, 3),
        dtype,
        {"padding": 1, "groups": 4},
    )

    # 2D -- the Qwen-Image VAE shape family
    x2, w2 = (1, 96, 64, 64), (96, 96, 3, 3)
    test_conv_implicit("2d_3x3_pad1", 2, x2, w2, dtype, {"padding": 1})
    test_conv_implicit("2d_1x1", 2, x2, (96, 96, 1, 1), dtype, {})
    test_conv_implicit(
        "2d_splitk2",
        2,
        x2,
        w2,
        dtype,
        {"padding": 1, "splitk": 2},
        ref_kw={"padding": 1},
    )

    # 1D
    test_conv_implicit("1d_3_pad1", 1, (1, 32, 128), (64, 32, 3), dtype, {"padding": 1})


if __name__ == "__main__":
    main()
