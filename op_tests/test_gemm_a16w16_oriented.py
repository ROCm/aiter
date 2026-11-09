# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
from aiter.ops.triton.gemm.basic.gemm_a16w16_oriented import gemm_a16w16_oriented
from aiter.test_common import benchmark, checkAllclose, run_perftest


def run_torch(x, weight, bias, dtype):
    result = x.float() @ weight.float().t()
    if bias is not None:
        result += bias.float()
    return result.to(dtype)


@benchmark()
def test_gemm(m, n, k, dtype, outdtype, layout, bias):
    torch.manual_seed(2026)
    # Both layouts keep W[N,K]. The strided case models a view, not a copy.
    step = 2 if layout == "strided" else 1
    x = torch.randn((m, k * step), device="cuda", dtype=dtype)[:, ::step]
    weight = torch.randn((n * step, k), device="cuda", dtype=dtype)[::step]
    bias_tensor = torch.randn((n,), device="cuda", dtype=outdtype) if bias else None
    y = torch.empty((m, n), device="cuda", dtype=outdtype)
    ref = run_torch(x, weight, bias_tensor, outdtype)
    candidates = {
        "triton": lambda: gemm_a16w16(
            x, weight, bias=bias_tensor, dtype=outdtype, y=y, backend="triton"
        ),
        "oriented": lambda: gemm_a16w16_oriented(
            x, weight, bias=bias_tensor, dtype=outdtype, out=y
        ),
    }
    flops = 2 * m * n * k
    nbytes = x.numel() * x.element_size() + weight.numel() * weight.element_size()
    nbytes += y.numel() * y.element_size()
    if bias_tensor is not None:
        nbytes += bias_tensor.numel() * bias_tensor.element_size()
    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        y.fill_(float("nan"))
        result, us = run_perftest(fn)
        assert result.data_ptr() == y.data_ptr()
        err = checkAllclose(
            ref.float(),
            result.float(),
            rtol=1e-2,
            atol=1e-2,
            tol_err_ratio=0.05,
            msg=name,
        )
        assert err <= 0.05, (name, err)
        if outdtype == torch.float32:
            rms = ((result - ref).square().mean() / ref.square().mean()).sqrt()
            assert rms.item() < 2e-5
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if get_gfx() != "gfx950":
        aiter.logger.warning("oriented A16W16 requires gfx950; skip %s", get_gfx())
        return
    parser = argparse.ArgumentParser(description="A16W16 orientation comparison")
    parser.add_argument(
        "-d", "--dtype", type=dtypes.str2Dtype, nargs="+", default=[torch.bfloat16]
    )
    parser.add_argument(
        "--outdtype",
        type=dtypes.str2Dtype,
        nargs="+",
        default=[torch.bfloat16, torch.float32],
    )
    # These are benchmark dimensions, not an assumed model/TP configuration.
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="+",
        default=[(64, 896, 7168), (128, 896, 7168), (256, 4480, 7168), (65, 97, 385)],
    )
    parser.add_argument(
        "--layout", choices=["contiguous", "strided"], nargs="+", default=["contiguous"]
    )
    parser.add_argument("--bias", type=int, choices=[0, 1], nargs="+", default=[0])
    args = parser.parse_args()
    rows = []
    for dtype, outdtype, shape, layout, bias in itertools.product(
        args.dtype, args.outdtype, args.mnk, args.layout, args.bias
    ):
        if outdtype not in (dtype, torch.float32):
            continue
        rows.append(test_gemm(*shape, dtype, outdtype, layout, bool(bias)))
    aiter.logger.info(
        "oriented A16W16 summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
