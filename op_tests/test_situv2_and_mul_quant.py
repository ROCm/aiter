# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.activation import situv2_and_mul_quant
from aiter.test_common import (
    benchmark,
    checkAllclose,
    run_perftest,
)

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950", "gfx1250"]

# Kimi-K3's values. Neither drives a dispatch branch, so one pair covers them.
BETA = 4.0
LINEAR_BETA = 25.0


def run_torch(x, beta=BETA, linear_beta=LINEAR_BETA):
    """Reference activation in fp32; not timed, not in the table."""
    d = x.shape[-1] // 2
    gate = x[..., :d].float()
    up = x[..., d:].float()
    return (
        beta
        * torch.tanh(gate / beta)
        * torch.sigmoid(gate)
        * (linear_beta * torch.tanh(up / linear_beta))
    )


@benchmark()
def test_situv2_and_mul_quant(m, d, dtype):
    x = torch.randn((m, 2 * d), dtype=dtype)
    ref = run_torch(x)
    ref_scale = ref.abs().amax(dim=-1, keepdim=True) / torch.finfo(dtypes.fp8).max

    # The model preallocates both outputs, so the test does too.
    out = torch.empty((m, d), dtype=dtypes.fp8)
    scale = torch.empty((m, 1), dtype=dtypes.fp32)

    _, us = run_perftest(situv2_and_mul_quant, out, x, scale, d, BETA, LINEAR_BETA)

    err = checkAllclose(
        ref,
        out.to(dtypes.fp32) * scale,
        rtol=8e-2,
        atol=3e-1,
        msg="situv2_and_mul_quant out",
    )
    # The scale is an amax over activations the kernel and torch compute to
    # slightly different last bits, so it is tight but not bit-exact.
    checkAllclose(
        ref_scale,
        scale,
        rtol=2e-5,
        atol=3e-8,
        msg="situv2_and_mul_quant scale",
    )

    # Reads 2*d bf16 per token, writes d fp8 plus one fp32 scale. Arithmetic is
    # ~11 ops per output element (three transcendentals counted as one each,
    # plus the quant multiply), which only serves to place the op on the
    # roofline -- it is memory bound at every shape here.
    nbytes = m * (2 * d * x.element_size() + d * out.element_size() + 4)
    flops = 11 * m * d
    return {
        "gfx": get_gfx(),
        "us": us,
        "TFLOPS": flops / us / 1e6,
        "TB/s": nbytes / us / 1e6,
        "err": err,
    }


def run_edge_shapes():
    """Two shapes the sweep cannot carry: an all-zero row, whose zero scale the
    randn inputs never produce, and an empty batch, which would only add a NaN
    row to the table. Correctness only, so both stay out of it."""
    d = 768
    out = torch.empty((4, d), dtype=dtypes.fp8)
    scale = torch.empty((4, 1), dtype=dtypes.fp32)
    situv2_and_mul_quant(
        out, torch.zeros((4, 2 * d), dtype=dtypes.bf16), scale, d, BETA, LINEAR_BETA
    )
    assert torch.count_nonzero(out.float()) == 0, "zero row: quantized output"
    assert torch.count_nonzero(scale) == 0, "zero row: scale"

    situv2_and_mul_quant(
        torch.empty((0, d), dtype=dtypes.fp8),
        torch.empty((0, 2 * d), dtype=dtypes.bf16),
        torch.empty((0, 1), dtype=dtypes.fp32),
        d,
        BETA,
        LINEAR_BETA,
    )
    aiter.logger.info("situv2_and_mul_quant edge shapes passed")


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "situv2_and_mul_quant unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default="bf16,",
        help="""Data type of the activation input.
    e.g.: -d bf16""",
    )
    parser.add_argument(
        "-m",
        "--m",
        type=int,
        nargs="*",
        default=[1, 32, 129, 1024, 8192, 32768],
        help="""Number of tokens.
    e.g.: -m 32""",
    )
    parser.add_argument(
        "-n",
        "--dim",
        dest="d",
        type=int,
        nargs="*",
        # 384 takes the single-wave path, 768 the compile-time specialization
        # Kimi-K3's shared experts run, 1024 the VecSize 16 path, and 4224 the
        # runtime-d dense MLP.
        default=[384, 768, 1024, 4224],
        help="""Hidden size per gate/up half.
    e.g.: -n 768""",
    )
    args = parser.parse_args()

    run_edge_shapes()

    for dtype in args.dtype:
        df = [
            test_situv2_and_mul_quant(m, d, dtype)
            for d, m in itertools.product(args.d, args.m)
        ]
        df = pd.DataFrame(df)
        aiter.logger.info(
            "situv2_and_mul_quant summary (markdown):\n%s", df.to_markdown(index=False)
        )


if __name__ == "__main__":
    main()
