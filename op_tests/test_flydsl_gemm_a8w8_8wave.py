# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness + perf for the FlyDSL 8-wave a8w8 ptpc bpreshuffle GEMM (gfx950).

The model path (``gemm_a8w8_bpreshuffle`` -> tuned ``flydsl_bpreshuffle_8w_*`` row):

    xq  : [M, K]  fp8, per-token fp32 scale [M, 1]
    wq  : [N, K]  fp8 in shuffle_weight (16, 16) layout, per-channel scale [N, 1]
    out : [M, N]  bf16, preallocated

A cross-wave LDS race on the last K tile used to corrupt 16-64 row (or column)
bands of a tile on some calls only, so every kernel is launched ``--repeat``
times on the same inputs and each result is checked.

Run:
    python op_tests/test_flydsl_gemm_a8w8_8wave.py
    python op_tests/test_flydsl_gemm_a8w8_8wave.py -s 12288,6144,7168 -r 50
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_bpreshuffle_flydsl
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import (
    benchmark,
    checkAllclose,
    run_perftest,
)

torch.set_default_device("cuda")

# The 8-wave kernel uses the CDNA4 MFMA_Scale atom: gfx950 only.
SUPPORTED_GFX = ["gfx950"]
SEED = 0

_KERNELS = [
    "flydsl_bpreshuffle_8w_256x256x128_F8_F8_B16_2x4",
    "flydsl_bpreshuffle_8w_256x256x128_F8_F8_B16_2x0",
    "flydsl_bpreshuffle_8w_128x256x128_F8_F8_B16_1x8",
]

# Tuned kimik3 shapes; before the fix (32768, 6400, 7168) failed on ~50% of calls.
_MNK = [
    (32768, 6400, 7168),
    (12288, 6144, 7168),
    (1000, 7168, 3584),
]


def run_torch(xq, wq, x_scale, w_scale, dtype=dtypes.bf16):
    # Reference only: fp32 dequant matmul, cast back. Not timed, not in the table.
    out = (xq.to(dtypes.fp32) @ wq.to(dtypes.fp32).t()) * x_scale * w_scale.t()
    return out.to(dtype)


@benchmark()
def test_gemm_8wave(m, n, k, kernel, repeat):
    torch.manual_seed(SEED)
    x = torch.randn((m, k), dtype=dtypes.bf16)
    w = torch.randn((n, k), dtype=dtypes.bf16)
    xq, x_scale = aiter.pertoken_quant(x, quant_dtype=dtypes.fp8)
    wq, w_scale = aiter.pertoken_quant(w, quant_dtype=dtypes.fp8)
    wq_shuffled = shuffle_weight(wq, layout=(16, 16))
    out = torch.empty((m, n), dtype=dtypes.bf16)
    ref = run_torch(xq, wq, x_scale, w_scale).to(dtypes.fp32)

    def _run():
        return gemm_a8w8_bpreshuffle_flydsl(
            xq, wq_shuffled, x_scale, w_scale, out, {"kernelName": kernel}
        )

    # The race is intermittent: check every launch, not just the last one.
    bad_calls = 0
    for _ in range(repeat):
        out.fill_(float("nan"))
        y = _run().to(dtypes.fp32)
        if (
            checkAllclose(
                ref, y, rtol=2e-2, atol=5e-2, tol_err_ratio=0.0, printLog=False
            )
            > 0
        ):
            bad_calls += 1

    y, us = run_perftest(_run)
    err = checkAllclose(
        ref, y.to(dtypes.fp32), rtol=2e-2, atol=5e-2, msg=f"{kernel}: 8wave gemm"
    )
    flops = 2 * m * n * k
    nbytes = m * k + n * k + m * n * out.element_size()
    return {
        "gfx": get_gfx(),
        "8wave us": us,
        "8wave TFLOPS": flops / us / 1e6,
        "8wave TB/s": nbytes / us / 1e6,
        "8wave err": err,
        "8wave bad calls": bad_calls,
    }


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("flydsl 8wave gemm unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=_MNK,
        help="""Shape of mnk.
        e.g.: -s 32768,6400,7168""",
    )
    parser.add_argument(
        "-k",
        "--kernel",
        type=str,
        nargs="*",
        default=_KERNELS,
        help="""8-wave kernelName(s).
        e.g.: -k flydsl_bpreshuffle_8w_256x256x128_F8_F8_B16_2x4""",
    )
    parser.add_argument(
        "-r",
        "--repeat",
        type=int,
        nargs="*",
        default=[20],
        help="""Launches checked per shape and kernel.
        e.g.: -r 50""",
    )
    args = parser.parse_args()

    rows = [
        test_gemm_8wave(m, n, k, kernel, repeat)
        for (m, n, k), kernel, repeat in itertools.product(
            args.mnk, args.kernel, args.repeat
        )
    ]
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "flydsl 8wave gemm summary (markdown):\n%s", df.to_markdown(index=False)
    )
    failed = df[(df["8wave bad calls"] > 0) | (df["8wave err"] > 0)]
    if not failed.empty:
        raise AssertionError(
            f"flydsl 8wave gemm mismatched the reference:\n{failed.to_string()}"
        )


if __name__ == "__main__":
    main()
