# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import functools
import itertools

import pandas as pd
import torch

from aiter import logger
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
from aiter.ops.triton.gemm.basic.gemm_a16w16_oriented import gemm_a16w16_oriented
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)


def generate_inputs(m, n, k):
    return (
        torch.randn((m, k), device="cuda", dtype=torch.bfloat16),
        torch.randn((n, k), device="cuda", dtype=torch.bfloat16),
        torch.empty((m, n), device="cuda", dtype=torch.bfloat16),
    )


def regular(x, w, out):
    return gemm_a16w16(x, w, y=out, dtype=out.dtype, backend="triton")


def main():
    if get_gfx() != "gfx950":
        logger.warning("oriented A16W16 requires gfx950; skip %s", get_gfx())
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--n", type=int, nargs="+", default=[224, 896, 4480])
    parser.add_argument("--k", type=int, nargs="+", default=[7168])
    args = parser.parse_args()
    rows = []
    for m, n, k in itertools.product(args.m, args.n, args.k):
        row = {"M": m, "N": n, "K": k}
        for name, fn in {"triton": regular, "oriented": gemm_a16w16_oriented}.items():
            bencher = RotatingBufferBench()
            bencher.build_pool(
                [functools.partial(generate_inputs, m, n, k)], [["x", "w", "out"]]
            )
            us = bencher.bench(fn) * 1000
            row[f"{name} us"] = us
            row[f"{name} TFLOPS"] = 2 * m * n * k / us / 1e6
            row[f"{name} TB/s"] = 2 * (m * k + n * k + m * n) / us / 1e6
        rows.append(row)
    logger.info(
        "oriented GEMM rotating-buffer benchmark:\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
