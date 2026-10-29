# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import sys

import pandas as pd
import torch
import torch.nn.functional as F
import triton

import aiter
from aiter.ops.triton.gemm.basic.gemm_a16w16_streamk import (
    _SUPPORTED_ARCHS,
    gemm_a16w16_streamk,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.test_common import benchmark, checkAllclose, run_perftest
from op_tests.triton_tests.gemm.basic.test_gemm_a16w16_streamk import generate_inputs

_SHAPES = [
    (8192, 8192, 8192),
    (16384, 16384, 16384),
    (8192, 5120, 2880),
    (4096, 2880, 4096),
    (512, 4096, 4096),
    (64, 4096, 4096),
]


@benchmark()
def bench_gemm_a16w16_streamk(M, N, K, layout, dtype):
    x, w = generate_inputs(M, N, K, dtype, layout=layout)
    ref = F.linear(x, w, bias=None)

    out, us = run_perftest(
        gemm_a16w16_streamk, x, w, num_rotate_args=4, use_cuda_event=True
    )
    checkAllclose(
        ref.float(),
        out.float(),
        rtol=1e-2,
        atol=1e-1,
        msg=f"gemm_a16w16_streamk M={M} N={N} K={K}",
    )

    # run_perftest syncs on every sample, which charges one dispatch latency per call. Graph
    # replay removes it, so the throughput columns come from the graph number.
    us_graph = (
        triton.testing.do_bench_cudagraph(lambda: gemm_a16w16_streamk(x, w), rep=100)
        * 1e3
    )

    tflops = 2.0 * M * N * K / us_graph * 1e-6
    gbps = (
        (
            x.numel() * x.element_size()
            + w.numel() * w.element_size()
            + out.numel() * out.element_size()
        )
        / us_graph
        * 1e-3
    )
    return {
        "gfx": get_arch(),
        "us": us,
        "us_graph": us_graph,
        "TFLOP/s": tflops,
        "GB/s": gbps,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the gluon StreamK a16w16 GEMM."
    )
    parser.add_argument(
        "--layout",
        default="TN",
        choices=["TN", "TT"],
        help="TN gives an N-major weight, TT a K-major one",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        metavar=("M", "N", "K"),
        help="benchmark one shape instead of the default set",
    )
    args = parser.parse_args()

    if not any(a in (get_arch() or "") for a in _SUPPORTED_ARCHS):
        print(f"gemm_a16w16_streamk needs one of {_SUPPORTED_ARCHS}, got {get_arch()}")
        return 1

    shapes = [tuple(args.shape)] if args.shape else _SHAPES
    rows = [
        bench_gemm_a16w16_streamk(M, N, K, args.layout, torch.bfloat16)
        for M, N, K in shapes
    ]
    aiter.logger.info(
        "gemm_a16w16_streamk summary (bfloat16, layout=%s):\n%s",
        args.layout,
        pd.DataFrame(rows).to_markdown(index=False),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
