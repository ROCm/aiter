# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host time per call of the FlyDSL a16w16 GEMM launch path (gfx950).

An eager caller pays the host side of every GEMM launch; for the small GEMMs of a
decode step or a short prefill that host time is longer than the kernel. This
measures it on the backend ``tgemm.mm`` dispatches a tuned ``flydsl`` row to,
``tuned_gemm.flydsl_gemm(..., config={"kernelName": ...})``: kernel-name parsing,
param build and validation, then the launch.

The kernel name is fixed per case, so the number does not depend on which tuned
tables are loaded and two aiter trees can be compared directly. The default cases
are the hyper-connection GEMMs of Qwen3.8-Flash-Next (``[M, 10240] x [324, 10240]``
and ``[M, 320] x [10240, 320]``, 96 calls per forward) with the configs its tuned
table picks.

Each repeat enqueues ``--calls`` launches back to back and synchronizes only after
the loop: ``host us`` is the wall time of the loop per call, ``cpu us`` the process
CPU time per call. ``kernel us`` is the device time from ``run_perftest``; while
it is below ``host us`` the loop is bound by the host, not by the GPU queue.

Run:
    python op_tests/op_benchmarks/flydsl/bench_flydsl_hgemm_host.py
    python op_tests/op_benchmarks/flydsl/bench_flydsl_hgemm_host.py --calls 2000 --repeat 9
    python op_tests/op_benchmarks/flydsl/bench_flydsl_hgemm_host.py \\
        -c 64,324,10240,flydsl_hgemm_abf16_wbf16_bf16_t32x32x64x8_ks8_w2x2x1_bias0_ktail0_gm0_pft_gfx950
"""

import argparse
import statistics
import time

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import checkAllclose, run_perftest
from aiter.tuned_gemm import flydsl_gemm

SUPPORTED_GFX = ["gfx950"]

_PREFIX = "flydsl_hgemm_abf16_wbf16_bf16_"
_SUFFIX = "_bias0_ktail0_gm0_pft_gfx950"
# (m, n, k, tile/split-k/waves part of the kernel name)
_QWEN38_HC = [
    (9, 324, 10240, "t16x16x128x8_ks8_w1x1x1"),
    (64, 324, 10240, "t32x32x64x8_ks8_w2x2x1"),
    (128, 324, 10240, "t64x32x128x4_ks8_w4x2x1"),
    (9, 10240, 320, "t16x64x64x6_ks1_w1x2x1"),
    (64, 10240, 320, "t32x64x64x5_ks1_w1x2x1"),
    (128, 10240, 320, "t64x64x64x3_ks1_w2x2x1"),
    (512, 10240, 320, "t128x96x64x2_ks1_w2x2x1"),
    (2048, 10240, 320, "t128x128x64x2_ks1_w4x2x1"),
]
DEFAULT_CASES = [(m, n, k, f"{_PREFIX}{cfg}{_SUFFIX}") for m, n, k, cfg in _QWEN38_HC]


def parse_case(text):
    m, n, k, name = text.split(",", 3)
    return int(m), int(n), int(k), name


def bench_case(m, n, k, kernel_name, calls, repeat):
    torch.manual_seed(0)
    x = torch.randn(m, k, dtype=dtypes.bf16, device="cuda")
    w = torch.randn(n, k, dtype=dtypes.bf16, device="cuda")
    config = {"kernelName": kernel_name}

    def launch():
        return flydsl_gemm(x, w, 0, None, dtypes.bf16, config=config)

    y = launch()
    ref = x.float() @ w.float().t()
    err = checkAllclose(ref, y.float(), rtol=5e-2, atol=5e-2, msg=f"{m}x{n}x{k}: ")
    _, kernel_us = run_perftest(launch)

    for _ in range(100):
        launch()
    torch.cuda.synchronize()
    host_us, cpu_us = [], []
    for _ in range(repeat):
        torch.cuda.synchronize()
        c0 = time.process_time()
        t0 = time.perf_counter()
        for _ in range(calls):
            launch()
        t1 = time.perf_counter()
        c1 = time.process_time()
        torch.cuda.synchronize()
        host_us.append((t1 - t0) / calls * 1e6)
        cpu_us.append((c1 - c0) / calls * 1e6)
    return {
        "shape": f"{m}x{n}x{k}",
        "config": kernel_name[len(_PREFIX) :].replace(_SUFFIX, ""),
        "host us": round(statistics.median(host_us), 2),
        "host us min-max": f"{min(host_us):.2f}-{max(host_us):.2f}",
        "cpu us": round(statistics.median(cpu_us), 2),
        "kernel us": round(kernel_us, 2),
        "err": err,
    }


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("flydsl hgemm unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="host time per FlyDSL a16w16 GEMM launch",
    )
    parser.add_argument(
        "-c",
        "--case",
        type=parse_case,
        nargs="*",
        default=DEFAULT_CASES,
        help="""m,n,k,kernelName of a tuned flydsl_hgemm row (bf16 in and out).
        e.g.: -c 64,10240,320,flydsl_hgemm_abf16_wbf16_bf16_t32x64x64x5_ks1_w1x2x1_bias0_ktail0_gm0_pft_gfx950""",
    )
    parser.add_argument(
        "--calls", type=int, default=1000, help="launches per timed loop"
    )
    parser.add_argument(
        "--repeat", type=int, default=7, help="timed loops per case; medians reported"
    )
    parser.add_argument("--csv", default=None, help="also write the table here")
    args = parser.parse_args()

    rows = [bench_case(*case, args.calls, args.repeat) for case in args.case]
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "flydsl a16w16 host time per call (aiter %s):\n%s",
        aiter.__file__,
        df.to_markdown(index=False),
    )
    if args.csv:
        df.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
