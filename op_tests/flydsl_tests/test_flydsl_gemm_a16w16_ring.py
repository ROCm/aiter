# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm
from aiter.ops.flydsl.kernels.gemm_a16w16_ring_gfx950 import gemm_a16w16_ring
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]
# 16384x384x5120 is the DeepSeek-V4.1 MoE router (torch.mm(x, W.t()) with W[N, K], prefill chunk);
# the others cover partial M and N tiles and the shortest K the four-stage ring accepts.
DEFAULT_MNK = [
    (16384, 384, 5120),
    (300, 384, 5120),
    (129, 200, 1024),
    (256, 384, 320),
]
# The generic gfx950 kernel at its best configuration for the router shape, for comparison.
GENERIC_CONFIG = {
    "block_m": 128,
    "block_n": 192,
    "block_k": 64,
    "stages": 4,
    "m_waves": 4,
    "n_waves": 2,
    "group_m": 4,
}


def run_torch(x, w, dtype):
    return (x.to(dtypes.fp32) @ w.to(dtypes.fp32).t()).to(dtype)


@benchmark()
def test_gemm_a16w16_ring(m, n, k, dtype):
    x = torch.randn(m, k, dtype=dtype)
    w = torch.randn(n, k, dtype=dtype)
    ref = run_torch(x, w, dtype)
    out = torch.empty(m, n, dtype=dtype)
    candidates = {"ring": lambda: gemm_a16w16_ring(x, w, out)}
    if m % GENERIC_CONFIG["block_m"] == 0 and n % GENERIC_CONFIG["block_n"] == 0:
        # The generic kernel needs whole tiles at this configuration.
        candidates["flydsl_ft"] = lambda: flydsl_hgemm(x, w, **GENERIC_CONFIG)

    flops = 2 * m * n * k
    nbytes = (m * k + n * k + m * n) * x.element_size()
    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        result, us = run_perftest(fn)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            result.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name}: gemm a16w16 ring",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("ring GEMM unsupported on %s; skipping", get_gfx())
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
        default=[dtypes.bf16, dtypes.fp16],
        help="""Input and output dtype.
        e.g.: -d bf16""",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=DEFAULT_MNK,
        help="""M, N, K; K must be a multiple of 64 with at least 5 K tiles, N a multiple of 8.
        e.g.: -s 16384,384,5120""",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        df = pd.DataFrame(
            [test_gemm_a16w16_ring(m, n, k, dtype) for (m, n, k) in args.mnk]
        )
        aiter.logger.info(
            "gemm_a16w16 ring summary (markdown):\n%s", df.to_markdown(index=False)
        )


if __name__ == "__main__":
    main()
