# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx950 BF16 GEMM on NN / TT / TN operand storage: correctness and latency.

Times ``gemm_a16w16_layout_opus`` on the stored operands (no copy) against what a
caller without it has to do: ``contiguous()`` on each transposed operand, then
the tuned NT path ``gemm_a16w16_opus``.

Usage:
    python3 op_tests/test_opus_a16w16_layout_gemm.py
    python3 op_tests/test_opus_a16w16_layout_gemm.py -l TN -s 64,2048,7168
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.opus import gemm_a16w16_layout_opus, gemm_a16w16_opus
from aiter.test_common import (
    benchmark,
    checkAllclose,
    run_perftest,
)

torch.set_default_device("cuda")

# The NN / TT / TN kernels are the gfx950 4-wave kids.
SUPPORTED_GFX = ["gfx950"]


def run_torch(A, B):
    # Reference only: fp32 math on the logical [M, K] / [N, K] operands. Not timed.
    return A.to(dtypes.fp32) @ B.to(dtypes.fp32).t()


def layout_supported(m, n, k, layout):
    """Shapes the 4-wave kids take: K tiles of 64 (>= 2); a [K, M] A needs M % 8
    and a [K, N] B needs N % 8 (16-byte rows)."""
    if k % 64 or k < 128:
        return False
    return not ((layout[0] == "T" and m % 8) or (layout[1] == "N" and n % 8))


@benchmark()
def test_layout_gemm(m, n, k, dtype, layout):
    # Logical A [m, k] / B [n, k]; the kernel sees them as the model stores them:
    #   A "N": [m, k]   A "T": [k, m]   B "T": [n, k]   B "N": [k, n]
    # passed as the transposed view of that storage, never a copy.
    A = torch.randn((m, k), dtype=dtype)
    B = torch.randn((n, k), dtype=dtype)
    x = A if layout[0] == "N" else A.t().contiguous().t()
    w = B if layout[1] == "T" else B.t().contiguous().t()
    ref = run_torch(A, B)

    candidates = {
        # stored operands, read in place
        "opus_layout": lambda: gemm_a16w16_layout_opus(x, w),
        # materialise [m, k] / [n, k], then the tuned NT path
        "contiguous+opus_nt": lambda: gemm_a16w16_opus(x.contiguous(), w.contiguous()),
    }

    # GEMM [m,k] @ [n,k]^T -> [m,n]:
    #   FLOPs = 2 * m * n * k (multiply-add); bytes = (A + B + Y) * dtype size.
    flops = 2 * m * n * k
    nbytes = (m * k + n * k + m * n) * A.element_size()

    ret = {"gfx": get_gfx_runtime()}
    for name, fn in candidates.items():
        # Stream events around each call, i.e. end-to-end latency (host launch
        # included): the profiler trace under-reports the pre-built .co kernels.
        out, us = run_perftest(fn, use_cuda_event=True)
        err = checkAllclose(
            ref,
            out.reshape(m, n).to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name}: a16w16 {layout} gemm",
        )
        # checkAllclose logs and returns the mismatch fraction; it does not raise.
        assert err == 0, f"{name}: {err:.2%} of elements outside rtol/atol"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "a16w16 NN/TT/TN gemm unsupported on %s; skipping", get_gfx_runtime()
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
        choices=[dtypes.d_dtypes["bf16"]],
        nargs="*",
        default="bf16,",
        metavar="{bf16}",
        help="""Data type.
        e.g.: -d bf16""",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (64, 2048, 7168),
            (336, 784, 2112),
            (1024, 1024, 4096),
            (4096, 1536, 512),
            (2048, 7168, 1024),
            (8192, 8192, 4096),
        ],
        help="""Shape of mnk.
        e.g.:   -s 64,2048,7168
                --mnk 64,2048,7168""",
    )
    parser.add_argument(
        "-l",
        "--layout",
        type=str,
        choices=["NN", "TT", "TN"],
        nargs="*",
        default=["NN", "TT", "TN"],
        help="""Operand storage (first letter A, second B):
        NN = A [m, k], B [k, n];  TT = A [k, m], B [n, k];  TN = A [k, m], B [k, n].
        e.g.: -l TN""",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        df = []
        for layout, (m, n, k) in itertools.product(args.layout, args.mnk):
            if not layout_supported(m, n, k, layout):
                aiter.logger.warning(
                    "skip %s (%d, %d, %d): not a 4-wave shape", layout, m, n, k
                )
                continue
            df.append(test_layout_gemm(m, n, k, dtype, layout))
        df = pd.DataFrame(df)
        aiter.logger.info(
            "opus a16w16 layout gemm summary (markdown):\n%s",
            df.to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
