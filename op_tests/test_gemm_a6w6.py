# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import os

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.core import get_asm_dir
from aiter.ops.gemm_op_a6w6 import (
    dequant_mxfp6_torch,
    quant_mxfp6_gemm,
    quant_mxfp6_torch,
)
from aiter.test_common import benchmark, checkAllclose, perftest, run_perftest

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)
SCALE_GROUP_SIZE = 32
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 30)


@perftest(num_iters=5)
def run_torch(x, w, dtype):
    # fp32 reference (on GPU) that matches the mxfp6 (E2M3, per-1x32 blockscale)
    # math the kernel approximates: quantize both operands, dequantize, matmul.
    # The packed GEMM accepts arbitrary K and pads it to its 128-wide tile. Mirror
    # that here because quant_mxfp6_torch itself requires complete scale groups.
    k = x.shape[1]
    padded_k = (k + 127) // 128 * 128
    if padded_k != k:
        x = torch.nn.functional.pad(x, (0, padded_k - k))
        w = torch.nn.functional.pad(w, (0, padded_k - k))
    xc, xs = quant_mxfp6_torch(x)
    wc, ws = quant_mxfp6_torch(w)
    xf = dequant_mxfp6_torch(xc, xs)
    wf = dequant_mxfp6_torch(wc, ws)
    return torch.mm(xf, wf.T).to(dtype)


@benchmark()
def test_gemm(dtype, M, N, K, kernel_name=None):
    from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx

    if get_gfx() not in ["gfx950"]:
        return
    ret = {}
    x = torch.randn((M, K), dtype=dtype)
    w = torch.randn((N, K), dtype=dtype)

    a, _avg_a = run_torch(x, w, dtype)

    # pack operands + scales into the kernel's mxfp6 layout (done once, untimed)
    xq, xs = quant_mxfp6_gemm(x)
    wq, ws = quant_mxfp6_gemm(w)

    c, us = run_perftest(
        aiter.gemm_a6w6,
        xq,
        wq,
        xs,
        ws,
        M,
        N,
        K,
        kernelName=kernel_name,
    )
    err = checkAllclose(a, c, msg="unified api", catastrophic_check=True)
    ret["us"] = us
    ret["TFLOPS"] = M * N * K * 2 / us / 1e6
    ret["TB/s"] = (x.nbytes + w.nbytes) / us / 1e6
    ret["err"] = err
    ret["kernel"] = kernel_name or "auto"
    return ret


def _manifest_kernel_names(M, N, K):
    manifest = os.path.join(get_asm_dir(), "f6gemm", "f6gemm_bf16_per1x32Fp6.csv")
    configs = pd.read_csv(manifest)
    padM = (M + 255) // 256 * 256
    padN = (N + 255) // 256 * 256
    padK = (K + 127) // 128 * 128
    compatible = (configs["swizzle_max_K"] <= 0) | (
        (padK > configs["swizzle_max_K"])
        | ((padM <= configs["swizzle_max_M"]) & (padN <= configs["swizzle_max_N"]))
    )
    return configs.loc[compatible, "knl_name"].astype(str).tolist()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        choices=[dtypes.d_dtypes["bf16"]],
        metavar="{bf16}",
        default=[dtypes.d_dtypes["bf16"]],
        help="""Data type.
        e.g.: -d bf16""",
    )
    parser.add_argument(
        "-mnk",
        "--shape",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            (8192, 8192, 8192),
            (16384, 16384, 16384),
            # transformer shapes
            (9450, 13824, 5120),
            (9450, 5120, 13824),
            # Exercise row, column, and contraction-dimension padding together.
            (257, 513, 129),
        ],
        help="""Shape of mnk.
        e.g. -mnk 8192,8192,8192""",
    )
    parser.add_argument(
        "--kernel-name",
        nargs="*",
        default=None,
        help="Explicit registered kernel name(s); default uses tuned dispatch.",
    )
    parser.add_argument(
        "--all-kernels",
        action="store_true",
        help="Test every compatible kernel in the gfx950 F6 manifest.",
    )
    args = parser.parse_args()

    if args.all_kernels and args.kernel_name:
        parser.error("--all-kernels and --kernel-name are mutually exclusive")
    if args.all_kernels:
        from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx

        if get_gfx() != "gfx950":
            aiter.logger.info("--all-kernels requires gfx950; skipping")
            return
    results = []
    for dtype in args.dtype:
        for m, n, k in args.shape:
            kernel_names = (
                _manifest_kernel_names(m, n, k)
                if args.all_kernels
                else (args.kernel_name if args.kernel_name else [None])
            )
            for kernel_name in kernel_names:
                results.append(test_gemm(dtype, m, n, k, kernel_name))
    frame = pd.DataFrame(results)
    aiter.logger.info(
        "gemm_a6w6 summary (markdown):\n%s", frame.to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
