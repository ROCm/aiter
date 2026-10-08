# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import os

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.core import get_asm_dir
from aiter.ops.gemm_op_a6w6 import (
    _select_gemm_a6w6_kernel,
    dequant_mxfp6_torch,
    gemm_a6w6_asm,
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

SPECIALIZED_CASES = (
    (
        (9450, 5120, 5120),
        "aiter_a6w6_m9472_n5120_k5120_m82",
        "f6gemm_dmabig_kernel_func",
    ),
    (
        (9450, 13824, 5120),
        "aiter_a6w6_m9472_n13824_k5120_m82",
        "f6gemm_dmabig_kernel_func",
    ),
    (
        (9450, 5120, 13824),
        "aiter_a6w6_m9472_n5120_k13824_m82",
        "f6gemm_dmabig_allk_kernel_func",
    ),
)
SPECIALIZED_SQUARE_KERNEL = SPECIALIZED_CASES[0][1]


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


test_gemm.__test__ = False


def _manifest_kernel_names(M, N, K):
    manifest = os.path.join(get_asm_dir(), "f6gemm", "f6gemm_bf16_per1x32Fp6.csv")
    configs = pd.read_csv(manifest)
    padM = (M + 255) // 256 * 256
    padN = (N + 255) // 256 * 256
    padK = (K + 127) // 128 * 128
    exact_compatible = (configs["exact_K"] <= 0) | (
        (padM == configs["exact_M"])
        & (padN == configs["exact_N"])
        & (padK == configs["exact_K"])
    )
    swizzle_compatible = (configs["swizzle_max_K"] <= 0) | (
        (padK > configs["swizzle_max_K"])
        | ((padM <= configs["swizzle_max_M"]) & (padN <= configs["swizzle_max_N"]))
    )
    compatible = exact_compatible & swizzle_compatible
    return configs.loc[compatible, "knl_name"].astype(str).tolist()


@pytest.mark.parametrize(
    "shape,specialized_kernel,safe_kernel",
    SPECIALIZED_CASES,
    ids=("square", "ffn_up", "ffn_down"),
)
@torch.no_grad()
def test_specialized_dispatch_matches_safe_kernel(
    shape, specialized_kernel, safe_kernel
):
    from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx

    if get_gfx() != "gfx950":
        pytest.skip("A6W6 assembly requires gfx950")
    M, N, K = shape
    padM = (M + 255) // 256 * 256
    padN = (N + 255) // 256 * 256
    torch.manual_seed(M + N + K)
    x = torch.randn((M, K), dtype=torch.bfloat16)
    w = torch.randn((N, K), dtype=torch.bfloat16)
    A, A_scale = quant_mxfp6_gemm(x)
    B, B_scale = quant_mxfp6_gemm(w)
    baseline = torch.empty((padM, padN), dtype=torch.bfloat16)
    actual = torch.empty_like(baseline)
    selected = _select_gemm_a6w6_kernel(M, N, K, None)
    assert selected == specialized_kernel
    gemm_a6w6_asm(A, B, A_scale, B_scale, baseline, K, safe_kernel)
    gemm_a6w6_asm(A, B, A_scale, B_scale, actual, K, selected)
    assert torch.equal(actual[:M, :N], baseline[:M, :N])


def test_manifest_excludes_shape_guarded_kernels_from_incompatible_sweeps():
    small = _manifest_kernel_names(257, 513, 129)
    assert not any("a6w6_m9472_" in name for name in small)
    square = _manifest_kernel_names(9450, 5120, 5120)
    assert SPECIALIZED_SQUARE_KERNEL in square
    assert "aiter_a6w6_m9472_n13824_k5120_m82" not in square
    assert "aiter_a6w6_m9472_n5120_k13824_m82" not in square


@torch.no_grad()
def test_specialized_kernel_rejects_incompatible_physical_shape():
    from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx

    if get_gfx() != "gfx950":
        pytest.skip("A6W6 assembly requires gfx950")
    M, N, K = 257, 513, 129
    padM = (M + 255) // 256 * 256
    padN = (N + 255) // 256 * 256
    padK = (K + 127) // 128 * 128
    x = torch.randn((M, K), dtype=torch.bfloat16)
    w = torch.randn((N, K), dtype=torch.bfloat16)
    A, A_scale = quant_mxfp6_gemm(x)
    B, B_scale = quant_mxfp6_gemm(w)
    out = torch.empty((padM, padN), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="requested physical shape"):
        gemm_a6w6_asm(A, B, A_scale, B_scale, out, padK, SPECIALIZED_SQUARE_KERNEL)


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
            (9450, 5120, 5120),
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
