# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for ``aiter.gemm_a8w8_mxscale`` (OCP MXFP8 dense GEMM) on
gfx1250.

The op (E4M3 x E4M3) consumes 1x32 E8M0 block scales and is routed through the
FlyDSL gfx1250 backend. Skipped on non-gfx1250 hardware.
"""

import pytest
import torch

import aiter
from aiter.utility import dtypes, fp4_utils
from aiter.ops.quant import per_1x32_f8_scale_f8_quant

from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.ops.flydsl.mxscale_layout import SCALE_BLOCK

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx1250",
    reason="MXScale GEMM ops require a gfx1250 device",
)


def _dequant_fp8(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    qf = q.view(torch.float8_e4m3fn).to(torch.float32)
    sf = fp4_utils.e8m0_to_f32(s.view(torch.uint8)).to(torch.float32)
    return qf * sf.repeat_interleave(SCALE_BLOCK, dim=1)


def _metrics(out: torch.Tensor, ref: torch.Tensor):
    out_f, ref_f = out.float(), ref.float()
    rel = (out_f - ref_f).abs().sum() / ref_f.abs().sum().clamp_min(1e-6)
    cos = torch.nn.functional.cosine_similarity(out_f.flatten(), ref_f.flatten(), dim=0)
    return rel.item(), cos.item()


@pytest.mark.parametrize(
    "M,N,K",
    [
        (256, 256, 256),
        (512, 1024, 512),
        (1, 4096, 4096),
        (333, 512, 1024),  # unaligned M
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gemm_a8w8_mxscale(M, N, K, dtype):
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 2.0
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 2.0

    aq, a_s = per_1x32_f8_scale_f8_quant(
        a, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0, shuffle=False
    )
    bq, b_s = per_1x32_f8_scale_f8_quant(
        b, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0, shuffle=False
    )

    ref = (_dequant_fp8(aq, a_s) @ _dequant_fp8(bq, b_s).t()).to(dtype)
    out = aiter.gemm_a8w8_mxscale(aq, bq, a_s, b_s, dtype=dtype)

    assert out.shape == (M, N)
    assert out.dtype == dtype
    rel, cos = _metrics(out, ref)
    assert cos > 0.99, f"cosine={cos} too low (M={M},N={N},K={K})"
    assert rel < 0.05, f"rel L1={rel} too high (M={M},N={N},K={K})"


def test_gemm_a8w8_mxscale_out_tensor():
    """The out= path must write into the caller-provided tensor."""
    torch.manual_seed(0)
    M, N, K = 512, 512, 512
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 2.0
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 2.0
    aq, a_s = per_1x32_f8_scale_f8_quant(
        a, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0, shuffle=False
    )
    bq, b_s = per_1x32_f8_scale_f8_quant(
        b, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0, shuffle=False
    )
    ref = (_dequant_fp8(aq, a_s) @ _dequant_fp8(bq, b_s).t()).to(torch.bfloat16)
    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    ret = aiter.gemm_a8w8_mxscale(aq, bq, a_s, b_s, out=out, dtype=torch.bfloat16)
    assert ret.data_ptr() == out.data_ptr()
    _, cos = _metrics(out, ref)
    assert cos > 0.99, f"cosine={cos} too low"


def test_gemm_a8w8_mxscale_split_k():
    """split_k > 1 uses the buffer-store atomic-accumulation path."""
    from aiter.ops.flydsl.mxscale_gemm import (
        flydsl_mxscale_gemm,
        flydsl_mxscale_kernel_name,
    )

    torch.manual_seed(0)
    M, N, K = 256, 256, 1024  # K big enough for split_k=2 + num_buffers=2
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 2.0
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 2.0
    aq, a_s = per_1x32_f8_scale_f8_quant(
        a, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0, shuffle=False
    )
    bq, b_s = per_1x32_f8_scale_f8_quant(
        b, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0, shuffle=False
    )
    ref = (_dequant_fp8(aq, a_s) @ _dequant_fp8(bq, b_s).t()).to(torch.bfloat16)
    name = flydsl_mxscale_kernel_name(
        data_format="fp8",
        out_dtype="bf16",
        tile_m=128,
        tile_n=128,
        tile_k=128,
        m_warp=2,
        n_warp=2,
        num_buffers=2,
        split_k=2,
    )
    out = flydsl_mxscale_gemm(
        aq, bq, a_s, b_s, data_format="fp8", out_dtype="bf16", kernel_name=name
    )
    _, cos = _metrics(out, ref)
    assert cos > 0.99, f"split_k cosine={cos} too low"


def _build_mxscale_inputs(M, N, K):
    """Random bf16 A/B quantized to OCP MXFP8 (E4M3 data + 1x32 E8M0 scales)."""
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 2.0
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 2.0
    aq, a_s = per_1x32_f8_scale_f8_quant(
        a, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0, shuffle=False
    )
    bq, b_s = per_1x32_f8_scale_f8_quant(
        b, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0, shuffle=False
    )
    return aq, bq, a_s, b_s


def _bench_gemm_mxscale(dtype, M, N, K, num_iters, num_warmup):
    from aiter.test_common import benchmark, checkAllclose, run_perftest

    @benchmark()
    def run(dtype, M, N, K):
        if K % SCALE_BLOCK != 0:
            raise ValueError(f"K={K} must be divisible by SCALE_BLOCK={SCALE_BLOCK}")
        torch.manual_seed(0)
        aq, bq, a_s, b_s = _build_mxscale_inputs(M, N, K)
        ref = (_dequant_fp8(aq, a_s) @ _dequant_fp8(bq, b_s).t()).to(dtype)

        out, us = run_perftest(
            aiter.gemm_a8w8_mxscale,
            aq,
            bq,
            a_s,
            b_s,
            dtype=dtype,
            num_iters=num_iters,
            num_warmup=num_warmup,
        )

        err = checkAllclose(ref, out, rtol=1e-2, atol=1e-2, msg=f"mxscale {M}x{N}x{K}")
        rel, cos = _metrics(out, ref)
        flops = 2.0 * M * N * K
        bytes_moved = aq.nbytes + bq.nbytes + a_s.nbytes + b_s.nbytes + out.nbytes
        return {
            "us": us,
            "TFLOPS": flops / us / 1e6,
            "TB/s": bytes_moved / us / 1e6,
            "cos": cos,
            "rel": rel,
            "err": err,
        }

    return run(dtype, M, N, K)


def _main():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Benchmark / test aiter.gemm_a8w8_mxscale (OCP MXFP8) on gfx1250.",
    )
    parser.add_argument(
        "--pytest",
        action="store_true",
        default=False,
        help="Run the pytest correctness suite instead of the benchmark.",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        choices=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp16"]],
        metavar="{bf16,fp16}",
        default=[dtypes.d_dtypes["bf16"]],
        help="Output dtype(s). e.g.: -d bf16 fp16",
    )
    parser.add_argument(
        "-mnk",
        "--shape",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (256, 256, 256),
            (512, 1024, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            (1, 4096, 4096),
            (8192, 8192, 8192),
        ],
        help="Shapes of (M, N, K). K must be divisible by 32.\n    e.g. -mnk 1024,1024,1024 2048,2048,2048",
    )
    parser.add_argument(
        "--iters", type=int, default=101, help="Timed iterations per shape."
    )
    parser.add_argument(
        "--warmup", type=int, default=2, help="Warmup iterations per shape."
    )
    args = parser.parse_args()

    if args.pytest:
        return pytest.main([__file__, "-v", "-s"])

    if not torch.cuda.is_available() or get_gfx() != "gfx1250":
        print("SKIP: MXScale GEMM requires a gfx1250 device; skipping benchmark.")
        return 0

    import pandas as pd

    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 1000)

    df = []
    for dtype in args.dtype:
        for m, n, k in args.shape:
            df.append(_bench_gemm_mxscale(dtype, m, n, k, args.iters, args.warmup))
    df = pd.DataFrame(df)
    try:
        summary = df.to_markdown(index=False)
    except ImportError:
        summary = df.to_string(index=False)
    aiter.logger.info("gemm_a8w8_mxscale summary:\n%s", summary)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main())
