# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Test/benchmark for the FlyDSL a8w8 blockscale bpreshuffle *batched* GEMM
interface ``aiter.gemm_a8w8_blockscale_bpreshuffle_bmm`` (gfx1250 only).

Exercises the four public-interface requirements:
  1. tuned-CSV shape selection   -- shapes present in the tuned bmm CSV
  2. default config for untuned  -- shapes not in the CSV fall back to a default
  3. torch.compile compatibility -- the op run through torch.compile
  4. correctness                 -- vs a torch blockwise-dequant reference

Usage:
    python op_tests/test_gemm_a8w8_blockscale_bpreshuffle_bmm.py
    python op_tests/test_gemm_a8w8_blockscale_bpreshuffle_bmm.py -b 2 -s 128,1024,4096
    python op_tests/test_gemm_a8w8_blockscale_bpreshuffle_bmm.py -l mbn --compile
"""

import argparse
import itertools

import aiter
import pandas as pd
import torch
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx1250"]
SCALE_BLOCK = 128


# ---------------------------------------------------------------------------
# fp8 / e8m0 helpers (self-contained; mirrors
# FlyDSL/tests/kernels/utils/gemm_common_utils.py).
# ---------------------------------------------------------------------------
def random_fp8_bytes(rows, cols):
    # Avoid NaN bytes (0x7F/0xFF): keep values < 126.
    return torch.randint(0, 126, (rows, cols), dtype=torch.uint8)


def random_e8m0(rows, cols, low_exp=127, high_exp=132):
    return torch.randint(low_exp, high_exp + 1, (rows, cols), dtype=torch.uint8)


def e8m0_to_f32(scale_u8):
    scale_u8 = scale_u8.view(torch.uint8)
    f32 = scale_u8.to(torch.int32) << 23
    f32[scale_u8 == 0] = 0x00400000
    f32[scale_u8 == 0xFF] = 0x7F800001
    return f32.view(dtypes.fp32)


def fp8_e4m3_to_f32(x):
    x = x.view(torch.uint8).to(torch.int32)
    sign, exp, mant = (x >> 7) & 1, (x >> 3) & 0xF, x & 0x7
    is_nan = (exp == 15) & (mant == 7)
    is_denorm = exp == 0
    f_normal = (1.0 + mant.float() / 8.0) * torch.pow(2.0, (exp.float() - 7.0))
    f_denorm = (mant.float() / 8.0) * (2.0**-6)
    r = torch.where(is_denorm, f_denorm, f_normal)
    r = torch.where(is_nan, torch.tensor(float("nan"), device=x.device), r)
    return torch.where(sign == 1, -r, r)


def preshuffle_b_16x16(b, rows, cols):
    b = b.view(rows // 16, 16, cols // 16, 16).permute(0, 2, 1, 3).contiguous()
    return b.view(rows, cols)


# ---------------------------------------------------------------------------
# Reference (per-batch blockwise dequant + matmul)
# ---------------------------------------------------------------------------
def run_torch(a_l, b_l, sa_l, sb_l, M, N, K, dtype=dtypes.bf16):
    out = []
    for a_u8, b_u8, sa_u8, sb_u8 in zip(a_l, b_l, sa_l, sb_l):
        a = fp8_e4m3_to_f32(a_u8)[:M, :K]
        b = fp8_e4m3_to_f32(b_u8)[:N, :K]
        sa = e8m0_to_f32(sa_u8).repeat_interleave(SCALE_BLOCK, dim=-1)[:M, :K]
        sb = (
            e8m0_to_f32(sb_u8)
            .repeat_interleave(SCALE_BLOCK, dim=0)
            .repeat_interleave(SCALE_BLOCK, dim=-1)[:N, :K]
        )
        out.append(torch.matmul(a * sa, (b * sb).T))
    return torch.stack(out).to(dtype)  # [B, M, N]


@benchmark()
def test_gemm(b, m, n, k, dtype, layout, do_compile=False):
    assert n % SCALE_BLOCK == 0, f"N={n} must be a multiple of {SCALE_BLOCK}"
    assert k % SCALE_BLOCK == 0, f"K={k} must be a multiple of {SCALE_BLOCK}"
    torch.manual_seed(42)
    scale_k, scale_n = k // SCALE_BLOCK, n // SCALE_BLOCK

    a_l = [random_fp8_bytes(m, k) for _ in range(b)]
    b_l = [random_fp8_bytes(n, k) for _ in range(b)]
    sa_l = [random_e8m0(m, scale_k) for _ in range(b)]
    sb_l = [random_e8m0(scale_n, scale_k) for _ in range(b)]

    ref = run_torch(a_l, b_l, sa_l, sb_l, m, n, k, dtype)  # [b, m, n]

    # Kernel inputs. Both layouts are logically [b, m, *]:
    #   bmn: contiguous [b, m, *].
    #   mbn: physically [m, b, *] (the batch dim interleaved with m), matching
    #        the deepseek-v4 grouped-output path -- see test_batched_gemm_bf16.py.
    if layout == "mbn":
        A = torch.stack(a_l, dim=1).contiguous().view(m * b, k)
        sa = torch.stack(sa_l, dim=1).contiguous()
        Out = torch.zeros(m, b, n, dtype=dtype)
    else:
        A = torch.cat(a_l, dim=0).contiguous()
        sa = torch.stack(sa_l, dim=0).contiguous()
        Out = torch.zeros(b, m, n, dtype=dtype)
    B = torch.cat([preshuffle_b_16x16(bi, n, k) for bi in b_l], dim=0).contiguous()
    sb = torch.stack(sb_l, dim=0).contiguous()

    A = A.view(dtypes.fp8)
    B = B.view(dtypes.fp8)
    layout_mbn = 1 if layout == "mbn" else 0

    op = aiter.gemm_a8w8_blockscale_bpreshuffle_bmm
    if do_compile:
        op = torch.compile(op, dynamic=False)

    def gemm_func():
        op(A, B, sa, sb, Out, m, n, k, b, layout_mbn)
        return Out

    out, us = run_perftest(gemm_func)
    out = out.permute(1, 0, 2).contiguous() if layout == "mbn" else out  # -> [b,m,n]

    err = checkAllclose(
        ref.to(dtypes.fp32),
        out.to(dtypes.fp32),
        rtol=1e-2,
        atol=5e-2,
        msg=f"a8w8 blockscale bpreshuffle bmm {layout} compile={do_compile}",
    )

    # FLOPs = 2 * b * m * n * k; bytes = (A + B + scales + out).
    flops = 2 * b * m * n * k
    nbytes = (
        b * m * k + b * n * k + sa.numel() + sb.numel() + b * m * n * Out.element_size()
    )
    return {
        "gfx": get_gfx(),
        "layout": layout,
        "compile": do_compile,
        "us": us,
        "TFLOPS": flops / us / 1e6,
        "TB/s": nbytes / us / 1e6,
        "err": err,
    }


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Test a8w8 blockscale bpreshuffle batched GEMM (gfx1250)",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp16"]],
        nargs="*",
        default="bf16,",
        metavar="{bf16,fp16}",
        help="Output dtype, e.g.: -d bf16",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[2, 16],
        help="Batch size(s), e.g.: -b 2 16",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            # Shapes present in a8w8_blockscale_bpreshuffle_tuned_bmm.csv (req 1):
            (1, 1024, 4096),
            (2, 1024, 4096),
            (4, 1024, 4096),
            (8, 1024, 4096),
            (16, 1024, 4096),
            (32, 1024, 4096),
            (64, 1024, 4096),
            (128, 1024, 4096),
            (192, 1024, 4096),
            (256, 1024, 4096),
            (384, 1024, 4096),
            (512, 1024, 4096),
            (768, 1024, 4096),
            (1024, 1024, 4096),
            (2048, 1024, 4096),
            (4096, 1024, 4096),
            (8192, 1024, 4096),
            (16384, 1024, 4096),
        ],
        help="Shape(s) of m,n,k, e.g.: -s 128,1024,4096",
    )
    parser.add_argument(
        "-l",
        "--layout",
        type=str,
        choices=["bmn", "mbn"],
        nargs="*",
        default=["mbn"],
        help="Logical [b,m,*] layout: bmn=contiguous, mbn=[m,b,*] interleaved.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Also run each config through torch.compile (req 4).",
    )
    args = parser.parse_args()

    if get_gfx() not in SUPPORTED_GFX:
        print(f"SKIP: a8w8 blockscale bpreshuffle bmm is gfx1250-only, got {get_gfx()}")
        return

    compile_opts = [False, True] if args.compile else [False]

    for dtype in args.dtype:
        df = []
        for layout, b, (m, n, k), do_compile in itertools.product(
            args.layout, args.batch, args.mnk, compile_opts
        ):
            ret = test_gemm(b, m, n, k, dtype, layout, do_compile=do_compile)
            df.append(ret)
        df = pd.DataFrame(df)
        aiter.logger.info(
            "a8w8_blockscale_bpreshuffle_bmm summary (markdown):\n%s",
            df.to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
