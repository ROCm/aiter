# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Op test for aiter.gemm_a16w8_asm: out = (A @ B.T) * B_scale with BF16
# activations, FP8 e4m3fn weights and one fp32 scale per weight row, for decode
# batches of 1 to 8 tokens on gfx950.
#
# One @benchmark function times both paths in one table:
#   "bf16"     : torch.mm with the weights in BF16, the GEMM that the FP8
#                weights replace.
#   "a16w8 asm": gemm_a16w8_asm.
# The asm path is checked against a float64 reference, and the `err` column is
# that check. CI runs this file with python3 and only sees the exit code, so
# main() raises when any `err` is not zero.
#
# Two kinds of input:
#   "exact": small integer weights, activations that are multiples of 1/64 and
#            power-of-two row scales, so every fp32 sum and the scaling are
#            exact. The output must then equal the reference bit for bit.
#   "randn": normal random weights quantized per row to FP8. The output must
#            match the reference within BF16 rounding.

import argparse

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.gemm_op_a16w8 import gemm_a16w8_asm
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]
FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max
# Per rank shapes at tensor parallel 8 of Qwen/Qwen3.8-2.4T-A95B: the linear
# attention input projection in_proj_qkvz (N 4608, K 8192) and output
# projection out_proj (N 8192, K 2048).
SHAPES = [(4608, 8192), (8192, 2048)]


def make_inputs(m, n, k, mode, seed=0):
    """Return A [m, k] bf16, B [n, k] fp8 e4m3fn and B_scale [n] fp32."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    if mode == "exact":
        # Every product is a multiple of 1/64 and every sum stays far below
        # 2^24 / 64, so fp32 holds every partial sum exactly. Scaling by a power
        # of two is exact too.
        a = torch.randint(-3, 4, (m, k), generator=gen).to(dtypes.bf16) / 64
        b = torch.randint(-3, 4, (n, k), generator=gen).float().to(FP8)
        b_scale = torch.ldexp(
            torch.ones(n), torch.randint(-6, -2, (n,), generator=gen)
        ).to(dtypes.fp32)
    else:
        a = torch.randn(m, k, generator=gen).to(dtypes.bf16)
        w = torch.randn(n, k, generator=gen) * k**-0.5
        b_scale = (w.abs().amax(dim=1) / FP8_MAX).to(dtypes.fp32)
        b = (w / b_scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return a, b, b_scale


def reference(a, b, b_scale):
    """Float64 sums, which are exact for these inputs, scaled and rounded to BF16."""
    return ((a.double() @ b.double().t()) * b_scale.double()).to(dtypes.bf16)


def run_bf16(a, b_bf16, out):
    torch.mm(a, b_bf16.t(), out=out)


def run_asm(a, b, b_scale, out):
    gemm_a16w8_asm(a, b, b_scale, out)


@benchmark()
def test_gemm_a16w8(m, n, k, mode):
    a, b, b_scale = make_inputs(m, n, k, mode)
    ref = reference(a, b, b_scale).float()
    b_bf16 = (b.float() * b_scale[:, None]).to(dtypes.bf16)
    out_bf16 = torch.empty(m, n, dtype=dtypes.bf16)
    out_asm = torch.full((m, n), float("nan"), dtype=dtypes.bf16)

    ret = {"gfx": get_gfx()}
    # The tensors are passed as arguments, so run_perftest rotates through
    # copies of them and the weights are read from memory, not from cache, as
    # they are in a real decode step.
    _, us_bf16 = run_perftest(run_bf16, a, b_bf16, out_bf16)
    _, us_asm = run_perftest(run_asm, a, b, b_scale, out_asm)
    torch.cuda.synchronize()

    if mode == "exact":
        rtol, atol = 0, 0
    else:
        # One bf16 step is at most 2^-8 relative, so rtol 1e-2 allows the fp32
        # sums of the kernel to round to the neighbour of the reference.
        rtol, atol = 1e-2, 1e-3
    err = checkAllclose(out_asm.float(), ref, rtol=rtol, atol=atol, msg="a16w8 asm")

    ret["bf16 us"] = us_bf16
    ret["bf16 TB/s"] = b_bf16.numel() * b_bf16.element_size() / us_bf16 / 1e6
    ret["a16w8 asm us"] = us_asm
    ret["a16w8 asm TB/s"] = (b.numel() + b_scale.numel() * 4) / us_asm / 1e6
    ret["a16w8 asm err"] = err
    ret["speedup"] = us_bf16 / us_asm
    return ret


def _expect_raises(fn, needle, desc):
    try:
        fn()
    except RuntimeError as e:
        assert needle in str(e), f"{desc}: unexpected error message: {e}"
        return
    raise AssertionError(f"{desc}: expected RuntimeError containing {needle!r}")


def _run_guard_checks():
    """Shapes and types the kernels do not cover must raise instead of writing garbage."""
    a, b, s = make_inputs(9, 4608, 8192, "randn")
    _expect_raises(lambda: gemm_a16w8_asm(a, b, s), "no kernel", "M 9")

    a, b, s = make_inputs(2, 4640, 8192, "randn")
    _expect_raises(lambda: gemm_a16w8_asm(a, b, s), "no kernel", "N 4640")

    a, b, s = make_inputs(2, 4608, 8192, "randn")
    _expect_raises(
        lambda: gemm_a16w8_asm(a, b.view(torch.float8_e4m3fnuz), s),
        "float8_e4m3fn",
        "fnuz weights",
    )
    a_wide = torch.empty(2, 2 * 8192, dtype=dtypes.bf16)[:, :8192]
    _expect_raises(lambda: gemm_a16w8_asm(a_wide, b, s), "contiguous", "strided A")
    _expect_raises(lambda: gemm_a16w8_asm(a, b, s[:-1]), "row scales", "short B_scale")
    out = torch.empty(2, 4608, dtype=dtypes.fp16)
    _expect_raises(lambda: gemm_a16w8_asm(a, b, s, out), "must be bf16", "fp16 out")

    a, b, s = make_inputs(0, 4608, 8192, "randn")
    assert gemm_a16w8_asm(a, b, s).shape == (0, 4608), "M 0 should be a no-op"

    aiter.logger.info(
        "guard checks passed: M 9, N 4640, fnuz weights, strided A, short B_scale and "
        "fp16 out raise, and M 0 is a no-op"
    )


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("gemm_a16w8_asm unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-m",
        type=int,
        nargs="*",
        default=[1, 2, 3, 4, 5, 6, 7, 8],
        help="number of tokens, 1 to 8",
    )
    parser.add_argument(
        "-nk",
        type=dtypes.str2tuple,
        nargs="*",
        default=SHAPES,
        help="N,K of the weight, for example -nk 4608,8192",
    )
    parser.add_argument(
        "--mode",
        type=str,
        nargs="*",
        choices=["exact", "randn"],
        default=["exact", "randn"],
    )
    args = parser.parse_args()

    df = pd.DataFrame(
        [
            test_gemm_a16w8(m, n, k, mode)
            for mode in args.mode
            for n, k in args.nk
            for m in args.m
        ]
    )
    aiter.logger.info(
        "gemm_a16w8_asm vs torch.mm in bf16:\n%s", df.to_markdown(index=False)
    )

    _run_guard_checks()

    bad = df[df["a16w8 asm err"] != 0]
    assert bad.empty, f"results differ from the reference:\n{bad.to_markdown()}"


if __name__ == "__main__":
    main()
