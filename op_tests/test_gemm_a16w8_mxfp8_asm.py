# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Op test for aiter.gemm_a16w8_mxfp8_asm: out = A @ dequant(B, B_scale).T with
# BF16 activations and MXFP8 weights (e4m3 + one UE8M0 exponent per 32 weights
# along K, the OCP MX layout), for decode batches of 1 to 64 tokens on gfx942.
#
# One @benchmark function times both paths in one table:
#   "bf16"     : torch.mm with the weights dequantized to BF16 once, the GEMM
#                that gfx942 runs for MXFP8 checkpoints today.
#   "asm"      : gemm_a16w8_mxfp8_asm on the prepared MXFP8 weights.
# The asm path is checked against a float64 reference, and the `err` column is
# that check. CI only sees the exit code, so main() raises when any `err` is
# not zero.
#
# Two kinds of input:
#   "exact": small integer weights with power-of-two group scales and
#            activations that are multiples of 1/64, so every fp32 partial sum
#            is exact in any order (split-K included). The output must then
#            equal the reference bit for bit.
#   "randn": normal random weights quantized to MXFP8. The output must match
#            the reference within BF16 rounding.
#
# The shapes are the tuned ones from
# aiter/configs/model_configs/dsv41_a16w8_mxfp8_asm_tuned_gemm.csv (per-rank
# projections of an MXFP8 checkpoint at tensor parallel 4) plus untuned shapes,
# which run the default kernel for M rows.

import argparse

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.ops.gemm_op_a16w8_mxfp8 import (
    gemm_a16w8_mxfp8_asm,
    gemm_a16w8_mxfp8_prepare_weight,
    is_gemm_a16w8_mxfp8_tuned,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942"]
FN = torch.float8_e4m3fn
FN_MAX = torch.finfo(FN).max
GROUP = 32
# Tuned: per-rank linears of DeepSeek-V4.1-Flash at tensor parallel 4 (MXFP8
# checkpoint): fused wq_a+wkv, wq_b, wo_a, wo_b, shared expert w1+w3 and w2,
# lm_head. Untuned: other common projection shapes, default kernel.
TUNED_SHAPES = [
    (1792, 5120),
    (8192, 1280),
    (2048, 4096),
    (5120, 2048),
    (1152, 5120),
    (5120, 576),
    (32320, 5120),
]
UNTUNED_SHAPES = [(4096, 4096), (2112, 7168), (7168, 2048), (1536, 7168)]


def make_inputs(m, n, k, mode, seed=0):
    """A [m, k] bf16, B [n, k] e4m3fn and B_scale [n, k / 32] UE8M0 (uint8)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    g = k // GROUP
    if mode == "exact":
        # Products are multiples of 2^-12 and every partial sum stays below
        # 2^7, so fp32 holds all of them exactly, in any summation order.
        a = torch.randint(-3, 4, (m, k), generator=gen).to(dtypes.bf16) / 64
        b = torch.randint(-3, 4, (n, k), generator=gen).float().to(FN)
        e = torch.randint(-6, -2, (n, g), generator=gen)
    else:
        a = torch.randn(m, k, generator=gen).to(dtypes.bf16)
        w = torch.randn(n, k, generator=gen) * k**-0.5
        amax = w.view(n, g, GROUP).abs().amax(-1).clamp_min(2.0**-100)
        e = torch.floor(torch.log2(amax)) - 8  # e4m3fn's largest power of two is 2^8
        q = w.view(n, g, GROUP) / torch.exp2(e)[..., None]
        b = q.clamp(-FN_MAX, FN_MAX).to(FN).view(n, k)
    b_scale = (e + 127).to(torch.uint8)
    # e4m3fn -0 (0x80) is NaN as e4m3fnuz; prepare_weight must clear it.
    b.view(torch.uint8).view(-1)[::997] = 0x80
    return a, b, b_scale


def dequant(b, b_scale):
    n, k = b.shape
    vals = b.double().view(n, k // GROUP, GROUP)
    return (vals * torch.exp2(b_scale.double() - 127)[..., None]).view(n, k)


def reference(a, b, b_scale):
    """Float64 sums of the dequantized weights, rounded to BF16."""
    return (a.double() @ dequant(b, b_scale).t()).to(dtypes.bf16)


def run_bf16(a, b_bf16, out):
    torch.mm(a, b_bf16.t(), out=out)


def run_asm(a, b, b_scale, out):
    gemm_a16w8_mxfp8_asm(a, b, b_scale, out)


@benchmark()
def test_gemm_a16w8_mxfp8(m, n, k, mode):
    a, b_fn, s = make_inputs(m, n, k, mode)
    ref = reference(a, b_fn, s).float()
    b, b_scale = gemm_a16w8_mxfp8_prepare_weight(b_fn, s)
    b_bf16 = dequant(b_fn, s).to(dtypes.bf16)
    out_bf16 = torch.empty(m, n, dtype=dtypes.bf16)
    out_asm = torch.full((m, n), float("nan"), dtype=dtypes.bf16)

    ret = {"gfx": get_gfx(), "tuned": is_gemm_a16w8_mxfp8_tuned(m, n, k)}
    # The tensors are passed as arguments, so run_perftest rotates through
    # copies of them and the weights are read from memory, not from cache, as
    # they are in a decode step.
    _, us_bf16 = run_perftest(run_bf16, a, b_bf16, out_bf16)
    _, us_asm = run_perftest(run_asm, a, b, b_scale, out_asm)
    torch.cuda.synchronize()

    if mode == "exact":
        rtol, atol = 0, 0
    else:
        # One BF16 step is at most 2^-8 relative, so rtol 1e-2 lets the fp32
        # sums of the kernel round to the neighbour of the reference.
        rtol, atol = 1e-2, 1e-3
    err = checkAllclose(out_asm.float(), ref, rtol=rtol, atol=atol, msg="mxfp8 asm")

    # Accuracy of both paths against the float64 reference: max |out - ref|
    # relative to max |ref|.
    scale = ref.abs().max().clamp_min(1e-30)
    ret["bf16 max rel err"] = ((out_bf16.float() - ref).abs().max() / scale).item()
    ret["asm max rel err"] = ((out_asm.float() - ref).abs().max() / scale).item()
    weight_bytes = b.numel() + b_scale.numel()
    flops = 2 * m * n * k
    ret["bf16 us"] = us_bf16
    ret["bf16 TFLOPS"] = flops / us_bf16 / 1e6
    ret["bf16 TB/s"] = b_bf16.numel() * 2 / us_bf16 / 1e6
    ret["asm us"] = us_asm
    ret["asm TFLOPS"] = flops / us_asm / 1e6
    ret["asm TB/s"] = weight_bytes / us_asm / 1e6
    ret["asm err"] = err
    return ret


def _expect_raises(fn, needle, desc):
    try:
        fn()
    except RuntimeError as e:
        assert needle in str(e), f"{desc}: unexpected error message: {e}"
        return
    raise AssertionError(f"{desc}: expected RuntimeError containing {needle!r}")


def _run_guard_checks():
    """Inputs the kernels do not take must raise instead of writing garbage."""
    a, b_fn, s = make_inputs(65, 1792, 5120, "randn")
    b, bs = gemm_a16w8_mxfp8_prepare_weight(b_fn, s)
    _expect_raises(lambda: gemm_a16w8_mxfp8_asm(a, b, bs), "no kernel", "M 65")

    a, b_fn, s = make_inputs(4, 1800, 5120, "randn")
    b, bs = gemm_a16w8_mxfp8_prepare_weight(b_fn, s)
    _expect_raises(lambda: gemm_a16w8_mxfp8_asm(a, b, bs), "no kernel", "N 1800")

    a, b_fn, s = make_inputs(4, 1792, 5120, "randn")
    b, bs = gemm_a16w8_mxfp8_prepare_weight(b_fn, s)
    _expect_raises(
        lambda: gemm_a16w8_mxfp8_asm(a, b_fn, s),
        "prepare_weight",
        "unprepared e4m3fn B",
    )
    _expect_raises(
        lambda: gemm_a16w8_mxfp8_asm(a, b_fn.view(torch.uint8), bs),
        "prepare_weight",
        "raw uint8 B",
    )
    if hasattr(torch, "float8_e8m0fnu"):
        _expect_raises(
            lambda: gemm_a16w8_mxfp8_asm(a, b, s.view(torch.float8_e8m0fnu)),
            "prepare_weight",
            "unprepared e8m0 B_scale",
        )
    _expect_raises(
        lambda: gemm_a16w8_mxfp8_asm(a.view(1, 4, 5120), b, bs), "2-D", "3-D A"
    )
    s_big = s.clone()
    s_big[0, 0] = 254
    _expect_raises(
        lambda: gemm_a16w8_mxfp8_prepare_weight(b_fn, s_big), "254", "exponent 254"
    )
    b_nan = b_fn.clone()
    b_nan.view(torch.uint8)[0, 1] = 0x7F
    _expect_raises(
        lambda: gemm_a16w8_mxfp8_prepare_weight(b_nan, s), "NaN", "e4m3fn NaN weight"
    )
    a_wide = torch.empty(4, 2 * 5120, dtype=dtypes.bf16)[:, :5120]
    _expect_raises(
        lambda: gemm_a16w8_mxfp8_asm(a_wide, b, bs), "contiguous", "strided A"
    )
    _expect_raises(
        lambda: gemm_a16w8_mxfp8_asm(a, b, bs[:, :-1].contiguous()),
        "B_scale",
        "short B_scale",
    )
    out = torch.empty(4, 1792, dtype=dtypes.fp16)
    _expect_raises(
        lambda: gemm_a16w8_mxfp8_asm(a, b, bs, out), "must be bf16", "fp16 out"
    )
    _expect_raises(
        lambda: gemm_a16w8_mxfp8_asm(
            a, b, bs, kernelName="a16w8gemm_bf16_mxfp8_tr1_tn1_nw2_sk", splitK=9
        ),
        "splitK",
        "splitK 9",
    )
    a0 = torch.empty(0, 5120, dtype=dtypes.bf16)
    assert gemm_a16w8_mxfp8_asm(a0, b, bs).shape == (0, 1792), "M 0 should be a no-op"

    # Split-K kernels leave their arrival counters at zero, so a CUDA graph of
    # several calls replays to the eager result.
    a, b_fn, s = make_inputs(12, 1152, 5120, "exact", seed=1)
    b, bs = gemm_a16w8_mxfp8_prepare_weight(b_fn, s)
    eager = gemm_a16w8_mxfp8_asm(a, b, bs).clone()
    outs = [torch.empty(12, 1152, dtype=dtypes.bf16) for _ in range(3)]
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for o in outs:
            gemm_a16w8_mxfp8_asm(a, b, bs, o)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for o in outs:
            gemm_a16w8_mxfp8_asm(a, b, bs, o)
    for _ in range(2):
        for o in outs:
            o.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert all(
            torch.equal(o, eager) for o in outs
        ), "graph replay differs from eager"

    # torch.compile traces the op into the custom op aiter::gemm_a16w8_mxfp8_asm_out
    # without graph breaks (fullgraph), with and without out=.
    compiled = torch.compile(
        lambda x: gemm_a16w8_mxfp8_asm(x, b, bs), fullgraph=True, dynamic=False
    )
    assert torch.equal(compiled(a), eager), "torch.compile differs from eager"
    out_c = torch.empty(12, 1152, dtype=dtypes.bf16)
    compiled_out = torch.compile(
        lambda x, o: gemm_a16w8_mxfp8_asm(x, b, bs, o), fullgraph=True, dynamic=False
    )
    compiled_out(a, out_c)
    assert torch.equal(out_c, eager), "torch.compile with out= differs from eager"

    aiter.logger.info(
        "guard checks passed: M 65, N 1800, unprepared weights or scales, 3-D A, "
        "strided A, short B_scale, fp16 out, splitK 9, and exponent 254 or NaN "
        "weights in prepare_weight raise; M 0 is a no-op; graph replay and "
        "torch.compile(fullgraph) equal eager"
    )


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "gemm_a16w8_mxfp8_asm unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-m",
        type=int,
        nargs="*",
        default=[1, 2, 7, 12, 16, 17, 24, 36, 48, 64],
        help="number of tokens, 1 to 64",
    )
    parser.add_argument(
        "-nk",
        type=dtypes.str2tuple,
        nargs="*",
        default=TUNED_SHAPES + UNTUNED_SHAPES,
        help="N,K of the weight, for example -nk 1792,5120",
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
            test_gemm_a16w8_mxfp8(m, n, k, mode)
            for mode in args.mode
            for n, k in args.nk
            for m in args.m
        ]
    )
    aiter.logger.info(
        "gemm_a16w8_mxfp8_asm vs torch.mm in bf16:\n%s", df.to_markdown(index=False)
    )

    _run_guard_checks()

    bad = df[df["asm err"] != 0]
    assert bad.empty, f"results differ from the reference:\n{bad.to_markdown()}"


if __name__ == "__main__":
    main()
