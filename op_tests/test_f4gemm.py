# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# A4W4 (F4GEMM) test/benchmark for gfx1250, modeled on test_gemm_a4w4.py
# and the aiter-op-test standard (candidates dict + run_perftest loop, a torch
# reference that is only compared (never timed/tabled), TFLOPS + TB/s per
# candidate, one markdown summary table, and a __main__ guard).
#
# Two candidates per (intype, shape, apre) row -- both resolve to the same
# gfx1250 F4GEMM .co but exercise different entrypoints:
#   gemm_a4w4 : the unified API the model calls (C++ heuristic picks the tile)
#   asm       : the low-level asm entry with the tile kernel forced by name
#
#   MXFP4 (intype=mxfp4): per_1x32 e8m0 scales, gfx1250 weight/scale shuffle
#   NVFP4 (intype=nvfp4): e4m3 per-16 scales + per-tensor global scales

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight_f4, shuffle_scale_f4
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility import fp4_utils
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 1000)

SUPPORTED_GFX = ["gfx1250"]  # gfx1250-only F4GEMM (preload SGPR) path
MXFP4_SCALE_BLOCK = 32
NVFP4_SCALE_BLOCK = 16


def _e4m3_to_f32(s: torch.Tensor) -> torch.Tensor:
    return s.view(torch.float8_e4m3fn).to(torch.float32)


def run_torch_mxfp4(xq, wq, xs, ws, dtype):
    # Reference only: fp32 math, cast back. Not timed, not in the table.
    x_f32 = fp4_utils.mxfp4_to_f32(xq)
    w_f32 = fp4_utils.mxfp4_to_f32(wq)
    xs = fp4_utils.e8m0_to_f32(xs).repeat_interleave(MXFP4_SCALE_BLOCK, dim=1)
    ws = fp4_utils.e8m0_to_f32(ws).repeat_interleave(MXFP4_SCALE_BLOCK, dim=1)
    return ((x_f32 * xs) @ (w_f32 * ws).T).to(dtype)


def run_torch_nvfp4(xq, wq, xs, ws, gA, gB, dtype):
    # Reference only: fp32 math, cast back. Not timed, not in the table.
    x_f32 = fp4_utils.mxfp4_to_f32(xq)
    w_f32 = fp4_utils.mxfp4_to_f32(wq)
    xs = _e4m3_to_f32(xs).repeat_interleave(NVFP4_SCALE_BLOCK, dim=1)
    ws = _e4m3_to_f32(ws).repeat_interleave(NVFP4_SCALE_BLOCK, dim=1)
    return (float(gA) * float(gB) * (x_f32 * xs) @ (w_f32 * ws).T).to(dtype)


def _prep_mxfp4(M, N, K, apre, dtype, init):
    if init == "random":
        # Reuse the per_1x32 e8m0 quant (same block-32 scales the mxfp4
        # kernel expects); only the shuffle differs from the gfx950 path.
        quant = aiter.get_triton_quant(aiter.QuantType.per_1x32)
        x = torch.randn((M, K), dtype=dtype)
        w = torch.randn((N, K), dtype=dtype)
        xq, xs = quant(x, shuffle=False)  # packed fp4 [*, K/2] + e8m0 [*, K/32]
        wq, ws = quant(w, shuffle=False)
        xq, wq = xq.view(torch.uint8), wq.view(torch.uint8)
        xs, ws = xs.view(torch.uint8), ws.view(torch.uint8)
    else:
        # Constant init, mirroring f4gemm.cpp data_init=0: A=0x22, B=0x33, and a
        # neutral e8m0 scale 0x7F (exp 0 -> 2^0 = 1.0). Stable/deterministic for perf.
        xq = torch.full((M, K // 2), 0x22, dtype=torch.uint8)
        wq = torch.full((N, K // 2), 0x33, dtype=torch.uint8)
        xs = torch.full((M, K // MXFP4_SCALE_BLOCK), 0x7F, dtype=torch.uint8)
        ws = torch.full((N, K // MXFP4_SCALE_BLOCK), 0x7F, dtype=torch.uint8)
    ref = run_torch_mxfp4(xq, wq, xs, ws, dtype)
    inp = dict(
        A=shuffle_weight_f4(xq) if apre else xq,
        B=shuffle_weight_f4(wq),
        sA=shuffle_scale_f4(xs, 7),
        sB=shuffle_scale_f4(ws, 7),
        gA=None,
        gB=None,
    )
    return inp, ref


def _prep_nvfp4(M, N, K, apre, dtype, init):
    if init == "random":
        # No per_1x16 e4m3 quant helper yet: random fp4 + e4m3 scales + globals.
        xq = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8)
        wq = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8)
        xs = torch.randint(0x20, 0x50, (M, K // NVFP4_SCALE_BLOCK), dtype=torch.uint8)
        ws = torch.randint(0x20, 0x50, (N, K // NVFP4_SCALE_BLOCK), dtype=torch.uint8)
        gA = gB = 0.5
    else:
        # Constant init, mirroring f4gemm.cpp data_init=0: A=0x22, B=0x33, neutral
        # e4m3 scale 0x38 (exp 7 = bias -> 1.0) and unit global scales.
        xq = torch.full((M, K // 2), 0x22, dtype=torch.uint8)
        wq = torch.full((N, K // 2), 0x33, dtype=torch.uint8)
        xs = torch.full((M, K // NVFP4_SCALE_BLOCK), 0x38, dtype=torch.uint8)
        ws = torch.full((N, K // NVFP4_SCALE_BLOCK), 0x38, dtype=torch.uint8)
        gA = gB = 1.0
    ref = run_torch_nvfp4(xq, wq, xs, ws, gA, gB, dtype)
    inp = dict(
        A=shuffle_weight_f4(xq) if apre else xq,
        B=shuffle_weight_f4(wq),
        sA=shuffle_scale_f4(xs, 8),
        sB=shuffle_scale_f4(ws, 8),
        gA=gA,  # NVFP4 per-tensor global scales (floats)
        gB=gB,
    )
    return inp, ref


@benchmark()  # (intype, M, N, K, apre, init, dtype) become the table's left columns
def test_gemm(intype, M, N, K, apre, init, dtype=dtypes.bf16):
    block = MXFP4_SCALE_BLOCK if intype == "mxfp4" else NVFP4_SCALE_BLOCK
    assert K % block == 0, f"K must be a multiple of {block}"
    prep = _prep_mxfp4 if intype == "mxfp4" else _prep_nvfp4
    inp, ref = prep(M, N, K, apre, dtype, init)

    def run_unified():
        # The path the model runs: unified API, C++ heuristic picks the tile.
        # NVFP4 per-tensor global scales are passed as tensors here.
        gA = None if inp["gA"] is None else torch.tensor(inp["gA"], dtype=torch.float32)
        gB = None if inp["gB"] is None else torch.tensor(inp["gB"], dtype=torch.float32)
        return aiter.gemm_a4w4(
            inp["A"],
            inp["B"],
            inp["sA"],
            inp["sB"],
            dtype=dtype,
            apreshuffle=bool(apre),
            bpreshuffle=True,
            global_A_scale=gA,
            global_B_scale=gB,
        )

    def run_asm():
        # See hsa/gfx1250/f4gemm/f4gemm.csv.
        pre = "ABpreShuffle" if apre else "BpreShuffle"
        base = f"f4gemm_bf16_{intype}_{pre}_256x256_4x4_ps"
        knl = f"_ZN5aiter{len(base)}{base}E"
        if intype == "nvfp4":
            return aiter.gemm_nvfp4_asm(
                inp["A"],
                inp["B"],
                inp["sA"],
                inp["sB"],
                inp["gA"],
                inp["gB"],
                dtype=dtype,
                a_preshuffle=bool(apre),
                kernelName=knl,
            )
        return aiter.gemm_mxfp4_asm(
            inp["A"],
            inp["B"],
            inp["sA"],
            inp["sB"],
            dtype=dtype,
            a_preshuffle=bool(apre),
            kernelName=knl,
        )

    candidates = {"gemm_a4w4": run_unified}
    candidates["asm"] = run_asm

    flops = 2 * M * N * K
    nbytes = (
        inp["A"].nbytes
        + inp["B"].nbytes
        + inp["sA"].nbytes
        + inp["sB"].nbytes
        + M * N * dtype.itemsize  # bf16 output
    )

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        err = checkAllclose(ref, out, rtol=1e-1, atol=1.0, msg=f"{intype} {name}")
        ret[f"{name} us"] = round(us, 2)
        ret[f"{name} TFLOPS"] = round(flops / us / 1e6, 1)
        ret[f"{name} TB/s"] = round(nbytes / us / 1e6, 2)
        ret[f"{name} err"] = err
    return ret


def main():
    # Whole-op arch gate goes HERE: @benchmark always returns the call-args dict,
    # so an in-fn return would still emit an args-only NaN row.
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "gemm_a4w4 (F4GEMM) unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Test/benchmark gfx1250 A4W4 (F4GEMM) via the unified gemm_a4w4 API",
    )
    parser.add_argument(
        "--intype",
        nargs="*",
        choices=["mxfp4", "nvfp4"],
        default=["mxfp4", "nvfp4"],
        help="fp4 input format(s) to sweep, e.g. --intype nvfp4",
    )
    parser.add_argument(
        "--apre",
        type=int,
        nargs="*",
        choices=[0, 1],
        default=[1],
        help="A-preshuffle sweep list: 1 preshuffles A, 0 sends it row-major",
    )
    parser.add_argument(
        "--init",
        nargs="*",
        choices=["constant", "random"],
        default=["constant", "random"],
        help="input-data init mode(s)"
        "  constant = A=0x22,B=0x33,neutral scales"
        "  random   = varied fp4 matrices + random scales",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        choices=[dtypes.d_dtypes["bf16"]],
        metavar="{bf16}",
        default=[dtypes.d_dtypes["bf16"]],
        help="output dtype, e.g. -d bf16",
    )
    parser.add_argument(
        "-mnk",
        "--shape",
        type=dtypes.str2tuple,
        nargs="*",
        # cluster(4x4)+persistent friendly for the 256x256 tile: M%1024, N%1024.
        default=[(16384, 16384, 16384)],
        help="(M,N,K) tuples, e.g. -mnk 2048,2048,2048 16384,16384,16384",
    )
    args = parser.parse_args()

    for dtype in args.dtype:  # one table per output dtype
        rows = [
            test_gemm(intype, M, N, K, apre, init, dtype=dtype)
            for intype, apre, init, (M, N, K) in itertools.product(
                args.intype, args.apre, args.init, args.shape
            )
        ]
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "gemm_a4w4 (F4GEMM) summary (markdown):\n%s", df.to_markdown(index=False)
        )


if __name__ == "__main__":
    main()
