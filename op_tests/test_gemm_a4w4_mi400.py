# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# A4W4 (F4GEMM) test/benchmark for gfx1250 (mi400), modeled on test_gemm_a4w4.py.
# Drives the unified ``aiter.gemm_a4w4`` API (which dispatches to the mi400
# F4GEMM path on gfx1250) and reuses the shared harness (run_torch reference,
# run_perftest, benchmark table, checkAllclose, fp4_utils).
#
#   MXFP4 (intype=7): reuses the per_1x32 e8m0 quant, gfx1250 weight/scale shuffle
#   NVFP4 (intype=8): e4m3 per-16 scales + per-tensor global scales
#
# Modes (mirrors the shader-side run.sh): func | perf | profile | all

import argparse

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

MXFP4_SCALE_BLOCK = 32
NVFP4_SCALE_BLOCK = 16


def _e4m3_to_f32(s: torch.Tensor) -> torch.Tensor:
    return s.view(torch.float8_e4m3fn).to(torch.float32)


def run_torch_mxfp4(xq, wq, xs, ws, dtype):
    x_f32 = fp4_utils.mxfp4_to_f32(xq)
    w_f32 = fp4_utils.mxfp4_to_f32(wq)
    xs = fp4_utils.e8m0_to_f32(xs).repeat_interleave(MXFP4_SCALE_BLOCK, dim=1)
    ws = fp4_utils.e8m0_to_f32(ws).repeat_interleave(MXFP4_SCALE_BLOCK, dim=1)
    return ((x_f32 * xs) @ (w_f32 * ws).T).to(dtype)


def run_torch_nvfp4(xq, wq, xs, ws, gA, gB, dtype):
    x_f32 = fp4_utils.mxfp4_to_f32(xq)
    w_f32 = fp4_utils.mxfp4_to_f32(wq)
    xs = _e4m3_to_f32(xs).repeat_interleave(NVFP4_SCALE_BLOCK, dim=1)
    ws = _e4m3_to_f32(ws).repeat_interleave(NVFP4_SCALE_BLOCK, dim=1)
    return (float(gA) * float(gB) * (x_f32 * xs) @ (w_f32 * ws).T).to(dtype)


def _prep_mxfp4(M, N, K, apre, dtype):
    # Reuse the per_1x32 e8m0 quant (same block-32 scales the mi400 mxfp4 kernel
    # expects); only the shuffle differs from the gfx950 path.
    quant = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    x = torch.randn((M, K), dtype=dtype)
    w = torch.randn((N, K), dtype=dtype)
    xq, xs = quant(x, shuffle=False)  # packed fp4 [*, K/2] + e8m0 [*, K/32]
    wq, ws = quant(w, shuffle=False)
    xq, wq = xq.view(torch.uint8), wq.view(torch.uint8)
    xs, ws = xs.view(torch.uint8), ws.view(torch.uint8)
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


def _prep_nvfp4(M, N, K, apre, dtype):
    # No per_1x16 e4m3 quant helper yet: random fp4 + e4m3 scales + global scales.
    xq = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8)
    wq = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8)
    xs = torch.randint(0x20, 0x50, (M, K // NVFP4_SCALE_BLOCK), dtype=torch.uint8)
    ws = torch.randint(0x20, 0x50, (N, K // NVFP4_SCALE_BLOCK), dtype=torch.uint8)
    gA = gB = 0.5
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


@benchmark()
def test_gemm(intype, M, N, K, apre, dtype=dtypes.bf16, mode="perf", tile="auto"):
    if get_gfx() not in ["gfx1250"]:
        return None

    block = MXFP4_SCALE_BLOCK if intype == "mxfp4" else NVFP4_SCALE_BLOCK
    assert K % block == 0, f"K must be a multiple of {block}"
    inp, ref = (_prep_mxfp4 if intype == "mxfp4" else _prep_nvfp4)(M, N, K, apre, dtype)

    needTrace = mode == "profile"
    num_iters = 5 if mode == "func" else 101

    if tile == "auto":
        # Unified API: tile picked by the C++ heuristic. NVFP4 global scales as tensors.
        gA = None if inp["gA"] is None else torch.tensor(inp["gA"], dtype=torch.float32)
        gB = None if inp["gB"] is None else torch.tensor(inp["gB"], dtype=torch.float32)
        out, us = run_perftest(
            aiter.gemm_a4w4,
            inp["A"],
            inp["B"],
            inp["sA"],
            inp["sB"],
            dtype=dtype,
            apreshuffle=bool(apre),
            bpreshuffle=True,
            global_A_scale=gA,
            global_B_scale=gB,
            num_iters=num_iters,
            needTrace=needTrace,
        )
        which = "gemm_a4w4"
    else:
        # Force a specific tile via the low-level asm entry (kernel per intype).
        which = f"f4gemm_{intype}_{tile}_apre{apre}"
        if intype == "nvfp4":
            out, us = run_perftest(
                aiter.gemm_nvfp4_asm,
                inp["A"],
                inp["B"],
                inp["sA"],
                inp["sB"],
                inp["gA"],
                inp["gB"],
                dtype=dtype,
                a_preshuffle=bool(apre),
                kernelName=which,
                num_iters=num_iters,
                needTrace=needTrace,
            )
        else:
            out, us = run_perftest(
                aiter.gemm_mxfp4_asm,
                inp["A"],
                inp["B"],
                inp["sA"],
                inp["sB"],
                dtype=dtype,
                a_preshuffle=bool(apre),
                kernelName=which,
                num_iters=num_iters,
                needTrace=needTrace,
            )
    err = checkAllclose(ref, out, rtol=1e-1, atol=1.0, msg=f"{intype} {which}")

    io_bytes = (
        inp["A"].nbytes
        + inp["B"].nbytes
        + inp["sA"].nbytes
        + inp["sB"].nbytes
        + out.nbytes
    )
    ret = {
        "intype": intype,
        "tile": tile,
        "apre": apre,
        "us": round(us, 2),
        "TFLOPS": round(M * N * K * 2 / us / 1e6, 1),
        "TB/s": round(io_bytes / us / 1e6, 2),
        "err": err,
    }
    if needTrace:
        ret["trace"] = f"./aiter_logs/gpu_id_{torch.cuda.current_device()}"
    return ret


def _str2tuple(s: str):
    return tuple(int(x) for x in s.split(","))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Test/benchmark gfx1250 A4W4 (F4GEMM) via the unified gemm_a4w4 API",
    )
    parser.add_argument(
        "--mode",
        choices=["func", "perf", "profile", "all"],
        default="perf",
        help="func=acc only, perf=acc+timing, profile=perf+trace, all=perf",
    )
    parser.add_argument("--intype", choices=["mxfp4", "nvfp4", "both"], default="nvfp4")
    parser.add_argument(
        "--tile",
        choices=["auto", "256x256"],
        default="auto",
        help="auto=unified gemm_a4w4 (C++ heuristic); 256x256 forces that tile",
    )
    parser.add_argument(
        "--apre",
        type=int,
        choices=[0, 1],
        default=1,
        help="A-preshuffle: 1 to preshuffle A, 0 to send row-major",
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
    # cluster(4x4)+persistent friendly for the 256x256 tile: M%1024, N%1024.
    parser.add_argument(
        "-mnk",
        "--shape",
        type=_str2tuple,
        nargs="*",
        default=[(1024, 2048, 2048), (2048, 4096, 4096)],
        help="(M,N,K) tuples, e.g. -mnk 1024,2048,2048 2048,4096,4096",
    )
    args = parser.parse_args()

    intypes = ["mxfp4", "nvfp4"] if args.intype == "both" else [args.intype]
    rows = []
    for dtype in args.dtype:
        for it in intypes:
            for M, N, K in args.shape:
                ret = test_gemm(
                    it, M, N, K, args.apre, dtype=dtype, mode=args.mode, tile=args.tile
                )
                if ret is not None and "us" in ret:
                    rows.append(ret)

    if rows and args.mode != "func":
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "gemm_a4w4_mi400 %s summary (markdown):\n%s",
            args.mode,
            df.to_markdown(index=False),
        )
        if args.mode == "profile":
            aiter.logger.info("profiler traces written under ./aiter_logs/")
