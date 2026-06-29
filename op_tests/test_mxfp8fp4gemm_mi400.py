# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Test + benchmark for gfx1250 MXFP8 x {MXFP8, MXFP4} GEMM (kernarg preload):
#   a8w8 -> gemm_a8w8_mxfp8: D = A @ B^T, A/B mxfp8 e4m3, e8m0 per-32 scales
#   a8w4 -> gemm_a8w4_mxfp8: D = A @ B^T, A mxfp8 e4m3, B mxfp4 e2m1, e8m0 per-32
#
# Modes (mirror the POC run.sh / run_compute.sh):
#   func    : correctness only (golden check vs a torch f32 reference)
#   perf    : correctness + latency/TFLOPS/TB-s summary table
#   profile : perf + a torch profiler trace dumped under ./aiter_logs
#
# Shape constraints (persistent + cluster, no partial-tile masking):
#   256x256 tile (cluster 4x4): M % 1024 == 0, N % 1024 == 0
#   64x512  tile (cluster 1x4): M % 64   == 0, N % 2048 == 0
#   all:    K % 128 == 0
# The .cu heuristic picks whichever registered tile fits the shape.

import argparse

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.ops.shuffle import (
    shuffle_mxfp8fp4_a,
    shuffle_mxfp8fp4_b,
    shuffle_mxfp8fp4_scale,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility import fp4_utils
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 1000)

MX_SCALE_BLOCK = 32


def _rand_mxfp8(rows: int, k: int) -> torch.Tensor:
    # Random mxfp8 (e4m3) activations/weights, exactly representable after cast.
    return (torch.randn((rows, k), dtype=torch.float32) * 2.0).to(torch.float8_e4m3fn)


def _rand_fp4_packed(rows: int, k: int) -> torch.Tensor:
    # Packed mxfp4: each uint8 carries two e2m1 nibbles -> shape (rows, k/2).
    assert k % 2 == 0
    return torch.randint(0, 256, (rows, k // 2), dtype=torch.uint8)


def _rand_e8m0_scale(rows: int, k: int) -> torch.Tensor:
    # e8m0 = unsigned 8-bit exponent, bias 127 (0x7F == 1.0). Keep the dynamic
    # range modest (exponent in [-2, 2]) to match the POC's validated init.
    return torch.randint(125, 130, (rows, k // MX_SCALE_BLOCK), dtype=torch.uint8)


def _fp8_to_f32(x_fp8: torch.Tensor) -> torch.Tensor:
    return x_fp8.to(torch.float32)


def _ref(intype, A, B, sA, sB, M, N):
    A_f32 = _fp8_to_f32(A)[:M]
    if intype == "a8w4":
        B_f32 = fp4_utils.mxfp4_to_f32(B)[:N]
    else:
        B_f32 = _fp8_to_f32(B)[:N]
    sA_f = fp4_utils.e8m0_to_f32(sA).repeat_interleave(MX_SCALE_BLOCK, dim=1)
    sB_f = fp4_utils.e8m0_to_f32(sB).repeat_interleave(MX_SCALE_BLOCK, dim=1)
    return (A_f32 * sA_f) @ (B_f32 * sB_f).T


def _prep(intype: str, M: int, N: int, K: int, apre: int):
    """Build raw + shuffled device tensors and the f32 golden reference."""
    A = _rand_mxfp8(M, K)
    if intype == "a8w4":
        B = _rand_fp4_packed(N, K)
    else:
        B = _rand_mxfp8(N, K)
    sA, sB = _rand_e8m0_scale(M, K), _rand_e8m0_scale(N, K)

    ref = _ref(intype, A, B, sA, sB, M, N).to(dtypes.bf16)

    inp = dict(
        A=shuffle_mxfp8fp4_a(A) if apre else A,  # B always preshuffled, A per `apre`
        B=shuffle_mxfp8fp4_b(B),
        sA=shuffle_mxfp8fp4_scale(sA),
        sB=shuffle_mxfp8fp4_scale(sB),
    )
    return inp, ref


def _run_kernel(intype, inp, apre, kernelName, num_iters, needTrace):
    fn = aiter.gemm_a8w4_mxfp8 if intype == "a8w4" else aiter.gemm_a8w8_mxfp8
    return run_perftest(
        fn,
        inp["A"],
        inp["B"],
        inp["sA"],
        inp["sB"],
        dtype=dtypes.bf16,
        a_preshuffle=bool(apre),
        kernelName=kernelName,
        num_iters=num_iters,
        needTrace=needTrace,
    )


@benchmark()
def test_gemm(intype, M, N, K, apre, kernelName="", mode="perf"):
    if get_gfx() not in ["gfx1250"]:
        return None

    assert K % MX_SCALE_BLOCK == 0, f"K must be a multiple of {MX_SCALE_BLOCK}"

    inp, ref = _prep(intype, M, N, K, apre)
    needTrace = mode == "profile"
    num_iters = 5 if mode == "func" else 101
    out, us = _run_kernel(intype, inp, apre, kernelName, num_iters, needTrace)

    err = checkAllclose(ref, out, rtol=1e-1, atol=1.0, msg=f"{intype} asm")

    io_bytes = (
        inp["A"].nbytes
        + inp["B"].nbytes
        + inp["sA"].nbytes
        + inp["sB"].nbytes
        + out.nbytes
    )
    ret = {
        "intype": intype,
        "apre": apre,
        "us": round(us, 2),
        "TFLOPS": round(M * N * K * 2 / us / 1e6, 1),
        "TB/s": round(io_bytes / us / 1e6, 2),
        "err": err,
    }
    if needTrace:
        gpu_id = torch.cuda.current_device()
        ret["trace"] = f"./aiter_logs/gpu_id_{gpu_id}"
    return ret


def _str2tuple(s: str):
    return tuple(int(x) for x in s.split(","))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Test/benchmark gfx1250 MXFP8x{FP8,FP4} (a8w8 / a8w4) ASM kernels",
    )
    parser.add_argument(
        "--mode",
        choices=["func", "perf", "profile", "all"],
        default="perf",
        help="func=acc only, perf=acc+timing, profile=perf+trace, all=perf",
    )
    parser.add_argument("--intype", choices=["a8w8", "a8w4", "both"], default="both")
    parser.add_argument(
        "--apre",
        type=int,
        choices=[0, 1],
        default=1,
        help="A-preshuffle: 1 to preshuffle A, 0 to send row-major",
    )
    # Defaults satisfy at least one registered tile each:
    #   (1024,2048,1024),(4096,8192,1024) -> 256x256; (512,16384,1024) -> 64x512.
    parser.add_argument(
        "-s",
        "--shape",
        type=_str2tuple,
        nargs="*",
        default=[(1024, 2048, 1024), (4096, 8192, 1024), (512, 16384, 1024)],
        help="(M,N,K) tuples, e.g. -s 1024,2048,1024 512,16384,1024",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="",
        help="Force a specific kernelName (bypass heuristic)",
    )
    args = parser.parse_args()

    intypes = ["a8w8", "a8w4"] if args.intype == "both" else [args.intype]
    rows = []
    for it in intypes:
        for M, N, K in args.shape:
            ret = test_gemm(
                it, M, N, K, args.apre, kernelName=args.kernel, mode=args.mode
            )
            if ret is not None:
                rows.append(ret)

    if rows and args.mode != "func":
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "mxfp8fp4gemm_mi400 %s summary (markdown):\n%s",
            args.mode,
            df.to_markdown(index=False),
        )
        if args.mode == "profile":
            aiter.logger.info("profiler traces written under ./aiter_logs/")
