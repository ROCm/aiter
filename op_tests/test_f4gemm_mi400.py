# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Test + benchmark for gfx1250 F4GEMM (preload SGPR mode):
#   gemm_mxfp4_asm: D = A * B^T with e8m0 per-32 scales (intype=7)
#   gemm_nvfp4_asm: D = A * B^T with e4m3 per-16 scales + GlobalScaleA/B (intype=8)
#
# Modes (mirrors the shader-side run.sh): func | perf | profile | all
#   func    : correctness only (golden check vs a torch f32 reference)
#   perf    : correctness + latency/TFLOPS/TB-s summary table
#   profile : perf + a torch profiler trace dumped under ./aiter_logs

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


def _e4m3_to_f32(scale_u8: torch.Tensor) -> torch.Tensor:
    return scale_u8.view(torch.float8_e4m3fn).to(torch.float32)


def _rand_fp4_packed(rows: int, k: int) -> torch.Tensor:
    # Packed fp4: each uint8 carries two 4-bit nibbles -> shape (rows, k/2).
    assert k % 2 == 0
    return torch.randint(0, 256, (rows, k // 2), dtype=torch.uint8)


def _rand_e8m0_scale(rows: int, k: int) -> torch.Tensor:
    # e8m0 = unsigned 8-bit exponent with bias 127. 0x7F = 1.0.
    return torch.randint(0x70, 0x88, (rows, k // MXFP4_SCALE_BLOCK), dtype=torch.uint8)


def _rand_e4m3_scale(rows: int, k: int) -> torch.Tensor:
    # Small positive e4m3 values; stay away from the NaN encoding (0x7F / 0xFF).
    return torch.randint(0x20, 0x50, (rows, k // NVFP4_SCALE_BLOCK), dtype=torch.uint8)


def _mxfp4_ref(A_fp4, B_fp4, sA_e8m0, sB_e8m0, M, N):
    A_f32 = fp4_utils.mxfp4_to_f32(A_fp4)[:M]
    B_f32 = fp4_utils.mxfp4_to_f32(B_fp4)[:N]
    sA = fp4_utils.e8m0_to_f32(sA_e8m0).repeat_interleave(MXFP4_SCALE_BLOCK, dim=1)
    sB = fp4_utils.e8m0_to_f32(sB_e8m0).repeat_interleave(MXFP4_SCALE_BLOCK, dim=1)
    return (A_f32 * sA) @ (B_f32 * sB).T


def _nvfp4_ref(A_fp4, B_fp4, sA_e4m3, sB_e4m3, gA, gB, M, N):
    A_f32 = fp4_utils.mxfp4_to_f32(A_fp4)[:M]
    B_f32 = fp4_utils.mxfp4_to_f32(B_fp4)[:N]
    sA = _e4m3_to_f32(sA_e4m3).repeat_interleave(NVFP4_SCALE_BLOCK, dim=1)
    sB = _e4m3_to_f32(sB_e4m3).repeat_interleave(NVFP4_SCALE_BLOCK, dim=1)
    return float(gA) * float(gB) * (A_f32 * sA) @ (B_f32 * sB).T


def _prep(intype: str, M: int, N: int, K: int, apre: int):
    """Build raw + shuffled device tensors and the f32 golden reference."""
    is_mx = intype == "mxfp4"
    A = _rand_fp4_packed(M, K)
    B = _rand_fp4_packed(N, K)
    gA = gB = 0.5
    if is_mx:
        sA, sB = _rand_e8m0_scale(M, K), _rand_e8m0_scale(N, K)
        ref = _mxfp4_ref(A, B, sA, sB, M, N).to(dtypes.bf16)
    else:
        sA, sB = _rand_e4m3_scale(M, K), _rand_e4m3_scale(N, K)
        ref = _nvfp4_ref(A, B, sA, sB, gA, gB, M, N).to(dtypes.bf16)

    intype_id = 7 if is_mx else 8
    inp = dict(
        A=shuffle_weight_f4(A) if apre else A,  # B always preshuffled; A per `apre`
        B=shuffle_weight_f4(B),
        sA=shuffle_scale_f4(sA, intype_id),
        sB=shuffle_scale_f4(sB, intype_id),
        gA=gA,
        gB=gB,
    )
    return inp, ref


def _run_kernel(intype, inp, apre, kernelName, num_iters, needTrace):
    """Time the asm kernel via run_perftest; returns (out, us)."""
    if intype == "mxfp4":
        return run_perftest(
            aiter.gemm_mxfp4_asm,
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
    return run_perftest(
        aiter.gemm_nvfp4_asm,
        inp["A"],
        inp["B"],
        inp["sA"],
        inp["sB"],
        inp["gA"],
        inp["gB"],
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

    scale_block = MXFP4_SCALE_BLOCK if intype == "mxfp4" else NVFP4_SCALE_BLOCK
    assert K % scale_block == 0, f"K must be a multiple of {scale_block}"

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
        description="Test/benchmark gfx1250 F4GEMM (MXFP4 / NVFP4) ASM kernels",
    )
    parser.add_argument(
        "--mode",
        choices=["func", "perf", "profile", "all"],
        default="perf",
        help="func=acc only, perf=acc+timing, profile=perf+trace, all=perf",
    )
    # mxfp4 and nvfp4 variants are registered in f4gemm.csv; each needs its
    # matching .co built (build_co.sh) before --intype can select it.
    parser.add_argument("--intype", choices=["mxfp4", "nvfp4", "both"], default="nvfp4")
    parser.add_argument(
        "--apre",
        type=int,
        choices=[0, 1],
        default=1,
        help="A-preshuffle: 1 to preshuffle A, 0 to send row-major",
    )
    # Defaults are cluster(4x4)+persistent friendly for the registered tiles:
    # 256x256 needs M%1024,N%1024; 128x512 needs M%512,N%2048.
    parser.add_argument(
        "-s",
        "--shape",
        type=_str2tuple,
        nargs="*",
        default=[(1024, 2048, 2048), (2048, 4096, 4096)],
        help="(M,N,K) tuples, e.g. -s 1024,2048,2048 2048,4096,4096",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="",
        help="Force a specific kernelName (bypass heuristic)",
    )
    args = parser.parse_args()

    intypes = ["mxfp4", "nvfp4"] if args.intype == "both" else [args.intype]
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
            "f4gemm_mi400 %s summary (markdown):\n%s",
            args.mode,
            df.to_markdown(index=False),
        )
        if args.mode == "profile":
            aiter.logger.info("profiler traces written under ./aiter_logs/")
