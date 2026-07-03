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
# Shape constraints (persistent + cluster):
#   256x256 tile (cluster 4x4): M % 1024 == 0, N % 1024 == 0
#   64x512  tile (cluster 1x4): N % 2048 == 0  (M unconstrained: partial M-tile
#                               allowed since M is not clustered)
#   all:    K % 128 == 0
# The .cu heuristic picks whichever registered tile fits the shape.

import argparse
import itertools

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
SUPPORTED_GFX = ["gfx1250"]  # ASM kernels are gfx1250-only (kernarg preload)


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


def _ref(intype, A, B, sA, sB, M, N):
    # Reference only: fp32 math, cast back. Not timed, not in the table.
    A_f32 = A.to(torch.float32)[:M]
    if intype == "a8w4":
        B_f32 = fp4_utils.mxfp4_to_f32(B)[:N]
    else:
        B_f32 = B.to(torch.float32)[:N]
    sA_f = fp4_utils.e8m0_to_f32(sA).repeat_interleave(MX_SCALE_BLOCK, dim=1)
    sB_f = fp4_utils.e8m0_to_f32(sB).repeat_interleave(MX_SCALE_BLOCK, dim=1)
    return (A_f32 * sA_f) @ (B_f32 * sB_f).T


def _const_mxfp8(rows: int, k: int, val: float) -> torch.Tensor:
    # Constant mxfp8 (e4m3): a single representable value, deterministic for perf.
    return torch.full((rows, k), val, dtype=torch.float32).to(torch.float8_e4m3fn)


def _prep(intype: str, M: int, N: int, K: int, apre: int, init: str):
    """Build raw + shuffled device tensors and the f32 golden reference.

    init="random"   : randn-derived mxfp8/mxfp4 + random e8m0 scales (varied).
                      Mirrors the POC sin_cos_init=1 (varied, non-zero) pattern.
    init="constant" : mirrors the POC sin_cos_init=10 -- every A/B element = 0.5
                      (exact in e4m3 and e2m1) with a neutral e8m0 scale 0x7F
                      (exp 0 -> 2^0 = 1.0). Deterministic/stable for perf.
    """
    if init == "constant":
        A = _const_mxfp8(M, K, 0.5)
        if intype == "a8w4":
            B = torch.full((N, K // 2), 0x11, dtype=torch.uint8)  # e2m1 nibble 0.5
        else:
            B = _const_mxfp8(N, K, 0.5)
        sA = torch.full((M, K // MX_SCALE_BLOCK), 0x7F, dtype=torch.uint8)
        sB = torch.full((N, K // MX_SCALE_BLOCK), 0x7F, dtype=torch.uint8)
    else:
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


@benchmark()  # intype, M, N, K, apre, init, ... become the table's left columns
def test_gemm(intype, M, N, K, apre, init="random", mode="perf"):
    assert K % MX_SCALE_BLOCK == 0, f"K must be a multiple of {MX_SCALE_BLOCK}"

    inp, ref = _prep(intype, M, N, K, apre, init)
    needTrace = mode == "profile"
    num_iters = 5 if mode == "func" else 101

    # Single ASM kernel under test, dispatched by intype. Faithful to the model
    # call: the wrapper allocates its own bf16 output (no preallocated buffer).
    fn = aiter.gemm_a8w4_mxfp8 if intype == "a8w4" else aiter.gemm_a8w8_mxfp8
    candidates = {
        "asm": lambda: fn(
            inp["A"],
            inp["B"],
            inp["sA"],
            inp["sB"],
            dtype=dtypes.bf16,
            a_preshuffle=bool(
                apre
            ),  # kernelName omitted -> .cu heuristic picks the tile
        ),
    }

    flops = 2 * M * N * K
    in_bytes = inp["A"].nbytes + inp["B"].nbytes + inp["sA"].nbytes + inp["sB"].nbytes

    ret = {"gfx": get_gfx()}
    for name, cand in candidates.items():
        out, us = run_perftest(cand, num_iters=num_iters, needTrace=needTrace)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-1,
            atol=1.0,
            msg=f"{intype} {name}",
        )
        io_bytes = in_bytes + out.nbytes
        ret[f"{name} us"] = round(us, 2)
        ret[f"{name} TFLOPS"] = round(flops / us / 1e6, 1)
        ret[f"{name} TB/s"] = round(io_bytes / us / 1e6, 2)
        ret[f"{name} err"] = err
        if needTrace:
            ret[f"{name} trace"] = f"./aiter_logs/gpu_id_{torch.cuda.current_device()}"
    return ret


def main():
    # Whole-op arch gate goes HERE, not inside test_gemm: @benchmark always
    # returns the call-args dict, so an in-fn `return` still emits an args-only row.
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "mxfp8fp4 gemm (a8w8/a8w4) unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Test/benchmark gfx1250 MXFP8x{FP8,FP4} (a8w8 / a8w4) ASM kernels",
    )
    parser.add_argument(
        "--mode",
        choices=["func", "perf", "profile"],
        default="perf",
        help="func=acc only, perf=acc+timing, profile=perf+trace",
    )
    parser.add_argument(
        "--intype",
        nargs="*",
        choices=["a8w8", "a8w4"],
        default=["a8w8", "a8w4"],
        help="input-type sweep list (a8w8 and/or a8w4)",
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
        "  constant = A=1.0, B=2.0/0x33, neutral e8m0 scales (deterministic)"
        "  random   = randn-derived mxfp8/mxfp4 + random e8m0 scales",
    )
    # Default (M,N,K) = union of the POC perf matrices (run.sh + run_compute.sh).
    # The .cu heuristic picks the registered tile that fits each shape.
    #   run_compute.sh perf -> 256x256 tile (cluster 4x4, BS=1, init 10):
    #     fp8: (16384,16384,8192) (16384,16384,16384) (8192,8192,16384)
    #     fp4: (16384,16384,16384) (16384,16384,32768) (8192,8192,32768) (8192,8192,65536)
    #   run.sh perf -> 64x512 tile (cluster 1x4, init {10,1}), BS=64 folded into
    #   M (M_eff = M*BS; aiter forces batch_size=1, so M*BS is FLOP-equivalent):
    #     fp8 (K=8192):  (1024,16384,8192)  (128,16384,8192)    # M=16*64, 2*64
    #     fp4 (K=16384): (1024,16384,16384) (128,16384,16384)   # M=16*64, 2*64
    # intype x shape is a full product, so each shape is run for both a8w8/a8w4.
    parser.add_argument(
        "-s",
        "-mnk",
        "--shape",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            # compute-bound
            (32768, 16384, 8192),
            (16384, 16384, 16384),
            # memory-bound
            # N16K x BS64
            # (16, 1048576, 16384),
            (2, 1048576, 16384),
            # (16, 1048576, 8192),
            (2, 1048576, 8192),
        ],
        help="(M,N,K) tuples, e.g. -s 16384,16384,8192 128,16384,16384",
    )
    args = parser.parse_args()

    rows = []
    for intype, (M, N, K), apre, init in itertools.product(
        args.intype, args.shape, args.apre, args.init
    ):
        rows.append(test_gemm(intype, M, N, K, apre, init=init, mode=args.mode))

    if rows and args.mode != "func":
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "mxfp8fp4gemm %s summary (markdown):\n%s",
            args.mode,
            df.to_markdown(index=False),
        )
        if args.mode == "profile":
            aiter.logger.info("profiler traces written under ./aiter_logs/")


if __name__ == "__main__":
    main()
