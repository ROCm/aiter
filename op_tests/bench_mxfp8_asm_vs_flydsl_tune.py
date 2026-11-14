#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fair ASM vs FlyDSL MXFP8 A/B for gemm-request shapes (jli/gfx1250-gemm-tune).

Usage::

    cd aiter
    export AITER_FLYDSL_MXFP8_SCHED_VGPR=1 AITER_FLYDSL_MXFP8_ASM_PIPE=1
    PYTHONPATH=. python op_tests/bench_mxfp8_asm_vs_flydsl_tune.py
"""

from __future__ import annotations

import os

import torch

import aiter
from aiter import dtypes
from aiter.benchmark_data_init import fill_fp8, fill_scale_e8m0, make_generator
from aiter.ops.gemm_op_a8w8 import (
    _get_mxfp8_bpreshuffle_config,
    _mxfp8_mxfp8_gemm_asm,
    _resolve_mxfp8_gemm_config,
    gemm_a8w8_mxfp8_bpreshuffle,
)
from aiter.ops.quant import per_group_quant_hip
from aiter.ops.shuffle import shuffle_mxfp8fp4_b, shuffle_mxfp8fp4_scale, shuffle_weight
from aiter.test_common import run_perftest

SHAPES = [
    (16384, 32768, 1280),
    (16384, 5120, 8192),
    (16384, 4608, 5120),
    (16384, 5120, 2304),
    (16384, 1792, 5120),
    (16384, 25600, 6144),
]
MX = 32
WARMUP = int(os.environ.get("WARMUP", "10"))
ITERS = int(os.environ.get("ITERS", "50"))


def tflops(m, n, k, us):
    return 2.0 * m * n * k / us / 1e6


def main():
    torch.set_default_device("cuda")
    print(
        "sched_vgpr=",
        os.environ.get("AITER_FLYDSL_MXFP8_SCHED_VGPR", "0"),
        "asm_pipe=",
        os.environ.get("AITER_FLYDSL_MXFP8_ASM_PIPE", "0"),
        "flydsl=",
        __import__("flydsl").__version__,
    )
    rows = []
    for M, N, K in SHAPES:
        print(f"\n=== {M},{N},{K} ===", flush=True)
        gen = make_generator(0)
        A = fill_fp8((M, K), "uniform", gen)
        B = fill_fp8((N, K), "uniform", gen)
        sA = fill_scale_e8m0((M, K // MX), "auto", gen)
        sB = fill_scale_e8m0((N, K // MX), "auto", gen)
        B_asm = shuffle_mxfp8fp4_b(B)
        sB_asm = shuffle_mxfp8fp4_scale(sB)
        kn_asm, sk = _resolve_mxfp8_gemm_config(M, N, K, False, dtypes.bf16, None, None)

        def run_asm():
            out = torch.empty((M, N), dtype=dtypes.bf16, device="cuda")
            _mxfp8_mxfp8_gemm_asm(A, B_asm, sA, sB_asm, out, kn_asm or None, 0, sk)
            return out

        for _ in range(3):
            run_asm()
        torch.cuda.synchronize()
        _, asm_us = run_perftest(run_asm, num_warmup=WARMUP, num_iters=ITERS)
        print(f"  ASM  {asm_us:.2f} us  {tflops(M,N,K,asm_us):.1f}T  knl={kn_asm}", flush=True)

        x = torch.randn(M, K, dtype=dtypes.bf16, device="cuda")
        weight = (torch.randn(N, K, device="cuda") * 0.02).to(dtypes.fp8)
        w_scale = torch.randint(122, 127, (N, K // MX), dtype=torch.uint8, device="cuda").view(
            dtypes.fp8_e8m0
        )
        xq, x_scale = per_group_quant_hip(
            x, quant_dtype=dtypes.fp8, group_size=32, scale_type=dtypes.fp8_e8m0
        )
        WQ = shuffle_weight(weight, layout=(16, 16))
        w_scale_sh = shuffle_mxfp8fp4_scale(w_scale)
        lib, kn_fly, sk_fly = _get_mxfp8_bpreshuffle_config(M, N, K)

        def run_fly():
            return gemm_a8w8_mxfp8_bpreshuffle(xq, WQ, x_scale, w_scale_sh, dtypes.bf16)

        for _ in range(3):
            run_fly()
        torch.cuda.synchronize()
        _, fly_us = run_perftest(run_fly, num_warmup=WARMUP, num_iters=ITERS)
        print(
            f"  Fly  {fly_us:.2f} us  {tflops(M,N,K,fly_us):.1f}T  knl={kn_fly}",
            flush=True,
        )
        rows.append(
            {
                "M": M,
                "N": N,
                "K": K,
                "asm_us": round(asm_us, 2),
                "asm_T": round(tflops(M, N, K, asm_us), 1),
                "fly_us": round(fly_us, 2),
                "fly_T": round(tflops(M, N, K, fly_us), 1),
                "asm/fly": round(fly_us / asm_us, 2),
            }
        )
        torch.cuda.empty_cache()

    try:
        import pandas as pd

        print("\n===== SUMMARY =====")
        print(pd.DataFrame(rows).to_markdown(index=False))
    except Exception:
        print(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
