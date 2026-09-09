# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compare the gfx950 MX GEMM formats -- a6w4 vs a6w6, a8w4, a4w4 -- on real
DeepSeek-R1 (TP=8) and Qwen3.8-27B (TP=1) linear shapes, swept over batch size.

Every path starts from the *same* bf16 tensors and then runs its own canonical
aiter quantization pipeline, so this compares formats, not quantizers:

  * a4w4 / a6w4 / a8w4 (FlyDSL) share one kernel and differ only in ``a_dtype``,
    which isolates activation precision exactly at fixed MXFP4 weights.
  * a4w4 and a6w6 (asm) are aiter's production paths and are the reality check.

All paths are tuned: the asm paths auto-select a shape-tuned kernel, and the
FlyDSL arm goes through pick_mx_tiles, which consults
aiter/configs/a6w4_flydsl_tuned_gemm.csv (regenerate with
tune_gemm_mx_a6w4.py in this directory).

Inputs are Gaussian, which is the *best* case for low-bit weights -- real
tensors have outliers that MXFP4 handles far worse than MXFP6, so the
a4w4-vs-a6w4 accuracy gap reported here is a lower bound.

Usage:
    python op_tests/op_benchmarks/hip/bench_gemm_mx_a6w4.py [--shapes r1.]
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from tune_gemm_mx_a6w4 import BATCH_SIZES, SHAPES

ITERS, WARMUP = 20, 5
PATHS = ["a4w4-fly", "a6w4-fly", "a8w4-fly", "a6w6-asm", "a4w4-asm", "bf16"]


def _bench(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(ITERS):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / ITERS * 1e6)
    return best


def _rel(out, ref) -> float:
    return float(torch.linalg.vector_norm(out.float() - ref)
                 / torch.linalg.vector_norm(ref))


def _runners(x_max, w, N, K):
    """Build {path: callable(M) -> out} for every format, prepped once."""
    import aiter
    import aiter.ops.flydsl.batched_gemm_mxfp4 as bg

    runners = {}

    w_codes, w_scales = bg.preshuffle_mx_weight(w)
    for a_dtype, name in (("fp4", "a4w4-fly"), ("fp6", "a6w4-fly"),
                          ("fp8", "a8w4-fly")):
        try:
            quant = {M: bg.quant_mx_act(x_max[:M].contiguous(), a_dtype)
                     for M in BATCH_SIZES}
        except Exception as exc:  # noqa: BLE001
            print(f"  {name} unavailable: {exc}", file=sys.stderr)
            continue

        def run(M, a_dtype=a_dtype, quant=quant):
            a_codes, a_scales = quant[M]
            return bg.flydsl_gemm_mxfp4(a_codes, w_codes, a_scales, w_scales, N,
                                        torch.bfloat16, a_dtype=a_dtype)
        runners[name] = run

    try:
        from aiter.ops.gemm_op_a6w6 import gemm_a6w6, quant_mxfp6_gemm

        Bq, Bs = quant_mxfp6_gemm(w)
        packed = {M: quant_mxfp6_gemm(x_max[:M].contiguous()) for M in BATCH_SIZES}

        def run_a6w6(M):
            A, As = packed[M]
            return gemm_a6w6(A, Bq, As, Bs, M, N, K)
        runners["a6w6-asm"] = run_a6w6
    except Exception as exc:  # noqa: BLE001
        print(f"  a6w6-asm unavailable: {exc}", file=sys.stderr)

    try:
        from aiter.ops.gemm_op_a4w4 import gemm_a4w4
        from aiter.ops.shuffle import shuffle_weight

        qf = aiter.get_triton_quant(aiter.QuantType.per_1x32)
        wq, ws = qf(w, shuffle=True)
        wsh = shuffle_weight(wq, layout=(16, 16))
        xq = {M: qf(x_max[:M].contiguous(), shuffle=True) for M in BATCH_SIZES}

        def run_a4w4(M):
            a, a_s = xq[M]
            got = gemm_a4w4(a, wsh, a_s, ws, bpreshuffle=True)
            got = got[0] if isinstance(got, tuple) else got
            return got[:M]
        runners["a4w4-asm"] = run_a4w4
    except Exception as exc:  # noqa: BLE001
        print(f"  a4w4-asm unavailable: {exc}", file=sys.stderr)

    runners["bf16"] = lambda M: x_max[:M] @ w.T
    return runners


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shapes", default="", help="substring filter on label")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no GPU", file=sys.stderr)
        return 1
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx950":
        print(f"gfx950-only benchmark, got {arch}", file=sys.stderr)
        return 1

    for label, N, K in [s for s in SHAPES if args.shapes in s[0]]:
        torch.manual_seed(0)
        x_max = torch.randn(max(BATCH_SIZES), K, device="cuda",
                            dtype=torch.bfloat16) * 0.5
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.5
        runners = _runners(x_max, w, N, K)
        order = [p for p in PATHS if p in runners]

        # Accuracy is a property of the format, not the batch size; report once.
        ref = x_max[:64].float() @ w.float().T
        errs = {p: _rel(runners[p](64), ref) for p in order}

        print(f"\n=== {label}  N={N} K={K} "
              f"({ITERS} iters, {WARMUP} warmup) ===")
        print("rel err: " + "  ".join(f"{p.split('-')[0]} {errs[p]:.3e}"
                                      for p in order if p != "bf16"))
        print(f"{'M':>5} " + " ".join(f"{p:>10}" for p in order))
        for M in BATCH_SIZES:
            us = {p: _bench(lambda p=p, M=M: runners[p](M)) for p in order}
            fastest = min(us, key=us.get)
            cells = []
            for p in order:
                mark = "*" if p == fastest else " "
                cells.append(f"{us[p]:9.1f}{mark}")
            print(f"{M:>5} " + " ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
