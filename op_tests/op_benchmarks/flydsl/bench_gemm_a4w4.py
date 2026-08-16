# SPDX-License-Identifier: MIT

import argparse

import torch

from aiter.ops.flydsl.gemm_a4w4 import (
    flydsl_gemm_a4w4,
    prepare_gemm_a4w4_weight,
)
from aiter.ops.triton.quant import dynamic_mxfp4_quant


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense FlyDSL inline A4W4 benchmark")
    parser.add_argument("--shape", default="16,1536,768")
    parser.add_argument("--bm", type=int, choices=(16, 64), default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    args = parser.parse_args()
    m, n, k = map(int, args.shape.split(","))

    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    w, w_scale = dynamic_mxfp4_quant(
        torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    )
    prepared = prepare_gemm_a4w4_weight(w, w_scale)
    flydsl_gemm_a4w4(a, prepared, _bm=args.bm)
    torch.cuda.synchronize()

    for _ in range(args.warmup):
        flydsl_gemm_a4w4(a, prepared, _bm=args.bm)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.rep):
        flydsl_gemm_a4w4(a, prepared, _bm=args.bm)
    end.record()
    end.synchronize()

    ms = start.elapsed_time(end) / args.rep
    print(
        f"M={m} N={n} K={k} BM={args.bm}: {ms * 1e3:.2f} us, "
        f"{2 * m * n * k / ms * 1e-9:.2f} TFLOP/s"
    )


if __name__ == "__main__":
    main()
