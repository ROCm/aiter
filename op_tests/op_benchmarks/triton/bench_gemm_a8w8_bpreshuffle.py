# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compare gfx1201 FP8 GEMMs using the same quantized operands and scales."""

import argparse
import json
from functools import partial

import torch
import triton.testing

from aiter.ops.gemm_op_a8w8 import gemm_a8w8_bpreshuffle
from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.shuffle import shuffle_weight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", default=["8192x2048", "4096x4096"])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4, 8, 16, 32])
    parser.add_argument("--cache-mode", choices=["graph", "flushed"], default="graph")
    args = parser.parse_args()
    if arch_info.get_arch() != "gfx1201":
        parser.error("this benchmark targets gfx1201")
    torch.manual_seed(17)

    for shape in args.shapes:
        n, k = map(int, shape.lower().split("x"))
        if n % 16 or k % 32:
            parser.error("comparison shapes require N % 16 == 0 and K % 32 == 0")
        weight = (torch.randn(n, k, device="cuda") * 0.25).to(torch.float8_e4m3fn)
        shuffled = shuffle_weight(weight)
        sw = torch.rand(n, 1, device="cuda") * 0.5 + 0.5
        for m in args.batches:
            x = (torch.randn(m, k, device="cuda") * 0.25).to(torch.float8_e4m3fn)
            sx = torch.rand(m, 1, device="cuda") * 0.5 + 0.5
            reference = ((x.float() @ weight.float().T) * sx * sw.T).bfloat16()
            variants = {
                "public": partial(gemm_a8w8_bpreshuffle, x, shuffled, sx, sw),
                "plain_triton": partial(gemm_a8w8, x, weight, sx, sw),
                "torch_rowwise": partial(
                    torch._scaled_mm,
                    x,
                    weight.T,
                    scale_a=sx,
                    scale_b=sw.T,
                    out_dtype=torch.bfloat16,
                ),
            }
            for name, fn in variants.items():
                output = fn()
                torch.testing.assert_close(output, reference, rtol=0.02, atol=0.01)
                if args.cache_mode == "graph":
                    ms = triton.testing.do_bench_cudagraph(
                        fn, rep=5, return_mode="median"
                    )
                else:
                    ms = triton.testing.do_bench(
                        fn, warmup=10, rep=30, return_mode="median"
                    )
                print(
                    json.dumps(
                        {
                            "m": m,
                            "n": n,
                            "k": k,
                            "variant": name,
                            "cache_mode": args.cache_mode,
                            "us": ms * 1000,
                            "max_abs_error": (output - reference).abs().max().item(),
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
