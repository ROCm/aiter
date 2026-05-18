# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import argparse
import pandas as pd
import aiter
from aiter import dtypes, QuantType
from aiter.test_common import perftest

torch.set_default_device("cuda")


@perftest()
def bench_add_rmsnorm_quant(input, residual, output, residual_out, scale, weight, eps):
    aiter.add_rmsnorm_quant(output, input, residual, residual_out, scale, weight, eps, 0)


def run(m, n, dtype, quant_dtype):
    input = torch.randn(m, n, dtype=dtype)
    residual = torch.randn(m, n, dtype=dtype)
    weight = torch.randn(n, dtype=dtype)
    output = torch.empty(m, n, dtype=quant_dtype)
    residual_out = torch.empty(m, n, dtype=dtype)
    scale = torch.empty(m, 1, dtype=dtypes.fp32)

    _, us = bench_add_rmsnorm_quant(
        input, residual, output, residual_out, scale, weight, 1e-5
    )

    # input + residual (read), weight (read), output (write), residual_out (write), scale (write)
    read_bytes = (input.nbytes + residual.nbytes + weight.nbytes)
    write_bytes = (output.nbytes + residual_out.nbytes + scale.nbytes)
    bw_gbs = (read_bytes + write_bytes) / us / 1024**3 * 1e6

    return {
        "m": m,
        "n": n,
        "dtype": str(dtype).split(".")[-1],
        "quant_dtype": str(quant_dtype).split(".")[-1],
        "us": round(us, 3),
        "bw(GB/s)": round(bw_gbs, 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark aiter.add_rmsnorm_quant (per-token, int8/fp8)"
    )
    parser.add_argument("-m", type=int, nargs="*", default=[8, 256, 2048, 2560, 32768])
    parser.add_argument("-n", type=int, nargs="*", default=[1024, 2048, 4096, 8192])
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.d_dtypes["bf16"]],
        choices=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp16"]],
    )
    parser.add_argument(
        "-q",
        "--quant_dtype",
        type=dtypes.str2Dtype,
        default=dtypes.d_dtypes["i8"],
        choices=[dtypes.d_dtypes["i8"], dtypes.d_dtypes["fp8"]],
    )
    args = parser.parse_args()

    rows = []
    for dtype in args.dtype:
        for n in args.n:
            for m in args.m:
                rows.append(run(m, n, dtype, args.quant_dtype))

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
