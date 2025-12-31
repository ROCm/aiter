# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter.test_common import checkAllclose, perftest, benchmark
from aiter import dtypes
import argparse
import pandas as pd


@perftest(num_iters=2)
def run_torch(input, x_scale, y_scale_dtype=dtypes.fp32, quant_dtype=dtypes.i8):
    output, y_scale = aiter.pertoken_quant(
        input, x_scale=x_scale, scale_dtype=y_scale_dtype, quant_dtype=quant_dtype
    )
    return output, y_scale


@perftest(num_iters=2)
def run_torch_topk(
    input, x_scale, topk_id, y_scale_dtype=dtypes.fp32, quant_dtype=dtypes.i8
):
    topk = topk_id.shape[-1]
    if input.shape[1] == 1:
        input = input.repeat(1, topk, 1)
    input = input * x_scale[topk_id]
    output, y_scale = aiter.pertoken_quant(
        input, scale_dtype=y_scale_dtype, quant_dtype=quant_dtype
    )
    return output, y_scale


@perftest()
def run_ck(input, x_scale, y_scale_dtype=dtypes.fp32, quant_dtype=dtypes.i8):
    # pad stride
    output = torch.empty_strided(
        input.shape,
        (input.shape[1] + 128, 1),
        dtype=quant_dtype,
        layout=input.layout,
        device=input.device,
    )
    y_scale = torch.empty(input.shape[0], 1, device="cuda", dtype=y_scale_dtype)
    aiter.smoothquant_fwd(output, input, x_scale, y_scale)
    return output, y_scale


@perftest()
def run_ck_moe_smoothquant(
    input, x_scale, topk_id, y_scale_dtype=dtypes.fp32, quant_dtype=dtypes.i8
):
    topk = topk_id.shape[-1]
    # pad stride
    output = torch.empty(
        (input.shape[0], topk, input.shape[-1]),
        dtype=quant_dtype,
        device=input.device,
    )
    y_scale = torch.empty((input.shape[0], topk, 1), device="cuda", dtype=y_scale_dtype)
    aiter.moe_smoothquant_fwd(output, input, x_scale, topk_id, y_scale)
    return output, y_scale


@perftest(num_iters=2)
def run_hip(
    input, x_scale, y_scale_dtype=dtypes.fp32, quant_dtype=dtypes.i8, topk_id=None
):
    output = torch.empty(
        input.shape,
        dtype=quant_dtype,
        device=input.device,
    )
    y_scale = torch.empty((*input.shape[:-1], 1), device="cuda", dtype=y_scale_dtype)
    aiter.smooth_per_token_scaled_quant(
        output, input, y_scale, x_scale, smooth_scale_map=topk_id
    )
    return output, y_scale


@benchmark()
def test_Smoothquant_instance(dtype, m, n, xscaleType, quant_dtype=dtypes.i8):
    dim = (m, n)
    input = torch.randn(dim, dtype=dtype, device="cuda")
    xscale = torch.randn(n, dtype=xscaleType, device="cuda")
    (a, yscale_a), avg_a = run_torch(input, x_scale=xscale, quant_dtype=quant_dtype)
    (b, yscale_b), avg_b = run_ck(input, x_scale=xscale, quant_dtype=quant_dtype)
    (c, yscale_c), avg_c = run_hip(input, x_scale=xscale, quant_dtype=quant_dtype)

    err_b = checkAllclose(a.to(dtypes.fp32), b.to(dtypes.fp32), rtol=0.01, atol=0.01)
    checkAllclose(yscale_a, yscale_b, rtol=1e-3, atol=1e-3)
    err_c = checkAllclose(a.to(dtypes.fp32), c.to(dtypes.fp32), rtol=0.01, atol=0.01)
    checkAllclose(yscale_a, yscale_c, rtol=1e-3, atol=1e-3)
    return {"ck us": avg_b, "err ck": err_b, "hip us": avg_c, "err hip": err_c}


@benchmark()
def test_topK_Smoothquant_instance(
    dtype, m, n, xscaleType, quant_dtype, topk=5, expert=128
):
    dim = (m, topk, n)
    input = torch.randn(dim, dtype=dtype, device="cuda")
    xscale = torch.randn((expert, n), dtype=xscaleType, device="cuda")
    topk_id = torch.randint(0, expert, (m, topk), dtype=dtypes.i32, device="cuda")
    (a, yscale_a), avg_a = run_torch_topk(
        input, x_scale=xscale, topk_id=topk_id, quant_dtype=quant_dtype
    )
    (c, yscale_c), avg_c = run_hip(
        input, x_scale=xscale, topk_id=topk_id, quant_dtype=quant_dtype
    )

    err_c = checkAllclose(a.to(dtypes.fp32), c.to(dtypes.fp32), rtol=0.01, atol=0.01)
    checkAllclose(yscale_a, yscale_c, rtol=1e-3, atol=1e-3)
    return {"hip us": avg_c, "err hip": err_c}


@benchmark()
def test_moe_Smoothquant_instance(
    dtype, m, n, xscaleType, quant_dtype, topk=2, expert=2
):
    dim = (m, 1, n)
    input = torch.ones(dim, dtype=dtype, device="cuda")
    xscale = torch.randn((expert, n), dtype=xscaleType, device="cuda")
    # topk_id = torch.randint(0, expert, (m, topk), dtype=dtypes.i32, device="cuda")
    topk_id = torch.tensor([list(range(0, topk))] * m, dtype=dtypes.i32, device="cuda")
    (a, yscale_a), avg_a = run_torch_topk(
        input, x_scale=xscale, topk_id=topk_id, quant_dtype=quant_dtype
    )
    (b, yscale_b), avg_b = run_hip(
        input.expand(-1, topk, -1),
        x_scale=xscale,
        topk_id=topk_id,
        quant_dtype=quant_dtype,
    )
    (c, yscale_c), avg_c = run_ck_moe_smoothquant(
        input, x_scale=xscale, topk_id=topk_id, quant_dtype=quant_dtype
    )

    err_b = checkAllclose(a.to(dtypes.fp32), b.to(dtypes.fp32), rtol=0.01, atol=0.01)
    checkAllclose(yscale_a, yscale_b, rtol=1e-3, atol=1e-3)

    _, ids = torch.sort(topk_id.view(-1))
    a = a.view(-1, n)[ids].view(m, topk, n)
    yscale_a = yscale_a.view(-1, 1)[ids].view(m, topk, 1)

    err_c = checkAllclose(a.to(dtypes.fp32), c.to(dtypes.fp32), rtol=0.01, atol=0.01)
    checkAllclose(yscale_a, yscale_c, rtol=1e-3, atol=1e-3)
    return {"hip us": avg_b, "hip err": err_b, "ck us": avg_c, "ckerr": err_c}


def test_Smoothquant(l_dtype: list, l_m: list, l_n: list):
    print("\nstart Smoothquant test")
    for scaleType in [dtypes.fp32]:
        for dtype in [dtypes.fp16, dtypes.bf16][1:]:
            for qtype in [dtypes.i8]:
                for n in l_n:
                    df = []
                    for m in l_m:
                        ret = test_Smoothquant_instance(
                            dtype, m, n, xscaleType=scaleType, quant_dtype=qtype
                        )
                        df.append(ret)
                    df = pd.DataFrame(df)
                    df_md = df.to_markdown(index=False)
                    aiter.logger.info("Smoothquant summary (markdown):\n%s", df_md)

    print("\nstart topk Smoothquant test")
    for scaleType in [dtypes.fp32]:
        for dtype in [dtypes.fp16, dtypes.bf16][1:]:
            for qtype in [dtypes.i8, dtypes.fp8][1:]:
                for n in l_n:
                    df = []
                    for m in l_m:
                        ret = test_topK_Smoothquant_instance(
                            dtype, m, n, xscaleType=scaleType, quant_dtype=qtype
                        )
                        df.append(ret)
                    df = pd.DataFrame(df)
                    df_md = df.to_markdown(index=False)
                    aiter.logger.info("Smoothquant_topk summary (markdown):\n%s", df_md)

    print("\nstart moe Smoothquant test")
    for scaleType in [dtypes.fp32]:
        for dtype in [dtypes.fp16, dtypes.bf16][1:]:
            for qtype in [dtypes.i8, dtypes.fp8][:1]:
                for n in l_n:
                    df = []
                    for m in l_m:
                        ret = test_moe_Smoothquant_instance(
                            dtype, m, n, xscaleType=scaleType, quant_dtype=qtype
                        )
                        df.append(ret)
                    df = pd.DataFrame(df)
                    df_md = df.to_markdown(index=False)
                    aiter.logger.info("Smoothquant_moe summary (markdown):\n%s", df_md)


if __name__ == "__main__":
    l_dtype = ["bf16", "fp16"]
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=str,
        choices=l_dtype,
        nargs="?",
        const=None,
        default=None,
        help="""Data type.
    e.g.: -d bf16""",
    )
    parser.add_argument(
        "-m",
        type=int,
        default=[1, 8, 16, 32, 48, 64, 128, 256, 1024],
        nargs="*",
        help="""M of mnk.
    e.g.: -m 32""",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=[5120],
        nargs="*",
        help="""N of mnk.
    e.g.: -n 1024""",
    )
    args = parser.parse_args()
    if args.dtype is None:
        l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
    else:
        l_dtype = [dtypes.d_dtypes[args.dtype]]
    test_Smoothquant(l_dtype, args.m, args.n)
