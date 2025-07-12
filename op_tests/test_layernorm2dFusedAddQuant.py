# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import aiter
import argparse
from aiter.test_common import checkAllclose, perftest
from aiter import dtypes


@perftest()
def run_torch(
    input, weight, bias, eps, residual=None, x_scale=None, y_scale_dtype=None
):
    if residual is None:
        residual_out = None
        output = F.layer_norm(
            input=input,
            normalized_shape=(input.shape[-1],),
            weight=weight,
            bias=bias,
            eps=eps,
        )
    else:
        residual_out = input + residual
        output = F.layer_norm(
            input=residual_out,
            normalized_shape=(input.shape[-1],),
            weight=weight,
            bias=bias,
            eps=eps,
        )
    if y_scale_dtype is None:
        y_scale = None
    else:
        output, y_scale = aiter.pertoken_quant(output, x_scale=x_scale)
    return output, residual_out, y_scale


@perftest()
def run_ck(input, weight, bias, eps, residual=None, x_scale=None, y_scale_dtype=None):
    if y_scale_dtype is None:
        y_scale = None
        if residual is None:
            residual_out = None
            output = aiter.layer_norm(input, weight, bias, eps)
        elif residual is not None:
            residual_out = torch.empty_like(input)
            output = torch.empty_like(input)
            aiter.layernorm2d_fwd_with_add(
                output, input, residual, residual_out, weight, bias, eps
            )
    elif x_scale is None:
        y_scale = torch.empty(input.shape[0], 1, dtype=y_scale_dtype, device="cuda")
        output = torch.empty(input.shape, dtype=dtypes.i8, device="cuda")
        if residual is None:
            residual_out = None
            aiter.layernorm2d_fwd_with_dynamicquant(
                output, input, y_scale, weight, bias, eps
            )
        elif residual is not None:
            residual_out = torch.empty_like(input)
            aiter.layernorm2d_fwd_with_add_dynamicquant(
                output, input, residual, residual_out, y_scale, weight, bias, eps
            )
    else:
        y_scale = torch.empty(input.shape[0], 1, dtype=y_scale_dtype, device="cuda")
        output = torch.empty(input.shape, dtype=dtypes.i8, device="cuda")
        if residual is None:
            residual_out = None
            aiter.layernorm2d_fwd_with_smoothquant(
                output, input, x_scale, y_scale, weight, bias, eps
            )
        elif residual is not None:
            residual_out = torch.empty_like(input)
            aiter.layernorm2d_fwd_with_add_smoothquant(
                output,
                input,
                residual,
                residual_out,
                x_scale,
                y_scale,
                weight,
                bias,
                eps,
            )

    return output, residual_out, y_scale


@perftest()
def run_asm(input, weight, bias, eps, residual, x_scale, y_scale_dtype):
    y_scale = torch.empty(input.shape[0], 1, dtype=y_scale_dtype, device="cuda")
    output = torch.empty(input.shape, dtype=dtypes.i8, device="cuda")
    residual_out = torch.empty_like(input)
    aiter.layernorm2d_with_add_smoothquant_asm(
        output, input, residual, residual_out, x_scale, y_scale, weight, bias, eps
    )
    return output, residual_out, y_scale


def test_layernorm2d_instance(dtype, m, n):
    dim = (m, n)
    input = torch.randn(dim, dtype=dtype, device="cuda")
    weight = torch.randn(n, dtype=dtype, device="cuda")
    bias = torch.randn(n, dtype=dtype, device="cuda")
    (a, *_), avg_a = run_torch(input, weight, bias, 1e-5)
    (b, *_), avg_b = run_ck(input, weight, bias, 1e-5)
    print(
        f"[perf] dim: {dim}, dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    )
    checkAllclose(a, b)
    print("[passed~]")


def test_layernorm2d_fuseAdd_instance(dtype, m, n):
    dim = (m, n)
    input = torch.randn(dim, dtype=dtype, device="cuda")
    weight = torch.randn(n, dtype=dtype, device="cuda")
    bias = torch.randn(n, dtype=dtype, device="cuda")
    res = torch.randn(dim, dtype=dtype, device="cuda")
    (a, res_a, *_), avg_a = run_torch(input, weight, bias, 1e-5, residual=res)
    (b, res_b, *_), avg_b = run_ck(input, weight, bias, 1e-5, residual=res)

    print(
        f"[perf] dim: {dim}, dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    )
    checkAllclose(a, b, rtol=1e-2, atol=1e-1)
    checkAllclose(res_a, res_b)
    print(" [passed~]")


def test_layernorm2d_fuseSmoothquant_instance(dtype, m, n, xscaleType, yscaleType):
    dim = (m, n)
    input = torch.randn(dim, dtype=dtype, device="cuda")
    weight = torch.randn(n, dtype=dtype, device="cuda")
    bias = torch.randn(n, dtype=dtype, device="cuda")
    xscale = torch.randn(n, dtype=xscaleType, device="cuda")
    (a, _, yscale_a), avg_a = run_torch(
        input, weight, bias, 1e-5, x_scale=xscale, y_scale_dtype=yscaleType
    )
    (b, _, yscale_b), avg_b = run_ck(
        input, weight, bias, 1e-5, x_scale=xscale, y_scale_dtype=yscaleType
    )

    print(
        f"[perf] dim: {dim}, dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    )
    checkAllclose(a, b, rtol=0, atol=1)
    checkAllclose(yscale_a, yscale_b, rtol=1e-3, atol=1e-3)
    print(" [passed~]")


def test_layernorm2d_fuseAdd_Smoothquant_instance(dtype, m, n, xscaleType, yscaleType):
    dim = (m, n)
    input = torch.randn(dim, dtype=dtype, device="cuda")
    weight = torch.randn(n, dtype=dtype, device="cuda")
    bias = torch.randn(n, dtype=dtype, device="cuda")
    res = torch.randn(dim, dtype=dtype, device="cuda")
    xscale = torch.randn(n, dtype=xscaleType, device="cuda")
    (c, res_c, yscale_c), avg_c = run_asm(
        input,
        weight,
        bias,
        1e-5,
        residual=res,
        x_scale=xscale,
        y_scale_dtype=yscaleType,
    )
    (a, res_a, yscale_a), avg_a = run_torch(
        input,
        weight,
        bias,
        1e-5,
        residual=res,
        x_scale=xscale,
        y_scale_dtype=yscaleType,
    )
    (b, res_b, yscale_b), avg_b = run_ck(
        input,
        weight,
        bias,
        1e-5,
        residual=res,
        x_scale=xscale,
        y_scale_dtype=yscaleType,
    )

    print(
        f"[perf] dim: {dim}, dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    )
    checkAllclose(a, b, rtol=0, atol=1)
    checkAllclose(res_a, res_b)
    checkAllclose(yscale_a, yscale_b, rtol=1e-3, atol=1e-3)
    print(" [passed~]")
    print(
        f"[perf] dim: {dim}, dtype: {dtype}, torch avg: {avg_a:<8.2f} us, asm avg: {avg_c:<8.2f} us, uplift: {avg_a/avg_c-1:<5.1%}"
    )
    checkAllclose(a, c, rtol=0, atol=1)
    checkAllclose(res_a, res_c)
    checkAllclose(yscale_a, yscale_c, rtol=1e-2, atol=1e-2)
    print(" [passed~]")


def test_layernorm2d_fuseDynamicquant_instance(dtype, m, n, yscaleType):
    dim = (m, n)
    input = torch.randn(dim, dtype=dtype, device="cuda")
    weight = torch.randn(n, dtype=dtype, device="cuda")
    bias = torch.randn(n, dtype=dtype, device="cuda")
    (a, _, yscale_a), avg_a = run_torch(
        input, weight, bias, 1e-5, y_scale_dtype=yscaleType
    )
    (b, _, yscale_b), avg_b = run_ck(
        input, weight, bias, 1e-5, y_scale_dtype=yscaleType
    )

    print(
        f"[perf] dim: {dim}, dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    )
    checkAllclose(a, b, rtol=0, atol=1)
    checkAllclose(yscale_a, yscale_b)
    print(" [passed~]")


def test_layernorm2d_fuseAdd_Dynamicquant_instance(dtype, m, n, yscaleType):
    dim = (m, n)
    input = torch.randn(dim, dtype=dtype, device="cuda")
    weight = torch.randn(n, dtype=dtype, device="cuda")
    bias = torch.randn(n, dtype=dtype, device="cuda")
    res = torch.randn(dim, dtype=dtype, device="cuda")
    (a, res_a, yscale_a), avg_a = run_torch(
        input, weight, bias, 1e-5, residual=res, y_scale_dtype=yscaleType
    )
    (b, res_b, yscale_b), avg_b = run_ck(
        input, weight, bias, 1e-5, residual=res, y_scale_dtype=yscaleType
    )

    print(
        f"[perf] dim: {dim}, dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    )
    checkAllclose(a, b, rtol=0, atol=1)
    checkAllclose(res_a, res_b)
    checkAllclose(yscale_a, yscale_b)
    print(" [passed~]")


def test_layernorm2d():
    print("\nstart layernorm2d test")
    for dtype in [dtypes.fp16, dtypes.bf16]:
        for m in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            for n in [4096, 8192, 16384, 32768, 65536]:
                test_layernorm2d_instance(dtype, m, n)


def test_layernorm2d_fuseAdd():
    print("\nstart layernorm2d fuse add test")
    for dtype in [dtypes.fp16, dtypes.bf16]:
        for m in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            for n in [4096, 8192, 16384, 32768, 65536]:
                test_layernorm2d_fuseAdd_instance(dtype, m, n)


def test_layernorm2d_fuseSmoothquant():
    print("\nstart layernorm2d fuse Smoothquant test")
    for scaleType in [dtypes.fp32]:
        for dtype in [dtypes.fp16, dtypes.bf16]:
            for m in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                for n in [10, 4096, 8192]:
                    test_layernorm2d_fuseSmoothquant_instance(
                        dtype, m, n, xscaleType=scaleType, yscaleType=scaleType
                    )


def test_layernorm2d_fuseAdd_Smoothquant():
    print("\nstart layernorm2d fuse add Smoothquant test")
    for scaleType in [dtypes.fp32]:
        for dtype in [dtypes.bf16]:
            for m in [2, 4, 8, 16, 32, 64, 128, 256]:
                for n in [8192]:
                    test_layernorm2d_fuseAdd_Smoothquant_instance(
                        dtype, m, n, xscaleType=scaleType, yscaleType=scaleType
                    )


def test_layernorm2d_fuseDynamicquant():
    print("\nstart layernorm2d fuse Smoothquant test")
    for scaleType in [dtypes.fp32]:
        for dtype in [dtypes.fp16, dtypes.bf16]:
            for m in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                for n in [1024, 2048]:
                    test_layernorm2d_fuseDynamicquant_instance(
                        dtype, m, n, yscaleType=scaleType
                    )


def test_layernorm2d_fuseAdd_Dynamicquant():
    print("\nstart layernorm2d fuse add Smoothquant test")
    for scaleType in [dtypes.fp32]:
        for dtype in [dtypes.fp16, dtypes.bf16]:
            for m in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                for n in [1024, 2048]:
                    test_layernorm2d_fuseAdd_Dynamicquant_instance(
                        dtype, m, n, yscaleType=scaleType
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        prog="test_layernorm2dFusedSQuant",
        description="Test ck layernorm2d Fused add and SmoothQuant",
    )
    parser.add_argument(
        "--mode",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        help="1: test_layernorm2d, \n2:test_layernorm2d_fuseAdd, \n"
        + "3:test_layernorm2d_fuseSmoothquant, \n4:test_layernorm2d_fuseAdd_Smoothquant"
        + "5:test_layernorm2d_fuseDynamicquant, \n6:test_layernorm2d_fuseAdd_Dynamicquant\n"
        + "    e.g.: --mode 1",
        default=1,
    )
    # parser.add_argument(
    #     "--GPUID",
    #     type=str,
    #     help="This script uses single GPU. Specify the GPU to use for tuning",
    #     default="0",
    # )
    args = parser.parse_args()
    if args.mode == 1:
        test_layernorm2d()
    elif args.mode == 2:
        test_layernorm2d_fuseAdd()
    elif args.mode == 3:
        test_layernorm2d_fuseSmoothquant()
    elif args.mode == 4:
        test_layernorm2d_fuseAdd_Smoothquant()
    elif args.mode == 5:
        test_layernorm2d_fuseDynamicquant()
    elif args.mode == 6:
        test_layernorm2d_fuseAdd_Dynamicquant()
