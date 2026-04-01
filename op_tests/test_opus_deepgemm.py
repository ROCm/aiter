# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import argparse
import aiter
from aiter import dtypes
from aiter.test_common import checkAllclose, run_perftest
from aiter.ops.deepgemm import deepgemm_opus


def run_torch_ref_scale(XQ, WQ, x_scale, w_scale, GROUP_M, GROUP_N, GROUP_K, out_dtype):
    batch, M, K = XQ.shape
    _, N, _ = WQ.shape

    x_fp32 = XQ.to(torch.float32)
    w_fp32 = WQ.to(torch.float32)

    if x_scale is not None:
        x_fp32 = x_fp32.reshape(batch, M // GROUP_M, GROUP_M, K // GROUP_K, GROUP_K)
        xs = x_scale[:, :, None, :, None]
        x_fp32 = (x_fp32 * xs).reshape(batch, M, K)

    if w_scale is not None:
        w_fp32 = w_fp32.reshape(batch, N // GROUP_N, GROUP_N, K // GROUP_K, GROUP_K)
        ws = w_scale[:, :, None, :, None]
        w_fp32 = (w_fp32 * ws).reshape(batch, N, K)

    out = torch.einsum("bmk,bnk->bmn", x_fp32, w_fp32)
    return out.to(out_dtype)


def run_torch_ref_noscale(XQ, WQ, out_dtype):
    x_fp32 = XQ.to(torch.float32)
    w_fp32 = WQ.to(torch.float32)
    out = torch.einsum("bmk,bnk->bmn", x_fp32, w_fp32)
    return out.to(out_dtype)


def test_a8w8_scale(batch, M, N, K, out_dtype=torch.float32):
    GROUP_M, GROUP_N, GROUP_K = 1, 128, 128
    assert M % GROUP_M == 0 and N % GROUP_N == 0 and K % GROUP_K == 0

    XQ = torch.randn(batch, M, K, device="cuda").to(dtypes.fp8)
    WQ = torch.randn(batch, N, K, device="cuda").to(dtypes.fp8)
    Y = torch.zeros(batch, M, N, device="cuda", dtype=out_dtype)

    num_groups_m = M // GROUP_M
    num_groups_n = N // GROUP_N
    num_groups_k = K // GROUP_K

    x_scale = torch.ones(batch, num_groups_m, num_groups_k, device="cuda", dtype=torch.float32)
    w_scale = torch.ones(batch, num_groups_n, num_groups_k, device="cuda", dtype=torch.float32)

    ref_out = run_torch_ref_scale(XQ, WQ, x_scale, w_scale, GROUP_M, GROUP_N, GROUP_K, out_dtype)

    Y, us = run_perftest(
        deepgemm_opus, XQ, WQ, Y, None, x_scale, w_scale,
    )

    err = checkAllclose(Y, ref_out, msg=f"a8w8_scale b={batch} m={M} n={N} k={K}", rtol=0.1, atol=0.1)
    tflops = 2.0 * batch * M * N * K / us / 1e6
    print(f"[a8w8_scale] batch={batch} M={M} N={N} K={K} | {us:.1f}us | {tflops:.2f} TFLOPs | err={err}")


def test_a8w8(batch, M, N, K, out_dtype=torch.float32):
    XQ = torch.randn(batch, M, K, device="cuda").to(dtypes.fp8)
    WQ = torch.randn(batch, N, K, device="cuda").to(dtypes.fp8)
    Y = torch.zeros(batch, M, N, device="cuda", dtype=out_dtype)

    ref_out = run_torch_ref_noscale(XQ, WQ, out_dtype)

    Y, us = run_perftest(
        deepgemm_opus, XQ, WQ, Y, None, None, None,
    )

    err = checkAllclose(Y, ref_out, msg=f"a8w8 b={batch} m={M} n={N} k={K}", rtol=0.1, atol=0.1)
    tflops = 2.0 * batch * M * N * K / us / 1e6
    print(f"[a8w8] batch={batch} M={M} N={N} K={K} | {us:.1f}us | {tflops:.2f} TFLOPs | err={err}")


def test_a16w16(batch, M, N, K, out_dtype=torch.bfloat16):
    XQ = torch.randn(batch, M, K, device="cuda", dtype=torch.bfloat16)
    WQ = torch.randn(batch, N, K, device="cuda", dtype=torch.bfloat16)
    Y = torch.zeros(batch, M, N, device="cuda", dtype=out_dtype)

    ref_out = run_torch_ref_noscale(XQ, WQ, out_dtype)

    Y, us = run_perftest(
        deepgemm_opus, XQ, WQ, Y, None, None, None,
    )

    err = checkAllclose(Y, ref_out, msg=f"a16w16 b={batch} m={M} n={N} k={K}", rtol=0.1, atol=0.5)
    tflops = 2.0 * batch * M * N * K / us / 1e6
    print(f"[a16w16] batch={batch} M={M} N={N} K={K} dtype={out_dtype} | {us:.1f}us | {tflops:.2f} TFLOPs | err={err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test opus_gemm kernels")
    parser.add_argument("-m", type=int, default=256)
    parser.add_argument("-n", type=int, default=512)
    parser.add_argument("-k", type=int, default=256)
    parser.add_argument("-b", "--batch", type=int, default=8)
    parser.add_argument(
        "-t", "--type",
        type=str,
        default="all",
        choices=["a8w8_scale", "a8w8", "a16w16", "all"],
        help="Kernel type to test",
    )
    parser.add_argument(
        "-d", "--dtype",
        type=str,
        default="fp32",
        choices=["fp32", "bf16"],
        help="Output dtype",
    )
    args = parser.parse_args()

    out_dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    if args.type in ("a8w8_scale", "all"):
        test_a8w8_scale(args.batch, args.m, args.n, args.k)

    if args.type in ("a8w8", "all"):
        test_a8w8(args.batch, args.m, args.n, args.k)

    if args.type in ("a16w16", "all"):
        k_a16 = max(args.k, 128)
        test_a16w16(args.batch, args.m, args.n, k_a16, out_dtype=torch.bfloat16)
