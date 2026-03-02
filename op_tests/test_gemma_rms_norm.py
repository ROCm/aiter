# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
import aiter
from aiter.test_common import checkAllclose, perftest
from aiter import dtypes
import argparse


def _gemma_rms_norm_ref_native(x, w, eps=1e-6, residual=None):
    """
    Reference from sglang GemmaRMSNorm.forward_native
    https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/layernorm.py
    """
    orig_dtype = x.dtype
    if residual is not None:
        x = x + residual
        residual_out = x.clone()
    else:
        residual_out = None
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * (1.0 + w.float())
    x = x.to(orig_dtype)
    return x, residual_out


@perftest()
def run_torch(input, weight, eps, residual=None):
    output, residual_out = _gemma_rms_norm_ref_native(
        input.clone(), weight, eps,
        residual=residual.clone() if residual is not None else None,
    )
    return output, residual_out


@perftest()
def run_gemma(input, weight, eps, residual=None):
    if residual is None:
        residual_out = None
        output = aiter.gemma_rmsnorm(input, weight, eps)
    else:
        output = input.clone()
        residual_out = residual.clone()
        aiter.gemma_fused_add_rmsnorm(output, residual_out, weight, eps)
    return output, residual_out


def _run_gemma_rmsnorm(dtype, m, n):
    dim = (m, n)
    input = torch.randn(dim, dtype=dtype, device="cuda")
    weight = torch.randn(n, dtype=dtype, device="cuda")
    (a, *_), avg_a = run_torch(input, weight, 1e-6)
    (b, *_), avg_b = run_gemma(input, weight, 1e-6)
    msg = (
        f"[perf] dim: {str(dim):<20}, dtype: {dtype}, torch avg: {avg_a:<8.2f} us, "
        f"gemma avg: {avg_b:<8.2f} us"
    )
    rtol, atol = (1e-2, 1e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)
    checkAllclose(a, b, rtol=rtol, atol=atol, msg=msg)


def _run_gemma_fused_add_rmsnorm(dtype, m, n):
    dim = (m, n)
    input = torch.randn(dim, dtype=dtype, device="cuda")
    weight = torch.randn(n, dtype=dtype, device="cuda")
    res = torch.randn(dim, dtype=dtype, device="cuda")
    (a, res_a, *_), avg_a = run_torch(input, weight, 1e-6, residual=res)
    (b, res_b, *_), avg_b = run_gemma(input, weight, 1e-6, residual=res)
    msg = (
        f"[perf] dim: {str(dim):<20}, dtype: {dtype}, torch avg: {avg_a:<8.2f} us, "
        f"gemma avg: {avg_b:<8.2f} us"
    )
    rtol, atol = (1e-2, 1e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)
    checkAllclose(a, b, rtol=rtol, atol=atol, msg=msg)
    checkAllclose(res_a, res_b, rtol=rtol, atol=atol, msg="gemma res check")


L_DTYPE_STR = ["fp16", "bf16"]
L_M = [1, 2, 128, 256, 8000, 16000]
L_N = [4096, 8192]

@pytest.mark.parametrize("dtype", [dtypes.d_dtypes[k] for k in L_DTYPE_STR])
@pytest.mark.parametrize("m", [1, 2, 8000, 16000])
@pytest.mark.parametrize("n", [4096, 8192])
def test_gemma_rmsnorm_pytest(dtype, m, n):
    _run_gemma_rmsnorm(dtype, m, n)


@pytest.mark.parametrize("dtype", [dtypes.d_dtypes[k] for k in L_DTYPE_STR])
@pytest.mark.parametrize("m", [1, 2, 8000, 16000])
@pytest.mark.parametrize("n", [4096, 8192])
def test_gemma_fused_add_rmsnorm_pytest(dtype, m, n):
    _run_gemma_fused_add_rmsnorm(dtype, m, n)


if __name__ == "__main__":
    l_dtype = L_DTYPE_STR
    l_m = L_M
    l_n = L_N
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Gemma RMSNorm test (fp16/bf16 only)",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=str,
        choices=l_dtype,
        nargs="?",
        const=None,
        default=None,
        help="Data type: fp16 or bf16",
    )
    parser.add_argument(
        "-m",
        "--m",
        type=int,
        nargs="?",
        default=None,
        help="Batch size (M)",
    )
    parser.add_argument(
        "-n",
        "--n",
        type=int,
        nargs="?",
        default=None,
        help="Hidden size (N)",
    )
    args = parser.parse_args()
    if args.dtype is None:
        l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
    else:
        l_dtype = [dtypes.d_dtypes[args.dtype]]
    if args.m is not None:
        l_m = [args.m]
    if args.n is not None:
        l_n = [args.n]

    print("\nstart gemma rmsnorm test(no residual) ---")
    for dtype in l_dtype:
        for m in l_m:
            for n in l_n:
                _run_gemma_rmsnorm(dtype, m, n)

    torch.cuda.synchronize()
    print("\nstart gemma rmsnorm fuse add test")
    for dtype in l_dtype:
        for m in l_m:
            for n in l_n:
                _run_gemma_fused_add_rmsnorm(dtype, m, n)


