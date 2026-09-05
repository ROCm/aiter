# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import argparse

import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.test_common import checkAllclose, perftest


@perftest()
def run_torch(input, weight, eps, residual=None):
    if residual is None:
        residual_out = None
        output = F.rms_norm(
            input=input, normalized_shape=(input.shape[-1],), weight=weight, eps=eps
        )
    else:
        residual_out = input + residual
        output = F.rms_norm(
            input=residual_out,
            normalized_shape=(input.shape[-1],),
            weight=weight,
            eps=eps,
        )
    return output, residual_out


@perftest()
def run_ck(input, weight, eps, residual=None, use_model_sensitive_rmsnorm=0):
    if residual is None:
        residual_out = None
        output = aiter.rms_norm(input, weight, eps, use_model_sensitive_rmsnorm)
    else:
        residual_out = torch.empty_like(input)
        output = torch.empty_like(input)
        aiter.rmsnorm2d_fwd_with_add(
            output,
            input,
            residual,
            residual_out,
            weight,
            eps,
            use_model_sensitive_rmsnorm=use_model_sensitive_rmsnorm,
        )
    return output, residual_out


@perftest()
def run_cu(input, weight, eps, residual=None):
    if residual is None:
        residual_out = None
        output = torch.empty_like(input)
        aiter.rms_norm_cu(output, input, weight, eps)
    else:
        aiter.fused_add_rms_norm_cu(input, residual, weight, eps)
        output = input
        residual_out = residual
    return output, residual_out


def test_rmsnorm2d(dtype, m, n):
    dim = (m, n)
    input = torch.randn(dim, dtype=dtype, device="cuda")
    weight = torch.randn(n, dtype=dtype, device="cuda")
    # q, k, v = torch.split(hidden_stats, [6*n, n, n], dim=1)
    # input = k
    (a, *_), avg_a = run_torch(input, weight, 1e-5)
    (b, *_), avg_b = run_ck(input, weight, 1e-5)
    (c, *_), avg_c = run_cu(input, weight, 1e-5)
    msg = f"[perf] dim: {dim!s:<20}, dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, cu avg: {avg_c:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    checkAllclose(a, b, msg=msg)
    checkAllclose(a, c, msg="cu")


def test_rmsnorm2d_fuseAdd(dtype, m, n):
    dim = (m, n)
    input = torch.randn(dim, dtype=dtype, device="cuda")
    weight = torch.randn(n, dtype=dtype, device="cuda")
    res = torch.randn(dim, dtype=dtype, device="cuda")
    # q, k, v = torch.split(hidden_stats, [6*n, n, n], dim=1)
    # input = k
    (a, res_a, *_), avg_a = run_torch(input, weight, 1e-5, residual=res)
    (b, res_b, *_), avg_b = run_ck(input, weight, 1e-5, residual=res)
    (c, res_c, *_), avg_c = run_ck(
        input, weight, 1e-5, residual=res, use_model_sensitive_rmsnorm=1
    )
    (_d, _res_d, *_), _avg_d = run_cu(input, weight, 1e-5, residual=res)

    msg = f"[perf] dim: {dim!s:<20}, dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, cu avg: {avg_c:<8.2f} us,uplift: {avg_a/avg_b-1:<5.1%}"
    checkAllclose(a, b, atol=0.03, msg=msg)
    checkAllclose(res_a, res_b, msg="ck res check (NO_SPECIFIC_MODEL)")

    checkAllclose(a, c, atol=0.03, msg=msg)
    checkAllclose(res_a, res_c, msg="ck res check (T5_MODEL_LIKE)")
    # checkAllclose(a, d, atol=0.03, msg='cu')
    # checkAllclose(res_a, res_d, atol=0.01, msg='cu res check')

    # gemma_norm: opus folds (weight + 1); reference uses weight (w + 1). Fresh tensors
    # because run_cu above rewrites input/res in place.
    gx = torch.randn(dim, dtype=dtype, device="cuda")
    gr = torch.randn(dim, dtype=dtype, device="cuda")
    gout = torch.empty_like(gx)
    gres = torch.empty_like(gx)
    aiter.rmsnorm2d_fwd_with_add(gout, gx, gr, gres, weight, 1e-5, gemma_norm=True)
    (g, gres_ref, *_), _ = run_torch(gx, (weight + 1).to(dtype), 1e-5, residual=gr)
    checkAllclose(g, gout, atol=0.03, msg="gemma out")
    checkAllclose(gres_ref, gres, msg="gemma res check")


# Rows whose byte length is not a multiple of 4 (#5044). The HIP rmsnorm kernel
# bounded each row buffer in whole dwords, so an unaligned row reached into the
# row after it -- reading its first element into the reduction and writing over
# it -- and reached past the tensor on the last row. Guard rows catch the write
# past the end; comparing every row against torch catches the write into the
# next row, which lands inside the tensor and so passes a guard check.
_GUARD_BYTE = 0x5A


def _with_guard_row(m, n, dtype):
    storage = torch.empty((m + 1, n), dtype=dtype, device="cuda")
    guard = storage[-1:].view(torch.uint8)
    guard.fill_(_GUARD_BYTE)
    return storage[:-1], guard


def _assert_guard_intact(guard, what):
    changed = torch.count_nonzero(guard != _GUARD_BYTE).item()
    assert changed == 0, f"{what}: {changed} bytes written past the last row"


def test_rmsnorm2d_unaligned(dtype, m, n):
    input, _ = _with_guard_row(m, n, dtype)
    input.normal_()
    weight = torch.randn(n, dtype=dtype, device="cuda")

    ref = F.rms_norm(input=input, normalized_shape=(n,), weight=weight, eps=1e-5)
    got = aiter.rms_norm(input, weight, 1e-5)
    torch.testing.assert_close(
        got, ref, atol=0.01, rtol=0.01, msg=f"rms_norm dim=({m}, {n}) dtype={dtype}"
    )
    print(f"[pass] rms_norm            dim: ({m}, {n}), dtype: {dtype}")


def test_rmsnorm2d_fuseAdd_unaligned(dtype, m, n):
    input, _ = _with_guard_row(m, n, dtype)
    input.normal_()
    residual, _ = _with_guard_row(m, n, dtype)
    residual.normal_()
    weight = torch.randn(n, dtype=dtype, device="cuda")
    out, out_guard = _with_guard_row(m, n, dtype)
    residual_out, residual_out_guard = _with_guard_row(m, n, dtype)

    aiter.rmsnorm2d_fwd_with_add(out, input, residual, residual_out, weight, 1e-5)

    residual_ref = input + residual
    ref = F.rms_norm(input=residual_ref, normalized_shape=(n,), weight=weight, eps=1e-5)
    where = f"dim=({m}, {n}) dtype={dtype}"
    torch.testing.assert_close(
        out, ref, atol=0.03, rtol=0.01, msg=f"rmsnorm2d_fwd_with_add out {where}"
    )
    torch.testing.assert_close(
        residual_out, residual_ref, msg=f"rmsnorm2d_fwd_with_add residual {where}"
    )
    _assert_guard_intact(out_guard, f"rmsnorm2d_fwd_with_add out {where}")
    _assert_guard_intact(
        residual_out_guard, f"rmsnorm2d_fwd_with_add residual_out {where}"
    )
    print(f"[pass] rmsnorm2d_fwd_with_add dim: ({m}, {n}), dtype: {dtype}")


# for dtype in [dtypes.fp16, dtypes.bf16]:
#     for m in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
#         for n in [4096, 8192, 16384, 32768, 65536]:
#             test_rmsnorm2d(dtype, m, n)

l_dtype = ["fp16", "bf16", "fp32"]
l_m = [1, 2, 4, 8, 16, 32, 64, 128, 256]
l_n = [4096, 8192, 16384, 32768, 65536]
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
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
    "--m",
    type=int,
    nargs="?",
    default=None,
    help="""M of mnk.
    e.g.: -m 32""",
)
parser.add_argument(
    "-n",
    "--n",
    type=int,
    nargs="?",
    default=None,
    help="""N of mnk.
    e.g.: -n 1024""",
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

print("\nstart fuse add test")
for dtype in l_dtype:
    for m in l_m:
        for n in l_n:
            test_rmsnorm2d_fuseAdd(dtype, m, n)

# One n per bin of the kernel's n<=512/1024/2048/4096/6144/8192 dispatch, plus the
# 7x769 shape from #5044. fp32 is routed to the opus backend, which is unaffected.
print("\nstart unaligned hidden-size test")
l_n_unaligned = [769, 1023, 2047, 4095, 6143, 8191]
l_m_unaligned = [1, 7, 33]
for dtype in [d for d in l_dtype if d.itemsize == 2]:
    for m in l_m_unaligned:
        for n in l_n_unaligned:
            test_rmsnorm2d_unaligned(dtype, m, n)
            test_rmsnorm2d_fuseAdd_unaligned(dtype, m, n)
