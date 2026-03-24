# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse

import torch
import pandas as pd

import aiter
from aiter import dtypes, rmsnorm2d_fwd
from aiter.test_common import benchmark, checkAllclose, perftest


def rms_norm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_f = x.float()
    w_f = weight.float()
    inv_rms = torch.rsqrt((x_f * x_f).mean(dim=-1, keepdim=True) + eps)
    return (x_f * inv_rms * w_f).to(dtype=x.dtype)


@perftest()
def run_aiter_split_qk_rmsnorm(
    q: torch.Tensor,
    q_weight: torch.Tensor,
    q_eps: float,
    k: torch.Tensor,
    k_weight: torch.Tensor,
    k_eps: float,
):
    q_ref = rmsnorm2d_fwd(q, q_weight, q_eps)
    k_ref = rmsnorm2d_fwd(k, k_weight, k_eps)
    return q_ref, k_ref


@perftest()
def run_aiter_fused_qk_rmsnorm(
    q: torch.Tensor,
    q_weight: torch.Tensor,
    q_eps: float,
    k: torch.Tensor,
    k_weight: torch.Tensor,
    k_eps: float,
):
    q_out, k_out = aiter.fused_qk_rmsnorm(q, q_weight, q_eps, k, k_weight, k_eps)
    return q_out, k_out


@benchmark()
def test_fused_qk_rmsnorm(
    dtype: torch.dtype,
    m: int,
    n1: int,
    n2: int,
    q_eps: float = 1e-5,
    k_eps: float = 1e-5,
):
    q = torch.randn((m, n1), dtype=dtype, device="cuda")
    k = torch.randn((m, n2), dtype=dtype, device="cuda")
    q_out = q.clone()
    k_out = k.clone()
    q_weight = torch.randn((n1,), dtype=dtype, device="cuda")
    k_weight = torch.randn((n2,), dtype=dtype, device="cuda")

    # Keep perf benchmark separate from accuracy check because fused_qk_rmsnorm
    # updates input tensors in-place.
    (q_ref, k_ref), avg_ref = run_aiter_split_qk_rmsnorm(
        q, q_weight, q_eps, k, k_weight, k_eps
    )
    (_, _), avg_opt = run_aiter_fused_qk_rmsnorm(q, q_weight, q_eps, k, k_weight, k_eps)

    q_out, k_out = aiter.fused_qk_rmsnorm(
        q_out, q_weight, q_eps, k_out, k_weight, k_eps
    )

    info = f"dtype:{dtype}, M:{m}, N1:{n1}, N2:{n2}"
    msg = (
        f"[perf] === {info} === "
        f"split_kernel avg: {avg_ref:<8.2f} us, fused_kernel avg: {avg_opt:<8.2f} us, "
        f"uplift: {avg_ref / avg_opt - 1:<5.1%}"
    )

    checkAllclose(q_ref, q_out, msg=f"{msg} (q)", rtol=1e-2, atol=1e-2)
    checkAllclose(k_ref, k_out, msg=f"{msg} (k)", rtol=1e-2, atol=1e-2)

    return {
        "dtype": str(dtype),
        "M": m,
        "N1": n1,
        "N2": n2,
        "split_kernel_us": avg_ref,
        "fused_kernel_us": avg_opt,
        "uplift": f"{avg_ref / avg_opt - 1:<5.1%}",
    }


l_dtype = ["fp16", "bf16"]
l_m = [1, 4, 5, 64, 1024, 8192]
l_n1 = [1024, 1536]
l_n2 = [512, 1024]

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Test fused_qk_rmsnorm op",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=l_dtype,
    nargs="*",
    default=None,
    help="Data type(s). e.g. -d bf16 or -d bf16 fp16",
)
parser.add_argument("-m", "--m", type=int, nargs="*", default=None, help="Rows M")
parser.add_argument("-n1", "--n1", type=int, nargs="*", default=None, help="Columns N1")
parser.add_argument("-n2", "--n2", type=int, nargs="*", default=None, help="Columns N2")
args = parser.parse_args()

if args.dtype is None:
    dtypes_to_test = [dtypes.d_dtypes[key] for key in l_dtype]
else:
    dtypes_to_test = [dtypes.d_dtypes[key] for key in args.dtype]
if args.m is not None:
    l_m = args.m
if args.n1 is not None:
    l_n1 = args.n1
if args.n2 is not None:
    l_n2 = args.n2

df = []
for dtype in dtypes_to_test:
    for m in l_m:
        for n1 in l_n1:
            for n2 in l_n2:
                ret = test_fused_qk_rmsnorm(dtype=dtype, m=m, n1=n1, n2=n2)
                df.append(ret)
df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("fused_qk_norm summary (markdown):\n%s", df_md)
