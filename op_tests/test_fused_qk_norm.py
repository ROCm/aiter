# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse

import torch

import aiter
from aiter import dtypes, rmsnorm2d_fwd
from aiter.test_common import benchmark, checkAllclose, perftest


def rms_norm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_f = x.float()
    w_f = weight.float()
    inv_rms = torch.rsqrt((x_f * x_f).mean(dim=-1, keepdim=True) + eps)
    return (x_f * inv_rms * w_f).to(dtype=x.dtype)


@perftest()
def run_torch_split_qk_rmsnorm(
    q: torch.Tensor,
    q_weight: torch.Tensor,
    q_eps: float,
    k: torch.Tensor,
    k_weight: torch.Tensor,
    k_eps: float,
):
    q_ref = rms_norm_forward(q, q_weight, q_eps)
    k_ref = rms_norm_forward(k, k_weight, k_eps)
    return q_ref, k_ref


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
    q_weight = torch.randn((n1,), dtype=dtype, device="cuda")
    k_weight = torch.randn((n2,), dtype=dtype, device="cuda")

    # (q_ref, k_ref), avg_torch = run_torch_split_qk_rmsnorm(
    #     q, q_weight, q_eps, k, k_weight, k_eps
    # )
    (q_ref, k_ref), avg_ref = run_aiter_split_qk_rmsnorm(
        q, q_weight, q_eps, k, k_weight, k_eps
    )
    # Keep perftest path clone-free so measured time is kernel-only.
    (_, _), avg_opt = run_aiter_fused_qk_rmsnorm(
        q.clone(), q_weight, q_eps, k.clone(), k_weight, k_eps
    )
    q_out = q.clone()
    k_out = k.clone()
    aiter.fused_qk_rmsnorm(q_out, q_weight, q_eps, k_out, k_weight, k_eps)

    info = f"dtype:{dtype}, M:{m}, N1:{n1}, N2:{n2}"
    msg = (
        f"[perf] === {info} === "
        f"torch avg: {avg_ref:<8.2f} us, aiter avg: {avg_opt:<8.2f} us, "
        f"uplift: {avg_ref / avg_opt - 1:<5.1%}"
    )

    checkAllclose(q_ref, q_out, msg=f"{msg} (q)", rtol=1e-2, atol=1e-2)
    checkAllclose(k_ref, k_out, msg=f"{msg} (k)", rtol=1e-2, atol=1e-2)

    bytes_q = q.numel() * q.element_size() * 2 + q_weight.numel() * q_weight.element_size()
    bytes_k = k.numel() * k.element_size() * 2 + k_weight.numel() * k_weight.element_size()
    total_bytes = bytes_q + bytes_k
    return {
        "dtype": str(dtype),
        "M": m,
        "N1": n1,
        "N2": n2,
        "fused_qk_us": avg_opt,
        "torch_us": avg_ref,
        "aiter_bw(TB/s)": total_bytes / (avg_opt * 1e6),
    }


if __name__ == "__main__":
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
        nargs="?",
        default=None,
        help="Data type. e.g. -d bf16",
    )
    parser.add_argument("-m", "--m", type=int, nargs="?", default=None, help="Rows M")
    parser.add_argument("-n1", "--n1", type=int, nargs="?", default=None, help="Columns N1")
    parser.add_argument("-n2", "--n2", type=int, nargs="?", default=None, help="Columns N2")
    args = parser.parse_args()

    if args.dtype is None:
        dtypes_to_test = [dtypes.d_dtypes[key] for key in l_dtype]
    else:
        dtypes_to_test = [dtypes.d_dtypes[args.dtype]]
    if args.m is not None:
        l_m = [args.m]
    if args.n1 is not None:
        l_n1 = [args.n1]
    if args.n2 is not None:
        l_n2 = [args.n2]

    for dtype in dtypes_to_test:
        for m in l_m:
            for n1 in l_n1:
                for n2 in l_n2:
                    test_fused_qk_rmsnorm(dtype=dtype, m=m, n1=n1, n2=n2)
