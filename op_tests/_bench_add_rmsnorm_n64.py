# SPDX-License-Identifier: MIT
# Temporary local benchmark for Blaze-O1 N=64 fused add RMSNorm.

import os

os.environ.setdefault("ROCR_VISIBLE_DEVICES", "0")

import torch
import torch.nn.functional as F
import triton

import aiter
from aiter.ops.triton._triton_kernels.normalization.fused_rmsnorm_add import (
    _triton_fused_rms_kernel,
)
from aiter.test_common import checkAllclose, run_perftest

EPS = 0.0078125
H20 = {59392: 7.744, 32768: 4.288}
CUST_TRITON = {59392: 13.321, 32768: 10.881}
CUST_HIP = {59392: 43.76, 32768: 22.2}


def run_hip(out, x, residual, residual_out, weight, eps):
    aiter.add_rmsnorm(out, x, residual, residual_out, weight, eps)
    return out, residual_out


def run_triton_row(out, x, residual, residual_out, weight, eps):
    m, n = x.shape
    block_n = max(triton.next_power_of_2(n), 32)
    _triton_fused_rms_kernel[(m,)](
        x,
        weight,
        residual,
        out,
        residual_out,
        eps,
        m,
        n,
        x.stride(0),
        residual.stride(0),
        out.stride(0),
        residual_out.stride(0),
        BLOCK_SIZE_N=block_n,
        FIRST_INPUT_RES=True,
    )
    return out, residual_out


def bench_one(m, n=64, run_triton=False):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    x = torch.randn((m, n), dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn(n, dtype=dtype, device="cuda")
    out = torch.empty_like(x)
    residual_out = torch.empty_like(x)

    ref_res = x + residual
    ref_out = F.rms_norm(ref_res, (n,), weight=weight, eps=EPS)

    print(f"\n=== JIT/warmup HIP m={m} ===", flush=True)
    run_hip(out, x, residual, residual_out, weight, EPS)
    torch.cuda.synchronize()

    (hip_out, hip_res), hip_us = run_perftest(
        run_hip,
        out,
        x,
        residual,
        residual_out,
        weight,
        EPS,
        num_warmup=20,
        num_iters=100,
    )
    hip_out_err = checkAllclose(ref_out, hip_out, msg=f"hip out m={m}")
    hip_res_err = checkAllclose(ref_res, hip_res, msg=f"hip res m={m}")

    tri_us = float("nan")
    tri_out_err = float("nan")
    tri_res_err = float("nan")
    if run_triton:
        print(f"=== JIT/warmup Triton-row m={m} ===", flush=True)
        run_triton_row(out, x, residual, residual_out, weight, EPS)
        torch.cuda.synchronize()
        (tri_out, tri_res), tri_us = run_perftest(
            run_triton_row,
            out,
            x,
            residual,
            residual_out,
            weight,
            EPS,
            num_warmup=20,
            num_iters=100,
        )
        tri_out_err = checkAllclose(ref_out, tri_out, msg=f"triton out m={m}")
        tri_res_err = checkAllclose(ref_res, tri_res, msg=f"triton res m={m}")

    bytes_moved = m * n * 8
    return {
        "m": m,
        "hip_us": hip_us,
        "triton_row_us": tri_us,
        "h20_us": H20[m],
        "cust_triton_us": CUST_TRITON[m],
        "cust_hip_old_us": CUST_HIP[m],
        "hip_out_err": float(hip_out_err),
        "hip_res_err": float(hip_res_err),
        "tri_out_err": float(tri_out_err),
        "tri_res_err": float(tri_res_err),
        "hip_GBs": bytes_moved / (hip_us * 1e-6) / 1e9,
        "h20_GBs": bytes_moved / (H20[m] * 1e-6) / 1e9,
        "hip_vs_h20": hip_us / H20[m],
        "hip_vs_cust_old": hip_us / CUST_HIP[m],
    }


def main():
    print("gpu:", torch.cuda.get_device_name(0))
    print("aiter:", aiter.__file__)
    print("torch:", torch.__version__, "hip:", torch.version.hip)

    run_triton = os.environ.get("BENCH_TRITON", "0") == "1"
    rows = [bench_one(59392, run_triton=run_triton), bench_one(32768, run_triton=run_triton)]
    print("\n===== SUMMARY =====")
    print(
        f"{'m':>8} {'HIP_us':>10} {'H20':>8} {'custTRI':>10} {'custHIP':>10} "
        f"{'HIP/H20':>8} {'HIP/custHIP':>12}"
    )
    for r in rows:
        print(
            f"{r['m']:8d} {r['hip_us']:10.3f} {r['h20_us']:8.3f} "
            f"{r['cust_triton_us']:10.3f} {r['cust_hip_old_us']:10.2f} "
            f"{r['hip_vs_h20']:8.2f} {r['hip_vs_cust_old']:12.2f}"
        )
    print("\nraw:")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
