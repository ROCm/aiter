# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Triton Grouped GEMM Benchmark.

Benchmarks grouped_gemm_fprop / grouped_gemm_dgrad / grouped_gemm_wgrad from
triton_grouped_gemm_ops.py and reports time (ms) and TFLOPS for each pass.
"""

import argparse
import sys
from datetime import datetime

import pandas as pd
import torch
import torch.utils.benchmark as benchmark
from tabulate import tabulate

from triton_grouped_gemm_ops import (
    grouped_gemm_dgrad,
    grouped_gemm_fprop,
    grouped_gemm_wgrad,
)


# ---------------------------------------------------------------------------
# Reference implementations (loop over groups, plain torch.matmul)
# ---------------------------------------------------------------------------


def _ref_fprop(x, w, split_sizes):
    """out[g] = x[g] @ w[g]^T"""
    outs = []
    offset = 0
    for g, m_g in enumerate(split_sizes.tolist()):
        if m_g > 0:
            outs.append(x[offset : offset + m_g] @ w[g].T)
        offset += m_g
    return torch.cat(outs, dim=0)


def _ref_dgrad(dy, w, split_sizes):
    """dx[g] = dy[g] @ w[g]"""
    outs = []
    offset = 0
    for g, m_g in enumerate(split_sizes.tolist()):
        if m_g > 0:
            outs.append(dy[offset : offset + m_g] @ w[g])
        offset += m_g
    return torch.cat(outs, dim=0)


def _ref_wgrad(dy, x, split_sizes):
    """dw[g] = dy[g]^T @ x[g]"""
    G = split_sizes.numel()
    N = dy.shape[1]
    K = x.shape[1]
    dw = torch.zeros(G, N, K, device=dy.device, dtype=dy.dtype)
    offset = 0
    for g, m_g in enumerate(split_sizes.tolist()):
        if m_g > 0:
            dw[g] = dy[offset : offset + m_g].T @ x[offset : offset + m_g]
        offset += m_g
    return dw


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_split_sizes(G, M, balanced, device):
    """Generate group sizes that sum to G*M."""
    total = G * M
    if balanced:
        base = total // G
        rem = total % G
        sizes = [base + (1 if i < rem else 0) for i in range(G)]
    else:
        torch.manual_seed(42)
        raw = torch.randint(1, M * 2, (G,), dtype=torch.int64)
        raw = (raw.float() / raw.sum() * total).long()
        raw[-1] = total - raw[:-1].sum()
        raw = raw.clamp(min=0)
        sizes = raw.tolist()
    return torch.tensor(sizes, dtype=torch.int64, device=device)


def _compute_snr(ref, test):
    """Signal-to-noise ratio in dB."""
    noise = (ref.float() - test.float()).norm()
    signal = ref.float().norm()
    if noise == 0:
        return float("inf")
    return 20 * torch.log10(signal / noise).item()


def get_platform_info():
    """Return (platform_string, gpu_name_slug)."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        slug = gpu_name.replace(" ", "_").replace("/", "_")
        return gpu_name, slug
    return "unknown", "unknown"


def gen_grouped_gemm_test_cases():
    """Representative (G, M, N, K) test cases for grouped GEMM."""
    cases = []
    for G in [1, 2, 4, 8]:
        for M in [128, 256, 512]:
            for N, K in [(2048, 1536), (4096, 4096), (1024, 2048)]:
                cases.append({"Case": f"G{G}_M{M}_N{N}_K{K}", "G": G, "M": M, "N": N, "K": K})
    return cases


# ---------------------------------------------------------------------------
# Correctness checks
# ---------------------------------------------------------------------------


def _check_correctness(out, ref, dtype, label):
    snr = _compute_snr(ref, out)
    threshold = 40 if dtype == torch.bfloat16 else 45
    ok = snr > threshold
    status = "PASS" if ok else "FAIL"
    print(f"  {label}: {status} (SNR={snr:.1f} dB, threshold={threshold} dB)")
    return ok


def check_correctness(G, M, N, K, dtype, balanced):
    """Run fprop / dgrad / wgrad correctness checks, return (fprop_ok, dgrad_ok, wgrad_ok)."""
    device = "cuda"
    split_sizes = _make_split_sizes(G, M, balanced, device)
    GM = int(split_sizes.sum().item())

    x = torch.randn(GM, K, device=device, dtype=dtype)
    w = torch.randn(G, N, K, device=device, dtype=dtype)
    dy = torch.randn(GM, N, device=device, dtype=dtype)

    fprop_ok = _check_correctness(grouped_gemm_fprop(x, w, split_sizes), _ref_fprop(x, w, split_sizes), dtype, "fprop")
    dgrad_ok = _check_correctness(grouped_gemm_dgrad(dy, w, split_sizes), _ref_dgrad(dy, w, split_sizes), dtype, "dgrad")
    wgrad_ok = _check_correctness(grouped_gemm_wgrad(dy, x, split_sizes), _ref_wgrad(dy, x, split_sizes), dtype, "wgrad")

    return fprop_ok, dgrad_ok, wgrad_ok


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


def profile_triton_grouped_gemm(G, M, N, K, dtype, balanced=True, warmup=20, iters=100):
    """Profile fprop / dgrad / wgrad for a single (G, M, N, K) configuration.

    Returns a dict with time (ms) and TFLOPS for each pass.
    """
    device = "cuda"
    split_sizes = _make_split_sizes(G, M, balanced, device)
    GM = int(split_sizes.sum().item())

    x = torch.randn(GM, K, device=device, dtype=dtype)
    w = torch.randn(G, N, K, device=device, dtype=dtype)
    dy_n = torch.randn(GM, N, device=device, dtype=dtype)

    # fprop: (GM, K) x (G, N, K)^T  -> (GM, N)  : 2 * GM * N * K FLOPs
    # dgrad: (GM, N) x (G, N, K)    -> (GM, K)  : 2 * GM * N * K FLOPs
    # wgrad: (N, GM) x (GM, K)      -> (G, N, K): 2 * GM * N * K FLOPs
    flops = 2 * GM * N * K

    # Correctness
    print("  Correctness:")
    fprop_ok, dgrad_ok, wgrad_ok = check_correctness(G, M, N, K, dtype, balanced)

    # Warm-up
    for _ in range(warmup):
        grouped_gemm_fprop(x, w, split_sizes)
        grouped_gemm_dgrad(dy_n, w, split_sizes)
        grouped_gemm_wgrad(dy_n, x, split_sizes)
    torch.cuda.synchronize()

    def fprop_fn():
        return grouped_gemm_fprop(x, w, split_sizes)

    def dgrad_fn():
        return grouped_gemm_dgrad(dy_n, w, split_sizes)

    def wgrad_fn():
        return grouped_gemm_wgrad(dy_n, x, split_sizes)

    fprop_timer = benchmark.Timer(stmt="fn()", globals={"fn": fprop_fn})
    dgrad_timer = benchmark.Timer(stmt="fn()", globals={"fn": dgrad_fn})
    wgrad_timer = benchmark.Timer(stmt="fn()", globals={"fn": wgrad_fn})

    fprop_ms = fprop_timer.timeit(iters).mean * 1e3
    dgrad_ms = dgrad_timer.timeit(iters).mean * 1e3
    wgrad_ms = wgrad_timer.timeit(iters).mean * 1e3

    fprop_tflops = flops / (fprop_ms * 1e-3) / 1e12
    dgrad_tflops = flops / (dgrad_ms * 1e-3) / 1e12
    wgrad_tflops = flops / (wgrad_ms * 1e-3) / 1e12

    print(f"  fprop  Mean time: {fprop_ms:.3f} ms | TFLOPS: {fprop_tflops:.2f}")
    print(f"  dgrad  Mean time: {dgrad_ms:.3f} ms | TFLOPS: {dgrad_tflops:.2f}")
    print(f"  wgrad  Mean time: {wgrad_ms:.3f} ms | TFLOPS: {wgrad_tflops:.2f}")

    return {
        "fprop_ok": fprop_ok,
        "dgrad_ok": dgrad_ok,
        "wgrad_ok": wgrad_ok,
        "fprop_ms": fprop_ms,
        "fprop_tflops": fprop_tflops,
        "dgrad_ms": dgrad_ms,
        "dgrad_tflops": dgrad_tflops,
        "wgrad_ms": wgrad_ms,
        "wgrad_tflops": wgrad_tflops,
    }


# ---------------------------------------------------------------------------
# Main benchmark driver
# ---------------------------------------------------------------------------


def benchmark_triton_grouped_gemm(dtype_name="bf16", balanced=True, output_csv=None):
    gpu_name, gpu_slug = get_platform_info()
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16

    test_cases = gen_grouped_gemm_test_cases()

    rows = []
    test_id = 0

    for case in test_cases:
        test_id += 1
        G, M, N, K = case["G"], case["M"], case["N"], case["K"]
        GM = G * M

        print(f"\n{'='*60}")
        print(
            f"TestID: {test_id}, Case: {case['Case']}, G: {G}, M: {M}, N: {N}, K: {K}, "
            f"GM: {GM}, dtype: {dtype_name}, balanced: {balanced}"
        )
        print(f"{'='*60}")

        try:
            results = profile_triton_grouped_gemm(G, M, N, K, dtype, balanced=balanced)

            all_ok = results["fprop_ok"] and results["dgrad_ok"] and results["wgrad_ok"]
            check_str = "PASS" if all_ok else "FAIL"

            row = {
                "TestID": test_id,
                "GPU": gpu_name,
                "Case": case["Case"],
                "G": G,
                "M": M,
                "N": N,
                "K": K,
                "GM": GM,
                "Dtype": dtype_name,
                "Balanced": balanced,
                "Check": check_str,
                "fprop Check": "PASS" if results["fprop_ok"] else "FAIL",
                "dgrad Check": "PASS" if results["dgrad_ok"] else "FAIL",
                "wgrad Check": "PASS" if results["wgrad_ok"] else "FAIL",
                "fprop Time (ms)": f"{results['fprop_ms']:.3f}",
                "fprop TFLOPS": f"{results['fprop_tflops']:.2f}",
                "dgrad Time (ms)": f"{results['dgrad_ms']:.3f}",
                "dgrad TFLOPS": f"{results['dgrad_tflops']:.2f}",
                "wgrad Time (ms)": f"{results['wgrad_ms']:.3f}",
                "wgrad TFLOPS": f"{results['wgrad_tflops']:.2f}",
            }

        except Exception as e:
            import traceback

            print(f"Failed: {e}")
            traceback.print_exc()
            row = {
                "TestID": test_id,
                "GPU": gpu_name,
                "Case": case["Case"],
                "G": G,
                "M": M,
                "N": N,
                "K": K,
                "GM": GM,
                "Dtype": dtype_name,
                "Balanced": balanced,
                "Check": "ERROR",
                "fprop Check": "ERROR",
                "dgrad Check": "ERROR",
                "wgrad Check": "ERROR",
                "fprop Time (ms)": "ERROR",
                "fprop TFLOPS": "0.00",
                "dgrad Time (ms)": "ERROR",
                "dgrad TFLOPS": "0.00",
                "wgrad Time (ms)": "ERROR",
                "wgrad TFLOPS": "0.00",
            }

        rows.append(row)

    df = pd.DataFrame(rows)
    print("\nFinal Results:")
    print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))

    for op in ("fprop", "dgrad", "wgrad"):
        avg = df[f"{op} TFLOPS"].astype(float).mean()
        print(f"Average {op} TFLOPS: {avg:.2f}")

    if output_csv:
        filename = output_csv
    else:
        timestamp = datetime.now().strftime("%Y%m%d")
        bal_tag = "balanced" if balanced else "unbalanced"
        filename = f"triton_grouped_gemm_{dtype_name}_{bal_tag}_{timestamp}_{gpu_slug}.csv"

    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Triton Grouped GEMM (fprop/dgrad/wgrad)")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16"],
        default="bf16",
        help="Data type: bf16 or fp16 (default: bf16)",
    )
    parser.add_argument(
        "--unbalanced",
        action="store_true",
        default=False,
        help="Use unbalanced group sizes (default: balanced)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output CSV filename",
    )
    args = parser.parse_args()
    benchmark_triton_grouped_gemm(
        dtype_name=args.dtype,
        balanced=not args.unbalanced,
        output_csv=args.output,
    )
