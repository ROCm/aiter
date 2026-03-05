###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""TransformerEngine Grouped GEMM Baseline Benchmark.

Reference: https://github.com/ROCm/TransformerEngine/blob/dev/benchmarks/linear/benchmark_grouped_linear.py
"""

import argparse
from datetime import datetime

import pandas as pd
import torch
import torch.utils.benchmark as benchmark
import transformer_engine as te
from tabulate import tabulate
from transformer_engine.common.recipe import (
    Float8BlockScaling,
    Float8CurrentScaling,
    Format,
    MXFP8BlockScaling,
)

# ---------------------------------------------------------------------------
# Helpers (no external config dependency)
# ---------------------------------------------------------------------------


def get_platform_info():
    """Return (platform_string, gpu_name_slug)."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        slug = gpu_name.replace(" ", "_").replace("/", "_")
        return gpu_name, slug
    return "unknown", "unknown"


def gen_grouped_gemm_test_cases():
    """Representative (B, M, N, K) test cases for grouped GEMM."""
    cases = []
    for B in [1, 2, 4, 8]:
        for M in [128, 256, 512]:
            for N, K in [(2048, 1536), (4096, 4096), (1024, 2048)]:
                cases.append({"Case": f"B{B}_M{M}_N{N}_K{K}", "B": B, "M": M, "N": N, "K": K})
    return cases


def grouped_gemm_ref(x, w, group_lens, trans_b=False):
    """Reference grouped GEMM: loop over groups, use torch.matmul."""
    outs = []
    offset = 0
    for g, m_g in enumerate(group_lens.tolist()):
        if m_g > 0:
            w_g = w[g].T if trans_b else w[g]
            outs.append(x[offset : offset + m_g] @ w_g)
        offset += m_g
    return torch.cat(outs, dim=0)


def compute_snr(ref, test):
    """Signal-to-noise ratio in dB."""
    noise = (ref.float() - test.float()).norm()
    signal = ref.float().norm()
    if noise == 0:
        return float("inf")
    return 20 * torch.log10(signal / noise).item()


def check_allclose(a, b, dtype):
    """Return True if tensors are close enough for the given dtype."""
    atol = 1e-2 if dtype == torch.bfloat16 else 1e-5
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-5
    return torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------

FP8_RECIPE_MAP = {
    "tensorwise": Float8CurrentScaling(fp8_format=Format.E4M3),
    "blockwise": Float8BlockScaling(fp8_format=Format.E4M3),
    "mx": MXFP8BlockScaling(fp8_format=Format.E4M3),
}


def check_grouped_gemm_te_correctness(layer, x, m_splits, grad_out, dtype):
    """Check correctness of TE GroupedLinear forward and backward (BF16)."""
    out = layer(x, m_splits)

    x_flat = x.detach()
    weights = []
    for i in range(len(m_splits)):
        weight_i = getattr(layer, f"weight{i}")
        weights.append(weight_i.detach().unsqueeze(0))
    w_stacked = torch.cat(weights, dim=0)
    group_lens = torch.tensor(m_splits, dtype=torch.int64, device=x.device)
    out_ref = grouped_gemm_ref(x_flat, w_stacked, group_lens, trans_b=True)
    fwd_correct = check_allclose(out.detach(), out_ref, dtype)

    x_ref = x.detach().clone().requires_grad_()
    w_ref = w_stacked.clone().requires_grad_()
    out_ref = grouped_gemm_ref(x_ref, w_ref, group_lens, trans_b=True)
    out_ref.backward(grad_out)
    out.backward(grad_out, retain_graph=True)
    bwd_correct = check_allclose(x.grad, x_ref.grad, dtype)

    x.grad = None
    layer.zero_grad()

    correct = fwd_correct and bwd_correct
    status = "PASS" if correct else "FAIL"
    print(f"Correctness Check: {status} (fwd={fwd_correct}, bwd={bwd_correct})")

    return correct


def check_grouped_gemm_te_fp8_correctness(layer, x, m_splits, grad_out, fp8_recipe):
    """Check correctness of TE GroupedLinear forward and backward (FP8) using SNR."""
    snr_threshold = 20

    with te.pytorch.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        out = layer(x, m_splits)

    x_flat = x.detach()
    weights = []
    for i in range(len(m_splits)):
        weight_i = getattr(layer, f"weight{i}")
        weights.append(weight_i.detach().unsqueeze(0))
    w_stacked = torch.cat(weights, dim=0)
    group_lens = torch.tensor(m_splits, dtype=torch.int64, device=x.device)
    out_ref = grouped_gemm_ref(x_flat, w_stacked, group_lens, trans_b=True)
    out_snr = compute_snr(out_ref, out.detach())

    x_ref = x.detach().clone().requires_grad_()
    w_ref = w_stacked.clone().requires_grad_()
    out_ref = grouped_gemm_ref(x_ref, w_ref, group_lens, trans_b=True)
    out_ref.backward(grad_out)
    out.backward(grad_out, retain_graph=True)
    dx_snr = compute_snr(x_ref.grad, x.grad)

    x.grad = None
    layer.zero_grad()

    correct = all(snr > snr_threshold for snr in [out_snr, dx_snr])
    status = "PASS" if correct else "FAIL"
    print(f"Correctness Check: {status} (out={out_snr:.1f}, dx={dx_snr:.1f}) threshold={snr_threshold}")

    return correct


def profile_grouped_gemm_te(B, M, N, K, dtype):
    """Profile BF16 Grouped GEMM using TE GroupedLinear."""
    device = "cuda"
    num_gemms = B
    in_features = K
    out_features = N
    total_m = B * M
    m_splits = [M] * B

    x = torch.randn((total_m, in_features), dtype=dtype, device=device, requires_grad=True)
    layer = te.pytorch.GroupedLinear(num_gemms, in_features, out_features, bias=False, params_dtype=dtype).to(
        device
    )

    out = layer(x, m_splits)
    grad_out = torch.randn_like(out)

    correct = check_grouped_gemm_te_correctness(layer, x, m_splits, grad_out, dtype)

    def fwd_func():
        return layer(x, m_splits)

    def fwd_bwd_func():
        out = layer(x, m_splits)
        out.backward(grad_out)

    fwd_total_flops = 2 * B * M * N * K
    bwd_total_flops = 2 * fwd_total_flops

    for _ in range(20):
        fwd_bwd_func()
    torch.cuda.synchronize()

    fwd_timer = benchmark.Timer(stmt="fn()", globals={"fn": fwd_func})
    fwd_measurement = fwd_timer.timeit(100)

    fwd_bwd_timer = benchmark.Timer(stmt="fn()", globals={"fn": fwd_bwd_func})
    fwd_bwd_measurement = fwd_bwd_timer.timeit(100)

    fwd_mean_time_ms = fwd_measurement.mean * 1e3
    fwd_bwd_mean_time_ms = fwd_bwd_measurement.mean * 1e3
    bwd_mean_time_ms = fwd_bwd_mean_time_ms - fwd_mean_time_ms

    fwd_tflops = fwd_total_flops / (fwd_mean_time_ms * 1e-3) / 1e12
    bwd_tflops = bwd_total_flops / (bwd_mean_time_ms * 1e-3) / 1e12
    print(f"Forward  Mean time: {fwd_mean_time_ms:.3f} ms | TFLOPS: {fwd_tflops:.2f}")
    print(f"Backward Mean time: {bwd_mean_time_ms:.3f} ms | TFLOPS: {bwd_tflops:.2f}")

    return fwd_mean_time_ms, fwd_tflops, bwd_mean_time_ms, bwd_tflops, correct


def profile_grouped_gemm_te_fp8(B, M, N, K, dtype, fp8_recipe):
    """Profile FP8 Grouped GEMM using TE GroupedLinear with fp8_autocast."""
    device = "cuda"
    num_gemms = B
    in_features = K
    out_features = N
    total_m = B * M
    m_splits = [M] * B

    x = torch.randn((total_m, in_features), dtype=dtype, device=device, requires_grad=True)
    layer = te.pytorch.GroupedLinear(num_gemms, in_features, out_features, bias=False, params_dtype=dtype).to(
        device
    )

    with te.pytorch.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        out = layer(x, m_splits)
    grad_out = torch.randn_like(out)

    correct = check_grouped_gemm_te_fp8_correctness(layer, x, m_splits, grad_out, fp8_recipe)

    def fwd_func():
        with te.pytorch.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            return layer(x, m_splits)

    def fwd_bwd_func():
        with te.pytorch.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            out = layer(x, m_splits)
        out.backward(grad_out)

    fwd_total_flops = 2 * B * M * N * K
    bwd_total_flops = 2 * fwd_total_flops

    for _ in range(20):
        fwd_bwd_func()
    torch.cuda.synchronize()

    fwd_timer = benchmark.Timer(stmt="fn()", globals={"fn": fwd_func})
    fwd_measurement = fwd_timer.timeit(100)

    fwd_bwd_timer = benchmark.Timer(stmt="fn()", globals={"fn": fwd_bwd_func})
    fwd_bwd_measurement = fwd_bwd_timer.timeit(100)

    fwd_mean_time_ms = fwd_measurement.mean * 1e3
    fwd_bwd_mean_time_ms = fwd_bwd_measurement.mean * 1e3
    bwd_mean_time_ms = fwd_bwd_mean_time_ms - fwd_mean_time_ms

    fwd_tflops = fwd_total_flops / (fwd_mean_time_ms * 1e-3) / 1e12
    bwd_tflops = bwd_total_flops / (bwd_mean_time_ms * 1e-3) / 1e12
    print(f"Forward  Mean time: {fwd_mean_time_ms:.3f} ms | TFLOPS: {fwd_tflops:.2f}")
    print(f"Backward Mean time: {bwd_mean_time_ms:.3f} ms | TFLOPS: {bwd_tflops:.2f}")

    return fwd_mean_time_ms, fwd_tflops, bwd_mean_time_ms, bwd_tflops, correct


def benchmark_grouped_gemm_te(dtype_name="bf16", granularity_name="tensorwise", output_csv=None):
    gpu_name, gpu_slug = get_platform_info()

    is_fp8 = dtype_name == "fp8"
    fp8_recipe = FP8_RECIPE_MAP.get(granularity_name) if is_fp8 else None

    test_cases = gen_grouped_gemm_test_cases()

    rows = []
    test_id = 0
    dtype = torch.bfloat16

    for case in test_cases:
        test_id += 1
        B, M, N, K = case["B"], case["M"], case["N"], case["K"]

        print(f"\n{'='*60}")
        if is_fp8:
            print(
                f"TestID: {test_id}, Case: {case['Case']}, B: {B}, M: {M}, N: {N}, K: {K}, "
                f"dtype: fp8, granularity: {granularity_name}"
            )
        else:
            print(f"TestID: {test_id}, Case: {case['Case']}, B: {B}, M: {M}, N: {N}, K: {K}, dtype: bf16")
        print(f"{'='*60}")

        try:
            if is_fp8:
                fwd_time_ms, fwd_tflops, bwd_time_ms, bwd_tflops, correct = profile_grouped_gemm_te_fp8(
                    B=B, M=M, N=N, K=K, dtype=dtype, fp8_recipe=fp8_recipe
                )
            else:
                fwd_time_ms, fwd_tflops, bwd_time_ms, bwd_tflops, correct = profile_grouped_gemm_te(
                    B=B, M=M, N=N, K=K, dtype=dtype
                )

            row = {
                "TestID": test_id,
                "GPU": gpu_name,
                "Case": case["Case"],
                "B": B,
                "M": M,
                "N": N,
                "K": K,
                "Dtype": dtype_name,
            }
            if is_fp8:
                row["Granularity"] = granularity_name
            row.update(
                {
                    "Check": "PASS" if correct else "FAIL",
                    "Forward Time (ms)": f"{fwd_time_ms:.2f}",
                    "Forward TFLOPS": f"{fwd_tflops:.2f}",
                    "Backward Time (ms)": f"{bwd_time_ms:.2f}",
                    "Backward TFLOPS": f"{bwd_tflops:.2f}",
                }
            )
            rows.append(row)

        except Exception as e:
            import traceback

            print(f"Failed: {str(e)}")
            traceback.print_exc()
            row = {
                "TestID": test_id,
                "GPU": gpu_name,
                "Case": case["Case"],
                "B": B,
                "M": M,
                "N": N,
                "K": K,
                "Dtype": dtype_name,
            }
            if is_fp8:
                row["Granularity"] = granularity_name
            row.update(
                {
                    "Check": "ERROR",
                    "Forward Time (ms)": "ERROR",
                    "Forward TFLOPS": "0.00",
                    "Backward Time (ms)": "ERROR",
                    "Backward TFLOPS": "0.00",
                }
            )
            rows.append(row)

    results = pd.DataFrame(rows)
    print("\nFinal Results:")
    print(tabulate(results, headers="keys", tablefmt="grid", showindex=False))

    avg_fwd_tflops = results["Forward TFLOPS"].astype(float).mean()
    avg_bwd_tflops = results["Backward TFLOPS"].astype(float).mean()
    print(f"\nAverage Forward TFLOPS: {avg_fwd_tflops:.2f}")
    print(f"Average Backward TFLOPS: {avg_bwd_tflops:.2f}")

    if output_csv:
        filename = output_csv
    else:
        timestamp = datetime.now().strftime("%Y%m%d")
        if is_fp8:
            filename = f"grouped_gemm_te_fp8_{granularity_name}_{timestamp}_{gpu_slug}.csv"
        else:
            filename = f"grouped_gemm_te_bf16_{timestamp}_{gpu_slug}.csv"
    results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark TransformerEngine Grouped GEMM (GroupedLinear)")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp8"],
        default="bf16",
        help="Data type: bf16 or fp8 (default: bf16)",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        choices=["tensorwise", "blockwise", "mx"],
        default="tensorwise",
        help="FP8 scaling granularity (only used when dtype=fp8, default: tensorwise)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output CSV filename",
    )
    args = parser.parse_args()
    benchmark_grouped_gemm_te(
        dtype_name=args.dtype, granularity_name=args.granularity, output_csv=args.output
    )
