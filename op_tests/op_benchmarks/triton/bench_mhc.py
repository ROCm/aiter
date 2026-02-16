# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for mHC (manifold-constrained Hyper Connection) fused kernel.

Measures performance of the Triton implementation across various input shapes
and configurations, reporting time, throughput (TFLOPS), and bandwidth.

Supports three benchmark modes:
- "mhc_lite": mHC-lite using convex combination of permutations (exact doubly stochastic, faster)
- "mhc": Full mHC with Sinkhorn-Knopp iterations (approximate doubly stochastic)
- "sinkhorn_knopp_only": Benchmark only the Sinkhorn-Knopp kernel in isolation
"""

import sys
import argparse
import logging
from itertools import product
from math import factorial
import torch
import triton

from aiter.ops.triton.fusions.mhc import mhc, fused_mhc, sinkhorn_knopp
from op_tests.triton_tests.utils.mhc_ref import generate_mhc_inputs
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    print_vgpr,
    get_caller_name_no_ext,
)

arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

# Configure logging before importing aiter modules
logging.basicConfig(
    level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s", force=True
)

def get_benchmark_configs(args):
    """Generate list of benchmark configurations based on args."""
    configs = []

    if args.M and args.n and args.C:
        configs.append((args.M, args.n, args.C))
    else:
        # Default configurations - typical mHC usage patterns
        # Format: (M, n, C) where:
        #   M: batch/sequence length
        #   n: stream parameter (manifold dimension)
        #   C: hidden dimension per stream
        Ms = [2**i for i in range(10, 15)]
        n = 4
        Cs = [128, 512, 4096, 2**15]
        # Sort by C (hidden dimension) to show scaling across C values
        configs = sorted(list(product(Ms, [n], Cs)), key=lambda x: (x[2], x[0]))

    return configs


def run_benchmark(args):
    """Run mHC benchmark with specified configuration."""
    dtype = arg_to_torch_dtype[args.dtype]
    mode = args.mode
    sinkhorn_iters = args.sinkhorn_iters

    configs = get_benchmark_configs(args)
    x_vals_list = configs
    x_names = ["M", "n", "C"]

    # Determine which metrics to report (following bench_diffusion_attention pattern)
    if args.metric == "all" or args.metric is None:
        line_vals = [
            "time(ms)",
            "throughput(TFLOPS)",
            "bandwidth(GB/s)",
            "arithmetic_intensity(FLOP/byte)",
        ]
    else:
        metric_map = {
            "time": "time(ms)",
            "throughput": "throughput(TFLOPS)",
            "bandwidth": "bandwidth(GB/s)",
            "arithmetic_intensity": "arithmetic_intensity(FLOP/byte)",
        }
        line_vals = [metric_map.get(args.metric, "throughput(TFLOPS)")]

    benchmark_name = get_caller_name_no_ext()
    if mode == "mhc_lite":
        benchmark_name += "_lite"
    elif mode == "mhc":
        benchmark_name += f"_sinkhorn-{sinkhorn_iters}iters"
    elif mode == "sinkhorn_knopp_only":
        benchmark_name += f"_sinkhorn_only-{sinkhorn_iters}iters"

    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_vals,
        styles=[("red", "-"), ("blue", "-"), ("yellow", "-"), ("green", "-")],
        ylabel="",
        plot_name=benchmark_name,
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench_mhc_kernel(M, n, C, provider):
        """Benchmark mHC kernel for given configuration."""
        # Generate inputs based on mode
        (
            x,
            phi_pre,
            phi_post,
            phi_res,
            alpha_pre,
            alpha_post,
            alpha_res,
            bias,
            n_streams,
        ) = generate_mhc_inputs(M, n, C, dtype, mode=mode)

        # Compute FLOPs for mHC or mHC-Lite operation
        nC = n * C
        n_factorial = factorial(n)
        n_squared = n * n

        # Determine phi_res output dimension based on mode
        n_res = n_factorial if mode == "mhc_lite" else n_squared
        N = n_res + 2 * n

        # Standard GEMM FLOPs (2*M*N*K for matrix multiply)
        # Eq 14: x @ phi for 3 streams
        # - x @ phi_pre: (M, nC) @ (nC, n) = 2*M*nC*n
        # - x @ phi_post: (M, nC) @ (nC, n) = 2*M*nC*n
        # - x @ phi_res: (M, nC) @ (nC, n_res) = 2*M*nC*n_res
        flops_matmul = 2.0 * M * nC * n + 2.0 * M * nC * n + 2.0 * M * nC * n_res

        # Eq 15: RMS normalization - M rows, each with nC elements
        # sqrt(sum(x^2)/K) requires: square + sum + divide + sqrt ≈ 4*nC ops per row
        flops_rms = 4.0 * M * nC

        # Eq 16: Scale and bias addition for 3 streams
        # - Multiply by (alpha/r): M*n + M*n + M*n_res FLOPs
        # - Add bias: M*n + M*n + M*n_res FLOPs
        # flops_scale_bias = 2.0 * M * (2*n + n_res)

        # Eq 17 & 18: Activation functions (fused in kernel, but count FLOPs)
        # - sigmoid(H_pre): ~10 ops per element for M*n elements
        # - 2*sigmoid(H_post): ~11 ops per element for M*n elements
        # - H_res activation: softmax (lite) or none (sinkhorn)
        # flops_activations = 10.0 * M * n + 11.0 * M * n

        # Eq 19: Sinkhorn-Knopp (separate kernel, log-domain implementation)
        # Each iteration: 2 normalizations (row + col) on M matrices of size (n, n)
        # Per normalization (log-domain): add, max, subtract, exp, sum, log operations
        # Operations per iteration per matrix:
        #   - Element-wise: add(n²) + subtract(n²) + exp(n²) ≈ 3n² (×2 for row+col)
        #   - Reductions: max(n) + sum(n) + log(n) ≈ 3n (×2 for row+col)
        # Total: ~(6n² + 6n) ≈ 6n² + 6n per iteration (for n=4: ~108 FLOPs)
        # Simplified: ~10*n² per iteration (accounting for expensive exp/log ops)
        flops_sinkhorn = 10.0 * M * n_squared * sinkhorn_iters

        # Calculate mode-specific FLOPs
        if mode == "mhc_lite":
            # Matrix multiply: (M, n_factorial) @ (n_factorial, n_squared) = 2*M*n_factorial*n_squared
            flops_softmax = 10.0 * M * n_factorial
            flops_matmul_lite = 2.0 * M * n_factorial * n_squared
            flops_lite = flops_softmax + flops_matmul_lite
            total_flops = (
                flops_matmul + flops_rms + flops_lite
            )  # + flops_scale_bias + flops_activations
        elif mode == "mhc":
            total_flops = (
                flops_matmul + flops_rms + flops_sinkhorn
            )  # + flops_scale_bias + flops_activations
        elif mode == "sinkhorn_knopp_only":
            total_flops = flops_sinkhorn

        # Compute memory traffic
        elem_size = 2  # bf16/fp16 = 2 bytes
        bias_size = 4  # bias is fp32 = 4 bytes

        # Memory reads:
        # - x: (M, nC)
        # - phi_pre, phi_post, phi_res: (nC, n), (nC, n), (nC, n_res)
        # - bias: (N,) in fp32
        # - For lite mode: perm_matrices (n_factorial, n_squared)
        mem_read = (
            M * nC * elem_size  # x
            + nC * n * elem_size  # phi_pre
            + nC * n * elem_size  # phi_post
            + nC * n_res * elem_size  # phi_res
            + N * bias_size  # bias
        )
        if mode == "mhc_lite":
            mem_read += n_factorial * n_squared * elem_size  # perm_matrices

        # Memory writes:
        # - out_pre: (M, n)
        # - out_post: (M, n)
        # - out_res: (M, n_squared) - always n_squared for output
        mem_write = (
            M * n * elem_size  # out_pre
            + M * n * elem_size  # out_post
            + M * n_squared * elem_size  # out_res
        )

        # Sinkhorn-Knopp memory traffic (in-place operations)
        if mode == "mhc":
            mem_sinkhorn = 2 * M * n_squared * elem_size * sinkhorn_iters
            mem_read += mem_sinkhorn / 2
            mem_write += mem_sinkhorn / 2
            total_mem = mem_read + mem_write
        elif mode == "mhc_lite":
            total_mem = mem_read + mem_write
        elif mode == "sinkhorn_knopp_only":
            # Sinkhorn-only: read/write H_res matrix
            total_mem = 2 * M * n_squared * elem_size * sinkhorn_iters

        # Create benchmark function
        if mode == "mhc":
            def fn():
                return mhc(
                    x,
                    phi_pre,
                    phi_post,
                    phi_res,
                    alpha_pre,
                    alpha_post,
                    alpha_res,
                    bias,
                    n_streams,
                    hres_mode="sinkhorn",
                    sinkhorn_iters=sinkhorn_iters,
                )
        elif mode == "mhc_lite":
            def fn():
                return mhc(
                    x,
                    phi_pre,
                    phi_post,
                    phi_res,
                    alpha_pre,
                    alpha_post,
                    alpha_res,
                    bias,
                    n_streams,
                    hres_mode="lite",
                )
        elif mode == "sinkhorn_knopp_only":
            # Sinkhorn-only: benchmark just the Sinkhorn-Knopp kernel
            # Pre-generate H_res for fair comparison (use sinkhorn mode for raw logits)
            x_sk, phi_pre_sk, phi_post_sk, phi_res_sk, _, _, _, bias_sk, _ = (
                generate_mhc_inputs(M, n, C, dtype, mode="mhc")
            )
            _, _, H_res_input = fused_mhc(
                x_sk,
                phi_pre_sk,
                phi_post_sk,
                phi_res_sk,
                alpha_pre,
                alpha_post,
                alpha_res,
                bias_sk,
                n_streams,
                hres_mode="sinkhorn",
            )
            H_res_3d = H_res_input.view(M, n, n)
            def fn():
                return sinkhorn_knopp(
                    H_res_3d, C=C, num_iters=sinkhorn_iters, out=H_res_3d
                )

        # Benchmark
        ms = triton.testing.do_bench(fn)

        # Return requested metric based on provider string (following bench_diffusion_attention pattern)
        if "ms" in provider:
            return ms
        elif "TFLOPS" in provider:
            return total_flops / (ms * 1e-3) / 1e12
        elif "GB/s" in provider:
            return total_mem / (ms * 1e-3) / 1e9
        elif "arithmetic_intensity" in provider:
            return total_flops / total_mem
        return ms

    bench_mhc_kernel.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="Benchmark mHC",
        description="Benchmark mHC (manifold-constrained Hyper Connection) kernel",
        allow_abbrev=False,
    )

    # Shape parameters
    parser.add_argument(
        "-M",
        type=int,
        default=None,
        help="Batch/sequence dimension (default: run suite of configs)",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=None,
        help="Stream parameter (manifold dimension, typically 4)",
    )
    parser.add_argument(
        "-C",
        type=int,
        default=None,
        help="Hidden dimension per stream (typically 1024)",
    )

    # Kernel configuration
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Data type for computation",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="mhc_lite",
        choices=["mhc", "mhc_lite", "sinkhorn_knopp_only"],
        help="Benchmark mode: 'mhc' (fused + sinkhorn_knopp), 'mhc_lite', 'sinkhorn_knopp_only'. Default: 'mhc_lite'",
    )
    parser.add_argument(
        "-sinkhorn_iters",
        type=int,
        default=20,
        help="Number of Sinkhorn-Knopp iterations for sinkhorn mode (default: 20)",
    )

    # Output options
    parser.add_argument(
        "-metric",
        nargs="?",
        const="throughput",
        choices=["all", "time", "throughput", "bandwidth", "arithmetic_intensity"],
        default=None,
        help="Metrics for the kernel benchmark (default: all)",
    )
    parser.add_argument(
        "-print_vgpr",
        action="store_true",
        default=False,
        help="Print VGPR usage for Triton kernels",
    )
    parser.add_argument(
        "-o",
        action="store_true",
        default=False,
        help="Write performance results to CSV file",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.print_vgpr:
        print("Retrieving VGPR usage for mHC Triton kernels...")

        def fun():
            return run_benchmark(args)

        print_vgpr(fun, get_caller_name_no_ext())
        return 0

    run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
