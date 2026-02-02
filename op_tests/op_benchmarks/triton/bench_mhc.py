# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for mHC (manifold-constrained Hyper Connection) fused kernel.

Measures performance of the Triton implementation across various input shapes
and configurations, reporting time, throughput (TFLOPS), and bandwidth.

Supports two modes:
- Standard mHC with Sinkhorn-Knopp (default)
- mHC-lite with Birkhoff-von Neumann reparameterization (--use_mhc_lite)
"""

import sys
import argparse
import logging
from itertools import product
from math import factorial
import torch
import triton

# Configure logging before importing aiter modules
logging.basicConfig(
    level=logging.INFO,
    format='[%(name)s] %(levelname)s: %(message)s',
    force=True
)

from aiter.ops.triton.fusions.mhc import mhc, fused_mhc, sinkhorn_knopp
from op_tests.triton_tests.utils.mhc_ref import generate_mhc_inputs, generate_mhc_lite_inputs
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    print_vgpr,
    get_caller_name_no_ext,
)

arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def get_benchmark_configs(args):
    """Generate list of benchmark configurations based on args."""
    configs = []
    
    if args.M and args.n and args.C:
        # Single configuration from command line
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
    use_mhc_lite = args.use_mhc_lite
    
    # Validate mode combinations
    if use_mhc_lite and mode != "full":
        raise ValueError(
            f"Only --mode full is valid with --use_mhc_lite. "
            f"Got --mode {mode}. mHC-lite produces exact doubly stochastic output "
            f"without Sinkhorn iterations, so 'fused' and 'sinkhorn' modes don't apply."
        )
    
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
    if use_mhc_lite:
        benchmark_name += "_lite"
    elif mode == "full":
        benchmark_name += f"_full_sinkhorn{sinkhorn_iters}"
    elif mode == "fused":
        benchmark_name += "_fused_only"
    elif mode == "sinkhorn":
        benchmark_name += f"_sinkhorn_only_{sinkhorn_iters}iters"
    
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
        if use_mhc_lite:
            x, phi_pre, phi_post, phi_res, alpha_pre, alpha_post, alpha_res, bias, n_streams = \
                generate_mhc_lite_inputs(M, n, C, dtype)
            n_factorial = factorial(n)
        else:
            x, phi_pre, phi_post, phi_res, alpha_pre, alpha_post, alpha_res, bias, n_streams = \
                generate_mhc_inputs(M, n, C, dtype)
        
        # Compute FLOPs for mHC operation
        nC = n * C
        n_squared = n * n
        
        if use_mhc_lite:
            # mHC-lite: phi_res has n! columns, then softmax + matmul with permutation matrices
            n_factorial = factorial(n)
            N = n_factorial + 2 * n
            
            # Matmul FLOPs: x @ phi_res is (M, nC) @ (nC, n!)
            flops_matmul = (
                2.0 * M * nC * n +           # x @ phi_pre
                2.0 * M * nC * n +           # x @ phi_post
                2.0 * M * nC * n_factorial   # x @ phi_res (n! columns)
            )
            
            # RMS normalization
            flops_rms = 3.0 * M * nC
            
            # Softmax over n! elements: exp + sum + divide ≈ 3 * n! per row
            flops_softmax = 3.0 * M * n_factorial
            
            # Permutation combination: (M, n!) @ (n!, n²) matmul
            flops_perm = 2.0 * M * n_factorial * n_squared
            
            total_flops = flops_matmul + flops_rms + flops_softmax + flops_perm
            
            # Memory traffic for lite mode
            elem_size = x.element_size()  # 2 for bf16/fp16, 4 for fp32
            bias_size = bias.element_size()  # bias dtype (typically fp32)
            mem_read = (
                M * nC * elem_size +           # x
                nC * n * elem_size +           # phi_pre
                nC * n * elem_size +           # phi_post
                nC * n_factorial * elem_size + # phi_res (n! columns)
                N * bias_size +                # bias
                n_factorial * n_squared * 4    # permutation matrices (float32)
            )
            mem_write = (
                M * n * elem_size +       # out_pre
                M * n * elem_size +       # out_post
                M * n_squared * elem_size # out_res
            )
            total_mem = mem_read + mem_write
        else:
            N = n_squared + 2 * n
            
            # Standard GEMM FLOPs (2*M*N*K for matrix multiply)
            flops_matmul = 2.0 * M * nC * n + 2.0 * M * nC * n + 2.0 * M * nC * n_squared
            
            # RMS normalization
            flops_rms = 4.0 * M * nC
            
            # Sinkhorn-Knopp
            flops_sinkhorn = 4.0 * M * n_squared * sinkhorn_iters
            
            if mode == "full":
                total_flops = flops_matmul + flops_rms + flops_sinkhorn
            elif mode == "fused":
                total_flops = flops_matmul + flops_rms
            elif mode == "sinkhorn":
                total_flops = flops_sinkhorn
            
            # Memory traffic
            elem_size = x.element_size()  # 2 for bf16/fp16, 4 for fp32
            bias_size = bias.element_size()  # bias dtype (typically fp32)
            mem_read = (
                M * nC * elem_size +
                nC * n * elem_size +
                nC * n * elem_size +
                nC * n_squared * elem_size +
                N * bias_size
            )
            mem_write = (
                M * n * elem_size +
                M * n * elem_size +
                M * n_squared * elem_size
            )
            mem_sinkhorn = 2 * M * n_squared * elem_size * sinkhorn_iters
            
            if mode == "full":
                mem_read += mem_sinkhorn / 2
                mem_write += mem_sinkhorn / 2
                total_mem = mem_read + mem_write
            elif mode == "fused":
                total_mem = mem_read + mem_write
            elif mode == "sinkhorn":
                total_mem = mem_sinkhorn
        
        # Create benchmark function
        if use_mhc_lite:
            fn = lambda: mhc(
                x, phi_pre, phi_post, phi_res,
                alpha_pre, alpha_post, alpha_res,
                bias, n_streams,
                hres_mode="lite"
            )
        elif mode == "full":
            fn = lambda: mhc(
                x, phi_pre, phi_post, phi_res,
                alpha_pre, alpha_post, alpha_res,
                bias, n_streams,
                sinkhorn_iters=sinkhorn_iters
            )
        elif mode == "fused":
            fn = lambda: fused_mhc(
                x, phi_pre, phi_post, phi_res,
                alpha_pre, alpha_post, alpha_res,
                bias, n_streams
            )
        elif mode == "sinkhorn":
            # Sinkhorn-only: benchmark just the Sinkhorn-Knopp kernel
            _, _, H_res_input = fused_mhc(
                x, phi_pre, phi_post, phi_res,
                alpha_pre, alpha_post, alpha_res,
                bias, n_streams
            )
            H_res_3d = H_res_input.view(M, n, n)
            fn = lambda: sinkhorn_knopp(H_res_3d, num_iters=sinkhorn_iters, out=H_res_3d)
        
        # Benchmark
        ms = triton.testing.do_bench(fn)
        
        # Return requested metric based on provider string
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
        help="Batch/sequence dimension (default: run suite of configs)"
    )
    parser.add_argument(
        "-n",
        type=int,
        default=None,
        help="Stream parameter (manifold dimension, typically 4)"
    )
    parser.add_argument(
        "-C",
        type=int,
        default=None,
        help="Hidden dimension per stream (typically 1024)"
    )
    
    # Kernel configuration
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Data type for computation"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "fused", "sinkhorn"],
        help="Benchmark mode: 'full' (fused+sinkhorn), 'fused' (fused kernel only), 'sinkhorn' (sinkhorn-only)"
    )
    parser.add_argument(
        "-sinkhorn_iters",
        type=int,
        default=20,
        help="Number of Sinkhorn-Knopp iterations (default: 20)"
    )
    parser.add_argument(
        "--use_mhc_lite",
        action="store_true",
        default=False,
        help="Use mHC-lite (Birkhoff-von Neumann) instead of Sinkhorn-Knopp. "
             "Only --mode full is valid with this option."
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
        help="Print VGPR usage for Triton kernels"
    )
    parser.add_argument(
        "-o",
        action="store_true",
        default=False,
        help="Write performance results to CSV file"
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
