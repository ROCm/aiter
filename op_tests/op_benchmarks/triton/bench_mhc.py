# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for mHC (manifold-constrained Hyper Connection) fused kernel.

Measures performance of the Triton implementation across various input shapes
and configurations, reporting time, throughput (TFLOPS), and bandwidth.
"""

import sys
import argparse
import torch
import triton
from aiter.ops.triton.fusions.mhc import mhc, fused_mhc
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
        configs = [
            # Small batch, typical streams
            (32, 4, 1024),
            (64, 4, 1024),
            (128, 4, 1024),
            (256, 4, 1024),
            
            # Larger batch
            (512, 4, 1024),
            (1024, 4, 1024),
            (2048, 4, 1024),
            
            # Different stream counts
            (128, 2, 1024),
            (128, 8, 1024),
            (128, 16, 1024),
            
            # Different hidden dimensions
            (128, 4, 512),
            (128, 4, 2048),
            (128, 4, 4096),
            
            # Large configurations
            (4096, 4, 1024),
            (8192, 4, 1024),
        ]
    
    return configs


def run_benchmark(args):
    """Run mHC benchmark with specified configuration."""
    dtype = arg_to_torch_dtype[args.dtype]
    with_sinkhorn = not args.no_sinkhorn
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
    if with_sinkhorn:
        benchmark_name += f"_sinkhorn{sinkhorn_iters}"
    else:
        benchmark_name += "_fused_only"
    
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
        # Generate inputs
        x, phi_pre, phi_post, phi_res, alpha_pre, alpha_post, alpha_res, bias, n_streams = \
            generate_mhc_inputs(M, n, C, dtype)
        
        # Compute FLOPs for mHC operation
        nC = n * C
        N = n * n + 2 * n
        
        # Eq 14: Matrix multiplication x @ phi for 3 streams
        # Each matmul: (M, nC) @ (nC, n_out) requires 2*M*nC*n_out FLOPs
        flops_pre = 2.0 * M * nC * n
        flops_post = 2.0 * M * nC * n
        flops_res = 2.0 * M * nC * (n * n)
        flops_matmul = flops_pre + flops_post + flops_res
        
        # Eq 15: RMS normalization - M rows, each with nC elements
        # sqrt(sum(x^2)/K) requires ~2*nC ops per row (square + sum)
        flops_rms = 2.0 * M * nC
        
        # Eq 16: Scaling and bias - negligible compared to matmul
        flops_scale = M * N
        
        # Eq 17-18: Activations (sigmoid) - ~10 FLOPs per element
        flops_activation = 10.0 * M * (n + n)
        
        # Eq 19: Sinkhorn-Knopp (if enabled)
        # Each iteration: 2 normalizations (row + col) on M matrices of size (n, n)
        # Each normalization: sum (n ops) + divide (n ops) = 2n ops per row/col
        # Total per iteration: 2 * 2 * M * n * n ops
        flops_sinkhorn = 0.0
        if with_sinkhorn:
            flops_sinkhorn = 4.0 * M * n * n * sinkhorn_iters
        
        total_flops = flops_matmul + flops_rms + flops_scale + flops_activation + flops_sinkhorn
        
        # Compute memory traffic
        elem_size = 2  # bf16/fp16 = 2 bytes
        bias_size = 4  # bias is fp32 = 4 bytes
        
        # Memory reads:
        # - x: (M, nC)
        # - phi_pre, phi_post, phi_res: (nC, n), (nC, n), (nC, n*n)
        # - bias: (N,) in fp32
        mem_read = (
            M * nC * elem_size +  # x
            nC * n * elem_size +  # phi_pre
            nC * n * elem_size +  # phi_post
            nC * n * n * elem_size +  # phi_res
            N * bias_size  # bias
        )
        
        # Memory writes:
        # - out_pre: (M, n)
        # - out_post: (M, n)
        # - out_res: (M, n*n)
        mem_write = (
            M * n * elem_size +  # out_pre
            M * n * elem_size +  # out_post
            M * n * n * elem_size  # out_res
        )
        
        # Sinkhorn-Knopp adds minimal memory traffic (in-place operations)
        if with_sinkhorn:
            # Read/write H_res multiple times during iterations
            mem_sinkhorn = 2 * M * n * n * elem_size * sinkhorn_iters
            mem_read += mem_sinkhorn / 2
            mem_write += mem_sinkhorn / 2
        
        total_mem = mem_read + mem_write
        
        # Create benchmark function
        if with_sinkhorn:
            fn = lambda: mhc(
                x, phi_pre, phi_post, phi_res,
                alpha_pre, alpha_post, alpha_res,
                bias, n_streams,
                sinkhorn_iters=sinkhorn_iters
            )
        else:
            fn = lambda: fused_mhc(
                x, phi_pre, phi_post, phi_res,
                alpha_pre, alpha_post, alpha_res,
                bias, n_streams
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
        "--no_sinkhorn",
        action="store_true",
        default=False,
        help="Benchmark fused_mhc only (skip Sinkhorn-Knopp)"
    )
    parser.add_argument(
        "-sinkhorn_iters",
        type=int,
        default=20,
        help="Number of Sinkhorn-Knopp iterations (default: 20)"
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
