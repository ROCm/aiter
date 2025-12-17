#!/usr/bin/env python3
"""
Benchmark MOE MXFP4 kernel with multiple M values and report summary statistics.

Usage:
    python bench_moe_mx_multi.py                    # Default M values
    python bench_moe_mx_multi.py --M 32 64 128 256  # Custom M values
    python bench_moe_mx_multi.py --model mixtral    # Specific model
"""

import argparse
import sys
import numpy as np
import triton
from aiter.ops.triton.utils.types import torch_to_triton_dtype
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.moe_op_mxfp4 import fused_moe_mxfp4
from op_tests.triton_tests.test_moe_mx import input_helper
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_available_models,
    get_model_configs,
)


def run_single_benchmark(M, N, K, E, top_k, a_dtype_str, b_dtype_str, routed_weight=False):
    """Run benchmark for a single configuration and return metrics."""
    (
        a_tri, b_tri, c_tri, c_tri_silu, a_scale, b_scale,
        a_mx_scales, b_mx_scales, topk_weights, topk_ids,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        top_k, config,
    ) = input_helper(M, N, K, top_k, E, a_dtype_str, b_dtype_str)
    
    # Calculate FLOPs
    flops = 2.0 * M * top_k * K * N
    if routed_weight:
        flops += M * top_k * N
    
    # Calculate memory
    mem_read = a_tri.numel() * a_tri.element_size() + b_tri.numel() * b_tri.element_size()
    mem_write = c_tri.numel() * c_tri.element_size()
    mem = mem_read + mem_write
    
    fn = lambda: fused_moe_mxfp4(
        a_tri, b_tri, c_tri, a_scale, b_scale,
        a_mx_scales, b_mx_scales, topk_weights, topk_ids,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        routed_weight, top_k, False, False, config,
        torch_to_triton_dtype[c_tri.dtype],
    )
    
    # Run benchmark with warmup
    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    tflops = flops / ms * 1e-9
    bandwidth = mem / (ms * 1e-3) * 1e-9
    
    return {
        'M': M, 'N': N, 'K': K, 'E': E, 'top_k': top_k,
        'time_ms': ms, 'tflops': tflops, 'bandwidth_gb_s': bandwidth
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-M MOE MXFP4 Benchmark")
    parser.add_argument('--M', type=int, nargs='+', default=[1, 16, 32, 64, 128, 256, 512],
                        help='List of M values to test')
    parser.add_argument('--model', type=str, default='mixtral', help='Model config')
    parser.add_argument('-model_configs', type=str, default='utils/model_configs.json')
    parser.add_argument('--routed-weight', action='store_true')
    parser.add_argument('-A', '--a-dtype', type=str, default='mxfp4_e2m1',
                        choices=['bf16', 'fp16', 'fp8_e5m2', 'mxfp4_e2m1'])
    args = parser.parse_args()
    
    if not arch_info.is_fp4_avail():
        print("MXFP4 not supported on this architecture")
        sys.exit(1)
    
    # Load model config
    configs = get_model_configs(config_path=args.model_configs, models=args.model)
    
    a_dtype_str = args.a_dtype
    b_dtype_str = "mxfp4_e2m1"
    
    print(f"=" * 80)
    print(f"MOE MXFP4 Multi-M Benchmark")
    print(f"Data types: A={a_dtype_str}, B={b_dtype_str}")
    print(f"M values: {args.M}")
    print(f"=" * 80)
    
    all_results = []
    
    for model_name, config in configs.items():
        N1 = config["intermediate_size"]
        K1 = config["hidden_size"]
        E = 8
        top_k = 2
        
        print(f"\n--- Model: {model_name} (N={N1}, K={K1}, E={E}, top_k={top_k}) ---")
        print(f"{'M':>6} | {'Time(ms)':>10} | {'TFLOPS':>10} | {'BW(GB/s)':>12}")
        print("-" * 50)
        
        results_for_model = []
        for M in args.M:
            try:
                result = run_single_benchmark(M, N1, K1, E, top_k, a_dtype_str, b_dtype_str, args.routed_weight)
                results_for_model.append(result)
                all_results.append(result)
                print(f"{M:>6} | {result['time_ms']:>10.4f} | {result['tflops']:>10.2f} | {result['bandwidth_gb_s']:>12.2f}")
            except Exception as e:
                print(f"{M:>6} | ERROR: {e}")
        
        if results_for_model:
            avg_time = np.mean([r['time_ms'] for r in results_for_model])
            avg_tflops = np.mean([r['tflops'] for r in results_for_model])
            avg_bw = np.mean([r['bandwidth_gb_s'] for r in results_for_model])
            print("-" * 50)
            print(f"{'AVG':>6} | {avg_time:>10.4f} | {avg_tflops:>10.2f} | {avg_bw:>12.2f}")
    
    # Overall summary
    if all_results:
        print(f"\n{'=' * 80}")
        print("OVERALL SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total configurations tested: {len(all_results)}")
        print(f"Average latency:    {np.mean([r['time_ms'] for r in all_results]):.4f} ms")
        print(f"Average TFLOPS:     {np.mean([r['tflops'] for r in all_results]):.2f}")
        print(f"Average Bandwidth:  {np.mean([r['bandwidth_gb_s'] for r in all_results]):.2f} GB/s")
        print(f"Peak TFLOPS:        {max([r['tflops'] for r in all_results]):.2f}")
        print(f"Peak Bandwidth:     {max([r['bandwidth_gb_s'] for r in all_results]):.2f} GB/s")


if __name__ == "__main__":
    main()



