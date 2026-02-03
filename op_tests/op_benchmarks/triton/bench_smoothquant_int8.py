# Benchmark for SmoothQuant Int8 Quantization Kernel
# Tests and benchmarks the smoothquant_int8.py triton kernels in isolation

import torch
import triton
import time
import argparse

from aiter.ops.triton._triton_kernels.moe.smoothquant_int8 import (
    _smoothquant_fuse_quant_kernel,
    _smoothquant_fuse_quant_kernel_single_pass,
)


# ---------------
# Triton Wrapper Functions
# ---------------


def smoothquant_quantize_triton_single_pass(
    x: torch.Tensor,
    smooth_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wrapper for single-pass triton kernel."""
    M, K = x.shape
    device = x.device
    
    x_int8 = torch.empty((M, K), dtype=torch.int8, device=device)
    x_scale = torch.empty((M,), dtype=torch.float32, device=device)
    
    smooth_scale = smooth_scale.to(torch.float32).contiguous()
    
    BLOCK_M = min(triton.next_power_of_2(M), 32)
    BLOCK_K = triton.next_power_of_2(K)
    grid = (triton.cdiv(M, BLOCK_M),)
    
    _smoothquant_fuse_quant_kernel_single_pass[grid](
        x, x.stride(0), x.stride(1),
        smooth_scale,
        x_int8, x_int8.stride(0), x_int8.stride(1),
        x_scale, 1,
        M, K,
        BLOCK_M, BLOCK_K,
        num_warps=4,
    )
    
    return x_int8, x_scale


def smoothquant_quantize_triton_two_pass(
    x: torch.Tensor,
    smooth_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wrapper for two-pass triton kernel."""
    M, K = x.shape
    device = x.device
    
    x_int8 = torch.empty((M, K), dtype=torch.int8, device=device)
    x_scale = torch.empty((M,), dtype=torch.float32, device=device)
    
    smooth_scale = smooth_scale.to(torch.float32).contiguous()
    
    BLOCK_M = min(triton.next_power_of_2(M), 32)
    BLOCK_K = 256
    grid = (triton.cdiv(M, BLOCK_M),)
    
    _smoothquant_fuse_quant_kernel[grid](
        x, x.stride(0), x.stride(1),
        smooth_scale,
        x_int8, x_int8.stride(0), x_int8.stride(1),
        x_scale, 1,
        M, K,
        BLOCK_M, BLOCK_K,
        num_warps=4,
    )
    
    return x_int8, x_scale


# ---------------
# Reference Implementation
# ---------------


def smoothquant_quantize_torch(
    x: torch.Tensor,
    smooth_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference implementation."""
    x_smooth = x.to(torch.float32) * smooth_scale[None, :]
    row_max = x_smooth.abs().max(dim=1).values
    INT8_MAX = 127.0
    row_scale = row_max / INT8_MAX + 1e-12
    x_scaled = x_smooth / row_scale[:, None]
    x_int8 = x_scaled.round().clamp(-127, 127).to(torch.int8)
    return x_int8, row_scale


# ---------------
# Correctness Check
# ---------------


def check_correctness(m, k, device="cuda"):
    """Verify kernel correctness."""
    torch.manual_seed(42)
    
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    smooth_scale = torch.randn((k,), device=device, dtype=torch.float32).abs() + 0.1
    
    # Reference
    ref_int8, ref_scale = smoothquant_quantize_torch(x, smooth_scale)
    
    # Triton
    if k <= 8192:
        tri_int8, tri_scale = smoothquant_quantize_triton_single_pass(x, smooth_scale)
        kernel_name = "single_pass"
    else:
        tri_int8, tri_scale = smoothquant_quantize_triton_two_pass(x, smooth_scale)
        kernel_name = "two_pass"
    
    # Compare scales
    scale_diff = (ref_scale - tri_scale).abs()
    scale_max_diff = scale_diff.max().item()
    scale_mean_diff = scale_diff.mean().item()
    
    # Compare int8 values
    int8_diff = (ref_int8.to(torch.int32) - tri_int8.to(torch.int32)).abs()
    int8_max_diff = int8_diff.max().item()
    int8_num_diff = (int8_diff > 0).sum().item()
    
    # Check if passed
    scale_passed = scale_max_diff < 1e-4
    int8_passed = int8_max_diff <= 1
    
    print(f"  M={m:5d}, K={k:5d} [{kernel_name:11s}] | "
          f"Scale: max_diff={scale_max_diff:.2e} {'✓' if scale_passed else '✗'} | "
          f"Int8: max_diff={int8_max_diff}, num_diff={int8_num_diff}/{m*k} {'✓' if int8_passed else '✗'}")
    
    return scale_passed and int8_passed


# ---------------
# Benchmarking
# ---------------


def benchmark_kernel(m, k, device="cuda", warmup=10, reps=100):
    """Benchmark the triton kernel."""
    torch.manual_seed(42)
    
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    smooth_scale = torch.randn((k,), device=device, dtype=torch.float32).abs() + 0.1
    
    # Pre-allocate outputs
    x_int8 = torch.empty((m, k), dtype=torch.int8, device=device)
    x_scale = torch.empty((m,), dtype=torch.float32, device=device)
    smooth_scale = smooth_scale.to(torch.float32).contiguous()
    
    # Choose kernel
    use_single_pass = k <= 8192
    BLOCK_M = min(triton.next_power_of_2(m), 32)
    
    if use_single_pass:
        BLOCK_K = triton.next_power_of_2(k)
        kernel = _smoothquant_fuse_quant_kernel_single_pass
        kernel_name = "single_pass"
    else:
        BLOCK_K = 256
        kernel = _smoothquant_fuse_quant_kernel
        kernel_name = "two_pass"
    
    grid = (triton.cdiv(m, BLOCK_M),)
    
    # Warmup
    for _ in range(warmup):
        kernel[grid](
            x, x.stride(0), x.stride(1),
            smooth_scale,
            x_int8, x_int8.stride(0), x_int8.stride(1),
            x_scale, 1,
            m, k,
            BLOCK_M, BLOCK_K,
            num_warps=4,
        )
    
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(reps):
        kernel[grid](
            x, x.stride(0), x.stride(1),
            smooth_scale,
            x_int8, x_int8.stride(0), x_int8.stride(1),
            x_scale, 1,
            m, k,
            BLOCK_M, BLOCK_K,
            num_warps=4,
        )
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    # Calculate metrics
    latency_us = (end - start) / reps * 1e6
    
    # Bytes moved:
    # - Read: x (bf16) + smooth_scale (fp32)
    # - Write: x_int8 (int8) + x_scale (fp32)
    bytes_read = m * k * 2 + k * 4  # bf16 input + fp32 smooth_scale
    bytes_write = m * k * 1 + m * 4  # int8 output + fp32 row_scale
    total_bytes = bytes_read + bytes_write
    
    throughput_gbps = total_bytes / latency_us * 1e-3  # GB/s
    
    return {
        "m": m,
        "k": k,
        "kernel": kernel_name,
        "latency_us": latency_us,
        "throughput_gbps": throughput_gbps,
        "bytes_read": bytes_read,
        "bytes_write": bytes_write,
        "total_bytes": total_bytes,
    }


def benchmark_pytorch_reference(m, k, device="cuda", warmup=10, reps=100):
    """Benchmark PyTorch reference for comparison."""
    torch.manual_seed(42)
    
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    smooth_scale = torch.randn((k,), device=device, dtype=torch.float32).abs() + 0.1
    
    # Warmup
    for _ in range(warmup):
        _ = smoothquant_quantize_torch(x, smooth_scale)
    
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(reps):
        _ = smoothquant_quantize_torch(x, smooth_scale)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    latency_us = (end - start) / reps * 1e6
    
    # Same bytes as triton
    bytes_read = m * k * 2 + k * 4
    bytes_write = m * k * 1 + m * 4
    total_bytes = bytes_read + bytes_write
    
    throughput_gbps = total_bytes / latency_us * 1e-3
    
    return {
        "m": m,
        "k": k,
        "kernel": "pytorch",
        "latency_us": latency_us,
        "throughput_gbps": throughput_gbps,
    }


def run_benchmarks(batch_sizes, k_sizes, device="cuda", compare_pytorch=True):
    """Run benchmarks across different sizes."""
    print("\n" + "="*80)
    print("SMOOTHQUANT INT8 KERNEL BENCHMARK")
    print("="*80)
    
    results = []
    
    for k in k_sizes:
        print(f"\n--- K = {k} ---")
        print(f"{'M':>6s} | {'Kernel':>11s} | {'Latency (us)':>12s} | {'Throughput (GB/s)':>17s} | {'Speedup':>8s}")
        print("-" * 70)
        
        for m in batch_sizes:
            # Triton benchmark
            tri_result = benchmark_kernel(m, k, device)
            results.append(tri_result)
            
            # PyTorch benchmark (optional)
            if compare_pytorch:
                torch_result = benchmark_pytorch_reference(m, k, device)
                speedup = torch_result["latency_us"] / tri_result["latency_us"]
                
                print(f"{m:6d} | {tri_result['kernel']:>11s} | {tri_result['latency_us']:12.2f} | {tri_result['throughput_gbps']:17.2f} | {speedup:7.2f}x")
                print(f"{m:6d} | {'pytorch':>11s} | {torch_result['latency_us']:12.2f} | {torch_result['throughput_gbps']:17.2f} | {'(ref)':>8s}")
            else:
                print(f"{m:6d} | {tri_result['kernel']:>11s} | {tri_result['latency_us']:12.2f} | {tri_result['throughput_gbps']:17.2f}")
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark SmoothQuant Int8 Kernel")
    parser.add_argument(
        "--check-correctness",
        action="store_true",
        help="Run correctness checks before benchmarking",
    )
    parser.add_argument(
        "--no-pytorch",
        action="store_true",
        help="Skip PyTorch reference comparison",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 16, 64, 256, 1024, 4096],
        help="Batch sizes (M dimension) to benchmark",
    )
    parser.add_argument(
        "--k-sizes",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096, 7168],
        help="K dimension sizes to benchmark",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=100,
        help="Number of benchmark repetitions",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Check correctness first
    if args.check_correctness:
        print("\n" + "="*80)
        print("CORRECTNESS CHECKS")
        print("="*80)
        
        test_cases = [
            (16, 256), (64, 512), (128, 1024), (256, 2048),
            (512, 4096), (1024, 7168), (33, 129), (100, 500),
        ]
        
        all_passed = True
        for m, k in test_cases:
            if not check_correctness(m, k):
                all_passed = False
        
        if all_passed:
            print("\n✓ All correctness checks passed!")
        else:
            print("\n✗ Some correctness checks failed!")
            exit(1)
    
    # Run benchmarks
    run_benchmarks(
        args.batch_sizes,
        args.k_sizes,
        compare_pytorch=not args.no_pytorch,
    )
