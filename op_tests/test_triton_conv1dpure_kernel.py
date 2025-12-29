#!/usr/bin/env python3
"""
Pure Kernel Time Benchmark for Triton causal_conv1d_update
Uses torch.cuda.Event for accurate kernel timing (no profiler overhead)
"""

import torch
import torch.nn.functional as F
import sys
import argparse
from pathlib import Path

# Add the parent directory to path to import aiter
sys.path.insert(0, str(Path(__file__).parent))

from aiter.ops.triton.causal_conv1d_triton import causal_conv1d_update


def torch_causal_conv1d_update_ref(x, conv_state, weight, bias=None, activation=None, cache_seqlens=None):
    """PyTorch reference implementation"""
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    assert conv_state.shape == (batch, dim, state_len)
    assert weight.shape == (dim, width)
    if cache_seqlens is None:
        x_new = torch.cat([conv_state, x], dim=-1).to(weight.dtype)
        conv_state.copy_(x_new[:, :, -state_len:])
    else:
        width_idx = torch.arange(-(width - 1), 0, dtype=torch.long, device=x.device).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        width_idx = torch.remainder(width_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        x_new = torch.cat([conv_state.gather(2, width_idx), x], dim=-1).to(weight.dtype)
        copy_idx = torch.arange(seqlen, dtype=torch.long, device=x.device).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        copy_idx = torch.remainder(copy_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        conv_state.scatter_(2, copy_idx, x)
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[:, :, -seqlen:]
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)


def benchmark_pure_kernel_time(batch, dim, seqlen, width=4, warmup_iters=10, bench_iters=100, device='cuda'):
    """
    Measure pure kernel execution time using torch.cuda.Event
    This excludes any Python/CPU overhead
    """
    state_len = width - 1
    silu_activation = "silu"
    
    # Initialize data
    torch.manual_seed(42)
    x = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float32)
    weight = torch.randn(dim, width, device=device, dtype=torch.float32)
    bias = torch.randn(dim, device=device, dtype=torch.float32)
    conv_state = torch.randn(batch, dim, state_len, device=device, dtype=torch.float32)
    
    # Warmup - important for GPU to reach stable state
    print(f"  Warming up ({warmup_iters} iterations)...", end='', flush=True)
    for _ in range(warmup_iters):
        conv_state_copy = conv_state.clone()
        out = causal_conv1d_update(
            x=x,
            conv_state=conv_state_copy,
            weight=weight,
            bias=bias,
            activation=silu_activation,
            cache_seqlens=None,
            conv_state_indices=None,
        )
    torch.cuda.synchronize()
    print(" Done")
    
    # Pure kernel timing using CUDA Events
    print(f"  Benchmarking ({bench_iters} iterations)...", end='', flush=True)
    
    # Create events for each iteration
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]
    
    # Run benchmark
    for i in range(bench_iters):
        conv_state_copy = conv_state.clone()
        
        # Record start
        start_events[i].record()
        
        # Execute kernel
        out = causal_conv1d_update(
            x=x,
            conv_state=conv_state_copy,
            weight=weight,
            bias=bias,
            activation=silu_activation,
            cache_seqlens=None,
            conv_state_indices=None,
        )
        
        # Record end
        end_events[i].record()
    
    # Wait for all operations to complete
    torch.cuda.synchronize()
    print(" Done")
    
    # Calculate timing statistics
    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    
    # Remove outliers (optional but recommended)
    import numpy as np
    times_ms = np.array(times_ms)
    
    # Statistics
    min_time_ms = np.min(times_ms)
    max_time_ms = np.max(times_ms)
    median_time_ms = np.median(times_ms)
    mean_time_ms = np.mean(times_ms)
    std_time_ms = np.std(times_ms)
    
    # Remove outliers using IQR method for more stable average
    Q1 = np.percentile(times_ms, 25)
    Q3 = np.percentile(times_ms, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_times = times_ms[(times_ms >= lower_bound) & (times_ms <= upper_bound)]
    
    if len(filtered_times) > 0:
        avg_time_ms = np.mean(filtered_times)
        outliers_removed = len(times_ms) - len(filtered_times)
    else:
        avg_time_ms = mean_time_ms
        outliers_removed = 0
    
    avg_time_us = avg_time_ms * 1000.0
    
    # Calculate throughput metrics
    total_elements = batch * dim * seqlen
    total_ops = total_elements * width * 2  # MACs
    gflops = (total_ops / avg_time_ms) / 1e6
    
    # Memory bandwidth
    total_bytes = (x.numel() + weight.numel() + bias.numel() + 
                   conv_state.numel() + out.numel()) * 4  # float32 = 4 bytes
    bandwidth_gb_s = total_bytes / (avg_time_ms / 1000.0) / 1e9
    
    return {
        'avg_time_us': avg_time_us,
        'avg_time_ms': avg_time_ms,
        'min_time_us': min_time_ms * 1000.0,
        'max_time_us': max_time_ms * 1000.0,
        'median_time_us': median_time_ms * 1000.0,
        'std_time_us': std_time_ms * 1000.0,
        'gflops': gflops,
        'bandwidth_gb_s': bandwidth_gb_s,
        'total_elements': total_elements,
        'outliers_removed': outliers_removed,
    }


def test_correctness(batch, dim, seqlen, width=4, device='cuda', tolerance=1e-3):
    """Quick correctness check"""
    state_len = width - 1
    silu_activation = "silu"
    
    torch.manual_seed(42)
    x = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float32)
    weight = torch.randn(dim, width, device=device, dtype=torch.float32)
    bias = torch.randn(dim, device=device, dtype=torch.float32)
    conv_state_triton = torch.randn(batch, dim, state_len, device=device, dtype=torch.float32)
    conv_state_ref = conv_state_triton.clone()
    
    # Run Triton
    out_triton = causal_conv1d_update(
        x=x.clone(),
        conv_state=conv_state_triton,
        weight=weight,
        bias=bias,
        activation=silu_activation,
        cache_seqlens=None,
        conv_state_indices=None,
    )
    
    # Run reference
    out_ref = torch_causal_conv1d_update_ref(
        x=x.clone(),
        conv_state=conv_state_ref,
        weight=weight,
        bias=bias,
        activation=silu_activation,
        cache_seqlens=None
    )
    
    # Compare
    diff = torch.abs(out_triton - out_ref)
    max_diff = torch.max(diff).item()
    
    return max_diff < tolerance, max_diff


def main():
    parser = argparse.ArgumentParser(description='Pure Kernel Time Benchmark for Triton')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    parser.add_argument('--iters', type=int, default=100, help='Benchmark iterations')
    parser.add_argument('--test-correctness', action='store_true', help='Run correctness test first')
    args = parser.parse_args()
    
    print("=" * 90)
    print("Triton causal_conv1d_update - Pure Kernel Time Benchmark")
    print("Using torch.cuda.Event for accurate GPU kernel timing")
    print("=" * 90)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        return
    
    device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device)
    print(f"\nGPU: {device_name}")
    print(f"Device: cuda:{device}\n")
    
    width = 4
    
    # Test configurations
    configs = [
        (1, 2048, 1, "Small (B=1, D=2048, L=1)"),
        (8, 2048, 1, "Medium Batch (B=8, D=2048, L=1)"),
        (32, 2048, 1, "Large Batch (B=32, D=2048, L=1)"),
        (1, 4096, 1, "Large Dim (B=1, D=4096, L=1)"),
        (16, 4096, 1, "Large (B=16, D=4096, L=1)"),
        (1, 2048, 16, "Long Seq (B=1, D=2048, L=16)"),
    ]
    
    # Optional correctness check
    if args.test_correctness:
        print("\n" + "=" * 90)
        print("CORRECTNESS CHECK")
        print("=" * 90)
        all_passed = True
        for batch, dim, seqlen, name in configs:
            passed, max_diff = test_correctness(batch, dim, seqlen, width)
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{name:<35} {status}  (max_diff={max_diff:.2e})")
            if not passed:
                all_passed = False
        
        if not all_passed:
            print("\n❌ Some correctness tests failed!")
            return
        print("\n✅ All correctness tests passed!\n")
    
    # Benchmark
    print("=" * 90)
    print("PURE KERNEL TIME BENCHMARK")
    print("=" * 90)
    print(f"Warmup iterations: {args.warmup}")
    print(f"Benchmark iterations: {args.iters}")
    print()
    
    results = []
    
    for batch, dim, seqlen, name in configs:
        print(f"\n{'='*90}")
        print(f"Config: {name}")
        print(f"  Batch={batch}, Dim={dim}, SeqLen={seqlen}, Width={width}")
        print('='*90)
        
        try:
            result = benchmark_pure_kernel_time(
                batch=batch,
                dim=dim,
                seqlen=seqlen,
                width=width,
                warmup_iters=args.warmup,
                bench_iters=args.iters,
            )
            
            print(f"\n  Results (outliers removed: {result['outliers_removed']}):")
            print(f"    Average time:  {result['avg_time_us']:>8.2f} μs  ({result['avg_time_ms']:.6f} ms)")
            print(f"    Min time:      {result['min_time_us']:>8.2f} μs")
            print(f"    Median time:   {result['median_time_us']:>8.2f} μs")
            print(f"    Max time:      {result['max_time_us']:>8.2f} μs")
            print(f"    Std dev:       {result['std_time_us']:>8.2f} μs")
            print(f"    Throughput:    {result['gflops']:>8.2f} GFLOPS")
            print(f"    Bandwidth:     {result['bandwidth_gb_s']:>8.2f} GB/s")
            
            results.append((name, batch, dim, seqlen, result))
            
        except Exception as e:
            print(f"\n  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print()
    print(f"{'Config':<35} {'Batch':>5} {'Dim':>6} {'SeqL':>5} {'Avg (μs)':>10} {'GFLOPS':>10} {'BW (GB/s)':>10}")
    print("-" * 90)
    
    for name, batch, dim, seqlen, result in results:
        print(f"{name:<35} {batch:>5} {dim:>6} {seqlen:>5} "
              f"{result['avg_time_us']:>10.2f} {result['gflops']:>10.2f} {result['bandwidth_gb_s']:>10.2f}")
    
    print("\n" + "=" * 90)
    print("✅ Benchmark completed!")
    print("=" * 90)


if __name__ == "__main__":
    main()

