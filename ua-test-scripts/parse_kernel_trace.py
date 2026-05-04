#!/usr/bin/env python3
"""
Parse rocprofv3 kernel trace and compute statistics excluding warmup.

This script identifies CK and Triton kernel calls and computes timing statistics
for benchmark iterations only (excluding warmup).
"""

import argparse
import csv
from collections import defaultdict


def parse_trace(trace_file, warmup_iterations=0):
    """Parse kernel trace CSV and collect timing data."""

    kernel_calls = defaultdict(list)  # kernel_name -> list of (start, end, duration_ns)

    with open(trace_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('Kernel_Name', '')
            if not name:
                continue

            start = int(row['Start_Timestamp'])
            end = int(row['End_Timestamp'])
            duration_ns = end - start

            kernel_calls[name].append((start, end, duration_ns))

    return kernel_calls


def identify_attention_kernels(kernel_calls):
    """Identify CK and Triton attention kernels."""
    ck_kernels = {}
    triton_kernels = {}
    triton_reduce_kernels = {}

    for name, calls in kernel_calls.items():
        if 'ck_tile::kentry' in name and 'UnifiedAttentionKernel' in name:
            ck_kernels[name] = calls
        elif 'kernel_unified_attention_2d' in name:
            triton_kernels[name] = calls
        elif 'kernel_unified_attention_3d' in name:
            triton_kernels[name] = calls
        elif ('fmha_fwd_splitkv_reduce' in name or
              'reduce_segments' in name or
              ('reduce' in name.lower() and 'fmha' in name.lower())):
            # Reduction kernels for 3D Triton attention
            # reduce_segments is the newer name for the split-KV reduction kernel
            triton_reduce_kernels[name] = calls

    return ck_kernels, triton_kernels, triton_reduce_kernels


def compute_stats(calls, skip_first_n=0):
    """Compute timing statistics, optionally skipping first N calls."""
    if skip_first_n >= len(calls):
        return None

    # Skip warmup iterations
    benchmark_calls = calls[skip_first_n:]

    if not benchmark_calls:
        return None

    durations_us = [d / 1e3 for (s, e, d) in benchmark_calls]
    total_ms = sum(durations_us) / 1e3

    return {
        'count': len(benchmark_calls),
        'total_ms': total_ms,
        'avg_us': total_ms * 1e3 / len(benchmark_calls),
        'min_us': min(durations_us),
        'max_us': max(durations_us),
        'median_us': sorted(durations_us)[len(durations_us) // 2],
    }


def main():
    parser = argparse.ArgumentParser(description="Parse rocprofv3 kernel trace")
    parser.add_argument("trace_file", help="Path to results_kernel_trace.csv")
    parser.add_argument("--warmup", type=int, default=0,
                       help="Number of warmup iterations to skip per kernel")
    parser.add_argument("--verbose", action="store_true",
                       help="Show all kernels, not just attention kernels")
    args = parser.parse_args()

    # Parse trace
    kernel_calls = parse_trace(args.trace_file)

    # Identify CK and Triton kernels
    ck_kernels, triton_kernels, triton_reduce_kernels = identify_attention_kernels(kernel_calls)

    print("=" * 100)
    print("Kernel Trace Analysis (Benchmark Iterations Only)")
    print("=" * 100)
    print(f"Warmup iterations skipped: {args.warmup}")
    print()

    # CK kernels
    if ck_kernels:
        print("-" * 100)
        print("CK Unified Attention Kernels")
        print("-" * 100)
        for name, calls in ck_kernels.items():
            stats = compute_stats(calls, skip_first_n=args.warmup)
            if stats:
                print(f"Kernel: ck_tile::UnifiedAttentionKernel")
                print(f"  Total calls (all):       {len(calls)}")
                print(f"  Warmup calls skipped:    {args.warmup}")
                print(f"  Benchmark calls:         {stats['count']}")
                print(f"  Total time:              {stats['total_ms']:.2f} ms")
                print(f"  Average per call:        {stats['avg_us']:.2f} us")
                print(f"  Median per call:         {stats['median_us']:.2f} us")
                print(f"  Min per call:            {stats['min_us']:.2f} us")
                print(f"  Max per call:            {stats['max_us']:.2f} us")
        print()

    # Triton kernels
    if triton_kernels:
        print("-" * 100)
        print("Triton Unified Attention Kernels")
        print("-" * 100)
        for name, calls in triton_kernels.items():
            stats = compute_stats(calls, skip_first_n=args.warmup)
            if stats:
                kernel_type = "2D" if "2d" in name else "3D" if "3d" in name else "unknown"
                print(f"Kernel: kernel_unified_attention_{kernel_type.lower()}")
                print(f"  Total calls (all):       {len(calls)}")
                print(f"  Warmup calls skipped:    {args.warmup}")
                print(f"  Benchmark calls:         {stats['count']}")
                print(f"  Total time:              {stats['total_ms']:.2f} ms")
                print(f"  Average per call:        {stats['avg_us']:.2f} us")
                print(f"  Median per call:         {stats['median_us']:.2f} us")
                print(f"  Min per call:            {stats['min_us']:.2f} us")
                print(f"  Max per call:            {stats['max_us']:.2f} us")

        # Show reduction kernels if present (for 3D variant)
        if triton_reduce_kernels:
            print()
            print("  Associated Reduction Kernels (for 3D):")
            for name, calls in triton_reduce_kernels.items():
                stats = compute_stats(calls, skip_first_n=args.warmup)
                if stats:
                    short_name = name[:60] + "..." if len(name) > 60 else name
                    print(f"    {short_name}")
                    print(f"      Benchmark calls:     {stats['count']}")
                    print(f"      Total time:          {stats['total_ms']:.2f} ms")
                    print(f"      Average per call:    {stats['avg_us']:.2f} us")
        print()

    # Comparison
    if ck_kernels and triton_kernels:
        ck_stats = compute_stats(list(ck_kernels.values())[0], skip_first_n=args.warmup)
        triton_stats = compute_stats(list(triton_kernels.values())[0], skip_first_n=args.warmup)

        # For 3D kernels, add reduction kernel time
        triton_total_time_ms = triton_stats['total_ms']
        if triton_reduce_kernels:
            for reduce_calls in triton_reduce_kernels.values():
                reduce_stats = compute_stats(reduce_calls, skip_first_n=args.warmup)
                if reduce_stats:
                    triton_total_time_ms += reduce_stats['total_ms']
                    # Update avg/median to include reduction
                    triton_stats['avg_us'] = triton_total_time_ms * 1e3 / triton_stats['count']
                    # Note: median is harder to compute correctly, so we'll use avg for 3D

        if ck_stats and triton_stats:
            print("=" * 100)
            variant = " (incl. reduction)" if triton_reduce_kernels else ""
            print(f"Performance Comparison (Benchmark Iterations){variant}")
            print("=" * 100)
            print(f"{'Metric':<30} {'CK':>15} {'Triton':>15} {'Speedup':>15}")
            print("-" * 100)
            print(f"{'Benchmark iterations':<30} {ck_stats['count']:>15} {triton_stats['count']:>15} {'-':>15}")
            print(f"{'Total time (ms)':<30} {ck_stats['total_ms']:>15.2f} {triton_total_time_ms:>15.2f} {'-':>15}")
            print(f"{'Average per iter (us)':<30} {ck_stats['avg_us']:>15.2f} {triton_stats['avg_us']:>15.2f} {triton_stats['avg_us']/ck_stats['avg_us']:>14.2f}x")

            # For 3D, median is less meaningful after adding reduce time
            if triton_reduce_kernels:
                print(f"{'Median per iter (us)':<30} {ck_stats['median_us']:>15.2f} {'(N/A for 3D)':>15} {'-':>15}")
            else:
                print(f"{'Median per iter (us)':<30} {ck_stats['median_us']:>15.2f} {triton_stats['median_us']:>15.2f} {triton_stats['median_us']/ck_stats['median_us']:>14.2f}x")

            print(f"{'Min per iter (us)':<30} {ck_stats['min_us']:>15.2f} {triton_stats['min_us']:>15.2f} {triton_stats['min_us']/ck_stats['min_us']:>14.2f}x")
            print(f"{'Max per iter (us)':<30} {ck_stats['max_us']:>15.2f} {triton_stats['max_us']:>15.2f} {triton_stats['max_us']/ck_stats['max_us']:>14.2f}x")

            if triton_reduce_kernels:
                print(f"\nNote: Triton 3D times include reduction kernel overhead")

            print("=" * 100)

    # Verbose mode - show all kernels
    if args.verbose:
        print()
        print("=" * 100)
        print("All Kernels Summary")
        print("=" * 100)
        kernel_summary = []
        for name, calls in kernel_calls.items():
            stats = compute_stats(calls, skip_first_n=args.warmup)
            if stats:
                short_name = name[:60] + "..." if len(name) > 60 else name
                kernel_summary.append((short_name, stats))

        # Sort by total time
        kernel_summary.sort(key=lambda x: x[1]['total_ms'], reverse=True)

        print(f"{'Kernel Name':<63} {'Calls':>8} {'Total (ms)':>12} {'Avg (us)':>12}")
        print("-" * 100)
        for name, stats in kernel_summary[:20]:
            print(f"{name:<63} {stats['count']:>8} {stats['total_ms']:>12.2f} {stats['avg_us']:>12.2f}")
        print("=" * 100)


if __name__ == "__main__":
    main()
