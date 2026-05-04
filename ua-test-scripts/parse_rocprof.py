#!/usr/bin/env python3
"""
Parse ROCProfiler CSV and generate kernel summary table.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def parse_rocprof_csv(csv_path):
    """Parse ROCProfiler CSV and aggregate kernel statistics."""
    stats = defaultdict(lambda: {'calls': 0, 'total_ns': 0, 'min_ns': float('inf'), 'max_ns': 0})

    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle different CSV formats (rocprof v1, v2, v3)
                name = row.get('Name', row.get('Kernel_Name', row.get('KernelName', '')))
                duration_str = row.get('DurationNs', row.get('TotalDurationNs', row.get('Total_Duration', row.get('Duration', ''))))

                if not name or name.startswith('#') or not duration_str:
                    continue

                try:
                    duration_ns = float(duration_str)
                    stats[name]['calls'] += 1
                    stats[name]['total_ns'] += duration_ns
                    stats[name]['min_ns'] = min(stats[name]['min_ns'], duration_ns)
                    stats[name]['max_ns'] = max(stats[name]['max_ns'], duration_ns)
                except (ValueError, KeyError):
                    continue

    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        return None

    return dict(stats)


def print_summary_table(stats, top_n=20, filter_pattern=None):
    """Print formatted summary table of kernel statistics."""
    if not stats:
        print("No kernel statistics found")
        return

    # Calculate total time
    total_time_ns = sum(s['total_ns'] for s in stats.values())

    # Filter if pattern provided
    if filter_pattern:
        stats = {k: v for k, v in stats.items() if filter_pattern.lower() in k.lower()}

    # Sort by total duration
    sorted_kernels = sorted(stats.items(), key=lambda x: x[1]['total_ns'], reverse=True)

    if not sorted_kernels:
        print(f"No kernels matching pattern: {filter_pattern}")
        return

    # Print header
    print()
    print("=" * 120)
    print("ROCProfiler Kernel Summary")
    print("=" * 120)
    print(f"{'Kernel Name':<65} {'Calls':>8} {'Total (ms)':>12} {'Avg (us)':>12} {'Min (us)':>12} {'Max (us)':>12} {'%':>7}")
    print("-" * 120)

    # Print top N kernels
    for name, data in sorted_kernels[:top_n]:
        calls = data['calls']
        total_ms = data['total_ns'] / 1e6
        avg_us = data['total_ns'] / calls / 1e3 if calls > 0 else 0
        min_us = data['min_ns'] / 1e3
        max_us = data['max_ns'] / 1e3
        pct = data['total_ns'] / total_time_ns * 100 if total_time_ns > 0 else 0

        # Truncate long kernel names
        display_name = name[:64] if len(name) > 65 else name

        print(f"{display_name:<65} {calls:>8} {total_ms:>12.4f} {avg_us:>12.2f} "
              f"{min_us:>12.2f} {max_us:>12.2f} {pct:>6.2f}%")

    print("-" * 120)
    print(f"{'Total GPU time':<65} {sum(s['calls'] for s in stats.values()):>8} "
          f"{total_time_ns/1e6:>12.4f} {'':>12} {'':>12} {'':>12} {'100.00%':>7}")
    print("=" * 120)
    print()


def identify_kernels(stats):
    """Identify CK and Triton kernels from the stats."""
    ck_kernels = {}
    triton_kernels = {}
    other_kernels = {}

    for name, data in stats.items():
        name_lower = name.lower()
        if 'fmha' in name_lower or 'unified_attention' in name_lower or 'ck_tile' in name_lower:
            ck_kernels[name] = data
        elif 'triton' in name_lower or '_attn_' in name_lower:
            triton_kernels[name] = data
        else:
            other_kernels[name] = data

    return ck_kernels, triton_kernels, other_kernels


def print_comparison(ck_stats, triton_stats):
    """Print CK vs Triton comparison."""
    ck_total = sum(s['total_ns'] for s in ck_stats.values()) if ck_stats else 0
    triton_total = sum(s['total_ns'] for s in triton_stats.values()) if triton_stats else 0
    ck_calls = sum(s['calls'] for s in ck_stats.values()) if ck_stats else 0
    triton_calls = sum(s['calls'] for s in triton_stats.values()) if triton_stats else 0

    print()
    print("=" * 80)
    print("CK vs Triton Kernel Comparison")
    print("=" * 80)
    print(f"{'Backend':<15} {'Kernels':>10} {'Total Calls':>15} {'Total Time (ms)':>20} {'Avg/Call (us)':>20}")
    print("-" * 80)

    if ck_total > 0:
        ck_avg = ck_total / ck_calls / 1e3 if ck_calls > 0 else 0
        print(f"{'CK':<15} {len(ck_stats):>10} {ck_calls:>15} {ck_total/1e6:>20.4f} {ck_avg:>20.2f}")

    if triton_total > 0:
        triton_avg = triton_total / triton_calls / 1e3 if triton_calls > 0 else 0
        print(f"{'Triton':<15} {len(triton_stats):>10} {triton_calls:>15} {triton_total/1e6:>20.4f} {triton_avg:>20.2f}")

    if ck_total > 0 and triton_total > 0:
        speedup = triton_total / ck_total
        print("-" * 80)
        print(f"Speedup (Triton/CK): {speedup:.4f}x")

    print("=" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(description="Parse ROCProfiler CSV and generate summary")
    parser.add_argument("csv_file", help="ROCProfiler CSV file")
    parser.add_argument("-n", "--top", type=int, default=20,
                        help="Show top N kernels (default: 20)")
    parser.add_argument("-f", "--filter", help="Filter kernels by name pattern")
    parser.add_argument("--compare", action="store_true",
                        help="Show CK vs Triton comparison")
    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # Parse CSV
    stats = parse_rocprof_csv(csv_path)
    if not stats:
        print("Error: No kernel data found in CSV", file=sys.stderr)
        sys.exit(1)

    # Print main summary table
    print_summary_table(stats, top_n=args.top, filter_pattern=args.filter)

    # Print comparison if requested
    if args.compare:
        ck_stats, triton_stats, _ = identify_kernels(stats)
        if ck_stats or triton_stats:
            print_comparison(ck_stats, triton_stats)

            # Show detailed breakdowns
            if ck_stats:
                print("\nCK Kernels Detail:")
                print_summary_table(ck_stats, top_n=10)

            if triton_stats:
                print("\nTriton Kernels Detail:")
                print_summary_table(triton_stats, top_n=10)


if __name__ == "__main__":
    main()
