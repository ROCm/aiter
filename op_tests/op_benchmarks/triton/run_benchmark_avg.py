#!/usr/bin/env python3
"""
Run benchmark multiple times and compute average metrics.
"""

import subprocess
import re
import sys
from collections import defaultdict
import argparse


def parse_benchmark_output(output):
    """Parse benchmark output and extract metrics per batch size."""
    metrics = {}

    # Look for lines like: batch:     1 | Total latency (us): 122.87 | Kernel latency (us): 62.81 | TFLOPS: 7.479 | TBPS: 3.74
    pattern = r"batch:\s+(\d+)\s+\|\s+Total latency \(us\):\s+([\d.]+)\s+\|\s+Kernel latency \(us\):\s+([\d.]+)\s+\|\s+TFLOPS:\s+([\d.]+)\s+\|\s+TBPS:\s+([\d.]+)"

    for line in output.split("\n"):
        match = re.search(pattern, line)
        if match:
            batch = int(match.group(1))
            total_latency = float(match.group(2))
            kernel_latency = float(match.group(3))
            tflops = float(match.group(4))
            tbps = float(match.group(5))

            metrics[batch] = {
                "total_latency": total_latency,
                "kernel_latency": kernel_latency,
                "tflops": tflops,
                "tbps": tbps,
            }

    return metrics


def run_benchmark(cmd, num_runs=10):
    """Run benchmark multiple times and collect metrics."""
    all_metrics = defaultdict(lambda: defaultdict(list))

    for i in range(num_runs):
        print(f"Running benchmark iteration {i + 1}/{num_runs}...", file=sys.stderr)

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            if result.returncode != 0:
                print(
                    f"Warning: Run {i + 1} failed with return code {result.returncode}",
                    file=sys.stderr,
                )
                print(f"stderr: {result.stderr}", file=sys.stderr)
                continue

            # Parse the output
            metrics = parse_benchmark_output(result.stdout)

            if not metrics:
                print(
                    f"Warning: Run {i + 1} produced no parseable metrics",
                    file=sys.stderr,
                )
                print(f"stdout: {result.stdout}", file=sys.stderr)
                continue

            # Store metrics for each batch size
            for batch, batch_metrics in metrics.items():
                for metric_name, value in batch_metrics.items():
                    all_metrics[batch][metric_name].append(value)

            print(f"  ✓ Completed iteration {i + 1}/{num_runs}", file=sys.stderr)

        except subprocess.TimeoutExpired:
            print(f"Warning: Run {i + 1} timed out", file=sys.stderr)
            continue
        except Exception as e:
            print(f"Warning: Run {i + 1} failed with error: {e}", file=sys.stderr)
            continue

    return all_metrics


def compute_averages(all_metrics):
    """Compute average metrics across all runs."""
    averages = {}

    for batch, metrics_dict in sorted(all_metrics.items()):
        averages[batch] = {}
        for metric_name, values in metrics_dict.items():
            if values:
                averages[batch][metric_name] = sum(values) / len(values)
                averages[batch][f"{metric_name}_count"] = len(values)

    return averages


def print_results(averages):
    """Print averaged results in a formatted table."""
    print("=" * 120)
    print("Averaged over 10 runs")
    print("=" * 120)

    for batch in sorted(averages.keys()):
        metrics = averages[batch]

        total_latency = metrics.get("total_latency", 0)
        kernel_latency = metrics.get("kernel_latency", 0)
        tflops = metrics.get("tflops", 0)
        tbps = metrics.get("tbps", 0)

        print(
            f"batch: {batch:5d} | Total latency (us): {total_latency:7.2f} | "
            f"Kernel latency (us): {kernel_latency:7.2f} | "
            f"TFLOPS: {tflops:#6.4g} | TBPS: {tbps:5.2f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark multiple times and compute averages"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of times to run the benchmark (default: 10)",
    )
    parser.add_argument(
        "--cmd", type=str, help="Command to run (if not provided, uses default)"
    )

    args = parser.parse_args()

    # Default command if not provided
    if args.cmd:
        cmd = args.cmd
    else:
        cmd = (
            "python bench_moe_gemm_a8w8_blockscale.py "
            "--shape 7168 4096  "  # MOE1 TP1
            "--experts 256 8 "
            # "--experts 1 1 "
            "--op-regex .*moe_gemm.* "
            "--act-dtype bs8 "
            "--w-dtype bs8 "
            "--act-per-row-bs True"
        )

    print(f"Running command: {cmd}", file=sys.stderr)
    print(f"Number of runs: {args.runs}", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    # Run benchmark multiple times
    all_metrics = run_benchmark(cmd, num_runs=args.runs)

    if not all_metrics:
        print("Error: No metrics collected from any run", file=sys.stderr)
        sys.exit(1)

    # Compute averages
    averages = compute_averages(all_metrics)

    # Print results
    print("=" * 80, file=sys.stderr)
    print_results(averages)


if __name__ == "__main__":
    main()
