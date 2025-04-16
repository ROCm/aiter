import subprocess
import os
import pandas as pd
from prettytable import PrettyTable


def run_profiling(triton_dir, batch_size, output_file):
    command = [
        "rocprof", "--stats", "-o", output_file, "python", f"{triton_dir}op_benchmarks/triton/bench_mha.py", "-b",
        str(batch_size), "-dtype", "fp16", "-layout", "bshd", "-causal", "True", "-model", "llama3_70B"
    ]
    subprocess.run(command, check=True)


def parse_profiling_output(output_file, kernel_names):
    df = pd.read_csv(output_file)
    results = {}
    for kernel in kernel_names:
        kernel_data = df[df['Name'].str.strip('"') == kernel]
        if not kernel_data.empty:
            results[kernel] = kernel_data['AverageNs'].iloc[0] / 1000.0
        else:
            results[kernel] = None

    # Calculate sum of other kernels
    other_kernels = df[~df['Name'].str.strip('"').isin(kernel_names)]
    other_kernels_sum = other_kernels['AverageNs'].sum() / 1000.0
    results['other_kernels_sum'] = other_kernels_sum

    return results


def main():
    triton_dir = os.environ.get("AITERDIR", "~/aiter")  # Default to ~/triton if not set
    output_file = os.path.expanduser("~/profiling.csv")
    kernel_names = ["_bwd_kernel_dkdvdq_causal.kd", "_bwd_kernel_dkdv_causal.kd", "_bwd_kernel_dq_causal.kd",]
    batch_sizes = [1, 4, 32]

    results = {B: {} for B in batch_sizes}
    for B in batch_sizes:
        print(f"Running profiling for B={B}...")
        run_profiling(triton_dir, B, output_file)
        output_stats_file = os.path.expanduser("~/profiling.stats.csv")
        kernel_results = parse_profiling_output(output_stats_file, kernel_names)
        results[B] = kernel_results

    table = PrettyTable()
    table.field_names = ["B"] + kernel_names + ["Other Kernels Sum (µs)"]
    for B in batch_sizes:
        row = [B] + [results[B].get(kernel, "N/A")
                     for kernel in kernel_names] + [results[B].get('other_kernels_sum', "N/A")]
        table.add_row(row)

    print("\nProfiling Summary (in microseconds):")
    print(table)


if __name__ == "__main__":
    main()
