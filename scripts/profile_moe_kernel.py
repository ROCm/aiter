#!/usr/bin/env python3
"""
Simple MOE kernel profiler using rocprofv3.
Runs the best kernel and collects performance counters.
"""

import subprocess
import csv
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

# Set up rocprofv3 path
os.environ['PATH'] = f"{os.environ['PATH']}:/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/bin"


# Essential counters
COUNTERS = [
    # Compute
    "SQ_INSTS_MFMA", "MfmaFlops", "MfmaUtil", "SQ_INSTS_VALU",
    # Memory
    "TCC_HIT_sum", "TCC_MISS_sum", "TCP_TCC_READ_REQ_sum", "TCP_TCC_WRITE_REQ_sum",
    # Coalescing
    "TCC_EA0_RDREQ_32B_sum", "TCC_EA0_RDREQ_64B_sum", "TCC_EA0_WRREQ_32B_sum",
    # LDS
    "SQ_LDS_BANK_CONFLICT", "SQ_LDS_READ", "SQ_LDS_WRITE",
    # Stalls
    "SQ_WAIT_ANY_CYCLES", "SQ_ACTIVE_CYCLES",
    # Utilization
    "GPU_UTIL", "SQ_WAVE_CYCLES", "GRBM_GUI_ACTIVE",
    # Precision
    "MfmaFlopsF16", "MfmaFlopsF32",
]

# MI300X specs
MI300X = {
    "peak_tflops_fp32": 91.1,
    "peak_tflops_fp16": 728.8,
    "bandwidth_gb_s": 6000.0,
    "ridge_point_fp32": 15.2,
    "ridge_point_fp16": 121.5,
    "num_cu": 384,
}


def load_kernels():
    """Load all kernel configs from CSV."""
    csv_path = Path("results/all_kernel_combinations.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} kernel configurations")
    return df


def run_rocprofv3(kernel, kernel_idx, total_kernels, num_runs=1):
    """Run rocprofv3 on kernel."""
    output_dir = Path("rocprofv3_results")
    output_dir.mkdir(exist_ok=True)
    
    config_name = (f"token{kernel['token']}_dim{kernel['model_dim']}_"
                   f"expert{kernel['expert']}_topk{kernel['topk']}")
    output_file = output_dir / f"rocprofv3_{config_name}"
    
    print(f"\n[{kernel_idx}/{total_kernels}] {kernel['kernel_name']}")
    print(f"  Config: {config_name}")
    print(f"  Time: {kernel['time_us']:.2f} µs, TFLOP/s: {kernel['tflops']:.2f}")
    
    # Create a temporary Python script to run (outside rocprofv3_results)
    temp_script = Path(f"run_{config_name}.py")
    script_content = f"""from aiter.ops.moe import fused_moe
import torch

a = torch.randn(1, 7168, device='cuda')
w = torch.randn(18432, 7168, device='cuda')
e = torch.ones(1, device='cuda', dtype=torch.int32)

for _ in range({num_runs}):
    fused_moe(a, w, e)
"""
    temp_script.write_text(script_content)
    
    # Build rocprofv3 command with counters
    # Use the current Python interpreter which has aiter installed
    python_exe = sys.executable
    counter_str = " ".join(COUNTERS)
    cmd = f'rocprofv3 --kernel-trace --pmc {counter_str} -d {output_dir} -o {output_file} -- {python_exe} {temp_script.absolute()}'
    
    print(f"  Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode != 0 and "rocprofv3" not in result.stderr:
            print(f"  ✗ Error: {result.stderr[:300]}")
            temp_script.unlink()
            return None
        if result.stdout:
            print(f"  Output:\n{result.stdout}")
        if result.stderr:
            print(f"  Stderr:\n{result.stderr}")
        print(f"  ✓ rocprofv3 completed")
        temp_script.unlink()
        return str(output_file)
    except subprocess.TimeoutExpired:
        print(f"  ✗ rocprofv3 timed out")
        temp_script.unlink()
        return None


def parse_and_save(output_file):
    """Parse rocprofv3 output and save to CSV."""
    if not output_file:
        return
    
    # Look for results in rocprofv3_results directory
    results_dir = Path("rocprofv3_results")
    if not results_dir.exists():
        print("No results directory found")
        return
    
    # Find and parse rocprofv3 output
    for csv_file in results_dir.glob("*.csv"):
        print(f"\nFound: {csv_file}")
        try:
            df = pd.read_csv(csv_file)
            print(f"Counters: {len(df.columns)} columns")
            print("\nKey metrics:")
            for col in ["MfmaFlops", "GPU_UTIL", "TCC_HIT_sum", "SQ_LDS_BANK_CONFLICT"]:
                if col in df.columns:
                    val = df[col].iloc[0] if len(df) > 0 else 0
                    print(f"  {col}: {val}")
        except Exception as e:
            print(f"Could not parse {csv_file}: {e}")


def main():
    print("=" * 60)
    print("MOE Kernel rocprofv3 Profiler - All Kernels")
    print("=" * 60)
    
    kernels = load_kernels()
    
    completed = 0
    failed = 0
    
    for idx, (_, kernel) in enumerate(kernels.iterrows(), 1):
        output = run_rocprofv3(kernel, idx, len(kernels), num_runs=1)
        if output:
            completed += 1
            parse_and_save(output)
        else:
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Summary: {completed} completed, {failed} failed")
    print("Check rocprofv3_results/ for all output files.")
    print("=" * 60)


if __name__ == "__main__":
    main()
