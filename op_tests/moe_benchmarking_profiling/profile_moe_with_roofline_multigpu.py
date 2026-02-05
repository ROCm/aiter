#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Profile MOE kernels using rooflineExtractor workflow with Multi-GPU support

Purpose:
    Profiles best-performing MOE kernels using rooflineExtractor's profile.py
    for sophisticated roofline analysis. Supports parallel execution across multiple GPUs.

Input:
    CSV file with best kernel selections
    Required columns: config_idx, token, model_dim, inter_dim, expert, topk,
                     act_type, dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1,
                     doweight_stage1, kernel_type, stage, block_m, kernel_name

Usage:
    python profile_moe_with_roofline_multigpu.py -i best_kernels.csv --num-gpus 2
"""

import subprocess
import sys
from pathlib import Path
import pandas as pd
import argparse
import shutil
import os
from multiprocessing import Pool, Manager
import time

# Check for required dependencies
try:
    import plotly
except ImportError:
    print("Error: plotly is required but not installed.")
    print("Install with: pip install plotly")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("Error: torch is required but not installed.")
    sys.exit(1)

# Add paths for tuner imports
current_dir = Path(__file__).parent
aiter_root = current_dir.parent.parent
sys.path.insert(0, str(aiter_root))

# Import shared utilities
import moe_utils

# Import script generation functions
from profile_kernels import (
    generate_asm_1stage_script,
    generate_ck_stage1_script,
    generate_ck_stage2_script,
    generate_asm_stage1_script,
)


def get_kernels(csv_file):
    """Load the CSV with best kernels."""
    csv_path = Path(csv_file)
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)
    return pd.read_csv(csv_file)


def create_kernel_execution_script(row_dict, script_path):
    """Create script using tuner.generate_data() for input generation."""
    row = pd.Series(row_dict)
    
    config_idx = int(row['config_idx'])
    kernel_name = row['kernel_name']
    kernel_type = str(row['kernel_type'])
    stage = str(row['stage'])
    
    token = int(row['token'])
    model_dim = int(row['model_dim'])
    inter_dim = int(row['inter_dim'])
    expert = int(row['expert'])
    topk = int(row['topk'])
    block_m = int(row['block_m'])
    
    dtype_str = str(row['dtype'])
    q_dtype_a_str = str(row['q_dtype_a'])
    q_dtype_w_str = str(row['q_dtype_w'])
    q_type_str = str(row['q_type'])
    act_type_str = str(row['act_type'])
    use_g1u1 = bool(row['use_g1u1'])
    doweight_stage1 = bool(row['doweight_stage1'])
    
    # Generate script
    if stage == "asm_1stage":
        code = generate_asm_1stage_script(config_idx, kernel_name, token, model_dim, inter_dim,
                                         expert, topk, block_m, dtype_str, q_dtype_a_str, q_dtype_w_str,
                                         q_type_str, act_type_str, use_g1u1, doweight_stage1)
    elif kernel_type == "ck" and stage == "stage2":
        code = generate_ck_stage2_script(config_idx, kernel_name, token, model_dim, inter_dim,
                                         expert, topk, block_m, dtype_str, q_dtype_a_str, q_dtype_w_str,
                                         q_type_str, act_type_str, use_g1u1, doweight_stage1)
    elif kernel_type == "ck" and stage == "stage1":
        code = generate_ck_stage1_script(config_idx, kernel_name, token, model_dim, inter_dim,
                                         expert, topk, block_m, dtype_str, q_dtype_a_str, q_dtype_w_str,
                                         q_type_str, act_type_str, use_g1u1, doweight_stage1)
    else:
        code = generate_asm_stage1_script(config_idx, kernel_name, stage, token, model_dim, inter_dim,
                                         expert, topk, block_m, dtype_str, q_dtype_a_str, q_dtype_w_str,
                                         q_type_str, act_type_str, use_g1u1, doweight_stage1)
    
    script_path.write_text(code)
    script_path.chmod(0o755)


def safe_append_to_csv(df, filepath, lock):
    """Thread-safe CSV append with lock."""
    with lock:
        if filepath.exists():
            df.to_csv(filepath, mode='a', header=False, index=False)
        else:
            df.to_csv(filepath, index=False)


def extract_and_append_target_kernel_locked(kernel_dir, config_idx, target_kernel_name, 
                                              combined_counters_file, combined_traces_file, lock):
    """Extract target kernel and append with thread-safe locking."""
    counters_file = kernel_dir / "counters.csv"
    trace_file = kernel_dir / "trace_kernel_trace.csv"
    
    if not (counters_file.exists() and trace_file.exists()):
        return False, "Missing files"
    
    df_trace = pd.read_csv(trace_file)
    
    # Determine search pattern
    if target_kernel_name.startswith('_ZN') and 'fmoe' in target_kernel_name:
        search_name = 'aiter::fmoe'
    elif target_kernel_name.startswith('moe_ck2stages'):
        search_name = 'kernel_moe_gemm'
    else:
        search_name = target_kernel_name
    
    # Match target kernel
    target_mask = df_trace['Kernel_Name'].str.contains(search_name, regex=False, na=False)
    df_target_trace = df_trace[target_mask].copy()
    
    if len(df_target_trace) == 0:
        return False, f"Target kernel not found (searched: '{search_name}')"
    
    new_kernel_name = f"cfg{config_idx}_{target_kernel_name}"
    df_target_trace['Kernel_Name'] = new_kernel_name
    
    # Match in counters
    df_counters = pd.read_csv(counters_file)
    target_mask = df_counters['Kernel_Name'].str.contains(search_name, regex=False, na=False)
    df_target_counters = df_counters[target_mask].copy()
    
    if len(df_target_counters) == 0:
        return False, "Target kernel not found in counters"
    
    df_target_counters['Kernel_Name'] = new_kernel_name
    
    # Thread-safe append
    safe_append_to_csv(df_target_counters, combined_counters_file, lock)
    safe_append_to_csv(df_target_trace, combined_traces_file, lock)
    
    # Clean up
    shutil.rmtree(kernel_dir)
    
    return True, f"Success ({len(df_target_trace)} dispatches)"


def profile_kernel_worker(args):
    """
    Worker function for parallel GPU execution.
    Each worker runs on a dedicated GPU with its own rooflineExtractor instance.
    """
    gpu_id, idx, row_dict, output_dir, scripts_dir, roofline_profile_script, arch, \
        combined_counters_file, combined_traces_file, keep_scripts, file_lock = args
    
    # CRITICAL: Set GPU for this worker process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['HIP_VISIBLE_DEVICES'] = str(gpu_id)
    
    row = pd.Series(row_dict)
    
    config_idx = int(row['config_idx'])
    kernel_type = str(row['kernel_type'])
    stage = str(row['stage'])
    block_m = int(row['block_m'])
    kernel_name = row['kernel_name']
    
    # Generate execution script
    script_name = f"cfg{config_idx}_{kernel_type}_{stage}_block{block_m}_{kernel_name}.py"
    script_path = Path(scripts_dir) / script_name
    
    try:
        create_kernel_execution_script(row_dict, script_path)
        
        # Create unique output directory
        kernel_output_dir = Path(output_dir) / f"cfg{config_idx}_{kernel_type}_{stage}_block{block_m}"
        kernel_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Build rooflineExtractor command
        cmd = [
            sys.executable,
            str(roofline_profile_script),
            "-o", str(kernel_output_dir.absolute()),
            "--arch", arch,
            "--",
            sys.executable, str(script_path.absolute())
        ]
        
        # Run profiling
        result = subprocess.run(
            cmd,
            cwd=Path(roofline_profile_script).parent,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode == 0:
            # Extract and append with lock
            success, msg = extract_and_append_target_kernel_locked(
                kernel_output_dir, config_idx, kernel_name,
                Path(combined_counters_file), Path(combined_traces_file), file_lock
            )
            
            # Clean up script
            if not keep_scripts and script_path.exists():
                script_path.unlink()
            
            return {
                'idx': idx,
                'gpu_id': gpu_id,
                'config_idx': config_idx,
                'kernel_name': kernel_name[:40],
                'success': success,
                'message': msg
            }
        else:
            error_msg = "HIP error" if ("HIP error" in result.stderr or "illegal memory" in result.stderr) else "Profiling failed"
            return {
                'idx': idx,
                'gpu_id': gpu_id,
                'config_idx': config_idx,
                'kernel_name': kernel_name[:40],
                'success': False,
                'message': error_msg
            }
            
    except subprocess.TimeoutExpired:
        return {'idx': idx, 'gpu_id': gpu_id, 'config_idx': config_idx, 'kernel_name': kernel_name[:40], 'success': False, 'message': "Timeout"}
    except Exception as e:
        return {'idx': idx, 'gpu_id': gpu_id, 'config_idx': config_idx, 'kernel_name': kernel_name[:40], 'success': False, 'message': str(e)}
    finally:
        # Always clean up script
        if not keep_scripts and script_path.exists():
            try:
                script_path.unlink()
            except:
                pass


def generate_combined_roofline(counters_file, traces_file, output_dir, roofline_extractor_script, arch):
    """Generate combined roofline plot."""
    print("\nGenerating combined roofline plot...")
    
    if not (counters_file.exists() and traces_file.exists()):
        print("  ✗ Missing required files")
        return False
    
    cmd = [
        sys.executable, str(roofline_extractor_script),
        "-c", str(counters_file.absolute()),
        "-r", str(traces_file.absolute()),
        "-p", "-d", "--arch", arch
    ]
    
    print(f"  Running: rooflineExtractor.py -c {counters_file.name} -r {traces_file.name} -p -d --arch {arch}")
    
    try:
        result = subprocess.run(cmd, cwd=roofline_extractor_script.parent,
                               capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            base_name = counters_file.stem
            html_source = counters_file.parent / f"{base_name}.html"
            if html_source.exists():
                shutil.move(str(html_source), str(output_dir / "moe_combined_roofline.html"))
                print(f"  ✓ Combined roofline plot saved")
            
            for suffix in ["_EXTRACTED.csv", "_EXTRACTED_AGG.csv"]:
                csv_source = counters_file.parent / f"{base_name}{suffix}"
                if csv_source.exists():
                    shutil.move(str(csv_source), str(output_dir / f"moe_only{suffix}"))
            
            if result.stdout:
                print("\n" + "="*80)
                print("ROOFLINE ANALYSIS SUMMARY")
                print("="*80)
                print(result.stdout)
            return True
        else:
            print(f"  ✗ Failed (exit code {result.returncode})")
            return False
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Profile MOE kernels with Multi-GPU support")
    parser.add_argument('-i', '--input', required=True, help='Input CSV file')
    parser.add_argument('-o', '--output-dir', default='results/roofline_profiling', help='Output directory')
    parser.add_argument('--arch', default='MI300X', choices=['MI250X', 'MI300A', 'MI300X', 'MI355X'])
    parser.add_argument('--keep-scripts', action='store_true', help='Keep scripts')
    parser.add_argument('--resume', action='store_true', help='Resume profiling')
    parser.add_argument('--num-gpus', type=int, default=None, help='Number of GPUs to use (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Detect GPUs
    if args.num_gpus is None:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("Warning: No GPUs detected, falling back to 1 worker")
            num_gpus = 1
    else:
        num_gpus = args.num_gpus
    
    # Setup paths
    kernels = get_kernels(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    scripts_dir = output_dir / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    
    # Locate rooflineExtractor and create per-GPU copies to avoid temp file conflicts
    roofline_base = Path(__file__).parent / "rooflineExtractor"
    
    if not roofline_base.exists():
        print(f"Error: rooflineExtractor not found at {roofline_base}")
        sys.exit(1)
    
    # Create per-GPU rooflineExtractor copies
    roofline_gpu_dirs = []
    for gpu_id in range(num_gpus):
        gpu_roofline_dir = Path(__file__).parent / f"rooflineExtractor_gpu{gpu_id}"
        
        # Copy if doesn't exist or force refresh
        if gpu_roofline_dir.exists():
            shutil.rmtree(gpu_roofline_dir)
        
        shutil.copytree(roofline_base, gpu_roofline_dir)
        print(f"Created rooflineExtractor instance for GPU{gpu_id}: {gpu_roofline_dir.name}")
        roofline_gpu_dirs.append(gpu_roofline_dir)
    
    print()
    
    roofline_extractor_script = roofline_base / "rooflineExtractor.py"  # Use original for final analysis
    
    # Setup combined output files
    combined_counters_file = output_dir / "moe_target_counters.csv"
    combined_traces_file = output_dir / "moe_target_traces.csv"
    
    if not args.resume:
        combined_counters_file.unlink(missing_ok=True)
        combined_traces_file.unlink(missing_ok=True)
    
    num_kernels = len(kernels)
    
    moe_utils.print_section_header("MOE ROOFLINE PROFILING - MULTI-GPU")
    print(f"GPUs available: {num_gpus}")
    print(f"Kernels to profile: {num_kernels}")
    print(f"Input: {args.input}")
    print(f"Output: {output_dir.absolute()}")
    print(f"Architecture: {args.arch}")
    print(f"Parallel workers: {num_gpus}")
    print(f"{'='*80}\n")
    
    # Create thread-safe lock for file writes
    manager = Manager()
    file_lock = manager.Lock()
    
    # Filter kernels to process
    kernels_to_process = []
    skipped_count = 0
    
    # Load already-profiled kernels for resume mode
    profiled_kernels = set()
    if args.resume and combined_traces_file.exists():
        df_existing = pd.read_csv(combined_traces_file)
        # Extract config_idx and kernel name from the renamed kernel names
        for kernel_name in df_existing['Kernel_Name'].unique():
            profiled_kernels.add(kernel_name)
        print(f"Resume mode: Found {len(profiled_kernels)} already-profiled kernels")
        print()
    
    for idx, (_, row) in enumerate(kernels.iterrows(), 1):
        error_value = str(row.get('error', '')).strip()
        if error_value == 'failed':
            skipped_count += 1
            continue
        
        # Check if already profiled (resume mode)
        config_idx = int(row['config_idx'])
        kernel_type = str(row['kernel_type'])
        stage = str(row['stage'])
        target_kernel_name = row['kernel_name']
        
        # Generate the renamed kernel name to check if it exists
        expected_kernel_name = f"cfg{config_idx}_{target_kernel_name}"
        
        if args.resume and expected_kernel_name in profiled_kernels:
            skipped_count += 1
            continue
        
        # Prepare worker args - use GPU-specific rooflineExtractor copy
        gpu_id = (len(kernels_to_process)) % num_gpus
        gpu_roofline_profile = roofline_gpu_dirs[gpu_id] / "profile.py"
        
        worker_args = (
            gpu_id, idx, row.to_dict(), str(output_dir), str(scripts_dir),
            str(gpu_roofline_profile), args.arch,
            str(combined_counters_file), str(combined_traces_file),
            args.keep_scripts, file_lock
        )
        kernels_to_process.append(worker_args)
    
    print(f"Processing {len(kernels_to_process)} kernels with {num_gpus} GPUs")
    print(f"Skipped: {skipped_count}")
    print(f"{'='*80}\n")
    
    # Run parallel profiling
    successful_count = 0
    failed_count = 0
    
    with Pool(processes=num_gpus) as pool:
        # Use imap for real-time progress reporting
        for result in pool.imap(profile_kernel_worker, kernels_to_process):
            idx = result['idx']
            gpu_id = result['gpu_id']
            success = result['success']
            message = result['message']
            kernel_name = result['kernel_name']
            config_idx = result['config_idx']
            
            status = "✓" if success else "✗"
            print(f"[{idx}/{num_kernels}] GPU{gpu_id} Config {config_idx}: {kernel_name} {status} {message}")
            
            if success:
                successful_count += 1
            else:
                failed_count += 1
    
    # Generate final roofline plot
    if combined_counters_file.exists() and combined_traces_file.exists():
        moe_utils.print_section_header("GENERATING ROOFLINE PLOT")
        generate_combined_roofline(combined_counters_file, combined_traces_file, output_dir, 
                                   roofline_extractor_script, args.arch)
    
    # Final summary
    moe_utils.print_section_header("PROFILING SUMMARY")
    print(f"Total kernels in input: {num_kernels}")
    print(f"Successfully profiled: {successful_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Failed: {failed_count}")
    print(f"\nOutput directory: {output_dir.absolute()}")
    print(f"\nKey output files:")
    print(f"  - Combined roofline plot: moe_combined_roofline.html")
    print(f"  - Target kernel counters: moe_target_counters.csv")
    print(f"  - Target kernel traces: moe_target_traces.csv")
    print(f"  - Detailed analysis: moe_only_EXTRACTED.csv")
    print(f"  - Aggregated metrics: moe_only_EXTRACTED_AGG.csv")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
