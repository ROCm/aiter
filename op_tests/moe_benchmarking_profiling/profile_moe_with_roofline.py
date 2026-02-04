#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Profile MOE kernels using rooflineExtractor workflow

Purpose:
    Profiles best-performing MOE kernels using rooflineExtractor's profile.py
    for sophisticated roofline analysis with memory hierarchy visualization.

Input:
    CSV file with best kernel selections from benchmark_and_analyze.py
    Required columns: config_idx, token, model_dim, inter_dim, expert, topk,
                     act_type, dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1,
                     doweight_stage1, kernel_type, stage, block_m, kernel_name

Output:
    - Individual roofline plots per kernel (HTML)
    - Combined roofline plot for all MoE kernels (HTML)
    - Aggregated counter data (CSV)

Usage:
    python profile_moe_with_roofline.py -i best_kernels.csv -o profiling_output
"""

import subprocess
import sys
from pathlib import Path
import pandas as pd
import argparse
import shutil

# Check for required dependencies
try:
    import plotly
except ImportError:
    print("Error: plotly is required for rooflineExtractor but not installed.")
    print("Install with: pip install plotly")
    sys.exit(1)

# Add paths for tuner imports
current_dir = Path(__file__).parent
aiter_root = current_dir.parent.parent
sys.path.insert(0, str(aiter_root))

# Import shared utilities
import moe_utils

# Import script generation functions from existing profile_kernels.py
# (We'll reuse the exact same script generation logic)
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


def create_kernel_execution_script(row, script_path):
    """
    Create script using tuner.generate_data() for input generation.
    Reuses exact logic from profile_kernels.py
    """
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
    
    # Generate script based on kernel type and stage
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
    # Make script executable
    script_path.chmod(0o755)


def extract_and_append_target_kernel(kernel_dir, config_idx, target_kernel_name, combined_counters_file, combined_traces_file):
    """
    Extract target kernel from profiling results, rename with config_idx, and append to combined files.
    Cleans up intermediate files after extraction.
    
    Returns:
        True if successful, False otherwise
    """
    counters_file = kernel_dir / "counters.csv"
    trace_file = kernel_dir / "trace_kernel_trace.csv"
    
    if not (counters_file.exists() and trace_file.exists()):
        print(f"    ✗ Missing files")
        return False
    
    # Read trace to find target kernel
    df_trace = pd.read_csv(trace_file)
    
    # Determine search pattern based on kernel name
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
        print(f"    ✗ Target kernel not found (searched: '{search_name}')")
        return False
    
    # Use the input kernel name (not the long demangled one) for cleaner plot labels
    # This gives us names like "cfg0__ZN5aiter..." or "cfg0_moe_ck2stages..." instead of
    # the very long demangled template names
    new_kernel_name = f"cfg{config_idx}_{target_kernel_name}"
    df_target_trace['Kernel_Name'] = new_kernel_name
    
    # Match and rename in counters
    df_counters = pd.read_csv(counters_file)
    target_mask = df_counters['Kernel_Name'].str.contains(search_name, regex=False, na=False)
    df_target_counters = df_counters[target_mask].copy()
    
    if len(df_target_counters) == 0:
        print(f"    ✗ Target kernel not found in counters")
        return False
    
    df_target_counters['Kernel_Name'] = new_kernel_name
    
    # Append to combined files (create if first kernel)
    if combined_counters_file.exists():
        df_target_counters.to_csv(combined_counters_file, mode='a', header=False, index=False)
    else:
        df_target_counters.to_csv(combined_counters_file, index=False)
    
    if combined_traces_file.exists():
        df_target_trace.to_csv(combined_traces_file, mode='a', header=False, index=False)
    else:
        df_target_trace.to_csv(combined_traces_file, index=False)
    
    print(f"    ✓ Extracted & appended ({len(df_target_trace)} dispatches)")
    
    # Clean up entire kernel directory after extraction
    # We don't need it anymore since data is in combined files
    shutil.rmtree(kernel_dir)
    print(f"    ✓ Cleaned up intermediate directory")
    
    return True


def profile_kernel_with_roofline(row, script_path, output_dir, roofline_profile_script, arch, combined_counters_file, combined_traces_file):
    """
    Profile a single kernel, extract target kernel data, and append to combined files.
    
    Returns:
        True if successful, False otherwise
    """
    config_idx = int(row['config_idx'])
    kernel_type = str(row['kernel_type'])
    stage = str(row['stage'])
    block_m = int(row['block_m'])
    kernel_name = row['kernel_name']
    
    # Create unique output directory for this kernel
    kernel_output_dir = output_dir / f"cfg{config_idx}_{kernel_type}_{stage}_block{block_m}"
    kernel_output_dir.mkdir(exist_ok=True, parents=True)
    
    # Build rooflineExtractor profile.py command
    cmd = [
        sys.executable,
        str(roofline_profile_script),
        "-o", str(kernel_output_dir.absolute()),
        "--arch", arch,
        "--",
        sys.executable, str(script_path.absolute())
    ]
    
    print(f"  Profiling: {roofline_profile_script.name} -- {script_path.name}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=roofline_profile_script.parent,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode == 0:
            # Extract target kernel and append to combined files
            success = extract_and_append_target_kernel(
                kernel_output_dir, config_idx, kernel_name,
                combined_counters_file, combined_traces_file
            )
            return success
        else:
            print(f"    ✗ Profiling failed (exit code {result.returncode})")
            if "HIP error" in result.stderr or "illegal memory" in result.stderr:
                print(f"      Kernel memory access bug")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"    ✗ Timeout")
        return False
    except Exception as e:
        print(f"    ✗ Exception: {e}")
        return False



def generate_combined_roofline(counters_file, traces_file, output_dir, roofline_extractor_script, arch):
    """
    Generate combined roofline plot for all MoE kernels using rooflineExtractor.py
    """
    print("\nGenerating combined roofline plot...")
    
    if not counters_file or not traces_file:
        print("  ✗ Missing required files for roofline generation")
        return False
    
    # Build rooflineExtractor.py command with absolute paths
    cmd = [
        sys.executable,
        str(roofline_extractor_script),
        "-c", str(counters_file.absolute()),
        "-r", str(traces_file.absolute()),
        "-p",  # Generate plots
        "-d",  # Dump data
        "--arch", arch
    ]
    
    print(f"  Running: rooflineExtractor.py -c {counters_file.name} -r {traces_file.name} -p -d --arch {arch}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=roofline_extractor_script.parent,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            # Determine HTML name based on input file name
            base_name = counters_file.stem  # e.g., "moe_only_counters" or "moe_all_counters"
            html_source = counters_file.parent / f"{base_name}.html"
            
            if html_source.exists():
                html_dest = output_dir / "moe_combined_roofline.html"
                shutil.move(str(html_source), str(html_dest))
                print(f"  ✓ Combined roofline plot: {html_dest}")
            
            # Move extracted CSVs to output directory
            for suffix in ["_EXTRACTED.csv", "_EXTRACTED_AGG.csv"]:
                csv_source = counters_file.parent / f"{base_name}{suffix}"
                if csv_source.exists():
                    csv_dest = output_dir / f"moe_only{suffix}"
                    shutil.move(str(csv_source), str(csv_dest))
                    print(f"  ✓ Saved: {csv_dest}")
            
            # Print analysis output
            if result.stdout:
                print("\n" + "="*80)
                print("ROOFLINE ANALYSIS SUMMARY")
                print("="*80)
                print(result.stdout)
            
            return True
        else:
            print(f"  ✗ Failed to generate combined roofline (exit code {result.returncode})")
            if result.stderr:
                print(f"    Error: {result.stderr[-500:]}")
            return False
            
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Profile MOE kernels with rooflineExtractor workflow"
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input CSV file with kernel configs from analyze_results.py'
    )
    parser.add_argument(
        '-o', '--output-dir',
        default='results/roofline_profiling',
        help='Output directory for results (default: results/roofline_profiling)'
    )
    parser.add_argument(
        '--arch',
        default='MI300X',
        choices=['MI250X', 'MI300A', 'MI300X', 'MI355X'],
        help='GPU architecture (default: MI300X)'
    )
    parser.add_argument(
        '--keep-scripts',
        action='store_true',
        help='Keep generated kernel execution scripts'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume profiling (skip already-profiled kernels)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    kernels = get_kernels(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    scripts_dir = output_dir / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    
    # Locate rooflineExtractor scripts (in same directory as this script)
    roofline_base = Path(__file__).parent / "rooflineExtractor"
    roofline_profile_script = roofline_base / "profile.py"
    roofline_extractor_script = roofline_base / "rooflineExtractor.py"
    
    if not roofline_profile_script.exists():
        print(f"Error: rooflineExtractor's profile.py not found at {roofline_profile_script}")
        print("Please ensure rooflineExtractor-main is in the expected location")
        sys.exit(1)
    
    num_kernels = len(kernels)
    
    moe_utils.print_section_header("MOE ROOFLINE PROFILING")
    print(f"Using rooflineExtractor workflow")
    print(f"Kernels to profile: {num_kernels}")
    print(f"Input: {args.input}")
    print(f"Output: {output_dir.absolute()}")
    print(f"Architecture: {args.arch}")
    print(f"Scripts: {'KEEP' if args.keep_scripts else 'DELETE after use'}")
    print(f"{'='*80}\n")
    
    # Setup combined output files (updated incrementally)
    combined_counters_file = output_dir / "moe_target_counters.csv"
    combined_traces_file = output_dir / "moe_target_traces.csv"
    
    # Remove existing combined files if not in resume mode
    if not args.resume:
        combined_counters_file.unlink(missing_ok=True)
        combined_traces_file.unlink(missing_ok=True)
    
    successful_count = 0
    skipped_count = 0
    failed_count = 0
    
    for idx, (_, row) in enumerate(kernels.iterrows(), 1):
        error_value = str(row.get('error', '')).strip()
        if error_value == 'failed':
            print(f"[{idx}/{num_kernels}] Skipping failed kernel: {row['kernel_name'][:40]}")
            skipped_count += 1
            continue
        
        config_idx = int(row['config_idx'])
        kernel_type = str(row['kernel_type'])
        stage = str(row['stage'])
        block_m = int(row['block_m'])
        kernel_name = row['kernel_name']
        
        # Check if already profiled (resume mode)
        kernel_output_dir = output_dir / f"cfg{config_idx}_{kernel_type}_{stage}_block{block_m}"
        if args.resume and kernel_output_dir.exists():
            counters_file = kernel_output_dir / "counters.csv"
            if counters_file.exists():
                print(f"[{idx}/{num_kernels}] [SKIP] Already profiled: {kernel_name[:40]}")
                skipped_count += 1
                continue
        
        print(f"[{idx}/{num_kernels}] Config {config_idx} {kernel_type} {stage}: {kernel_name[:40]}")
        
        # Generate execution script
        script_name = f"cfg{config_idx}_{kernel_type}_{stage}_block{block_m}_{kernel_name}.py"
        script_path = scripts_dir / script_name
        create_kernel_execution_script(row, script_path)
        
        # Profile and incrementally append to combined files
        success = profile_kernel_with_roofline(
            row, script_path, output_dir, roofline_profile_script, args.arch,
            combined_counters_file, combined_traces_file
        )
        
        if success:
            successful_count += 1
        else:
            failed_count += 1
        
        # Clean up script
        if not args.keep_scripts and script_path.exists():
            script_path.unlink()
        
        print()
    
    # Generate final roofline plot
    if combined_counters_file.exists() and combined_traces_file.exists():
        moe_utils.print_section_header("GENERATING ROOFLINE PLOT")
        generate_combined_roofline(combined_counters_file, combined_traces_file, output_dir, roofline_extractor_script, args.arch)
    
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
