#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
MOE Kernel Benchmarking and Analysis Script

Purpose:
    Benchmarks ALL available MOE kernel implementations and select
    best kernels from the results. Produces three output CSVs in one execution.

Input:
    CSV file with MOE configurations (one config per row)
    Required columns: token, model_dim, inter_dim, expert, topk, act_type, dtype, 
                     q_dtype_a, q_dtype_w, q_type, use_g1u1, doweight_stage1

Outputs:
    1. {output}_benchmark_results.csv - All kernel benchmark results
    2. {output}_all_combinations.csv - All valid kernel combinations, sorted by time
    3. {output}_best_kernels.csv - Best kernels selected for profiling

Usage:
    python benchmark_and_analyze.py -i configs/example_config.csv -o results
"""

import sys
import os
from pathlib import Path

# Add paths for imports
current_dir = Path(__file__).parent
aiter_root = current_dir.parent.parent
sys.path.insert(0, str(aiter_root))
sys.path.insert(0, str(aiter_root / "csrc/ck_gemm_moe_2stages_codegen"))

import pandas as pd
import torch
from datetime import datetime
import argparse

# Import aiter and tune modules
import aiter
from aiter import QuantType, ActivationType
from aiter import dtypes
import gemm_moe_tune
from gemm_moe_tune import FmoeTuner
from gemm_moe_tune_triton import gen_triton_1stage_task

# Import shared utilities
import moe_utils


def run_kernel_benchmarks(config_file: str, 
                          output_file: str,
                          error_log_file: str,
                          resume: bool = False,
                          force: bool = False,
                          num_gpus: int = 0,
                          include_triton: bool = False) -> pd.DataFrame:
    """
    Run all kernel combinations for configurations in the input file.
    
    Args:
        config_file: Path to input CSV with configuration
        output_file: Path to output CSV with all results
        error_log_file: Path to error log file
        resume: If True, skip already-benchmarked configs
        force: If True, ignore existing results and re-run all
        
    Returns:
        DataFrame with all benchmark results
    """
    
    # Read configuration
    if not os.path.exists(config_file):
        print(f"Error: Config file {config_file} not found!")
        return None
        
    config_df = pd.read_csv(config_file)
    print(f"Loaded {len(config_df)} configurations from {config_file}")
    
    # Load existing results for resume mode
    existing_results = []
    existing_configs = set()
    
    if resume and os.path.exists(output_file) and not force:
        print(f"\n--resume mode: Loading existing results from {output_file}")
        existing_df = pd.read_csv(output_file)
        existing_configs = set(existing_df['config_idx'].unique())
        existing_results = existing_df.to_dict('records')
        print(f"  Found results for {len(existing_configs)} configs")
        print(f"  Will benchmark {len(config_df) - len(existing_configs)} remaining configs")
    elif force:
        print("\n--force mode: Re-running all configs")
    elif os.path.exists(output_file):
        print(f"\nWARNING: Output file {output_file} already exists!")
        print("  Use --resume to skip completed configs")
        print("  Use --force to overwrite")
        return None
    
    # Open error log file
    error_log = open(error_log_file, 'w')
    error_log.write(f"MOE Kernel Benchmark Error Log\n")
    error_log.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    error_log.write(f"Config file: {config_file}\n")
    error_log.write(f"="*80 + "\n\n")
    error_log.flush()
    
    # Initialize tuner
    key = [
        "cu_num", "token", "model_dim", "inter_dim", "expert", "topk",
        "act_type", "dtype", "q_dtype_a", "q_dtype_w", "q_type",
        "use_g1u1", "doweight_stage1",
    ]
    resultList = [
        "block_m", "ksplit", "us1", "kernelName1", "err1",
        "us2", "kernelName2", "err2", "us", "run_1stage",
        "tflops", "bw",
    ]
    
    tuner = FmoeTuner("fmoeTuner", key, resultList, "MOE kernel test")
    
    # ============================================================================
    # MAIN BENCHMARKING LOOP
    # ============================================================================
    all_results = []
    
    for idx, row in config_df.iterrows():
        # Use config_idx from input file if it exists, otherwise use row index
        config_idx = row.get('config_idx', idx)
        
        # Skip if already benchmarked (resume mode)
        if config_idx in existing_configs:
            print(f"\n[SKIP] Config {config_idx} already benchmarked (use --force to re-run)")
            continue
        
        moe_utils.print_section_header(
            f"Processing configuration {idx + 1}/{len(config_df)} (config_idx={config_idx})",
            char='='
        )
        moe_utils.print_config_summary(row)
        
        # Parse string representations to Python objects
        dtype = eval(row['dtype'])
        q_dtype_a = eval(row['q_dtype_a'])
        q_dtype_w = eval(row['q_dtype_w'])
        q_type = eval(row['q_type'])
        act_type = eval(row['act_type'])
        
        cu_num = tuner.get_cu_num()
        
        # Package configuration
        info = (
            cu_num, row['token'], row['model_dim'], row['inter_dim'],
            row['expert'], row['topk'], act_type, dtype,
            q_dtype_a, q_dtype_w, q_type, row['use_g1u1'], row['doweight_stage1'],
        )
        
        # Generate kernel tasks
        blockMs = [16, 32, 64, 128]
        print("\nGenerating kernel tasks...")
        
        tasks_asm = []
        tasks_ck = []
        tasks_1stage = []
        
        try:
            tasks_asm = tuner.gen_2stages_asm1_task(info, blockMs)
        except Exception as e:
            error_msg = str(e)
            print(f"  WARNING: ASM kernel generation failed: {error_msg[:100]}")
            error_log.write(f"[{datetime.now().strftime('%H:%M:%S')}] ASM Generation Failed\n")
            error_log.write(f"  Config#{idx}: token={row['token']}, model_dim={row['model_dim']}, expert={row['expert']}, topk={row['topk']}\n")
            error_log.write(f"  Error: {error_msg}\n\n")
            error_log.flush()
        
        try:
            tasks_ck = tuner.gen_2stages_task(info, blockMs)
        except Exception as e:
            error_msg = str(e)
            print(f"  WARNING: CK kernel generation failed: {error_msg[:100]}")
            error_log.write(f"[{datetime.now().strftime('%H:%M:%S')}] CK Generation Failed\n")
            error_log.write(f"  Config#{idx}: token={row['token']}, model_dim={row['model_dim']}, expert={row['expert']}, topk={row['topk']}\n")
            error_log.write(f"  Error: {error_msg}\n\n")
            error_log.flush()
        
        try:
            tasks_1stage = tuner.gen_1stage_asm_task(info)
        except Exception as e:
            error_msg = str(e)
            print(f"  WARNING: 1-stage kernel generation failed: {error_msg[:100]}")
            error_log.write(f"[{datetime.now().strftime('%H:%M:%S')}] 1-Stage Generation Failed\n")
            error_log.write(f"  Config#{idx}: token={row['token']}, model_dim={row['model_dim']}, expert={row['expert']}, topk={row['topk']}\n")
            error_log.write(f"  Error: {error_msg}\n\n")
            error_log.flush()
        
        tasks_triton = []
        if include_triton:
            try:
                tasks_triton = gen_triton_1stage_task(info)
            except Exception as e:
                error_msg = str(e)
                print(f"  WARNING: Triton kernel generation failed: {error_msg[:100]}")
                error_log.write(f"[{datetime.now().strftime('%H:%M:%S')}] Triton Generation Failed\n")
                error_log.write(f"  Config#{idx}: token={row['token']}, model_dim={row['model_dim']}, expert={row['expert']}, topk={row['topk']}\n")
                error_log.write(f"  Error: {error_msg}\n\n")
                error_log.flush()
        
        total_tasks = len(tasks_asm) + len(tasks_ck) + len(tasks_1stage) + len(tasks_triton)
        print(f"Total kernels to test: {total_tasks}")
        print(f"  - ASM stage1: {len(tasks_asm)}")
        print(f"  - CK 2-stage: {len(tasks_ck)}")
        print(f"  - ASM 1-stage: {len(tasks_1stage)}")
        print(f"  - Triton e2e: {len(tasks_triton)}")
        
        if total_tasks == 0:
            print("No kernels available for this configuration!")
            continue
            
        # Run all kernels
        print(f"\nBenchmarking all kernels using {num_gpus if num_gpus > 0 else 'all available'} GPU(s)...")
        from aiter.utility.mp_tuner import mp_tuner
        
        all_tasks = tasks_asm + tasks_ck + tasks_1stage + tasks_triton
        in_data = [(len(all_tasks), ())]
        
        try:
            results = mp_tuner(
                all_tasks, in_data, mp_num=num_gpus, fast_mode=True,
                shape_grouped=False, timeout=300, verbose=False
            )
        except Exception as e:
            error_msg = str(e)
            print(f"ERROR: Kernel execution crash for this configuration!")
            print(f"  Error: {error_msg[:100]}")
            print(f"  Skipping this configuration and continuing...")
            
            error_log.write(f"[{datetime.now().strftime('%H:%M:%S')}] EXECUTION CRASH\n")
            error_log.write(f"  Config#{idx}: token={row['token']}, model_dim={row['model_dim']}, inter_dim={row['inter_dim']}\n")
            error_log.write(f"  expert={row['expert']}, topk={row['topk']}, q_type={q_type}, act={act_type}\n")
            error_log.write(f"  Tasks: ASM={len(tasks_asm)}, CK={len(tasks_ck)}, 1-stage={len(tasks_1stage)}, Triton={len(tasks_triton)}\n")
            error_log.write(f"  Error: {error_msg}\n")
            error_log.write(f"-"*80 + "\n\n")
            error_log.flush()
            continue
        
        # Measure quantization overhead
        print("\nMeasuring inter-stage quantization overhead...")
        quant_time_us = 0.0
        
        if q_type != QuantType.No:
            (
                input, a1_qt, w1_qt, w2_qt, w1_qt_shffle, w2_qt_shffle,
                sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
                topk_ids, topk_weights, moe_buf, a1_scale, w1_scale, w2_scale,
            ) = tuner.generate_data(
                row['token'], row['model_dim'], row['inter_dim'],
                row['expert'], row['topk'], dtype, q_dtype_a, q_dtype_w,
                q_type, row['use_g1u1'], 64
            )
            
            out1_ref = tuner.run_torch_moe_stage1(
                a1_qt, w1_qt, w2_qt, topk_weights, topk_ids,
                a1_scale, w1_scale, dtype, act_type, q_type, 
                row['doweight_stage1'], row['topk']
            )
            
            from aiter.test_common import run_perftest
            
            if q_type == QuantType.per_1x128:
                def quant_func():
                    a2_qt, a2_scale = aiter.pertoken_quant(
                        out1_ref.view(row['token'], -1, 128), quant_dtype=q_dtype_a
                    )
                    return a2_qt, a2_scale
            else:
                torch_quant = aiter.get_torch_quant(q_type)
                def quant_func():
                    return torch_quant(out1_ref, quant_dtype=q_dtype_a)
            
            _, quant_time_us = run_perftest(quant_func, num_iters=10, num_warmup=2)
            print(f"  Quantization overhead: {quant_time_us:.2f} us")
        else:
            print(f"  Quantization overhead: 0.00 us (No quantization)")
        
        # Process results
        print(f"\nProcessing {len(results)} results...")
        
        for result in results:
            info_tuple, us, err = result
            
            if not isinstance(info_tuple, tuple) or len(info_tuple) < 4:
                print(f"Warning: Unexpected info structure: {info_tuple}")
                continue
                
            key_info, stage, kernel_name, block_m = info_tuple
            
            try:
                tflops, bw = tuner.calculate((key_info, stage, kernel_name, block_m, us, err))
            except Exception as calc_err:
                print(f"Warning: Failed to calculate metrics for {kernel_name}: {calc_err}")
                tflops, bw = 0, 0
            
            kernel_type = moe_utils.categorize_kernel_type(kernel_name, stage)
            kernel_params = moe_utils.parse_kernel_parameters(kernel_name, kernel_type, stage)
            
            result_dict = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'config_idx': config_idx,
                'token': row['token'],
                'model_dim': row['model_dim'],
                'inter_dim': row['inter_dim'],
                'expert': row['expert'],
                'topk': row['topk'],
                'act_type': str(act_type),
                'dtype': str(dtype),
                'q_dtype_a': str(q_dtype_a),
                'q_dtype_w': str(q_dtype_w),
                'q_type': str(q_type),
                'use_g1u1': row['use_g1u1'],
                'doweight_stage1': row['doweight_stage1'],
                'kernel_type': kernel_type,
                'stage': stage,
                'block_m': block_m,
                'kernel_name': kernel_name,
                'tile_m': kernel_params.get('tile_m'),
                'tile_n': kernel_params.get('tile_n'),
                'tile_k': kernel_params.get('tile_k'),
                'block_size': kernel_params.get('block_size'),
                'waves_m': kernel_params.get('waves_m'),
                'waves_n': kernel_params.get('waves_n'),
                'kernel_version': kernel_params.get('version'),
                'time_us': us,
                'quant_time_us': quant_time_us if stage in ['stage1', 'stage2'] else 0,
                'error': f"{err:.2%}" if err < 1.0 else "failed",
                'tflops': tflops if tflops > 0 else 0,
                'bandwidth_gb': bw if bw > 0 else 0
            }
            
            all_results.append(result_dict)
            
            status = "PASS" if err < 0.01 else "WARN" if err < 1.0 else "FAIL"
            print(f"[{status}] {kernel_type:6} {stage:12} block_m={block_m:3} "
                  f"time={us:8.2f}us err={err*100:5.1f}% {kernel_name[:50]}")
        
        # Incremental save - include both existing and new results
        if all_results:
            # Combine existing results with new results for incremental save
            combined_for_save = existing_results + all_results
            temp_df = pd.DataFrame(combined_for_save)
            temp_df.to_csv(output_file, index=False)
            print(f"  Saved {len(combined_for_save)} total results ({len(existing_results)} existing + {len(all_results)} new) to {output_file}")
    
    # Final processing
    error_log.write(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    error_log.close()
    
    # Merge with existing results if in resume mode
    if existing_results and all_results:
        print(f"\nMerging {len(existing_results)} existing + {len(all_results)} new results...")
        combined_results = existing_results + all_results
        results_df = pd.DataFrame(combined_results)
    elif existing_results:
        print(f"\nNo new results - using {len(existing_results)} existing results")
        results_df = pd.DataFrame(existing_results)
    elif all_results:
        results_df = pd.DataFrame(all_results)
    else:
        print("No results collected!")
        return None
    
    # Final save with sorting
    results_df = results_df.sort_values(['config_idx', 'token', 'time_us'])
    results_df.to_csv(output_file, index=False)
    
    moe_utils.print_section_header("BENCHMARKING COMPLETE")
    print(f"Saved {len(results_df)} total kernel results to {output_file}")
    print(f"  Existing: {len(existing_results)}")
    print(f"  New: {len(all_results)}")
    print(f"Error log saved to: {error_log_file}")
    
    # Print summary
    print("\nSummary by kernel type:")
    summary = results_df.groupby('kernel_type').agg({
        'time_us': ['count', 'min', 'mean', 'max'],
        'tflops': ['max'],
        'bandwidth_gb': ['max']
    }).round(2)
    print(summary)
    
    return results_df


def generate_all_combinations(valid_df: pd.DataFrame, output_file: str) -> pd.DataFrame:
    """Generate all valid kernel combinations."""
    unique_configs = valid_df['config_idx'].unique()
    moe_utils.print_section_header(f"Generating combinations for {len(unique_configs)} configurations")
    
    all_combinations = []
    
    for config_idx in unique_configs:
        config_data = valid_df[valid_df['config_idx'] == config_idx]
        config_info = config_data.iloc[0]
        
        stage1 = config_data[config_data['stage'] == 'stage1']
        stage2 = config_data[config_data['stage'] == 'stage2']
        onestage = config_data[config_data['stage'].isin(['asm_1stage', 'triton_1stage'])]
        
        quant_time = config_data['quant_time_us'].iloc[0]
        
        # 2-stage combinations
        for _, s1 in stage1.iterrows():
            s2_matches = stage2[stage2['block_m'] == s1['block_m']]
            
            for _, s2 in s2_matches.iterrows():
                total_time = s1['time_us'] + quant_time + s2['time_us']
                
                all_combinations.append({
                    'config_idx': config_idx,
                    'token': config_info['token'],
                    'model_dim': config_info['model_dim'],
                    'inter_dim': config_info['inter_dim'],
                    'expert': config_info['expert'],
                    'topk': config_info['topk'],
                    'act_type': config_info['act_type'],
                    'q_type': config_info['q_type'],
                    'dtype': config_info['dtype'],
                    'q_dtype_a': config_info['q_dtype_a'],
                    'q_dtype_w': config_info['q_dtype_w'],
                    'use_g1u1': config_info['use_g1u1'],
                    'doweight_stage1': config_info['doweight_stage1'],
                    'approach': '2-stage',
                    'block_m': s1['block_m'],
                    'stage1_kernel_name': s1['kernel_name'],
                    'stage1_kernel_type': s1['kernel_type'],
                    'stage1_time_us': s1['time_us'],
                    'stage1_error': s1['error'],
                    'stage1_tflops': s1['tflops'],
                    'quant_time_us': quant_time,
                    'stage2_kernel_name': s2['kernel_name'],
                    'stage2_kernel_type': s2['kernel_type'],
                    'stage2_time_us': s2['time_us'],
                    'stage2_error': s2['error'],
                    'stage2_tflops': s2['tflops'],
                    'total_time_us': total_time,
                })
        
        # 1-stage kernels
        for _, ks in onestage.iterrows():
            all_combinations.append({
                'config_idx': config_idx,
                'token': config_info['token'],
                'model_dim': config_info['model_dim'],
                'inter_dim': config_info['inter_dim'],
                'expert': config_info['expert'],
                'topk': config_info['topk'],
                'act_type': config_info['act_type'],
                'q_type': config_info['q_type'],
                'dtype': config_info['dtype'],
                'q_dtype_a': config_info['q_dtype_a'],
                'q_dtype_w': config_info['q_dtype_w'],
                'use_g1u1': config_info['use_g1u1'],
                'doweight_stage1': config_info['doweight_stage1'],
                'approach': '1-stage',
                'block_m': ks['block_m'],
                'stage1_kernel_name': ks['kernel_name'],
                'stage1_kernel_type': ks['kernel_type'],
                'stage1_time_us': ks['time_us'],
                'stage1_error': ks['error'],
                'stage1_tflops': ks['tflops'],
                'quant_time_us': 0,
                'stage2_kernel_name': None,
                'stage2_kernel_type': None,
                'stage2_time_us': 0,
                'stage2_error': None,
                'stage2_tflops': 0,
                'total_time_us': ks['time_us'],
            })
    
    result_df = pd.DataFrame(all_combinations)
    result_df = result_df.sort_values(['config_idx', 'total_time_us'])
    result_df.to_csv(output_file, index=False)
    
    print(f"Generated {len(result_df)} kernel combinations:")
    print(f"  2-stage combinations: {len(result_df[result_df['approach'] == '2-stage'])}")
    print(f"  1-stage kernels: {len(result_df[result_df['approach'] == '1-stage'])}")
    print(f"Saved to: {output_file}")
    
    return result_df


def select_best_kernels(valid_df: pd.DataFrame, output_file: str) -> pd.DataFrame:
    """Select the best kernels for each configuration, including both ASM and CK variants."""
    unique_configs = valid_df['config_idx'].unique()
    moe_utils.print_section_header(f"Selecting best kernels for {len(unique_configs)} configurations")
    
    profiling_rows = []
    
    for config_idx in unique_configs:
        config_data = valid_df[valid_df['config_idx'] == config_idx]
        config_info = config_data.iloc[0]
        
        # Get list of selected kernels (includes best + opposite type for each stage)
        selected_kernels = moe_utils.select_best_kernels_for_config(config_data)
        
        print(f"\nConfig {config_idx}: Selected {len(selected_kernels)} kernels")
        moe_utils.print_config_summary(config_info)
        
        # Get quantization time
        quant_time = config_data['quant_time_us'].iloc[0] if len(config_data) > 0 else 0
        
        # Add all selected kernels
        for kernel in selected_kernels:
            profiling_rows.append({
                'config_idx': config_idx,
                'token': config_info['token'],
                'model_dim': config_info['model_dim'],
                'inter_dim': config_info['inter_dim'],
                'expert': config_info['expert'],
                'topk': config_info['topk'],
                'act_type': config_info['act_type'],
                'dtype': config_info['dtype'],
                'q_dtype_a': config_info['q_dtype_a'],
                'q_dtype_w': config_info['q_dtype_w'],
                'q_type': config_info['q_type'],
                'use_g1u1': config_info['use_g1u1'],
                'doweight_stage1': config_info['doweight_stage1'],
                'kernel_type': kernel['kernel_type'],
                'stage': kernel['stage'],
                'block_m': kernel['block_m'],
                'kernel_name': kernel['kernel_name'],
                'time_us': kernel['time_us'],
                'quant_time_us': quant_time if kernel['stage'] in ['stage1', 'stage2'] else 0,
                'error': kernel['error'],
                'tflops': kernel['tflops'],
                'bandwidth_gb': kernel['bandwidth_gb'],
            })
            
            print(f"  {kernel['stage']:12} {kernel['kernel_type']:4} {kernel['kernel_name'][:45]:45} {kernel['time_us']:8.2f} μs")
    
    profiling_df = pd.DataFrame(profiling_rows)
    profiling_df.to_csv(output_file, index=False)
    
    print(f"\n{'-'*80}")
    print(f"Total kernels selected: {len(profiling_rows)}")
    print(f"  Stage1: {len(profiling_df[profiling_df['stage'] == 'stage1'])}")
    print(f"  Stage2: {len(profiling_df[profiling_df['stage'] == 'stage2'])}")
    print(f"  1-stage: {len(profiling_df[profiling_df['stage'].isin(['asm_1stage', 'triton_1stage'])])}")
    print(f"  ASM kernels: {len(profiling_df[profiling_df['kernel_type'] == 'asm'])}")
    print(f"  CK kernels: {len(profiling_df[profiling_df['kernel_type'] == 'ck'])}")
    print(f"  Triton kernels: {len(profiling_df[profiling_df['kernel_type'] == 'triton'])}")
    print(f"Saved to: {output_file}")
    
    return profiling_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="""
MOE Kernel Benchmarking Tool

Benchmarks ALL available MOE kernel implementations (ASM, CK, Triton) for given configurations
and produces three analysis outputs: benchmark results, all combinations, and best kernels.

Examples:
  # Benchmark from CSV file (ASM/CK only)
  python benchmark.py -i configs/sample.csv -o results/output
  
  # Include Triton kernels
  python benchmark.py -i configs/sample.csv -o results/output --include-triton
  
  # Quick test with single config
  python benchmark.py --config "1,4096,1536,16,8,ActivationType.Silu,torch.bfloat16,torch.bfloat16,torch.bfloat16,QuantType.No,1,0" -o results/test --include-triton
  
  # Resume interrupted run
  python benchmark.py -i configs/sample.csv -o results/output --resume
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-i", "--input",
        default="configs/configs_for_benchmark.csv",
        help="Input configuration CSV file (or use --config for single config)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Single config as comma-separated values: token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1 (e.g., '1,4096,1536,16,8,ActivationType.Silu,torch.bfloat16,torch.bfloat16,torch.bfloat16,QuantType.No,1,0')"
    )
    parser.add_argument(
        "-o", "--output",
        default="results/benchmark",
        help="Output base name (default: results/benchmark, generates 3 files: _benchmark_results, _all_combinations, _best_kernels)"
    )
    parser.add_argument(
        "--error-threshold",
        type=float,
        default=50.0,
        help="Maximum error percentage to consider valid (default: 50.0)"
    )
    parser.add_argument(
        "--resume",
        action='store_true',
        help="Resume from existing results (skip already-benchmarked configs)"
    )
    parser.add_argument(
        "--force",
        action='store_true',
        help="Force re-run all configs (ignore existing results)"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=0,
        help="Number of GPUs to use (default: 0 = use all available GPUs)"
    )
    parser.add_argument(
        "--include-triton",
        action='store_true',
        help="Include Triton e2e kernels in benchmarking (default: False)"
    )
    
    args = parser.parse_args()
    
    # Handle --config argument (single config as comma-separated string)
    if args.config:
        # Parse comma-separated config
        values = args.config.split(',')
        if len(values) != 12:
            print(f"Error: --config requires 12 comma-separated values, got {len(values)}")
            print("Expected: token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1")
            return
        config_data = {
            'config_idx': [0],
            'token': [int(values[0])],
            'model_dim': [int(values[1])],
            'inter_dim': [int(values[2])],
            'expert': [int(values[3])],
            'topk': [int(values[4])],
            'act_type': [values[5].strip()],
            'dtype': [values[6].strip()],
            'q_dtype_a': [values[7].strip()],
            'q_dtype_w': [values[8].strip()],
            'q_type': [values[9].strip()],
            'use_g1u1': [int(values[10])],
            'doweight_stage1': [int(values[11])]
        }
        import tempfile
        temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_df = pd.DataFrame(config_data)
        temp_df.to_csv(temp_csv.name, index=False)
        temp_csv.close()
        input_file = temp_csv.name
        print("Using config from command line argument")
    else:
        input_file = args.input
    
    # Generate output filenames
    benchmark_file = f"{args.output}_benchmark_results.csv"
    combinations_file = f"{args.output}_all_combinations.csv"
    best_kernels_file = f"{args.output}_best_kernels.csv"
    error_log_file = f"{args.output}_errors.log"
    
    moe_utils.print_section_header("MOE KERNEL BENCHMARKING")
    print(f"Input config: {input_file}")
    print(f"Output files:")
    print(f"  1. {benchmark_file} (all benchmark results)")
    print(f"  2. {combinations_file} (all combinations)")
    print(f"  3. {best_kernels_file} (best kernels for profiling)")
    print(f"  4. {error_log_file} (error log)")
    print(f"Error threshold: {args.error_threshold}%")
    
    # Show GPU configuration
    import torch
    available_gpus = torch.cuda.device_count()
    gpus_to_use = args.gpus if args.gpus > 0 else available_gpus
    print(f"GPUs: {gpus_to_use} (available: {available_gpus})")
    
    # STEP 1: Run benchmarks
    results_df = run_kernel_benchmarks(input_file, benchmark_file, error_log_file, 
                                      resume=args.resume, force=args.force,
                                      num_gpus=args.gpus, include_triton=args.include_triton)
    
    if results_df is None:
        print("\nNo results generated. Check your configuration.")
        return
    
    # STEP 2: Compare results
    moe_utils.print_section_header("COMPARING BENCHMARK RESULTS")
    
    # Filter valid results
    valid_df = moe_utils.filter_valid_results(results_df, args.error_threshold)
    print(f"Valid results (error < {args.error_threshold}%): {len(valid_df)}")
    
    if len(valid_df) == 0:
        print(f"\nERROR: No valid kernels found (all have >{args.error_threshold}% error or failed)")
        return
    
    # Generate all combinations
    combinations_df = generate_all_combinations(valid_df, combinations_file)
    
    # Select best kernels
    best_kernels_df = select_best_kernels(valid_df, best_kernels_file)
    
    # Clean up temp file if using --config
    if args.config:
        try:
            os.unlink(input_file)
        except:
            pass
    
    # Final summary
    moe_utils.print_section_header("PIPELINE COMPLETE")
    print("Output files generated:")
    print(f"  1. {benchmark_file}")
    print(f"     - {len(results_df)} total kernel results")
    print(f"  2. {combinations_file}")
    print(f"     - {len(combinations_df)} kernel combinations")
    print(f"  3. {best_kernels_file}")
    print(f"     - {len(best_kernels_df)} best kernels for profiling")
    
    print(f"\nNext step: Run profiling script")
    print(f"  python profile_kernels.py -i {best_kernels_file}")


if __name__ == "__main__":
    main()
