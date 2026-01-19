# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
MOE Kernel Benchmarking Tool

Purpose:
    Comprehensively benchmark all available MOE (Mixture of Experts) kernel implementations
    for given configurations. Tests ASM kernels, CK 2-stage kernels, and 1-stage kernels,
    then identifies the fastest option for each configuration.

Input:
    CSV file with MOE configurations (one config per row)
    Required columns: token, model_dim, inter_dim, expert, topk, act_type, dtype, 
                     q_dtype_a, q_dtype_w, q_type, use_g1u1, doweight_stage1
    
    Each row is analyzed independently - configurations can differ in any parameter
    (token count, dimensions, quantization, activation, etc.)

Output:
    CSV file with all kernel benchmark results including:
    - Individual kernel timings for each configuration
    - Error rates (verification accuracy)
    - TFLOPS and bandwidth metrics
    - Best kernel recommendation per row/configuration

Usage:
    python benchmark_moe_kernels.py -i configs/trace_moe_config.csv -o results.csv

Key Features:
    - Uses tune.py's kernel generation functions
    - Matches tune.py's 50% error threshold (ERR_RATIO=0.5)
    - Each CSV row analyzed independently
    - Includes inter-stage quantization overhead for 2-stage kernels
    - Compares 1-stage vs 2-stage approaches per configuration
"""

import sys
import os
from pathlib import Path

# Add paths for imports
current_dir = Path(__file__).parent
aiter_root = current_dir.parent
sys.path.insert(0, str(aiter_root))
sys.path.insert(0, str(aiter_root / "csrc/ck_gemm_moe_2stages_codegen"))

import pandas as pd
import torch
from datetime import datetime
import re

# Now import aiter and tune modules
import aiter
from aiter import QuantType, ActivationType
from aiter import dtypes
import gemm_moe_tune
from gemm_moe_tune import FmoeTuner


def parse_kernel_parameters(kernel_name, kernel_type, stage):
    """
    Parse kernel parameters from kernel name.
    
    Returns dict with: tile_m, tile_n, tile_k, block_size, waves_m, waves_n, version
    """
    params = {
        'tile_m': None,
        'tile_n': None, 
        'tile_k': None,
        'block_size': None,
        'waves_m': None,
        'waves_n': None,
        'version': None,
    }
    
    # Parse CK kernel names
    if 'moe_ck2stages' in kernel_name:
        # Example: moe_ck2stages_gemm1_256x32x64x128_1x4_TypeCast_v3_Nswizzle0_Quant2_MulRoutedWeight1_silu_B16_B16_B16
        parts = kernel_name.split('_')
        
        # Extract tile sizes: 256x32x64x128 = block_size x M x N x K
        for part in parts:
            if 'x' in part and part[0].isdigit():
                tiles = part.split('x')
                if len(tiles) == 4:
                    params['block_size'] = int(tiles[0])
                    params['tile_m'] = int(tiles[1])
                    params['tile_n'] = int(tiles[2])
                    params['tile_k'] = int(tiles[3])
                elif len(tiles) == 2:  # Waves: 1x4
                    params['waves_m'] = int(tiles[0])
                    params['waves_n'] = int(tiles[1])
        
        # Extract version: v1, v2, v3
        for part in parts:
            if part.startswith('v') and part[1:].isdigit():
                params['version'] = int(part[1:])
    
    # Parse ASM kernel names - extract tile sizes from name
    elif '_ZN' in kernel_name or 'fmoe_' in kernel_name:
        # ASM 2-stage: fmoe_stage1_bf16_pertokenFp8_doweight_g1u1_64x128_2tg_pf3
        # ASM 1-stage: fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_32x256
        
        # Look for pattern like 32x256 or 64x128
        tile_match = re.search(r'_(\d+)x(\d+)', kernel_name)
        if tile_match:
            params['tile_m'] = int(tile_match.group(1))
            params['tile_n'] = int(tile_match.group(2))
        
        # Extract version from pf2, pf3
        if 'pf2' in kernel_name:
            params['version'] = 2
        elif 'pf3' in kernel_name:
            params['version'] = 3
    
    return params


def run_all_kernel_tests(config_file="trace_moe_config.csv", 
                         output_file="all_kernel_results.csv",
                         error_log_file="benchmark_errors.log"):
    """
    Run all kernel combinations for configurations in the input file.
    
    Args:
        config_file: Path to input CSV with configuration
        output_file: Path to output CSV with all results
        error_log_file: Path to error log file
    """
    
    # Read configuration
    if not os.path.exists(config_file):
        print(f"Error: Config file {config_file} not found!")
        return
        
    config_df = pd.read_csv(config_file)
    print(f"Loaded {len(config_df)} configurations from {config_file}")
    
    # Open error log file
    error_log = open(error_log_file, 'w')
    error_log.write(f"MOE Kernel Benchmark Error Log\n")
    error_log.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    error_log.write(f"Config file: {config_file}\n")
    error_log.write(f"="*80 + "\n\n")
    error_log.flush()
    
    # Initialize tuner
    key = [
        "cu_num",
        "token",
        "model_dim",
        "inter_dim",
        "expert",
        "topk",
        "act_type",
        "dtype",
        "q_dtype_a",
        "q_dtype_w",
        "q_type",
        "use_g1u1",
        "doweight_stage1",
    ]
    resultList = [
        "block_m",
        "ksplit",
        "us1",
        "kernelName1",
        "err1",
        "us2",
        "kernelName2",
        "err2",
        "us",
        "run_1stage",
        "tflops",
        "bw",
    ]
    
    tuner = FmoeTuner("fmoeTuner", key, resultList, "MOE kernel test")
    
    # ============================================================================
    # MAIN BENCHMARKING LOOP
    # Each row in the CSV is processed independently
    # ============================================================================
    all_results = []
    
    for idx, row in config_df.iterrows():
        print(f"\n{'='*80}")
        print(f"Processing configuration {idx + 1}/{len(config_df)}:")
        print(f"Token={row['token']}, Model={row['model_dim']}, Inter={row['inter_dim']}, "
              f"Expert={row['expert']}, TopK={row['topk']}")
        
        # Parse string representations to Python objects
        # CSV stores enums as strings like "ActivationType.Silu"
        dtype = eval(row['dtype'])          # torch.bfloat16, torch.float16, etc.
        q_dtype_a = eval(row['q_dtype_a'])  # Activation quant dtype
        q_dtype_w = eval(row['q_dtype_w'])  # Weight quant dtype
        q_type = eval(row['q_type'])        # QuantType.No, per_Token, per_1x128, etc.
        act_type = eval(row['act_type'])    # ActivationType.Silu, Gelu, etc.
        
        cu_num = tuner.get_cu_num()  # Get GPU compute unit count per config
        
        # Package configuration into tuple format expected by tune.py functions
        info = (
            cu_num,                    # GPU CU count
            row['token'],              # Token count
            row['model_dim'],          # Model dimension (hidden size)
            row['inter_dim'],          # Intermediate dimension (FFN size)
            row['expert'],             # Number of experts
            row['topk'],               # Top-K experts to route to
            act_type,                  # Activation function
            dtype,                     # Output dtype
            q_dtype_a,                 # Activation quantization dtype
            q_dtype_w,                 # Weight quantization dtype
            q_type,                    # Quantization type
            row['use_g1u1'],           # Gate+Up in one weight (g1u1)
            row['doweight_stage1'],    # Apply routing weights in stage1
        )
        
        # ========================================================================
        # KERNEL GENERATION
        # Generate all possible kernel tasks using tune.py's functions
        # ========================================================================
        
        # Block sizes to test (tile sizes for GEMM operations)
        # These values match tune.py and cover common padding scenarios  
        blockMs = [16, 32, 64, 128]
        
        print("\nGenerating kernel tasks...")
        
        # Try to generate tasks - may fail if kernels don't exist or can't build
        tasks_asm = []
        tasks_ck = []
        tasks_1stage = []
        
        try:
            # 1. ASM kernels for 2-stage (assembly-based implementations)
            tasks_asm = tuner.gen_2stages_asm1_task(info, blockMs)
        except Exception as e:
            error_msg = str(e)
            print(f"  WARNING: ASM kernel generation failed: {error_msg[:100]}")
            error_log.write(f"[{datetime.now().strftime('%H:%M:%S')}] ASM Generation Failed\n")
            error_log.write(f"  Config#{idx}: token={row['token']}, model_dim={row['model_dim']}, expert={row['expert']}, topk={row['topk']}\n")
            error_log.write(f"  Error: {error_msg}\n\n")
            error_log.flush()
        
        try:
            # 2. CK kernels for 2-stage (Composable Kernel library)
            tasks_ck = tuner.gen_2stages_task(info, blockMs)
        except Exception as e:
            error_msg = str(e)
            print(f"  WARNING: CK kernel generation failed: {error_msg[:100]}")
            error_log.write(f"[{datetime.now().strftime('%H:%M:%S')}] CK Generation Failed\n")
            error_log.write(f"  Config#{idx}: token={row['token']}, model_dim={row['model_dim']}, expert={row['expert']}, topk={row['topk']}\n")
            error_log.write(f"  Error: {error_msg}\n\n")
            error_log.flush()
        
        try:
            # 3. 1-stage kernels (fused approach)
            tasks_1stage = tuner.gen_1stage_asm_task(info)
        except Exception as e:
            error_msg = str(e)
            print(f"  WARNING: 1-stage kernel generation failed: {error_msg[:100]}")
            error_log.write(f"[{datetime.now().strftime('%H:%M:%S')}] 1-Stage Generation Failed\n")
            error_log.write(f"  Config#{idx}: token={row['token']}, model_dim={row['model_dim']}, expert={row['expert']}, topk={row['topk']}\n")
            error_log.write(f"  Error: {error_msg}\n\n")
            error_log.flush()
        
        total_tasks = len(tasks_asm) + len(tasks_ck) + len(tasks_1stage)
        print(f"Total kernels to test: {total_tasks}")
        print(f"  - ASM stage1: {len(tasks_asm)} (assembly)")
        print(f"  - CK 2-stage: {len(tasks_ck)} (Composable Kernel)")
        print(f"  - 1-stage: {len(tasks_1stage)} (fused)")
        
        if total_tasks == 0:
            print("No kernels available for this configuration!")
            continue
            
        # Run all kernels using mp_tuner
        print("\nBenchmarking all kernels...")
        from aiter.utility.mp_tuner import mp_tuner
        
        all_tasks = tasks_asm + tasks_ck + tasks_1stage
        in_data = [(len(all_tasks), ())]
        
        # Execute all kernel tasks using mp_tuner utility from tune.py
        # FIXED: Added missing parameters to match tuner's call signature
        try:
            results = mp_tuner(
                all_tasks, 
                in_data, 
                mp_num=1, 
                fast_mode=True,  # FIX: Must specify fast_mode explicitly
                shape_grouped=False,
                timeout=300,  # FIX: Added timeout parameter
                verbose=False  # FIX: Added verbose parameter
            )
        except Exception as e:
            error_msg = str(e)
            print(f"ERROR: Kernel execution crash for this configuration!")
            print(f"  Error: {error_msg[:100]}")
            print(f"  Skipping this configuration and continuing...")
            
            # Log execution crash with details
            error_log.write(f"[{datetime.now().strftime('%H:%M:%S')}] EXECUTION CRASH\n")
            error_log.write(f"  Config#{idx}: token={row['token']}, model_dim={row['model_dim']}, inter_dim={row['inter_dim']}\n")
            error_log.write(f"  expert={row['expert']}, topk={row['topk']}, q_type={q_type}, act={act_type}\n")
            error_log.write(f"  Tasks: ASM={len(tasks_asm)}, CK={len(tasks_ck)}, 1-stage={len(tasks_1stage)}\n")
            error_log.write(f"  Error: {error_msg}\n")
            error_log.write(f"-"*80 + "\n\n")
            error_log.flush()
            continue
        
        # ========================================================================
        # QUANTIZATION OVERHEAD MEASUREMENT
        # For 2-stage approaches, measure the quantization cost between stages
        # ========================================================================
        print("\nBenchmarking inter-stage quantization...")
        quant_time_us = 0.0
        
        if q_type != QuantType.No:
            # Generate test data for quantization
            (
                input, a1_qt, w1_qt, w2_qt, w1_qt_shffle, w2_qt_shffle,
                sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
                topk_ids, topk_weights, moe_buf, a1_scale, w1_scale, w2_scale,
            ) = tuner.generate_data(
                row['token'], row['model_dim'], row['inter_dim'],
                row['expert'], row['topk'], dtype, q_dtype_a, q_dtype_w,
                q_type, row['use_g1u1'], 64
            )
            
            # Stage1 output for quantization test
            out1_ref = tuner.run_torch_moe_stage1(
                a1_qt, w1_qt, w2_qt, topk_weights, topk_ids,
                a1_scale, w1_scale, dtype, act_type, q_type, 
                row['doweight_stage1'], row['topk']
            )
            
            # Benchmark quantization (stage1 output -> stage2 input)
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
            print(f"  Quantization time: {quant_time_us:.2f} us")
        else:
            print(f"  Quantization time: 0.00 us (No quantization)")
        
        # ========================================================================
        # RESULTS PROCESSING
        # Convert raw benchmark results into structured data
        # ========================================================================
        print(f"\nProcessing {len(results)} results...")
        
        # FIXED: Proper two-step unpacking to handle mp_tuner result format
        for result in results:
            # Unpack the result tuple: (info, us, err)
            info_tuple, us, err = result
            
            # Unpack the info tuple: (key_info, stage, kernel_name, block_m)
            if not isinstance(info_tuple, tuple) or len(info_tuple) < 4:
                print(f"Warning: Unexpected info structure: {info_tuple}")
                continue
                
            key_info, stage, kernel_name, block_m = info_tuple
            
            # Calculate performance metrics from timing
            # calculate() expects: (key, stage, kernelName, block_m, us, err)
            try:
                tflops, bw = tuner.calculate((key_info, stage, kernel_name, block_m, us, err))
            except Exception as calc_err:
                print(f"Warning: Failed to calculate metrics for {kernel_name}: {calc_err}")
                tflops, bw = 0, 0
            
            # Categorize kernel implementation type from kernel name and stage
            # ASM kernels have mangled names starting with _ZN or contain aiter::
            # CK kernels start with moe_ck2stages
            if stage == "asm_1stage" or "_ZN" in kernel_name or "aiter::" in kernel_name or "fmoe_" in kernel_name:
                kernel_type = "asm"
            elif "moe_ck2stages" in kernel_name or "ck_moe" in kernel_name:
                kernel_type = "ck"
            elif stage in ["stage1", "stage2"]:
                # Fallback: if stage1/stage2 but not CK name, likely ASM
                kernel_type = "asm" if "_ZN" in kernel_name else "ck"
            else:
                kernel_type = "1stage"
            
            # Parse kernel parameters from kernel name
            kernel_params = parse_kernel_parameters(kernel_name, kernel_type, stage)
            
            # Build result record containing all information
            result_dict = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'config_idx': idx,
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
                # Kernel parameters parsed from name
                'tile_m': kernel_params.get('tile_m'),
                'tile_n': kernel_params.get('tile_n'),
                'tile_k': kernel_params.get('tile_k'),
                'block_size': kernel_params.get('block_size'),
                'waves_m': kernel_params.get('waves_m'),
                'waves_n': kernel_params.get('waves_n'),
                'kernel_version': kernel_params.get('version'),
                # Performance metrics
                'time_us': us,
                'quant_time_us': quant_time_us if stage in ['stage1', 'stage2'] else 0,
                'error': f"{err:.2%}" if err < 1.0 else "failed",
                'tflops': tflops if tflops > 0 else 0,
                'bandwidth_gb': bw if bw > 0 else 0
            }
            
            all_results.append(result_dict)
            
            # Print result with simplified status
            status = "PASS" if err < 0.01 else "WARN" if err < 1.0 else "FAIL"
            print(f"[{status}] {kernel_type:6} {stage:12} block_m={block_m:3} "
                  f"time={us:8.2f}us err={err*100:5.1f}% {kernel_name[:50]}")
        
        # Incremental save after each configuration
        # This ensures results are preserved even if benchmarking is interrupted
        if all_results:
            temp_df = pd.DataFrame(all_results)
            temp_df.to_csv(output_file, index=False)
            print(f"  Saved {len(all_results)} results so far to {output_file}")
    
    # Final save (already done incrementally, but ensures latest state)
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Sort by configuration and performance
        results_df = results_df.sort_values(['config_idx', 'token', 'time_us'])
        
        # Save to file
        results_df.to_csv(output_file, index=False)
        print(f"\n{'='*80}")
        print(f"Saved {len(results_df)} kernel results to {output_file}")
        
        # Print summary statistics per token value
        print("\nSummary by token and kernel type:")
        for token_val in sorted(results_df['token'].unique()):
            token_data = results_df[results_df['token'] == token_val]
            print(f"\nToken={token_val}:")
            summary = token_data.groupby('kernel_type').agg({
                'time_us': ['count', 'min', 'mean', 'max'],
                'tflops': ['max'],
                'bandwidth_gb': ['max']
            }).round(2)
            print(summary)
        
        print(f"\n{'='*80}")
        print("BEST KERNEL SELECTION PER TOKEN VALUE")
        print(f"{'='*80}\n")
        
        # Filter to successful results (error < 50%, matching tune.py's errRatio=0.5)
        valid_results = results_df[results_df['error'] != 'failed'].copy()
        
        # Extract error percentage as float
        # tune.py uses errRatio=0.5 (decimal), which equals 50% (percentage)
        # Relationship: err_ratio = 0.5 ←→ err_perc = 50.0
        valid_results['error_pct'] = valid_results['error'].str.rstrip('%').astype(float)
        ERR_RATIO = 0.5  # Same as tune.py default
        valid_results = valid_results[valid_results['error_pct'] < (ERR_RATIO * 100)]
        
        if len(valid_results) == 0:
            print(f"WARNING: No valid kernels found (all have >{ERR_RATIO*100}% error or failed)")
            return results_df
        
        # ========================================================================
        # ANALYSIS: Best Kernel Selection Per Configuration
        # Each configuration (row) analyzed independently
        # ========================================================================
        
        # Process each unique configuration separately
        for token_val in sorted(valid_results['token'].unique()):
            print(f"\n{'='*80}")
            print(f"Token = {token_val}")
            print(f"{'='*80}\n")
            
            # Filter results for this specific token count
            token_results = valid_results[valid_results['token'] == token_val]
            stage1_results = token_results[token_results['stage'] == 'stage1']
            stage2_results = token_results[token_results['stage'] == 'stage2']
            onestage_results = token_results[token_results['stage'] == 'asm_1stage']
            
            # Get quantization overhead for this configuration
            quant_time = token_results['quant_time_us'].iloc[0] if len(token_results) > 0 else 0
            
            # --- 2-Stage Kernel Analysis ---
            best_2stage_combo = None
            if len(stage1_results) > 0 and len(stage2_results) > 0:
                print("2-Stage Combinations (by block_m with quantization overhead):")
                print(f"{'-'*80}")
                
                block_m_combos = []
                # For 2-stage kernels, stage1 and stage2 must use same block_m
                # This is a hardware/algorithmic requirement
                for block_m in sorted(token_results['block_m'].unique()):
                    s1_for_blockm = stage1_results[stage1_results['block_m'] == block_m]
                    s2_for_blockm = stage2_results[stage2_results['block_m'] == block_m]
                
                    if len(s1_for_blockm) > 0 and len(s2_for_blockm) > 0:
                        # Find fastest stage1 and stage2 kernels for this block_m
                        best_s1 = s1_for_blockm.loc[s1_for_blockm['time_us'].idxmin()]
                        best_s2 = s2_for_blockm.loc[s2_for_blockm['time_us'].idxmin()]
                        
                        # Calculate total 2-stage time: stage1 + quantization + stage2
                        combo_time = best_s1['time_us'] + quant_time + best_s2['time_us']
                        combo_tflops = best_s1['tflops'] + best_s2['tflops']
                        
                        block_m_combos.append({
                            'block_m': block_m,
                            'total_time_us': combo_time,
                            'stage1_time_us': best_s1['time_us'],
                            'quant_time_us': quant_time,
                            'stage2_time_us': best_s2['time_us'],
                            'stage1_kernel': best_s1['kernel_name'],
                            'stage2_kernel': best_s2['kernel_name'],
                            'stage1_error': best_s1['error'],
                            'stage2_error': best_s2['error'],
                            'combined_tflops': combo_tflops,
                        })
                        
                        print(f"  block_m={block_m}: {combo_time:.2f} us")
                        print(f"    Stage1: {best_s1['time_us']:.2f} us (err={best_s1['error']})")
                        print(f"    Quant:  {quant_time:.2f} us")
                        print(f"    Stage2: {best_s2['time_us']:.2f} us (err={best_s2['error']})")
                
                # Select the block_m with minimum total time
                if block_m_combos:
                    best_2stage_combo = min(block_m_combos, key=lambda x: x['total_time_us'])
                    
                    print(f"\n{'-'*80}")
                    print("Best 2-Stage Combination:")
                    print(f"  block_m: {best_2stage_combo['block_m']}")
                    print(f"  Stage1: {best_2stage_combo['stage1_kernel'][:60]}")
                    print(f"    Time: {best_2stage_combo['stage1_time_us']:.2f} us (err={best_2stage_combo['stage1_error']})")
                    print(f"  Quantization: {best_2stage_combo['quant_time_us']:.2f} us")
                    print(f"  Stage2: {best_2stage_combo['stage2_kernel'][:60]}")
                    print(f"    Time: {best_2stage_combo['stage2_time_us']:.2f} us (err={best_2stage_combo['stage2_error']})")
                    print(f"  Combined Total: {best_2stage_combo['total_time_us']:.2f} us")
            
            # --- 1-Stage Kernel Analysis ---
            best_1stage = None
            if len(onestage_results) > 0:
                # Find fastest 1-stage kernel (fused approach)
                best_1stage = onestage_results.loc[onestage_results['time_us'].idxmin()]
                print(f"\n{'-'*80}")
                print("Best 1-Stage Kernel:")
                print(f"  Kernel: {best_1stage['kernel_name'][:60]}")
                print(f"  block_m: {best_1stage['block_m']}")
                print(f"  Time: {best_1stage['time_us']:.2f} us (err={best_1stage['error']})")
                print(f"  TFLOPS: {best_1stage['tflops']:.2f}")
            
            # --- Final Recommendation ---
            # Compare 2-stage vs 1-stage and recommend fastest
            print(f"\n{'-'*80}")
            print(f"RECOMMENDATION for Token={token_val}:")
            
            candidates = []
            if best_2stage_combo:
                candidates.append(('2-Stage', best_2stage_combo['total_time_us']))
            if best_1stage is not None:
                candidates.append(('1-Stage', best_1stage['time_us']))
            
            if candidates:
                fastest = min(candidates, key=lambda x: x[1])
                print(f"  BEST: {fastest[0]}: {fastest[1]:.2f} us")
                
                if len(candidates) > 1:
                    slower = max(candidates, key=lambda x: x[1])
                    speedup = slower[1] / fastest[1]
                    print(f"  Speedup vs {slower[0]}: {speedup:.2f}x faster")
    
    # Close error log file
    error_log.write(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    error_log.close()
    
    if not all_results:
        print("No results collected!")
        return None
    
    print(f"\nError log saved to: {error_log_file}")    
    return results_df


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test all MOE kernel combinations")
    parser.add_argument(
        "-i", "--input",
        default="trace_moe_config.csv",
        help="Input configuration CSV file"
    )
    parser.add_argument(
        "-o", "--output",
        default="all_kernel_results.csv",
        help="Output results CSV file"
    )
    
    args = parser.parse_args()
    
    print("MOE Kernel Comprehensive Testing")
    print("=" * 80)
    
    # Run tests
    results = run_all_kernel_tests(args.input, args.output)
    
    if results is not None:
        print(f"\nTesting complete! Results saved to {args.output}")
    else:
        print("\nNo results generated. Check your configuration.")


if __name__ == "__main__":
    main()
