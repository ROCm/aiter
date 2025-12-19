#!/usr/bin/env python3
"""
Test all MOE kernel combinations for a given configuration.
Reads config from input_config_to_test.csv and outputs all kernel results.
"""

import sys
import os
from pathlib import Path

# Add paths for imports
current_dir = Path(__file__).parent
aiter_root = current_dir.parent
sys.path.insert(0, str(aiter_root))
sys.path.insert(0, str(aiter_root / "hsa/gfx942/fmoe_2stages"))

import pandas as pd
import torch

# Now import aiter and tune modules
import aiter
from aiter import QuantType, ActivationType
from aiter import dtypes
import tune
from tune import FmoeTuner


def run_all_kernel_tests(config_file="configs/input_config_to_test.csv", 
                         output_file="all_kernel_results.csv"):
    """
    Run all kernel combinations for configurations in the input file.
    
    Args:
        config_file: Path to input CSV with configuration
        output_file: Path to output CSV with all results
    """
    
    # Read configuration
    if not os.path.exists(config_file):
        print(f"Error: Config file {config_file} not found!")
        return
        
    config_df = pd.read_csv(config_file)
    print(f"Loaded {len(config_df)} configurations from {config_file}")
    
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
    
    # Process each configuration
    all_results = []
    
    for idx, row in config_df.iterrows():
        print(f"\n{'='*80}")
        print(f"Processing configuration {idx + 1}/{len(config_df)}:")
        print(f"Token={row['token']}, Model={row['model_dim']}, Inter={row['inter_dim']}, "
              f"Expert={row['expert']}, TopK={row['topk']}")
        
        # Prepare configuration
        dtype = eval(row['dtype'])
        q_dtype_a = eval(row['q_dtype_a'])
        q_dtype_w = eval(row['q_dtype_w'])
        q_type = eval(row['q_type'])
        act_type = eval(row['act_type'])
        
        cu_num = tuner.get_cu_num()
        
        info = (
            cu_num,
            row['token'],
            row['model_dim'],
            row['inter_dim'],
            row['expert'],
            row['topk'],
            act_type,
            dtype,
            q_dtype_a,
            q_dtype_w,
            q_type,
            row['use_g1u1'],
            row['doweight_stage1'],
        )
        
        # Generate all tasks (ASM stage1, CK stages, 1-stage)
        # Use same block_m values as tune.py
        blockMs = [16, 32, 64, 128]
        
        print("\nGenerating kernel tasks...")
        tasks_asm = tuner.gen_2stages_asm1_task(info, blockMs)
        tasks_ck = tuner.gen_2stages_task(info, blockMs)
        tasks_1stage = tuner.gen_1stage_asm_task(info)
        
        total_tasks = len(tasks_asm) + len(tasks_ck) + len(tasks_1stage)
        print(f"Total kernels to test: {total_tasks}")
        print(f"  - ASM stage1: {len(tasks_asm)}")
        print(f"  - CK 2-stage: {len(tasks_ck)}")
        print(f"  - 1-stage: {len(tasks_1stage)}")
        
        if total_tasks == 0:
            print("No kernels available for this configuration!")
            continue
            
        # Run all kernels using mp_tuner
        print("\nBenchmarking all kernels...")
        from aiter.utility.mp_tuner import mp_tuner
        
        all_tasks = tasks_asm + tasks_ck + tasks_1stage
        in_data = [(len(all_tasks), ())]
        
        # Run benchmarks (mp_num=1 means sequential, shape_grouped=False for individual results)
        results = mp_tuner(all_tasks, in_data, mp_num=1, shape_grouped=False)
        
        # Benchmark quantization between stages
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
        
        # Process results
        print(f"\nProcessing {len(results)} results...")
        
        for (key_info, stage, kernel_name, block_m), us, err in results:
            # Calculate TFLOPS and bandwidth
            tflops, bw = tuner.calculate((key_info, stage, kernel_name, block_m, us, err))
            
            # Determine kernel type
            if "asm" in stage:
                kernel_type = "asm"
            elif stage in ["stage1", "stage2"]:
                kernel_type = "ck"
            else:
                kernel_type = "1stage"
            
            # Store result
            result_dict = {
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
                'time_us': us,
                'quant_time_us': quant_time_us if stage in ['stage1', 'stage2'] else 0,
                'error': f"{err:.2%}" if err < 1.0 else "failed",
                'tflops': tflops if tflops > 0 else 0,
                'bandwidth_gb': bw if bw > 0 else 0
            }
            
            all_results.append(result_dict)
            
            # Print result
            status = "✓" if err < 0.01 else "⚠" if err < 1.0 else "✗"
            print(f"{status} {kernel_type:6} {stage:12} block_m={block_m:3} "
                  f"time={us:8.2f}us err={err:6.2%} "
                  f"TFLOPS={tflops:7.2f} BW={bw:7.2f}GB/s "
                  f"{kernel_name[:50]}")
    
    # Save all results to CSV
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
            print(f"⚠️  No valid kernels found (all have >{ERR_RATIO*100}% error or failed)")
            return results_df
        
        # Analyze each token value separately
        for token_val in sorted(valid_results['token'].unique()):
            print(f"\n{'='*80}")
            print(f"Token = {token_val}")
            print(f"{'='*80}\n")
            
            token_results = valid_results[valid_results['token'] == token_val]
            stage1_results = token_results[token_results['stage'] == 'stage1']
            stage2_results = token_results[token_results['stage'] == 'stage2']
            onestage_results = token_results[token_results['stage'] == 'asm_1stage']
            
            # Get quant_time for this token
            quant_time = token_results['quant_time_us'].iloc[0] if len(token_results) > 0 else 0
            
            best_2stage_combo = None
            if len(stage1_results) > 0 and len(stage2_results) > 0:
                # Group by block_m and find best combination for each block_m
                print("2-Stage Combinations (by block_m with quantization overhead):")
                print(f"{'-'*80}")
                
                block_m_combos = []
                for block_m in sorted(token_results['block_m'].unique()):
                    s1_for_blockm = stage1_results[stage1_results['block_m'] == block_m]
                    s2_for_blockm = stage2_results[stage2_results['block_m'] == block_m]
                
                    if len(s1_for_blockm) > 0 and len(s2_for_blockm) > 0:
                        # Find best stage1 and stage2 for this block_m
                        best_s1 = s1_for_blockm.loc[s1_for_blockm['time_us'].idxmin()]
                        best_s2 = s2_for_blockm.loc[s2_for_blockm['time_us'].idxmin()]
                        
                        # Include quantization time between stages
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
                
                # Select best block_m combination for this token
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
            
            # Analyze 1-stage kernels for this token
            best_1stage = None
            if len(onestage_results) > 0:
                best_1stage = onestage_results.loc[onestage_results['time_us'].idxmin()]
                print(f"\n{'-'*80}")
                print("Best 1-Stage Kernel:")
                print(f"  Kernel: {best_1stage['kernel_name'][:60]}")
                print(f"  block_m: {best_1stage['block_m']}")
                print(f"  Time: {best_1stage['time_us']:.2f} us (err={best_1stage['error']})")
                print(f"  TFLOPS: {best_1stage['tflops']:.2f}")
            
            # Final recommendation for this token
            print(f"\n{'-'*80}")
            print(f"RECOMMENDATION for Token={token_val}:")
            
            candidates = []
            if best_2stage_combo:
                candidates.append(('2-Stage', best_2stage_combo['total_time_us']))
            if best_1stage is not None:
                candidates.append(('1-Stage', best_1stage['time_us']))
            
            if candidates:
                fastest = min(candidates, key=lambda x: x[1])
                print(f"  {fastest[0]}: {fastest[1]:.2f} us")
                
                if len(candidates) > 1:
                    slower = max(candidates, key=lambda x: x[1])
                    speedup = slower[1] / fastest[1]
                    print(f"  Speedup vs {slower[0]}: {speedup:.2f}x faster")
        
    else:
        print("No results collected!")
        
    return results_df if all_results else None


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test all MOE kernel combinations")
    parser.add_argument(
        "-i", "--input",
        default="configs/input_config_to_test.csv",
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
