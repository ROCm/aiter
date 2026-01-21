#!/usr/bin/env python3
"""
Generate comprehensive kernel combination report.

For each config:
- 2-stage: All combinations of (stage1 kernel, stage2 kernel) with same block_m
- 1-stage: All available 1-stage kernels
- Sorted by total time (fastest first)

Outputs:
1. All combinations: {output_base}_all_combinations.csv
2. Best kernels for profiling: {output_base}_best_for_profiling.csv
"""

import pandas as pd
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Generate kernel combination reports')
    parser.add_argument('-i', '--input', required=True, help='Input CSV file with benchmark results')
    parser.add_argument('-o', '--output', required=True, help='Output base name (will generate two files)')
    
    args = parser.parse_args()
    
    input_file = args.input
    output_base = args.output
    
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)

    # Filter valid results
    valid_df = df[df['error'] != 'failed'].copy()
    valid_df['error_pct'] = valid_df['error'].str.rstrip('%').astype(float)
    valid_df = valid_df[valid_df['error_pct'] < 50.0]

    print(f"Loaded {len(valid_df)} valid results")

    # Get unique configs
    unique_configs = valid_df['config_idx'].unique()
    print(f"Processing {len(unique_configs)} configurations...")

    all_combinations = []

    for config_idx in unique_configs:
        config_data = valid_df[valid_df['config_idx'] == config_idx]
        
        # Get config info
        config_info = config_data.iloc[0]
        
        # Separate by stage
        stage1 = config_data[config_data['stage'] == 'stage1']
        stage2 = config_data[config_data['stage'] == 'stage2']
        onestage = config_data[config_data['stage'] == 'asm_1stage']
        
        quant_time = config_data['quant_time_us'].iloc[0]
        
        # 2-stage combinations
        for _, s1 in stage1.iterrows():
            # Find stage2 kernels with same block_m
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

    # Create dataframe
    result_df = pd.DataFrame(all_combinations)

    # Sort by config_idx and total_time (fastest first)
    result_df = result_df.sort_values(['config_idx', 'total_time_us'])

    # Save to CSV
    all_combos_file = f'{output_base}_all_combinations.csv'
    result_df.to_csv(all_combos_file, index=False)

    print(f"\nGenerated {len(result_df)} kernel combinations")
    print(f"  2-stage combinations: {len(result_df[result_df['approach'] == '2-stage'])}")
    print(f"  1-stage kernels: {len(result_df[result_df['approach'] == '1-stage'])}")
    print(f"\nSaved to: {all_combos_file}")

    # Show sample
    print("\nSample (first config, top 5 fastest):")
    sample = result_df[result_df['config_idx'] == result_df['config_idx'].iloc[0]].head(5)
    print(sample[['approach', 'stage1_kernel_name', 'stage2_kernel_name', 
                  'stage1_time_us', 'quant_time_us', 'stage2_time_us', 'total_time_us']])

    print("\nAll kernel combinations ready for analysis!")

    # ============================================================================
    # PART 2: Generate best kernels for profiling
    # ============================================================================
    print("\n" + "="*80)
    print("GENERATING BEST KERNELS FOR PROFILING")
    print("="*80)

    profiling_rows = []

    for config_idx in unique_configs:
        config_data = valid_df[valid_df['config_idx'] == config_idx]
        config_info = config_data.iloc[0]
        
        stage1 = config_data[config_data['stage'] == 'stage1']
        stage2 = config_data[config_data['stage'] == 'stage2']
        onestage = config_data[config_data['stage'] == 'asm_1stage']
        quant_time = config_data['quant_time_us'].iloc[0]
        
        # Find best 2-stage combination
        best_2stage_combo = None
        best_2stage_time = float('inf')
        
        for _, s1 in stage1.iterrows():
            s2_matches = stage2[stage2['block_m'] == s1['block_m']]
            
            for _, s2 in s2_matches.iterrows():
                total_time = s1['time_us'] + quant_time + s2['time_us']
                
                if total_time < best_2stage_time:
                    best_2stage_time = total_time
                    best_2stage_combo = (s1, s2)
        
        # Find best 1-stage kernel
        best_1stage = None
        if len(onestage) > 0:
            best_1stage = onestage.loc[onestage['time_us'].idxmin()]
        
        # Select overall best
        candidates = []
        if best_2stage_combo:
            candidates.append(('2-stage', best_2stage_time, best_2stage_combo))
        if best_1stage is not None:
            candidates.append(('1-stage', best_1stage['time_us'], best_1stage))
        
        if not candidates:
            print(f"Config {config_idx}: No valid kernels!")
            continue
        
        fastest = min(candidates, key=lambda x: x[1])
        approach, best_time, kernel_info = fastest
        
        print(f"\nConfig {config_idx}: {approach} selected ({best_time:.2f} μs)")
        
        if approach == '2-stage':
            s1, s2 = kernel_info
            
            # Stage1 row
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
                'kernel_type': s1['kernel_type'],
                'stage': s1['stage'],
                'block_m': s1['block_m'],
                'kernel_name': s1['kernel_name'],
                'time_us': s1['time_us'],
                'quant_time_us': quant_time,
                'error': s1['error'],
                'tflops': s1['tflops'],
                'bandwidth_gb': s1['bandwidth_gb'],
            })
            
            # Stage2 row
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
                'kernel_type': s2['kernel_type'],
                'stage': s2['stage'],
                'block_m': s2['block_m'],
                'kernel_name': s2['kernel_name'],
                'time_us': s2['time_us'],
                'quant_time_us': quant_time,
                'error': s2['error'],
                'tflops': s2['tflops'],
                'bandwidth_gb': s2['bandwidth_gb'],
            })
            
            print(f"  Stage1: {s1['kernel_name'][:50]} - {s1['time_us']:.2f} μs")
            print(f"  Stage2: {s2['kernel_name'][:50]} - {s2['time_us']:.2f} μs")
        
        else:  # 1-stage
            ks = kernel_info
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
                'kernel_type': ks['kernel_type'],
                'stage': ks['stage'],
                'block_m': ks['block_m'],
                'kernel_name': ks['kernel_name'],
                'time_us': ks['time_us'],
                'quant_time_us': 0,
                'error': ks['error'],
                'tflops': ks['tflops'],
                'bandwidth_gb': ks['bandwidth_gb'],
            })
            
            print(f"  Kernel: {ks['kernel_name'][:50]} - {ks['time_us']:.2f} μs")

    # Save best kernels for profiling
    profiling_df = pd.DataFrame(profiling_rows)
    profiling_output_file = f'{output_base}_best_for_profiling.csv'
    profiling_df.to_csv(profiling_output_file, index=False)

    print("\n" + "="*80)
    print("BEST KERNELS FOR PROFILING")
    print("="*80)
    print(f"Total configs: {len(unique_configs)}")
    print(f"Total kernels to profile: {len(profiling_rows)}")
    print(f"  Stage1: {len(profiling_df[profiling_df['stage'] == 'stage1'])}")
    print(f"  Stage2: {len(profiling_df[profiling_df['stage'] == 'stage2'])}")
    print(f"  1-stage: {len(profiling_df[profiling_df['stage'] == 'asm_1stage'])}")
    print(f"\nSaved to: {profiling_output_file}")
    print("\nUsage:")
    print(f"  python profile_moe_kernel_fixed.py -i {profiling_output_file} -o profiling_output")
    print("="*80)


if __name__ == '__main__':
    main()
