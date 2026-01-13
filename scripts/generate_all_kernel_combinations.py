#!/usr/bin/env python3
"""
Generate comprehensive kernel combination report.

For each config:
- 2-stage: All combinations of (stage1 kernel, stage2 kernel) with same block_m
- 1-stage: All available 1-stage kernels
- Sorted by total time (fastest first)
"""

import pandas as pd

print("Loading results.csv...")
df = pd.read_csv('results/results.csv')

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
output_file = 'results/all_kernel_combinations.csv'
result_df.to_csv(output_file, index=False)

print(f"\nGenerated {len(result_df)} kernel combinations")
print(f"  2-stage combinations: {len(result_df[result_df['approach'] == '2-stage'])}")
print(f"  1-stage kernels: {len(result_df[result_df['approach'] == '1-stage'])}")
print(f"\nSaved to: {output_file}")

# Show sample
print("\nSample (first config, top 5 fastest):")
sample = result_df[result_df['config_idx'] == result_df['config_idx'].iloc[0]].head(5)
print(sample[['approach', 'stage1_kernel_name', 'stage2_kernel_name', 
              'stage1_time_us', 'quant_time_us', 'stage2_time_us', 'total_time_us']])

print("\nAll kernel combinations ready for analysis!")
