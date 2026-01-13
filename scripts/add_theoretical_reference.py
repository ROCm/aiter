#!/usr/bin/env python3
"""
Calculate theoretical performance reference for each MoE configuration.

Uses roofline model based on:
- FLOPs required
- Memory bandwidth required
- GPU peak performance (MI300X specs)
"""

import pandas as pd
import numpy as np


def get_dtype_bytes(dtype_str):
    """Get bytes per element for a dtype."""
    if 'bfloat16' in str(dtype_str) or 'float16' in str(dtype_str):
        return 2
    elif 'fp8' in str(dtype_str).lower() or 'float8' in str(dtype_str):
        return 1
    elif 'int8' in str(dtype_str):
        return 1
    elif 'int4' in str(dtype_str) or 'fp4' in str(dtype_str):
        return 0.5
    else:
        return 2  # Default to FP16


def calculate_theoretical_time(row, efficiency=0.7):
    """
    Calculate theoretical minimum time for a MoE configuration.
    
    Args:
        row: DataFrame row with config info
        efficiency: Efficiency factor (0.7 = 70% of peak)
    
    Returns:
        Theoretical minimum time in microseconds
    """
    token = row['token']
    model_dim = row['model_dim']
    inter_dim = row['inter_dim']
    topk = row['topk']
    
    # Get data type sizes
    dtype_bytes = get_dtype_bytes(row['dtype'])
    weight_bytes = get_dtype_bytes(row['q_dtype_w'])
    
    # FLOPs calculation (Stage1 + Stage2)
    # Stage1: token x (topk*inter_dim*2) @ (topk*inter_dim*2) x model_dim
    # Stage2: (token*topk) x inter_dim @ inter_dim x model_dim
    # Each GEMM: 2*M*N*K FLOPs
    
    stage1_flops = 2 * token * topk * (inter_dim * 2) * model_dim
    stage2_flops = 2 * (token * topk) * model_dim * inter_dim
    total_flops = stage1_flops + stage2_flops
    
    # Memory traffic (bytes)
    # Inputs + Weights + Outputs + intermediate values
    input_bytes = token * model_dim * dtype_bytes
    w1_bytes = topk * (inter_dim * 2) * model_dim * weight_bytes
    w2_bytes = topk * model_dim * inter_dim * weight_bytes
    intermediate_bytes = token * topk * inter_dim * dtype_bytes  # Between stages
    output_bytes = token * model_dim * dtype_bytes
    
    total_bytes = input_bytes + w1_bytes + w2_bytes + intermediate_bytes + output_bytes
    
    # GPU specs (MI300X)
    # Adjust peak TFLOPS based on data type
    if 'fp8' in str(row['q_dtype_w']).lower():
        peak_tflops = 2600  # FP8 peak
    else:
        peak_tflops = 1300  # FP16/BF16 peak
    
    peak_bandwidth_gbs = 5300  # 5.3 TB/s
    
    # Calculate theoretical times
    compute_time_us = (total_flops / (peak_tflops * 1e12 * efficiency)) * 1e6
    memory_time_us = (total_bytes / (peak_bandwidth_gbs * 1e9 * efficiency)) * 1e6
    
    # Theoretical minimum is the bottleneck
    theoretical_min_us = max(compute_time_us, memory_time_us)
    
    # Add routing overhead estimation (empirical: ~5-10% of compute time)
    routing_overhead_us = compute_time_us * 0.08
    
    theoretical_total_us = theoretical_min_us + routing_overhead_us
    
    return {
        'theoretical_time_us': theoretical_total_us,
        'compute_bound_time_us': compute_time_us,
        'memory_bound_time_us': memory_time_us,
        'bottleneck': 'compute' if compute_time_us > memory_time_us else 'memory',
        'total_flops': total_flops,
        'total_bytes': total_bytes,
        'arithmetic_intensity': total_flops / total_bytes if total_bytes > 0 else 0
    }


# Load all_kernel_combinations.csv
print("Loading all_kernel_combinations.csv...")
df = pd.read_csv('all_kernel_combinations.csv')

print(f"Loaded {len(df)} kernel combinations")

# Get unique configs
config_cols = ['token', 'model_dim', 'inter_dim', 'expert', 'topk',
               'act_type', 'q_type', 'dtype', 'q_dtype_a', 'q_dtype_w']

unique_configs = df[config_cols].drop_duplicates()
print(f"Calculating theoretical times for {len(unique_configs)} unique configs...")

# Calculate theoretical time for each unique config
theoretical_times = []

for idx, config in unique_configs.iterrows():
    theory = calculate_theoretical_time(config)
    
    config_dict = config.to_dict()
    config_dict.update(theory)
    theoretical_times.append(config_dict)

theory_df = pd.DataFrame(theoretical_times)

# Merge theoretical times back to main dataframe
df_with_theory = df.merge(
    theory_df,
    on=config_cols,
    how='left'
)

# Calculate efficiency metrics
df_with_theory['slowdown_factor'] = df_with_theory['total_time_us'] / df_with_theory['theoretical_time_us']

# Keep only fastest kernel per config
best_per_config = df_with_theory.loc[df_with_theory.groupby('config_idx')['total_time_us'].idxmin()].copy()

# Keep only essential columns
keep_cols = [
    'config_idx', 'token', 'model_dim', 'inter_dim', 'expert', 'topk',
    'act_type', 'q_type', 'dtype', 'q_dtype_a', 'q_dtype_w',
    'use_g1u1', 'doweight_stage1',
    'approach', 'block_m',
    'stage1_kernel_name', 'stage1_kernel_type', 'stage1_time_us',
    'quant_time_us',
    'stage2_kernel_name', 'stage2_kernel_type', 'stage2_time_us',
    'total_time_us',
    'theoretical_time_us',  # Keep only this theoretical metric
    'slowdown_factor'        # Keep only this efficiency metric
]

# Filter to only existing columns
keep_cols = [col for col in keep_cols if col in best_per_config.columns]
best_per_config = best_per_config[keep_cols]

# Save best per config only
output_file = 'best_kernels_with_reference.csv'
best_per_config.to_csv(output_file, index=False)

print(f"\nSaved {len(best_per_config)} best kernels to: {output_file}")
print(f"Each row = fastest kernel for one config")
print(f"Key columns: theoretical_time_us, slowdown_factor")

# Print summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nTheoretical Performance:")
print(f"  Average theoretical time: {theory_df['theoretical_time_us'].mean():.2f} us")
print(f"  Compute-bound configs: {(theory_df['bottleneck'] == 'compute').sum()}")
print(f"  Memory-bound configs: {(theory_df['bottleneck'] == 'memory').sum()}")

print(f"\nBest Kernel Efficiency:")
print(f"  Average slowdown: {best_per_config['slowdown_factor'].mean():.2f}x")
print(f"  Best slowdown: {best_per_config['slowdown_factor'].min():.2f}x")
print(f"  Worst slowdown: {best_per_config['slowdown_factor'].max():.2f}x")
print(f"  Configs with slowdown >2x: {(best_per_config['slowdown_factor'] > 2.0).sum()}")

print(f"\nTop 5 Most Efficient:")
best = best_per_config.nsmallest(5, 'slowdown_factor')
print(best[['config_idx', 'token', 'approach', 'total_time_us', 'theoretical_time_us', 
            'slowdown_factor']])

print(f"\nTop 5 Least Efficient:")
worst = best_per_config.nlargest(5, 'slowdown_factor')
print(worst[['config_idx', 'token', 'approach', 'total_time_us', 'theoretical_time_us',
             'slowdown_factor']])

print("\nTheoretical reference times calculated!")
print("Use 'slowdown_factor' column to identify underperforming kernels")
