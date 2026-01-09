#!/usr/bin/env python3
"""
Analyze MOE kernel benchmark results.

Reads results CSV from benchmark_moe_kernels.py and:
1. Groups kernels by configuration
2. Finds best 2-stage combinations (stage1 + quant + stage2)
3. Finds best 1-stage kernels
4. Compares and recommends fastest approach per config

Usage:
    python analyze_benchmark_results.py -i results.csv
"""

import pandas as pd
import argparse


def analyze_results(results_file):
    """
    Analyze benchmark results and find best kernels per configuration.
    
    Args:
        results_file: Path to CSV file with benchmark results
    """
    
    print("MOE Kernel Results Analysis")
    print("=" * 80)
    
    # Read results
    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} kernel results from {results_file}")
    
    # Filter to valid results (error < 50%)
    valid_df = df[df['error'] != 'failed'].copy()
    valid_df['error_pct'] = valid_df['error'].str.rstrip('%').astype(float)
    valid_df = valid_df[valid_df['error_pct'] < 50.0]
    
    print(f"Valid results (error < 50%): {len(valid_df)}")
    
    # Get unique configurations
    config_cols = ['config_idx', 'token', 'model_dim', 'inter_dim', 'expert', 'topk',
                   'act_type', 'q_type', 'dtype', 'q_dtype_a', 'q_dtype_w']
    
    unique_configs = valid_df[config_cols].drop_duplicates()
    print(f"Unique configurations tested: {len(unique_configs)}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS PER CONFIGURATION")
    print(f"{'='*80}\n")
    
    # Analyze each configuration
    for _, config in unique_configs.iterrows():
        print(f"\n{'='*80}")
        print(f"Config #{config['config_idx']}: "
              f"token={config['token']}, model_dim={config['model_dim']}, "
              f"expert={config['expert']}, topk={config['topk']}")
        print(f"  quant={config['q_type']}, act={config['act_type']}")
        print(f"{'='*80}")
        
        # Filter results for this config
        config_results = valid_df[valid_df['config_idx'] == config['config_idx']]
        
        # Separate by stage
        stage1_results = config_results[config_results['stage'] == 'stage1']
        stage2_results = config_results[config_results['stage'] == 'stage2']
        onestage_results = config_results[config_results['stage'] == 'asm_1stage']
        
        # Get quantization time (same for all in this config)
        quant_time = config_results['quant_time_us'].iloc[0] if len(config_results) > 0 else 0
        
        # Analyze 2-stage combinations
        best_2stage = None
        if len(stage1_results) > 0 and len(stage2_results) > 0:
            print("\n2-Stage Combinations:")
            print("-" * 80)
            
            combinations = []
            for block_m in sorted(config_results['block_m'].unique()):
                s1 = stage1_results[stage1_results['block_m'] == block_m]
                s2 = stage2_results[stage2_results['block_m'] == block_m]
                
                if len(s1) > 0 and len(s2) > 0:
                    best_s1 = s1.loc[s1['time_us'].idxmin()]
                    best_s2 = s2.loc[s2['time_us'].idxmin()]
                    
                    total = best_s1['time_us'] + quant_time + best_s2['time_us']
                    
                    combinations.append({
                        'block_m': block_m,
                        'total_time': total,
                        's1_time': best_s1['time_us'],
                        's1_kernel': best_s1['kernel_name'],
                        's1_error': best_s1['error'],
                        's2_time': best_s2['time_us'],
                        's2_kernel': best_s2['kernel_name'],
                        's2_error': best_s2['error'],
                    })
                    
                    print(f"  block_m={block_m}: {total:.2f} us")
                    print(f"    Stage1: {best_s1['time_us']:.2f} us (err={best_s1['error']})")
                    print(f"    Quant:  {quant_time:.2f} us")
                    print(f"    Stage2: {best_s2['time_us']:.2f} us (err={best_s2['error']})")
            
            if combinations:
                best_2stage = min(combinations, key=lambda x: x['total_time'])
                print(f"\n  BEST 2-Stage: block_m={best_2stage['block_m']}, "
                      f"total={best_2stage['total_time']:.2f} us")
        
        # Analyze 1-stage kernels
        best_1stage = None
        if len(onestage_results) > 0:
            print("\n1-Stage Kernels:")
            print("-" * 80)
            
            best_1stage = onestage_results.loc[onestage_results['time_us'].idxmin()]
            print(f"  BEST: {best_1stage['kernel_name'][:60]}")
            print(f"    Time: {best_1stage['time_us']:.2f} us")
            print(f"    Error: {best_1stage['error']}")
        
        # Final recommendation
        print(f"\n{'='*80}")
        print("RECOMMENDATION:")
        
        candidates = []
        if best_2stage:
            candidates.append(('2-Stage', best_2stage['total_time']))
        if best_1stage is not None:
            candidates.append(('1-Stage', best_1stage['time_us']))
        
        if candidates:
            winner = min(candidates, key=lambda x: x[1])
            print(f"  WINNER: {winner[0]} - {winner[1]:.2f} us")
            
            if len(candidates) > 1:
                loser = max(candidates, key=lambda x: x[1])
                speedup = loser[1] / winner[1]
                print(f"  Speedup vs {loser[0]}: {speedup:.2f}x faster")
        else:
            print("  No valid kernels for this configuration!")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MOE kernel benchmark results"
    )
    parser.add_argument(
        '-i', '--input',
        default='results.csv',
        help='Input results CSV file from benchmark_moe_kernels.py'
    )
    
    args = parser.parse_args()
    
    analyze_results(args.input)
    
    print(f"\n{'='*80}")
    print("Analysis complete!")


if __name__ == "__main__":
    main()
