# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Extract MOE configuration from trace files.
"""

import json
import sys
import pandas as pd
from pathlib import Path


def extract_moe_configs_from_trace(trace_file):
    """
    Extract MOE kernel configurations from PyTorch trace file.
    
    Returns:
        List of MOE configurations found in the trace
    """
    print(f"Loading trace file: {trace_file}")
    
    with open(trace_file, 'r') as f:
        trace_data = json.load(f)
    
    events = trace_data.get('traceEvents', [])
    
    # Find aiter::fused_moe_ events to extract configurations
    moe_configs = []
    
    for event in events:
        name = event.get('name', '')
        
        # Look only for aiter::fused_moe_
        if name != 'aiter::fused_moe_':
            continue
            
        args = event.get('args', {})
        input_dims = args.get('Input Dims', [])
        input_types = args.get('Input type', [])
        
        if not input_dims or len(input_dims) < 3:
            continue
        
        # Extract configuration from dimensions
        # Format: [[token, model_dim], [expert, inter_dim*2, model_dim], [expert, model_dim, inter_dim], ...]
        try:
            # First dimension: [token, model_dim]
            if len(input_dims[0]) >= 2:
                # We don't extract token for now as trace files may have varying token counts across calls and across prefill and decode phases
                model_dim = input_dims[0][1]
            else:
                continue
            
            # Second dimension: w1 [expert, inter_dim*2, model_dim]
            if len(input_dims[1]) >= 3:
                expert = input_dims[1][0]
                w1_dim = input_dims[1][1]
            else:
                continue
            
            # Third dimension: w2 [expert, model_dim, inter_dim]
            if len(input_dims[2]) >= 3:
                inter_dim = input_dims[2][2]
            else:
                continue
            
            # Check if g1u1 (w1 has 2*inter_dim)
            use_g1u1 = (w1_dim == inter_dim * 2)
            
            # Get topk from args
            topk = 8  # Default
            for dim in input_dims[3:5]:
                if len(dim) == 2:
                    topk = dim[1]
                    break
            
            # Map PyTorch C++ types to aiter types
            def map_dtype(cpp_type):
                type_map = {
                    'c10::BFloat16': 'torch.bfloat16',
                    'c10::Half': 'torch.float16',
                    'c10::Float16': 'torch.float16',
                    'c10::Float': 'torch.float32',
                    'c10::Float8_e4m3fn': 'aiter.dtypes.fp8',
                    'c10::Float8_e4m3fnuz': 'aiter.dtypes.fp8',
                    'c10::Float8_e5m2': 'aiter.dtypes.fp8',
                    'c10::Int8': 'torch.int8',
                    'c10::Byte': 'torch.uint8',
                }
                return type_map.get(cpp_type, 'torch.bfloat16')
            
            # Extract parameters from Concrete Inputs
            concrete_inputs = args.get('Concrete Inputs', [])
            
            # Position 6: activation (0=Silu, 1=Gelu, 2=Swiglu)
            # Position 7: quant_type (0=No, 2=per_Token, 5=per_1x128, etc.)
            # Position 8: doweight_stage1 (False/True)
            
            act_type = "ActivationType.Silu"
            if len(concrete_inputs) > 6 and concrete_inputs[6] in ['0', '1', '2']:
                act_code = int(concrete_inputs[6])
                if act_code == 0:
                    act_type = "ActivationType.Silu"
                elif act_code == 1:
                    act_type = "ActivationType.Gelu"
                elif act_code == 2:
                    act_type = "ActivationType.Swiglu"
            
            # Map quant_type integer to QuantType enum
            quant_type_map = {
                '0': 'QuantType.No',
                '1': 'QuantType.per_Tensor',
                '2': 'QuantType.per_Token',
                '3': 'QuantType.per_Token_full',
                '4': 'QuantType.per_1x32',
                '5': 'QuantType.per_1x128',
                '6': 'QuantType.per_128x128',
            }
            
            q_type = "QuantType.No"
            if len(concrete_inputs) > 7 and concrete_inputs[7] in quant_type_map:
                q_type = quant_type_map[concrete_inputs[7]]
            
            doweight_stage1 = 0
            if len(concrete_inputs) > 8:
                doweight_val = concrete_inputs[8]
                if doweight_val in ['True', 'true', '1']:
                    doweight_stage1 = 1
            
            # Extract dtypes from Input Types
            # input_types[0]: hidden_states dtype (output dtype)
            # input_types[1]: w1 weight dtype
            # input_types[2]: w2 weight dtype (can be used for activation quant dtype)
            hidden_dtype = input_types[0] if len(input_types) > 0 else 'c10::BFloat16'
            w1_dtype = input_types[1] if len(input_types) > 1 else 'c10::BFloat16'
            w2_dtype = input_types[2] if len(input_types) > 2 else 'c10::BFloat16'
            
            dtype = map_dtype(hidden_dtype)      # Output dtype
            q_dtype_w = map_dtype(w1_dtype)      # Weight quant dtype
            q_dtype_a = map_dtype(w1_dtype)      # Activation quant dtype (extracted from input_types)
            
            config = {
                'model_dim': model_dim,
                'inter_dim': inter_dim,
                'expert': expert,
                'topk': topk,
                'act_type': act_type,
                'dtype': dtype,
                'q_dtype_a': q_dtype_a,
                'q_dtype_w': q_dtype_w,
                'q_type': q_type,
                'use_g1u1': 1 if use_g1u1 else 0,
                'doweight_stage1': doweight_stage1,
            }
            
            moe_configs.append(config)
            
        except (IndexError, ValueError) as e:
            continue
    
    return moe_configs


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze trace file for MOE configurations")
    parser.add_argument(
        "trace_file",
        help="Path to PyTorch trace JSON file"
    )
    parser.add_argument(
        "-o", "--output",
        default="configs/trace_moe_config.csv",
        help="Output configuration CSV file"
    )
    
    args = parser.parse_args()
    
    # Extract configurations
    configs = extract_moe_configs_from_trace(args.trace_file)
    
    if not configs:
        print("No MOE configurations found in trace!")
        return
    
    print(f"\nFound {len(configs)} aiter::fused_moe_ calls")
    
    # Get unique configurations (no token)
    df = pd.DataFrame(configs)
    
    # Group by configuration
    config_cols = ['model_dim', 'inter_dim', 'expert', 'topk', 
                   'act_type', 'dtype', 'q_dtype_a', 'q_dtype_w', 'q_type',
                   'use_g1u1', 'doweight_stage1']
    
    # Add token column with empty value for output CSV
    df['token'] = ''
    
    grouped = df.groupby(config_cols).size().reset_index(name='count')
    
    print("\nExtracted MOE Configuration:")
    print("=" * 80)
    
    # Check if all kernel calls have the same configuration
    if len(grouped) > 1:
        print(f"WARNING: Found {len(grouped)} different configurations in trace!")
        print("Showing all configurations:\n")
        for idx, row in grouped.iterrows():
            print(f"Config {idx+1} (used {row['count']} times):")
            for col in config_cols:
                print(f"  {col}: {row[col]}")
            print()
    
    if len(grouped) > 0:
        row = grouped.iloc[0]
        print(f"Using configuration from {row['count']} kernel calls:")
        print(f"  Model Dim: {row['model_dim']}")
        print(f"  Inter Dim: {row['inter_dim']}")
        print(f"  Expert: {row['expert']}")
        print(f"  TopK: {row['topk']}")
        print(f"  Quant: {row['q_type']}")
        print(f"  dtype: {row['dtype']}")
        print(f"  Q_dtype_a: {row['q_dtype_a']}, Q_dtype_w: {row['q_dtype_w']}")
        print(f"  Activation: {row['act_type']}")
        print(f"  use_g1u1: {row['use_g1u1']}, doweight_stage1: {row['doweight_stage1']}")
    
    
    token_values = [
        1,
        2,
        4,
        8,
        16,     
        32,     
        64,     
        128,    
        256,    
        512,    
        1024,   
        2048,   
        4096,   
        8192,   
        16384,
    ]
    
    print(f"\n{'='*80}")
    print(f"{len(token_values)} token configuration")
    
    # Replicate config for each token value
    if len(grouped) > 0:
        base_config = grouped.iloc[0]
        output_rows = []
        
        for token in token_values:
            row_dict = {'token': token}
            for col in config_cols:
                row_dict[col] = base_config[col]
            output_rows.append(row_dict)
        
        output_df = pd.DataFrame(output_rows)
        # Reorder columns to have token first
        cols = ['token'] + config_cols
        output_df = output_df[cols]
        output_df.to_csv(args.output, index=False)
        
        print(f"Saved {len(output_rows)} configurations with token values: {token_values}")
    else:
        print("No configuration found!")
        return
    

if __name__ == "__main__":
    main()
