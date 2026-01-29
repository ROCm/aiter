#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Extract MOE Configurations from PyTorch Trace Files

Purpose:
    Extracts MOE kernel configurations from PyTorch profiler trace JSON files.
    Generates a CSV file with configurations ready for benchmarking pipeline.

Input:
    PyTorch trace JSON file (from torch.profiler)
    - Looks for 'aiter::fused_moe_' events
    - Extracts dimensions, dtypes, quantization settings, etc.

Output:
    CSV file with MOE configurations for multiple token counts
    - Columns: token, model_dim, inter_dim, expert, topk, act_type, dtype,
              q_dtype_a, q_dtype_w, q_type, use_g1u1, doweight_stage1
    - One configuration replicated across multiple token values

Usage:
    python extract_configs.py trace.json -o config.csv
"""

import json
import sys
from pathlib import Path
import pandas as pd
import argparse
from typing import List, Dict, Optional

# Add parent directory to path for moe_utils import
sys.path.insert(0, str(Path(__file__).parent))
import moe_utils


# ============================================================================
# DTYPE MAPPING
# ============================================================================

def map_cpp_dtype_to_torch(cpp_type: str) -> str:
    """
    Map C++ PyTorch dtype strings to Python torch dtype strings.
    
    Args:
        cpp_type: C++ dtype string from trace (e.g., 'c10::BFloat16')
        
    Returns:
        Python torch dtype string (e.g., 'torch.bfloat16')
    """
    type_map = {
        'c10::BFloat16': 'torch.bfloat16',
        'c10::Half': 'torch.float16',
        'c10::Float16': 'torch.float16',
        'c10::Float': 'torch.float32',
        'c10::Float8_e4m3fn': 'torch.float8_e4m3fn',
        'c10::Float8_e4m3fnuz': 'torch.float8_e4m3fnuz',
        'c10::Float8_e5m2': 'torch.float8_e5m2',
        'c10::Int8': 'torch.int8',
        'c10::Byte': 'torch.uint8',
    }
    return type_map.get(cpp_type, 'torch.bfloat16')


def map_activation_code(act_code: int) -> str:
    """
    Map activation code to ActivationType enum string.
    
    Args:
        act_code: Integer activation code (0=Silu, 1=Gelu, 2=Swiglu)
        
    Returns:
        ActivationType enum string
    """
    activation_map = {
        0: 'ActivationType.Silu',
        1: 'ActivationType.Gelu',
        2: 'ActivationType.Swiglu',
    }
    return activation_map.get(act_code, 'ActivationType.Silu')


def map_quant_type_code(quant_code: str) -> str:
    """
    Map quantization type code to QuantType enum string.
    
    Args:
        quant_code: String quantization code
        
    Returns:
        QuantType enum string
    """
    quant_type_map = {
        '0': 'QuantType.No',
        '1': 'QuantType.per_Tensor',
        '2': 'QuantType.per_Token',
        '3': 'QuantType.per_Token_full',
        '4': 'QuantType.per_1x32',
        '5': 'QuantType.per_1x128',
        '6': 'QuantType.per_128x128',
    }
    return quant_type_map.get(quant_code, 'QuantType.No')


# ============================================================================
# TRACE EXTRACTION
# ============================================================================

def extract_moe_configs_from_trace(trace_file: str) -> List[Dict]:
    """
    Extract MOE kernel configurations from PyTorch trace file.
    
    Args:
        trace_file: Path to PyTorch profiler trace JSON file
        
    Returns:
        List of MOE configuration dictionaries extracted from trace events
    """
    print(f"Loading trace file: {trace_file}")
    
    try:
        with open(trace_file, 'r') as f:
            trace_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON file: {e}")
        return []
    except FileNotFoundError:
        print(f"ERROR: Trace file not found: {trace_file}")
        return []
    
    events = trace_data.get('traceEvents', [])
    print(f"Found {len(events)} total events in trace")
    
    # Find aiter::fused_moe_ events to extract configurations
    moe_configs = []
    moe_event_count = 0
    
    for event in events:
        name = event.get('name', '')
        
        # Look only for aiter::fused_moe_ events
        if name != 'aiter::fused_moe_':
            continue
        
        moe_event_count += 1
        args = event.get('args', {})
        input_dims = args.get('Input Dims', [])
        input_types = args.get('Input type', [])
        
        if not input_dims or len(input_dims) < 3:
            print(f"  Warning: Event {moe_event_count} has insufficient dimensions")
            continue
        
        # Extract configuration from dimensions
        # Format: [[token, model_dim], [expert, inter_dim*2, model_dim], 
        #          [expert, model_dim, inter_dim], ...]
        try:
            # First dimension: [token, model_dim]
            if len(input_dims[0]) >= 2:
                model_dim = input_dims[0][1]
            else:
                continue
            
            # Second dimension: w1 [expert, inter_dim*2 or inter_dim, model_dim]
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
            
            # Check if g1u1 (w1 has 2*inter_dim, else just inter_dim)
            use_g1u1 = (w1_dim == inter_dim * 2)
            
            # Get topk from args (usually in dimensions 3-4)
            topk = 8  # Default
            for dim in input_dims[3:5]:
                if len(dim) == 2:
                    topk = dim[1]
                    break
            
            # Extract parameters from Concrete Inputs
            concrete_inputs = args.get('Concrete Inputs', [])
            
            # Position 6: activation (0=Silu, 1=Gelu, 2=Swiglu)
            act_type = "ActivationType.Silu"
            if len(concrete_inputs) > 6 and concrete_inputs[6] in ['0', '1', '2']:
                act_code = int(concrete_inputs[6])
                act_type = map_activation_code(act_code)
            
            # Position 7: quant_type
            q_type = "QuantType.No"
            if len(concrete_inputs) > 7:
                q_type = map_quant_type_code(concrete_inputs[7])
            
            # Position 8: doweight_stage1
            doweight_stage1 = 0
            if len(concrete_inputs) > 8:
                doweight_val = concrete_inputs[8]
                if doweight_val in ['True', 'true', '1']:
                    doweight_stage1 = 1
            
            # Extract dtypes from Input Types
            hidden_dtype = input_types[0] if len(input_types) > 0 else 'c10::BFloat16'
            w1_dtype = input_types[1] if len(input_types) > 1 else 'c10::BFloat16'
            
            dtype = map_cpp_dtype_to_torch(hidden_dtype)
            q_dtype_w = map_cpp_dtype_to_torch(w1_dtype)
            q_dtype_a = map_cpp_dtype_to_torch(w1_dtype)
            
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
            
        except (IndexError, ValueError, KeyError) as e:
            print(f"  Warning: Failed to parse event {moe_event_count}: {e}")
            continue
    
    print(f"Successfully extracted {len(moe_configs)} MOE configurations from {moe_event_count} events")
    return moe_configs


def generate_config_csv(configs: List[Dict], 
                       output_file: str,
                       token_values: List[int]) -> None:
    """
    Generate configuration CSV file with multiple token values.
    
    Args:
        configs: List of configuration dictionaries
        output_file: Output CSV filepath
        token_values: List of token counts to generate configs for
    """
    if not configs:
        print("No configurations to save!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(configs)
    
    # Group by configuration (excluding token if present)
    config_cols = ['model_dim', 'inter_dim', 'expert', 'topk', 
                   'act_type', 'dtype', 'q_dtype_a', 'q_dtype_w', 'q_type',
                   'use_g1u1', 'doweight_stage1']
    
    grouped = df.groupby(config_cols).size().reset_index(name='count')
    
    print(f"\n{'='*80}")
    print("EXTRACTED CONFIGURATIONS")
    print(f"{'='*80}")
    
    # Check for multiple configurations
    if len(grouped) > 1:
        print(f"WARNING: Found {len(grouped)} different configurations in trace!")
        print("Showing all configurations:\n")
        for idx, row in grouped.iterrows():
            print(f"Config {idx+1} (appears {row['count']} times in trace):")
            for col in config_cols:
                print(f"  {col}: {row[col]}")
            print()
        print("Using first configuration for output...\n")
    
    # Use first (most common) configuration
    if len(grouped) > 0:
        base_config = grouped.iloc[0]
        
        print(f"Configuration (from {base_config['count']} kernel calls):")
        print(f"  Model Dim: {base_config['model_dim']}")
        print(f"  Inter Dim: {base_config['inter_dim']}")
        print(f"  Expert: {base_config['expert']}")
        print(f"  TopK: {base_config['topk']}")
        print(f"  Quant Type: {base_config['q_type']}")
        print(f"  Output dtype: {base_config['dtype']}")
        print(f"  Q dtype (activation): {base_config['q_dtype_a']}")
        print(f"  Q dtype (weight): {base_config['q_dtype_w']}")
        print(f"  Activation: {base_config['act_type']}")
        print(f"  use_g1u1: {base_config['use_g1u1']}")
        print(f"  doweight_stage1: {base_config['doweight_stage1']}")
        
        # Replicate config for each token value
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
        
        # Save to CSV
        output_df.to_csv(output_file, index=False)
        
        print(f"\n{'='*80}")
        print(f"Generated {len(output_rows)} configuration rows")
        print(f"Token values: {token_values}")
        print(f"Saved to: {output_file}")
        print(f"{'='*80}")
    else:
        print("ERROR: No valid configuration found in trace!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract MOE configurations from PyTorch trace files"
    )
    parser.add_argument(
        "trace_file",
        help="Path to PyTorch trace JSON file"
    )
    parser.add_argument(
        "-o", "--output",
        default="configs/extracted_config.csv",
        help="Output configuration CSV file (default: configs/extracted_config.csv)"
    )
    parser.add_argument(
        "--tokens",
        type=int,
        nargs='+',
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        help="Token values to generate configs for (default: 1, 2, 4, 8, 16, ...)"
    )
    
    args = parser.parse_args()
    
    # Verify input file exists
    trace_path = Path(args.trace_file)
    if not trace_path.exists():
        print(f"ERROR: Trace file not found: {args.trace_file}")
        return 1
    
    moe_utils.print_section_header("MOE CONFIGURATION EXTRACTION")
    print(f"Input trace: {args.trace_file}")
    print(f"Output CSV: {args.output}")
    print(f"Token sweep: {len(args.tokens)} values ({min(args.tokens)} to {max(args.tokens)})")
    
    # Extract configurations from trace
    configs = extract_moe_configs_from_trace(args.trace_file)
    
    if not configs:
        print("\nERROR: No MOE configurations found in trace!")
        print("Make sure the trace contains 'aiter::fused_moe_' events")
        return 1
    
    # Generate CSV with token sweep
    generate_config_csv(configs, args.output, args.tokens)
    
    # Success message
    moe_utils.print_section_header("EXTRACTION COMPLETE")
    print(f"Configuration file ready: {args.output}")
    print(f"\nNext step: Run benchmark and analysis")
    print(f"  python benchmark_and_analyze.py -i {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
