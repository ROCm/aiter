#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Shared utilities for MOE kernel benchmarking and profiling pipeline.

This module contains common functionality used across the benchmarking,
analysis, and profiling scripts to avoid code duplication.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd


# ============================================================================
# CSV SCHEMA DEFINITIONS (for reference/documentation)
# ============================================================================

# Input configuration columns (fixed format from tuned_configs_for_benchmark.csv)
CONFIG_COLUMNS = [
    'token', 'model_dim', 'inter_dim', 'expert', 'topk',
    'act_type', 'dtype', 'q_dtype_a', 'q_dtype_w', 'q_type',
    'use_g1u1', 'doweight_stage1'
]

# Benchmark results columns (output from 01_benchmark_and_analyze.py)
BENCHMARK_RESULT_COLUMNS = [
    'timestamp', 'config_idx',
    'token', 'model_dim', 'inter_dim', 'expert', 'topk',
    'act_type', 'dtype', 'q_dtype_a', 'q_dtype_w', 'q_type',
    'use_g1u1', 'doweight_stage1',
    'kernel_type', 'stage', 'block_m', 'kernel_name',
    'tile_m', 'tile_n', 'tile_k', 'block_size', 'waves_m', 'waves_n', 'kernel_version',
    'time_us', 'quant_time_us', 'error', 'tflops', 'bandwidth_gb'
]

# Best kernels columns (output from 01_benchmark_and_analyze.py, input to 02_profile_kernels.py)
BEST_KERNELS_COLUMNS = [
    'config_idx',
    'token', 'model_dim', 'inter_dim', 'expert', 'topk',
    'act_type', 'dtype', 'q_dtype_a', 'q_dtype_w', 'q_type',
    'use_g1u1', 'doweight_stage1',
    'kernel_type', 'stage', 'block_m', 'kernel_name',
    'time_us', 'quant_time_us', 'error', 'tflops', 'bandwidth_gb'
]


# ============================================================================
# KERNEL PARAMETER PARSING
# ============================================================================

def parse_kernel_parameters(kernel_name: str, kernel_type: str, stage: str) -> Dict[str, Optional[int]]:
    """
    Parse kernel parameters from kernel name.
    
    Args:
        kernel_name: Full kernel name string
        kernel_type: Type of kernel ('asm', 'ck', '1stage')
        stage: Stage identifier ('stage1', 'stage2', 'asm_1stage')
    
    Returns:
        Dictionary with parsed parameters: tile_m, tile_n, tile_k, block_size,
        waves_m, waves_n, version
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


# ============================================================================
# KERNEL CATEGORIZATION
# ============================================================================

def categorize_kernel_type(kernel_name: str, stage: str) -> str:
    """
    Categorize kernel implementation type from kernel name and stage.
    
    Args:
        kernel_name: Full kernel name string
        stage: Stage identifier
    
    Returns:
        Kernel type: 'asm', 'ck', or '1stage'
    """
    # ASM kernels have mangled names starting with _ZN or contain aiter::
    # CK kernels start with moe_ck2stages
    if stage == "asm_1stage" or "_ZN" in kernel_name or "aiter::" in kernel_name or "fmoe_" in kernel_name:
        return "asm"
    elif "moe_ck2stages" in kernel_name or "ck_moe" in kernel_name:
        return "ck"
    elif stage in ["stage1", "stage2"]:
        # Fallback: if stage1/stage2 but not CK name, likely ASM
        return "asm" if "_ZN" in kernel_name else "ck"
    else:
        return "1stage"


# ============================================================================
# KERNEL SELECTION AND ANALYSIS
# ============================================================================

def filter_valid_results(df: pd.DataFrame, error_threshold_pct: float = 50.0) -> pd.DataFrame:
    """
    Filter benchmark results to only valid kernels (error < threshold).
    
    Args:
        df: DataFrame with benchmark results
        error_threshold_pct: Maximum error percentage to consider valid (default 50%)
    
    Returns:
        Filtered DataFrame with only valid results
    """
    # Filter out failed results
    valid_df = df[df['error'] != 'failed'].copy()
    
    # Extract error percentage as float
    valid_df['error_pct'] = valid_df['error'].str.rstrip('%').astype(float)
    
    # Filter by error threshold
    valid_df = valid_df[valid_df['error_pct'] < error_threshold_pct]
    
    return valid_df


def select_best_kernels_for_config(config_results: pd.DataFrame) -> Tuple[Optional[Dict], Optional[pd.Series], str]:
    """
    Select the overall best kernel approach for a single configuration.
    
    Compares 2-stage vs 1-stage and returns the fastest approach.
    
    Args:
        config_results: DataFrame with all results for a single configuration
    
    Returns:
        Tuple of (best_2stage_combo, best_1stage, winner_type)
        where winner_type is '2-stage' or '1-stage'
    """
    # Separate by stage
    stage1_results = config_results[config_results['stage'] == 'stage1']
    stage2_results = config_results[config_results['stage'] == 'stage2']
    onestage_results = config_results[config_results['stage'] == 'asm_1stage']
    
    # Get quantization time (same for all in this config)
    quant_time = config_results['quant_time_us'].iloc[0] if len(config_results) > 0 else 0
    
    # Find best 2-stage combination
    best_2stage = None
    best_2stage_time = float('inf')
    
    if len(stage1_results) > 0 and len(stage2_results) > 0:
        # For each block_m value, find best stage1 + stage2 combination
        for block_m in sorted(stage1_results['block_m'].unique()):
            s1_for_blockm = stage1_results[stage1_results['block_m'] == block_m]
            s2_for_blockm = stage2_results[stage2_results['block_m'] == block_m]
            
            if len(s1_for_blockm) > 0 and len(s2_for_blockm) > 0:
                # Find fastest stage1 and stage2 kernels for this block_m
                best_s1 = s1_for_blockm.loc[s1_for_blockm['time_us'].idxmin()]
                best_s2 = s2_for_blockm.loc[s2_for_blockm['time_us'].idxmin()]
                
                # Calculate total 2-stage time
                total_time = best_s1['time_us'] + quant_time + best_s2['time_us']
                
                if total_time < best_2stage_time:
                    best_2stage_time = total_time
                    best_2stage = {
                        'block_m': block_m,
                        'total_time_us': total_time,
                        'stage1': best_s1,
                        'stage2': best_s2,
                        'quant_time_us': quant_time
                    }
    
    # Find best 1-stage kernel
    best_1stage = None
    if len(onestage_results) > 0:
        best_1stage = onestage_results.loc[onestage_results['time_us'].idxmin()]
    
    # Compare and select winner
    candidates = []
    if best_2stage:
        candidates.append(('2-stage', best_2stage['total_time_us']))
    if best_1stage is not None:
        candidates.append(('1-stage', best_1stage['time_us']))
    
    if not candidates:
        return None, None, 'none'
    
    winner_type, _ = min(candidates, key=lambda x: x[1])
    
    return best_2stage, best_1stage, winner_type


# ============================================================================
# LOGGING AND REPORTING
# ============================================================================

def print_section_header(title: str, char: str = '=', width: int = 80) -> None:
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(title)
    print(f"{char * width}\n")


def print_config_summary(config_row: pd.Series) -> None:
    """Print a summary of a configuration."""
    print(f"Token={config_row['token']}, Model={config_row['model_dim']}, "
          f"Inter={config_row['inter_dim']}, Expert={config_row['expert']}, "
          f"TopK={config_row['topk']}")
    print(f"  Quant={config_row['q_type']}, Act={config_row['act_type']}")
