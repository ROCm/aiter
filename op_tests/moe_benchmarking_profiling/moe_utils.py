#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Shared utilities for MOE kernel benchmarking and profiling pipeline.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd


# ============================================================================
# CSV SCHEMA DEFINITIONS
# ============================================================================

# Input configuration columns
CONFIG_COLUMNS = [
    'token', 'model_dim', 'inter_dim', 'expert', 'topk',
    'act_type', 'dtype', 'q_dtype_a', 'q_dtype_w', 'q_type',
    'use_g1u1', 'doweight_stage1'
]

# Benchmark results columns (output from benchmark_and_analyze.py)
BENCHMARK_RESULT_COLUMNS = [
    'timestamp', 'config_idx',
    'token', 'model_dim', 'inter_dim', 'expert', 'topk',
    'act_type', 'dtype', 'q_dtype_a', 'q_dtype_w', 'q_type',
    'use_g1u1', 'doweight_stage1',
    'kernel_type', 'stage', 'block_m', 'kernel_name',
    'tile_m', 'tile_n', 'tile_k', 'block_size', 'waves_m', 'waves_n', 'kernel_version',
    'time_us', 'quant_time_us', 'error', 'tflops', 'bandwidth_gb'
]

# Best kernels columns (output from benchmark_and_analyze.py, input to profile_kernels.py)
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
        Kernel type: 'asm', 'ck', 'triton', or '1stage'
    """
    if stage == "triton_1stage" or "triton_e2e_moe" in kernel_name:
        return "triton"
    elif stage == "asm_1stage" or "_ZN" in kernel_name or "aiter::" in kernel_name or "fmoe_" in kernel_name:
        return "asm"
    elif "moe_ck2stages" in kernel_name or "ck_moe" in kernel_name:
        return "ck"
    elif stage in ["stage1", "stage2"]:
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


def select_best_kernels_for_config(config_results: pd.DataFrame) -> List[pd.Series]:
    """
    Select best kernels for profiling, ensuring both ASM and CK coverage.
    
    For each config, selects:
    - Best stage1 kernel + best of opposite type (ASM/CK)
    - Best stage2 kernel + best of opposite type (ASM/CK)
    - Best 1-stage kernel
    
    Args:
        config_results: DataFrame with all results for a single configuration
    
    Returns:
        List of kernel Series to profile (deduplicated)
    """
    selected_kernels = []
    seen_kernels = set()  # Track (stage, kernel_name) to avoid duplicates
    
    # Separate by stage
    stage1_results = config_results[config_results['stage'] == 'stage1']
    stage2_results = config_results[config_results['stage'] == 'stage2']
    onestage_results = config_results[config_results['stage'] == 'asm_1stage']
    
    # === STAGE 1 SELECTION ===
    if len(stage1_results) > 0:
        # Best stage1 overall
        best_s1 = stage1_results.loc[stage1_results['time_us'].idxmin()]
        key = ('stage1', best_s1['kernel_name'])
        if key not in seen_kernels:
            seen_kernels.add(key)
            selected_kernels.append(best_s1)
        
        # Best of opposite type
        if best_s1['kernel_type'] == 'asm':
            ck_s1 = stage1_results[stage1_results['kernel_type'] == 'ck']
            if len(ck_s1) > 0:
                best_ck_s1 = ck_s1.loc[ck_s1['time_us'].idxmin()]
                key = ('stage1', best_ck_s1['kernel_name'])
                if key not in seen_kernels:
                    seen_kernels.add(key)
                    selected_kernels.append(best_ck_s1)
        else:  # best is CK
            asm_s1 = stage1_results[stage1_results['kernel_type'] == 'asm']
            if len(asm_s1) > 0:
                best_asm_s1 = asm_s1.loc[asm_s1['time_us'].idxmin()]
                key = ('stage1', best_asm_s1['kernel_name'])
                if key not in seen_kernels:
                    seen_kernels.add(key)
                    selected_kernels.append(best_asm_s1)
    
    # === STAGE 2 SELECTION ===
    stage2_results = config_results[config_results['stage'] == 'stage2']
    if len(stage2_results) > 0:
        # Best stage2 overall
        best_s2 = stage2_results.loc[stage2_results['time_us'].idxmin()]
        key = ('stage2', best_s2['kernel_name'])
        if key not in seen_kernels:
            seen_kernels.add(key)
            selected_kernels.append(best_s2)
        
        # Best of opposite type
        if best_s2['kernel_type'] == 'asm':
            ck_s2 = stage2_results[stage2_results['kernel_type'] == 'ck']
            if len(ck_s2) > 0:
                best_ck_s2 = ck_s2.loc[ck_s2['time_us'].idxmin()]
                key = ('stage2', best_ck_s2['kernel_name'])
                if key not in seen_kernels:
                    seen_kernels.add(key)
                    selected_kernels.append(best_ck_s2)
        else:  # best is CK
            asm_s2 = stage2_results[stage2_results['kernel_type'] == 'asm']
            if len(asm_s2) > 0:
                best_asm_s2 = asm_s2.loc[asm_s2['time_us'].idxmin()]
                key = ('stage2', best_asm_s2['kernel_name'])
                if key not in seen_kernels:
                    seen_kernels.add(key)
                    selected_kernels.append(best_asm_s2)
    
    # === 1-STAGE SELECTION ===
    onestage_results = config_results[config_results['stage'].isin(['asm_1stage', 'triton_1stage'])]
    if len(onestage_results) > 0:
        best_1s = onestage_results.loc[onestage_results['time_us'].idxmin()]
        key = (best_1s['stage'], best_1s['kernel_name'])
        if key not in seen_kernels:
            seen_kernels.add(key)
            selected_kernels.append(best_1s)
    
    return selected_kernels


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
