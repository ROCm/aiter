#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
MOE Profiling Results Analysis Script

Purpose:
    Analyzes profiling results from profile_kernels.py and generates
    performance analyses and visualizations.

Input:
    CSV file with profiling results (includes hardware counters)

Usage:
    The usage is shown in analysis_notebook.py

Analysis Types:
    - roofline: Generate roofline plots
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from typing import Union, List, Dict

# Import plotting libraries
import plotly.graph_objects as go

# Add parent directory to path for moe_utils import
sys.path.insert(0, str(Path(__file__).parent))
import moe_utils


# ============================================================================
# GPU HARDWARE SPECIFICATIONS (Configurable)
# ============================================================================

# Supported GPU specifications
GPU_SPECS = {
    'MI300X': {
        'peak_bw_gb_s': 5300.0,  # GB/s (HBM3 memory bandwidth)
        'peak_compute': {
            "torch.bfloat16": 1307.4,       # TFLOPs/s (Matrix BF16: 2048 FLOPS/clock/CU × 304 CUs × 2.1 GHz)
            "torch.float8_e4m3fnuz": 2614.9,  # TFLOPs/s (Matrix FP8: 4096 FLOPS/clock/CU × 304 CUs × 2.1 GHz)
            "torch.int8": 2614.9,           # TOPs/s (Matrix INT8: 4096 OPS/clock/CU × 304 CUs × 2.1 GHz)
        },
        'gpu_name': 'MI300X'
    },
    'MI350': {
        'peak_bw_gb_s': 6400.0,  # GB/s (estimated HBM3e bandwidth)
        'peak_compute': {
            "torch.bfloat16": 1300.0,       # TFLOPs/s (estimated ~2x MI300X)
            "torch.float8_e4m3fnuz": 2600.0,  # TFLOPs/s (FP8 MFMA)
            "torch.int8": 2600.0,           # TOPs/s (INT8 MFMA)
        },
        'gpu_name': 'MI350'
    },
}

# Default GPU (can be overridden via command line)
DEFAULT_GPU = 'MI300X'


def get_gpu_specs(gpu_name: str = DEFAULT_GPU) -> Dict:
    """
    Get GPU hardware specifications for roofline analysis.
    
    Args:
        gpu_name: GPU model name ('MI300X' or 'MI350')
        
    Returns:
        Dictionary with GPU specifications
    """
    if gpu_name not in GPU_SPECS:
        print(f"Warning: Unknown GPU '{gpu_name}', using {DEFAULT_GPU}")
        gpu_name = DEFAULT_GPU
    
    specs = GPU_SPECS[gpu_name]
    
    # Calculate ridge points from specs
    peak_bw = specs['peak_bw_gb_s']
    peak_compute = specs['peak_compute']
    ridge_points = {dt: (pc * 1000.0) / peak_bw for dt, pc in peak_compute.items()}
    
    return {
        'peak_bw_gb_s': peak_bw,
        'peak_compute': peak_compute,
        'ridge_points': ridge_points,
        'gpu_name': specs['gpu_name']
    }


# ============================================================================
# DATA SUMMARY
# ============================================================================

def generate_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate concise summary statistics of profiling data.
    
    Args:
        df: DataFrame with prepared profiling results (after prepare_roofline_data)
        
    Returns:
        DataFrame with summary statistics in single column
    """
    summary = {
        'Total Kernels': len(df),
        'Kernel Types': ', '.join(sorted(df['kernel_type'].unique())),
        'Data Types': ', '.join(sorted(df['dtype'].unique())),
        'Quantization Types': ', '.join(sorted(df['q_type'].unique())),
        'Token Counts': ', '.join(map(str, sorted(df['token'].unique()))),
        'Model Dims': ', '.join(map(str, sorted(df['model_dim'].unique()))),
        'Inter Dims': ', '.join(map(str, sorted(df['inter_dim'].unique()))),
        'Experts': ', '.join(map(str, sorted(df['expert'].unique()))),
        'Top-K': ', '.join(map(str, sorted(df['topk'].unique()))),
        'Time (µs)': f"{df['time_us'].min():.1f} - {df['time_us'].max():.1f} (avg: {df['time_us'].mean():.1f})",
        'TFLOPs/s': f"{df['tflops_mfma'].min():.2f} - {df['tflops_mfma'].max():.2f} (avg: {df['tflops_mfma'].mean():.2f})",
        'Bandwidth (GB/s)': f"{df['bandwidth_gb'].min():.1f} - {df['bandwidth_gb'].max():.1f} (avg: {df['bandwidth_gb'].mean():.1f})",
        'OI (FLOP/Byte)': f"{df['OI'].min():.2f} - {df['OI'].max():.2f} (avg: {df['OI'].mean():.2f})",
        'Error (%)': f"{df['error_pct'].min():.2f} - {df['error_pct'].max():.2f} (avg: {df['error_pct'].mean():.2f})"
    }
    
    return pd.DataFrame([summary]).T.rename(columns={0: 'Value'})


# ============================================================================
# ROOFLINE ANALYSIS (Modular Functions)
# ============================================================================

def prepare_roofline_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for roofline analysis.
    
    Computes:
    - Operational Intensity (OI): FLOP / Byte
    - Actual TFLOPs/s from MFMA counters
    - Actual Bandwidth from HW counters (replaces theoretical value from CSV)
    - Error percentage as float
    
    Args:
        df: DataFrame with profiling results
        
    Returns:
        DataFrame with added computed columns
    """
    df = df.copy()
    
    # Operational Intensity = FLOP / Byte
    df["OI"] = df["MFMA_FLOPS"] / ((df["FETCH_SIZE"] + df["WRITE_SIZE"]) * 1024)
    
    # TFLOPs/s using actual MFMA operations
    # Formula: TFLOP/s = MFMA_FLOPS / (time_us / 10^6) / 10^12
    #                  = MFMA_FLOPS / time_us / 10^6
    df["tflops_mfma"] = df["MFMA_FLOPS"] / df["time_us"] / 1e6
    
    # Recalculate bandwidth from actual HW counters (FETCH_SIZE + WRITE_SIZE in KB)
    # Formula: Bandwidth (GB/s) = Total_Bytes / time_us / 1000
    #          where Total_Bytes = (FETCH_SIZE + WRITE_SIZE) * 1024
    df["bandwidth_gb"] = ((df["FETCH_SIZE"] + df["WRITE_SIZE"]) * 1024) / df["time_us"] / 1000
    
    # Parse error percentage
    if df["error"].dtype == object:
        df["error_pct"] = df["error"].str.rstrip("%").astype(float)
    else:
        df["error_pct"] = df["error"]
    
    return df


def create_hover_text(row: pd.Series) -> str:
    """Generate detailed hover tooltip text for individual kernels."""
    return (
        f"<b>{row['kernel_name']}</b><br>"
        f"cfg_idx={row['config_idx']}, token={row['token']}, "
        f"mdim={row['model_dim']}, idim={row['inter_dim']}<br>"
        f"expert={row['expert']}, topk={row['topk']}<br>"
        f"dtype={row['dtype']}, q_dtype_a={row['q_dtype_a']}, q_dtype_w={row['q_dtype_w']}<br>"
        f"q_type={row['q_type']}, act={row['act_type']}<br>"
        f"<br><b>Performance:</b><br>"
        f"TFLOPs/s (MFMA actual): {row['tflops_mfma']:.2f}<br>"
        f"BW: {row['bandwidth_gb']:.1f} GB/s<br>"
        f"Time: {row['time_us']:.1f} µs<br>"
        f"Error: {row['error']}<br>"
        f"OI: {row['OI']:.3f} FLOP/Byte"
    )


def aggregate_2stage_kernels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ALL valid 2-stage combinations per config, ranked by performance.
    
    For each config_idx:
    - Creates all valid stage1 + stage2 combinations (matching block_m)
    - Ranks combinations by total time (1st, 2nd, 3rd best, etc.)
    - Computes combined metrics for each combination
    
    Args:
        df: DataFrame with profiling results
        
    Returns:
        DataFrame with all 2-stage combinations, including rank
    """
    df_2stage = df[df['stage'].isin(['stage1', 'stage2'])].copy()
    
    all_combinations = []
    
    for config_idx in df_2stage['config_idx'].unique():
        config_data = df_2stage[df_2stage['config_idx'] == config_idx]
        
        stage1_kernels = config_data[config_data['stage'] == 'stage1']
        stage2_kernels = config_data[config_data['stage'] == 'stage2']
        
        if len(stage1_kernels) == 0 or len(stage2_kernels) == 0:
            continue
        
        # Create all valid combinations for this config
        config_combinations = []
        
        for _, s1 in stage1_kernels.iterrows():
            for _, s2 in stage2_kernels.iterrows():
                # Only combine if block_m matches
                if s1['block_m'] != s2['block_m']:
                    continue
                
                # Aggregate metrics
                total_mfma_flops = s1['MFMA_FLOPS'] + s2['MFMA_FLOPS']
                total_time_us = s1['time_us'] + s2['time_us']
                quant_time_us = s1['quant_time_us']
                total_time_with_quant = total_time_us + quant_time_us
                
                # Memory transfers
                total_fetch = s1['FETCH_SIZE'] + s2['FETCH_SIZE']
                total_write = s1['WRITE_SIZE'] + s2['WRITE_SIZE']
                total_bytes = (total_fetch + total_write) * 1024
                
                # Combined metrics
                combined_oi = total_mfma_flops / total_bytes if total_bytes > 0 else 0
                combined_tflops_mfma = total_mfma_flops / total_time_with_quant / 1e6
                combined_bandwidth_gb = total_bytes / total_time_with_quant / 1000 if total_time_with_quant > 0 else 0
                
                config_combinations.append({
                    'config_idx': config_idx,
                    'kernel_name_stage1': s1['kernel_name'],
                    'kernel_name_stage2': s2['kernel_name'],
                    'kernel_name': f"2-stage: {s1['kernel_type']}/{s2['kernel_type']}",
                    'kernel_type': '2-stage',
                    'stage': '2-stage-combined',
                    'time_us': total_time_with_quant,
                    'time_stage1_us': s1['time_us'],
                    'time_stage2_us': s2['time_us'],
                    'quant_time_us': quant_time_us,
                    'MFMA_FLOPS': total_mfma_flops,
                    'OI': combined_oi,
                    'tflops_mfma': combined_tflops_mfma,
                    'bandwidth_gb': combined_bandwidth_gb,
                    'error': f"{(float(s1['error'].rstrip('%')) + float(s2['error'].rstrip('%'))) / 2:.2f}%",
                    'error_pct': (s1['error_pct'] + s2['error_pct']) / 2,
                    # Preserve metadata
                    'token': s1['token'],
                    'model_dim': s1['model_dim'],
                    'inter_dim': s1['inter_dim'],
                    'expert': s1['expert'],
                    'topk': s1['topk'],
                    'dtype': s1['dtype'],
                    'q_dtype_a': s1['q_dtype_a'],
                    'q_dtype_w': s1['q_dtype_w'],
                    'q_type': s1['q_type'],
                    'act_type': s1['act_type'],
                })
        
        # Rank combinations for this config by total time (fastest = rank 1)
        config_combinations.sort(key=lambda x: x['time_us'])
        for rank, combo in enumerate(config_combinations, 1):
            combo['rank'] = rank
            all_combinations.append(combo)
    
    return pd.DataFrame(all_combinations)


def create_hover_text_2stage(row: pd.Series) -> str:
    """Generate detailed hover tooltip text for 2-stage combined kernels."""
    return (
        f"<b>2-Stage Combined Kernel</b><br>"
        f"<br><b>Stage 1:</b> {row['kernel_name_stage1'][:40]}<br>"
        f"<b>Stage 2:</b> {row['kernel_name_stage2'][:40]}<br>"
        f"<br>cfg_idx={row['config_idx']}, token={row['token']}, "
        f"mdim={row['model_dim']}, idim={row['inter_dim']}<br>"
        f"expert={row['expert']}, topk={row['topk']}<br>"
        f"dtype={row['dtype']}, q_dtype_a={row['q_dtype_a']}, q_dtype_w={row['q_dtype_w']}<br>"
        f"q_type={row['q_type']}, act={row['act_type']}<br>"
        f"<br><b>Combined Performance:</b><br>"
        f"TFLOPs/s (MFMA actual): {row['tflops_mfma']:.2f}<br>"
        f"BW: {row['bandwidth_gb']:.1f} GB/s<br>"
        f"<br><b>Time Breakdown:</b><br>"
        f"Total Time: {row['time_us']:.1f} µs<br>"
        f"  • Stage1: {row['time_stage1_us']:.1f} µs<br>"
        f"  • Stage2: {row['time_stage2_us']:.1f} µs<br>"
        f"  • Quantization: {row['quant_time_us']:.1f} µs<br>"
        f"<br>Avg Error: {row['error']}<br>"
        f"Combined OI: {row['OI']:.3f} FLOP/Byte"
    )


def build_roofline_plot(df: pd.DataFrame,
                       gpu_specs: Dict,
                       color_by: str = "kernel_type",
                       title: str = None) -> go.Figure:
    """
    Build roofline plot with hardware performance boundaries.
    
    Args:
        df: DataFrame with prepared roofline data
        gpu_specs: GPU hardware specifications dictionary
        color_by: Column to use for coloring points
        title: Plot title (auto-generated if None)
        
    Returns:
        Plotly Figure object
    """
    if title is None:
        title = f"{gpu_specs['gpu_name']} MoE Kernels Roofline Plot"
    
    fig = go.Figure()
    
    # Extract specs
    peak_bw = gpu_specs['peak_bw_gb_s']
    peak_compute = gpu_specs['peak_compute']
    ridge_points = gpu_specs['ridge_points']
    
    # Color palette
    palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    # Add data points grouped by color_by column
    for idx, (grp, gdf) in enumerate(df.groupby(color_by, sort=True)):
        hover = gdf.apply(create_hover_text, axis=1)
        
        fig.add_trace(go.Scatter(
            x=gdf["OI"],
            y=gdf["tflops_mfma"],
            mode="markers",
            name=str(grp)[:30],
            marker=dict(
                size=8,
                color=palette[idx % len(palette)],
                line=dict(width=0.5, color='white'),
                opacity=0.75
            ),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            showlegend=True
        ))
    
    # Determine x-axis range
    oi_min = df["OI"].min()
    oi_max = df["OI"].max()
    all_ridge_vals = list(ridge_points.values())
    x_min = max(1e-3, min(oi_min * 0.7, min(all_ridge_vals) * 0.5))
    x_max = max(oi_max * 1.3, max(all_ridge_vals) * 2.0)
    x_range = np.logspace(np.log10(x_min), np.log10(x_max), 300)
    
    # Draw roofline boundaries for each dtype
    colors_roof = {
        "torch.bfloat16": "red",
        "torch.float8_e4m3fnuz": "orange",
        "torch.int8": "brown"
    }
    
    for dtype in sorted(peak_compute.keys()):
        peak_tflops = peak_compute[dtype]
        ridge_oi = ridge_points[dtype]
        dt_label = dtype.split(".")[-1] if "." in dtype else dtype
        color = colors_roof.get(dtype, "red")
        
        # Memory-bound roofline (diagonal line up to ridge)
        mem_x = x_range[x_range <= ridge_oi]
        mem_y = (peak_bw / 1000.0) * mem_x
        if len(mem_x) > 0:
            fig.add_trace(go.Scatter(
                x=mem_x, y=mem_y,
                mode="lines",
                name=f"Mem roof ({dt_label}): {peak_bw} GB/s",
                line=dict(color=color, width=2.5, dash="dash"),
                showlegend=True,
                hovertemplate=f"{dtype}<br>Memory-bound<br>BW={peak_bw} GB/s<extra></extra>"
            ))
        
        # Compute-bound roofline (horizontal line from ridge onward)
        comp_x = x_range[x_range >= ridge_oi]
        comp_y = np.full_like(comp_x, peak_tflops)
        if len(comp_x) > 0:
            fig.add_trace(go.Scatter(
                x=comp_x, y=comp_y,
                mode="lines",
                name=f"Comp roof ({dt_label}): {peak_tflops} TFLOPs/s",
                line=dict(color=color, width=2.5, dash="dot"),
                showlegend=True,
                hovertemplate=f"{dtype}<br>Compute-bound<br>Peak={peak_tflops} TFLOPs/s<extra></extra>"
            ))
        
        # Ridge point marker
        fig.add_trace(go.Scatter(
            x=[ridge_oi], y=[peak_tflops],
            mode="markers+text",
            name=f"Ridge ({dt_label}): {ridge_oi:.1f} FLOP/Byte",
            marker=dict(symbol="star", size=14, color="darkred",
                       line=dict(width=2, color="white")),
            text=[f"{ridge_oi:.1f}"],
            textposition="top center",
            textfont=dict(size=11, color="darkred"),
            showlegend=True,
            hovertemplate=f"{dtype}<br>Ridge: {ridge_oi:.2f} FLOP/Byte<br>Peak: {peak_tflops} TFLOPs/s<extra></extra>"
        ))
        
        # Vertical line at ridge
        fig.add_vline(
            x=ridge_oi,
            line_dash="dot",
            line_color=color,
            line_width=1.5,
            opacity=0.5,
            annotation_text=f"{dt_label}",
            annotation_position="top"
        )
    
    # Layout
    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Operational Intensity (FLOP/Byte)",
            type="log",
            showgrid=True,
            gridcolor="lightgray",
            range=[np.log10(x_min), np.log10(x_max)]
        ),
        yaxis=dict(
            title="Performance (TFLOPs/s - MFMA Actual)",
            type="log",
            showgrid=True,
            gridcolor="lightgray"
        ),
        width=1400,
        height=850,
        template="plotly_white",
        hovermode="closest",
        legend=dict(
            x=0.02, y=0.98,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1
        )
    )
    
    return fig


def build_roofline_2stage_plot(df_individual: pd.DataFrame,
                               df_combined: pd.DataFrame,
                               gpu_specs: Dict,
                               title: str = None) -> go.Figure:
    """
    Build roofline plot showing both individual and 2-stage combined kernels.
    
    Args:
        df_individual: DataFrame with individual kernel results
        df_combined: DataFrame with 2-stage aggregated results
        gpu_specs: GPU hardware specifications dictionary
        title: Plot title (auto-generated if None)
        
    Returns:
        Plotly Figure object
    """
    if title is None:
        title = f"{gpu_specs['gpu_name']} MoE Roofline - Complete Landscape"
    
    fig = go.Figure()
    
    # Extract specs
    peak_bw = gpu_specs['peak_bw_gb_s']
    peak_compute = gpu_specs['peak_compute']
    ridge_points = gpu_specs['ridge_points']
    
    # Filter for 1-stage kernels only (complete MoE runs)
    df_1stage = df_individual[df_individual['stage'] == 'asm_1stage'].copy()
    
    # Add 1-stage kernels (blue circles)
    if len(df_1stage) > 0:
        hover = df_1stage.apply(create_hover_text, axis=1)
        
        fig.add_trace(go.Scatter(
            x=df_1stage["OI"],
            y=df_1stage["tflops_mfma"],
            mode="markers",
            name="1-stage MoE",
            marker=dict(
                size=8,
                color='#1f77b4',
                line=dict(width=0.5, color='white'),
                opacity=0.7,
                symbol='circle'
            ),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            showlegend=True
        ))
    
    # Add 2-stage combined kernels with rank-based coloring
    if len(df_combined) > 0:
        # Color palette for ranks (1st, 2nd, 3rd, 4th+)
        rank_colors = {
            1: '#1f77b4',  # Blue - 1st best
            2: '#ff7f0e',  # Orange - 2nd best
            3: '#2ca02c',  # Green - 3rd best
        }
        default_color = '#d62728'  # Red - 4th+ best
        
        # Group by rank and plot
        for rank in sorted(df_combined['rank'].unique()):
            rank_data = df_combined[df_combined['rank'] == rank]
            color = rank_colors.get(rank, default_color)
            
            hover = rank_data.apply(
                lambda r: (
                    f"<b>Rank #{r['rank']} for Config</b><br>"
                    f"<b>Stage 1:</b> {r['kernel_name_stage1'][:35]}<br>"
                    f"<b>Stage 2:</b> {r['kernel_name_stage2'][:35]}<br>"
                    f"<br>cfg_idx={r['config_idx']}, token={r['token']}, "
                    f"mdim={r['model_dim']}, idim={r['inter_dim']}<br>"
                    f"<br><b>Performance:</b><br>"
                    f"TFLOPs/s: {r['tflops_mfma']:.2f}<br>"
                    f"Total Time: {r['time_us']:.1f} µs<br>"
                    f"  • Stage1: {r['time_stage1_us']:.1f} µs<br>"
                    f"  • Stage2: {r['time_stage2_us']:.1f} µs<br>"
                    f"OI: {r['OI']:.3f} FLOP/Byte"
                ),
                axis=1
            )
            
            rank_label = f"Rank {rank}" if rank <= 3 else f"Rank 4+"
            
            fig.add_trace(go.Scatter(
                x=rank_data["OI"],
                y=rank_data["tflops_mfma"],
                mode="markers",
                name=f"2-stage ({rank_label})",
                marker=dict(
                    size=8,
                    color=color,
                    line=dict(width=0.5, color='white'),
                    opacity=0.75,
                    symbol='square'
                ),
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                showlegend=True
            ))
    
    # OI range from both datasets
    if len(df_1stage) > 0 and len(df_combined) > 0:
        oi_min = min(df_1stage["OI"].min(), df_combined["OI"].min())
        oi_max = max(df_1stage["OI"].max(), df_combined["OI"].max())
    elif len(df_1stage) > 0:
        oi_min = df_1stage["OI"].min()
        oi_max = df_1stage["OI"].max()
    elif len(df_combined) > 0:
        oi_min = df_combined["OI"].min()
        oi_max = df_combined["OI"].max()
    else:
        oi_min, oi_max = 1, 100
    
    all_ridge_vals = list(ridge_points.values())
    x_min = max(1e-3, min(oi_min * 0.7, min(all_ridge_vals) * 0.5))
    x_max = max(oi_max * 1.3, max(all_ridge_vals) * 2.0)
    x_range = np.logspace(np.log10(x_min), np.log10(x_max), 300)
    
    # Draw rooflines
    colors_roof = {"torch.bfloat16": "red", "torch.float8_e4m3fnuz": "orange", "torch.int8": "brown"}
    
    for dtype in sorted(peak_compute.keys()):
        peak_tflops = peak_compute[dtype]
        ridge_oi = ridge_points[dtype]
        dt_label = dtype.split(".")[-1] if "." in dtype else dtype
        color = colors_roof.get(dtype, "red")
        
        # Memory-bound
        mem_x = x_range[x_range <= ridge_oi]
        mem_y = (peak_bw / 1000.0) * mem_x
        if len(mem_x) > 0:
            fig.add_trace(go.Scatter(
                x=mem_x, y=mem_y, mode="lines",
                name=f"Mem roof ({dt_label})",
                line=dict(color=color, width=2.5, dash="dash"),
                showlegend=True
            ))
        
        # Compute-bound
        comp_x = x_range[x_range >= ridge_oi]
        comp_y = np.full_like(comp_x, peak_tflops)
        if len(comp_x) > 0:
            fig.add_trace(go.Scatter(
                x=comp_x, y=comp_y, mode="lines",
                name=f"Comp roof ({dt_label})",
                line=dict(color=color, width=2.5, dash="dot"),
                showlegend=True
            ))
        
        # Ridge marker
        fig.add_trace(go.Scatter(
            x=[ridge_oi], y=[peak_tflops],
            mode="markers+text",
            name=f"Ridge ({dt_label})",
            marker=dict(symbol="star", size=14, color="darkred",
                       line=dict(width=2, color="white")),
            text=[f"{ridge_oi:.1f}"],
            textposition="top center",
            showlegend=True
        ))
        
        fig.add_vline(x=ridge_oi, line_dash="dot", line_color=color,
                     line_width=1.5, opacity=0.5,
                     annotation_text=f"{dt_label}", annotation_position="top")
    
    # Layout
    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Operational Intensity (FLOP/Byte)",
            type="log",
            showgrid=True,
            gridcolor="lightgray",
            range=[np.log10(x_min), np.log10(x_max)]
        ),
        yaxis=dict(
            title="Performance (TFLOPs/s - MFMA Actual)",
            type="log",
            showgrid=True,
            gridcolor="lightgray"
        ),
        width=1400,
        height=850,
        template="plotly_white",
        hovermode="closest",
        legend=dict(
            x=0.02, y=0.98,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1
        )
    )
    
    return fig


def create_config_label(row: pd.Series, group_by_cols: List[str] = None) -> str:
    """
    Create configuration label for grouping.
    
    Args:
        row: DataFrame row
        group_by_cols: List of columns to include in label. If None, uses all config columns.
    
    Returns:
        Formatted configuration label string
    """
    if group_by_cols is None:
        group_by_cols = ['token', 'model_dim', 'inter_dim', 'expert', 'topk', 
                        'dtype', 'q_dtype_a', 'q_dtype_w', 'q_type', 'act_type']
    
    # Create shortened column names for compact labels
    col_abbrev = {
        'token': 't', 'model_dim': 'md', 'inter_dim': 'id', 'expert': 'exp',
        'topk': 'k', 'dtype': 'dt', 'q_dtype_a': 'qa', 'q_dtype_w': 'qw',
        'q_type': 'qt', 'act_type': 'act'
    }
    
    parts = []
    for col in group_by_cols:
        if col in row.index:
            abbrev = col_abbrev.get(col, col)
            val = str(row[col]).replace('torch.', '').replace('QuantType.', '').replace('ActivationType.', '')
            parts.append(f"{abbrev}={val}")
    
    return ", ".join(parts)


def calculate_efficiency_gap(df: pd.DataFrame, gpu_specs: Dict) -> pd.DataFrame:
    """
    Calculate roofline efficiency for each kernel.
    
    For each kernel, determines if it's memory-bound or compute-bound,
    then calculates the efficiency percentage relative to the roofline.
    
    Args:
        df: DataFrame with OI, tflops_mfma, and dtype columns
        gpu_specs: GPU hardware specifications
        
    Returns:
        DataFrame with added efficiency_pct and bound_type columns
    """
    df = df.copy()
    
    efficiencies = []
    bound_types = []
    roofline_perfs = []
    
    peak_bw = gpu_specs['peak_bw_gb_s']
    peak_compute = gpu_specs['peak_compute']
    ridge_points = gpu_specs['ridge_points']
    
    for _, row in df.iterrows():
        dtype = row['dtype']
        oi = row['OI']
        actual_perf = row['tflops_mfma']
        
        # Get ridge point for this dtype
        ridge = ridge_points.get(dtype, ridge_points['torch.bfloat16'])
        
        if oi < ridge:
            # Memory-bound: roofline = (BW / 1000) × OI
            roofline_perf = (peak_bw / 1000.0) * oi
            bound_type = 'memory'
        else:
            # Compute-bound: roofline = Peak_Compute
            roofline_perf = peak_compute.get(dtype, peak_compute['torch.bfloat16'])
            bound_type = 'compute'
        
        # Calculate efficiency as percentage of roofline (100% = at roofline, 0% = no performance)
        efficiency_pct = min(100, (actual_perf / roofline_perf) * 100)
        
        efficiencies.append(efficiency_pct)
        bound_types.append(bound_type)
        roofline_perfs.append(roofline_perf)
    
    df['efficiency_pct'] = efficiencies
    df['bound_type'] = bound_types
    df['roofline_perf'] = roofline_perfs
    
    return df


def build_efficiency_gap_plot(df: pd.DataFrame,
                              gpu_specs: Dict,
                              x_axis: str = 'token',
                              color_by: str = 'kernel_type',
                              use_config_grouping: bool = False,
                              group_by_cols: List[str] = None,
                              title: str = None) -> go.Figure:
    """
    Build roofline efficiency plot.
    
    Args:
        df: DataFrame with efficiency data
        gpu_specs: GPU hardware specifications
        x_axis: Column to use for x-axis (token, model_dim, inter_dim, etc.) - ignored if use_config_grouping=True
        color_by: Column to use for coloring points
        use_config_grouping: If True, group by configuration columns instead of single parameter
        group_by_cols: List of columns to group by when use_config_grouping=True. If None, uses all columns.
        title: Plot title (auto-generated if None)
        
    Returns:
        Plotly Figure object
    """
    if title is None:
        title = f"{gpu_specs['gpu_name']} MoE Kernels - Roofline Efficiency"
    
    # Create config labels if needed
    df = df.copy()
    if use_config_grouping:
        df['config_label'] = df.apply(lambda r: create_config_label(r, group_by_cols), axis=1)
        x_col = 'config_label'
        x_title = "Configuration"
        tick_angle = -45
    else:
        x_col = x_axis
        x_title = f"{x_axis}"
        tick_angle = 0
    
    fig = go.Figure()
    
    # Color palette
    palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    # Group by color_by and bound_type for different markers
    for idx, (grp, gdf) in enumerate(df.groupby(color_by, sort=True)):
        color = palette[idx % len(palette)]
        
        # Memory-bound kernels (circles)
        memory_bound = gdf[gdf['bound_type'] == 'memory']
        if len(memory_bound) > 0:
            hover = memory_bound.apply(
                lambda r: (
                    f"<b>{r['kernel_name']}</b><br>"
                    f"<b>Memory-Bound</b><br>"
                    f"cfg_idx={r['config_idx']}, token={r['token']}, "
                    f"mdim={r['model_dim']}, idim={r['inter_dim']}<br>"
                    f"expert={r['expert']}, topk={r['topk']}<br>"
                    f"dtype={r['dtype']}, q_dtype_a={r['q_dtype_a']}, q_dtype_w={r['q_dtype_w']}<br>"
                    f"q_type={r['q_type']}, act={r['act_type']}<br>"
                    f"<br><b>Performance:</b><br>"
                    f"Efficiency: {r['efficiency_pct']:.1f}%<br>"
                    f"TFLOPs/s (actual): {r['tflops_mfma']:.2f}<br>"
                    f"Roofline: {r['roofline_perf']:.2f} TFLOPs/s<br>"
                    f"BW: {r['bandwidth_gb']:.1f} GB/s<br>"
                    f"Time: {r['time_us']:.1f} µs<br>"
                    f"Error: {r['error']}<br>"
                    f"OI: {r['OI']:.3f} FLOP/Byte"
                ),
                axis=1
            )
            
            fig.add_trace(go.Scatter(
                x=memory_bound[x_col],
                y=memory_bound['efficiency_pct'],
                mode='markers',
                name=f"{str(grp)[:25]} (mem)",
                marker=dict(
                    size=10,
                    color=color,
                    symbol='circle',
                    line=dict(width=0.5, color='white'),
                    opacity=0.75
                ),
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                showlegend=True
            ))
        
        # Compute-bound kernels (squares)
        compute_bound = gdf[gdf['bound_type'] == 'compute']
        if len(compute_bound) > 0:
            hover = compute_bound.apply(
                lambda r: (
                    f"<b>{r['kernel_name']}</b><br>"
                    f"<b>Compute-Bound</b><br>"
                    f"cfg_idx={r['config_idx']}, token={r['token']}, "
                    f"mdim={r['model_dim']}, idim={r['inter_dim']}<br>"
                    f"expert={r['expert']}, topk={r['topk']}<br>"
                    f"dtype={r['dtype']}, q_dtype_a={r['q_dtype_a']}, q_dtype_w={r['q_dtype_w']}<br>"
                    f"q_type={r['q_type']}, act={r['act_type']}<br>"
                    f"<br><b>Performance:</b><br>"
                    f"Efficiency: {r['efficiency_pct']:.1f}%<br>"
                    f"TFLOPs/s (actual): {r['tflops_mfma']:.2f}<br>"
                    f"Roofline: {r['roofline_perf']:.2f} TFLOPs/s<br>"
                    f"BW: {r['bandwidth_gb']:.1f} GB/s<br>"
                    f"Time: {r['time_us']:.1f} µs<br>"
                    f"Error: {r['error']}<br>"
                    f"OI: {r['OI']:.3f} FLOP/Byte"
                ),
                axis=1
            )
            
            fig.add_trace(go.Scatter(
                x=compute_bound[x_col],
                y=compute_bound['efficiency_pct'],
                mode='markers',
                name=f"{str(grp)[:25]} (comp)",
                marker=dict(
                    size=10,
                    color=color,
                    symbol='square',
                    line=dict(width=0.5, color='white'),
                    opacity=0.75
                ),
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                showlegend=True
            ))
    
    # Get unique x-axis values and sort them
    unique_x_vals = sorted(df[x_col].unique())
    
    # Layout
    fig.update_layout(
        title=title,
        xaxis=dict(
            title=x_title,
            showgrid=True,
            gridcolor="lightgray",
            type='category',
            categoryorder='array',
            categoryarray=unique_x_vals,
            tickangle=tick_angle
        ),
        yaxis=dict(
            title="Roofline Efficiency (%)",
            showgrid=True,
            gridcolor="lightgray",
            range=[0, 100]
        ),
        width=1400,
        height=850,
        template="plotly_white",
        hovermode="closest",
        legend=dict(
            x=0.02, y=0.98,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1
        )
    )
    
    # Add reference line at 100% (perfect efficiency)
    fig.add_hline(
        y=100,
        line_dash="dash",
        line_color="green",
        line_width=2,
        opacity=0.5,
        annotation_text="100% Efficiency (at roofline)",
        annotation_position="right"
    )
    
    return fig


def generate_roofline_analysis(input_file: str, output_dir: Union[str, Path], gpu_name: str = DEFAULT_GPU) -> None:
    """
    Generate roofline analysis plots and save as HTML files.
    
    Args:
        input_file: Path to profiling results CSV
        output_dir: Directory to save output files (str or Path)
        gpu_name: GPU model name for hardware specs
    """
    moe_utils.print_section_header("ROOFLINE ANALYSIS")
    
    # Convert output_dir to Path if it's a string
    output_dir = Path(output_dir)
    
    # Get GPU specifications
    gpu_specs = get_gpu_specs(gpu_name)
    print(f"Using GPU specs: {gpu_specs['gpu_name']}")
    print(f"  Peak bandwidth: {gpu_specs['peak_bw_gb_s']} GB/s")
    print(f"  Peak compute (BF16): {gpu_specs['peak_compute']['torch.bfloat16']} TFLOPs/s")
    print(f"  Peak compute (FP8): {gpu_specs['peak_compute']['torch.float8_e4m3fnuz']} TFLOPs/s")
    
    # Load and prepare data
    print(f"\nLoading profiling data: {input_file}")
    df = pd.read_csv(input_file)
    df = prepare_roofline_data(df)
    print(f"Loaded {len(df)} profiled kernels")
    
    # Generate individual kernel roofline plot
    print("\nGenerating individual kernel roofline plot...")
    fig_individual = build_roofline_plot(
        df,
        gpu_specs,
        color_by="kernel_type",
        title=f"{gpu_specs['gpu_name']} MoE Kernels Roofline (Individual)"
    )
    
    individual_html = output_dir / "roofline_individual.html"
    fig_individual.write_html(
        str(individual_html),
        config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True}
    )
    print(f"  Saved: {individual_html}")
    
    # Generate 2-stage combined roofline plot
    print("\nGenerating 2-stage combined roofline plot...")
    df_combined = aggregate_2stage_kernels(df)
    
    if len(df_combined) > 0:
        print(f"  Aggregated {len(df_combined)} 2-stage kernel pairs")
        
        fig_combined = build_roofline_2stage_plot(
            df,
            df_combined,
            gpu_specs,
            title=f"{gpu_specs['gpu_name']} MoE Roofline - Complete Landscape"
        )
        
        combined_html = output_dir / "roofline_2stage.html"
        fig_combined.write_html(
            str(combined_html),
            config={'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True}
        )
        print(f"  Saved: {combined_html}")
    else:
        print("  No 2-stage kernel pairs found in data")
    
    print("\nRoofline analysis complete!")


# ============================================================================
# SUMMARY STATISTICS (Future Extension Point)
# ============================================================================

def generate_summary_statistics(input_file: str, output_dir: Path) -> None:
    """
    Generate performance summary statistics.
    
    Args:
        input_file: Path to profiling results CSV
        output_dir: Directory to save output files
    """
    moe_utils.print_section_header("SUMMARY STATISTICS")
    print("Feature not yet implemented - placeholder for future extension")
    # TODO: Add statistical analysis, best/worst performers, etc.


# ============================================================================
# MAIN ANALYSIS FRAMEWORK
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze MOE kernel profiling results"
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input CSV file with profiling results from 02_profile_kernels.py"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="results/analysis",
        help="Output directory for analysis results (default: results/analysis)"
    )
    parser.add_argument(
        "--analyses",
        nargs='+',
        default=['roofline'],
        choices=['roofline', 'summary', 'all'],
        help="Analysis types to run (default: roofline)"
    )
    parser.add_argument(
        "--gpu",
        default=DEFAULT_GPU,
        choices=list(GPU_SPECS.keys()),
        help=f"GPU model for hardware specs (default: {DEFAULT_GPU})"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Expand 'all' analysis type
    if 'all' in args.analyses:
        analyses = ['roofline', 'summary']
    else:
        analyses = args.analyses
    
    moe_utils.print_section_header("MOE PROFILING ANALYSIS")
    print(f"Input: {args.input}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Analyses to run: {', '.join(analyses)}")
    
    # Verify input file exists
    if not Path(args.input).exists():
        print(f"\nERROR: Input file not found: {args.input}")
        return 1
    
    # Run requested analyses
    for analysis_type in analyses:
        if analysis_type == 'roofline':
            generate_roofline_analysis(args.input, output_dir, args.gpu)
        elif analysis_type == 'summary':
            generate_summary_statistics(args.input, output_dir)
    
    # Final summary
    moe_utils.print_section_header("ANALYSIS COMPLETE")
    print(f"Output directory: {output_dir.absolute()}")
    print("\nGenerated files:")
    for html_file in output_dir.glob("*.html"):
        print(f"  - {html_file.name}")
    for csv_file in output_dir.glob("*.csv"):
        print(f"  - {csv_file.name}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
