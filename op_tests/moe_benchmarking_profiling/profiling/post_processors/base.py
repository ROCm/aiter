#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Abstract base class for post-processors.

Provides common logic for processing profiling results:
- Reading counters.csv and trace_kernel_trace.csv
- Computing execution time from timestamps
- Filtering to target kernel
- Adding common columns (cfg_idx, kernel_name, execution_time_us, etc.)

Operator-specific subclasses implement:
- Kernel name matching pattern
- Kernel name from config
- Operator-specific columns
"""

import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from profiling.script_generators.base import ScriptConfig


# Columns to drop from trace/counters output
COLUMNS_TO_DROP = [
    'Kind',
    'Agent_Id', 
    'Queue_Id',
    'Stream_Id',
    'Thread_Id',
    'Kernel_Id',
    'Kernel_Name',  # We use kernel_name from config instead
    'Correlation_Id',
    'Dispatch_Id',
]


@dataclass
class ProcessedResult:
    """Result from processing a single kernel's profiling output."""
    success: bool
    message: str
    cfg_idx: int
    df: Optional[pd.DataFrame] = None
    num_dispatches: int = 0


class PostProcessor(ABC):
    """
    Abstract base class for post-processing profiling results.
    
    Subclasses implement operator-specific logic for:
    - Finding target kernel in trace/counters
    - Providing kernel name from config
    - Adding operator-specific columns
    
    Usage:
        processor = MoePostProcessor()
        result_df = processor.combine_results(output_dir, configs)
    """
    
    @abstractmethod
    def get_kernel_search_pattern(self, config: ScriptConfig) -> str:
        """
        Return pattern to find target kernel in Kernel_Name column.
        
        The pattern is used with str.contains() for substring matching.
        
        Args:
            config: The script configuration
            
        Returns:
            Search pattern string
        """
        pass
    
    @abstractmethod
    def get_kernel_name(self, config: ScriptConfig) -> Optional[str]:
        """
        Return kernel name from config for output CSV.
        
        If None is returned, the demangled name from trace will be used.
        
        Args:
            config: The script configuration
            
        Returns:
            Kernel name string or None to use trace name
        """
        pass
    
    @abstractmethod
    def get_cfg_idx(self, config: ScriptConfig) -> int:
        """
        Return the configuration index for this kernel.
        
        For operators with config_idx in their config, return that.
        For others, return the sequential script index.
        
        Args:
            config: The script configuration
            
        Returns:
            Configuration index
        """
        pass
    
    @abstractmethod
    def get_config_columns(self, config: ScriptConfig) -> Dict[str, Any]:
        """
        Return operator-specific columns to add to output.
        
        These columns are specific to the operator (e.g., token, expert for MOE).
        Do NOT include cfg_idx or kernel_name - those are handled separately.
        
        Args:
            config: The script configuration
            
        Returns:
            Dictionary of column_name -> value
        """
        pass
    
    def process_single(
        self,
        kernel_dir: Path,
        config: ScriptConfig,
        fallback_idx: int,
    ) -> ProcessedResult:
        """
        Process profiling results for a single kernel.
        
        Reads counters.csv and trace_kernel_trace.csv, filters to target kernel,
        computes execution time, and adds common + operator-specific columns.
        
        Args:
            kernel_dir: Directory containing counters.csv and trace_kernel_trace.csv
            config: The script configuration
            fallback_idx: Fallback index if operator doesn't provide cfg_idx
            
        Returns:
            ProcessedResult with combined DataFrame
        """
        counters_file = kernel_dir / "counters.csv"
        trace_file = kernel_dir / "trace_kernel_trace.csv"
        
        # Get cfg_idx from operator (or use fallback)
        cfg_idx = self.get_cfg_idx(config)
        if cfg_idx is None:
            cfg_idx = fallback_idx
        
        # Check files exist
        if not counters_file.exists():
            return ProcessedResult(
                success=False,
                message="counters.csv not found",
                cfg_idx=cfg_idx,
            )
        
        if not trace_file.exists():
            return ProcessedResult(
                success=False,
                message="trace_kernel_trace.csv not found",
                cfg_idx=cfg_idx,
            )
        
        try:
            df_counters = pd.read_csv(counters_file)
            df_trace = pd.read_csv(trace_file)
        except Exception as e:
            return ProcessedResult(
                success=False,
                message=f"Error reading CSV: {e}",
                cfg_idx=cfg_idx,
            )
        
        # Get search pattern from operator
        search_pattern = self.get_kernel_search_pattern(config)
        
        # Filter trace to target kernel
        trace_mask = df_trace['Kernel_Name'].str.contains(
            search_pattern, regex=False, na=False
        )
        df_trace_filtered = df_trace[trace_mask].copy()
        
        if len(df_trace_filtered) == 0:
            return ProcessedResult(
                success=False,
                message=f"Target kernel not found (searched: '{search_pattern}')",
                cfg_idx=cfg_idx,
            )
        
        # Filter counters to target kernel
        counters_mask = df_counters['Kernel_Name'].str.contains(
            search_pattern, regex=False, na=False
        )
        df_counters_filtered = df_counters[counters_mask].copy()
        
        if len(df_counters_filtered) == 0:
            return ProcessedResult(
                success=False,
                message=f"Target kernel not found in counters (searched: '{search_pattern}')",
                cfg_idx=cfg_idx,
            )
        
        # Get kernel name from operator (or use first match from trace)
        kernel_name = self.get_kernel_name(config)
        if kernel_name is None:
            kernel_name = df_trace_filtered['Kernel_Name'].iloc[0]
        
        # Compute execution time from timestamps (nanoseconds -> microseconds)
        if 'Start_Timestamp' in df_trace_filtered.columns and 'End_Timestamp' in df_trace_filtered.columns:
            df_trace_filtered['execution_time_us'] = (
                df_trace_filtered['End_Timestamp'] - df_trace_filtered['Start_Timestamp']
            ) / 1000.0
        
        # Add common columns
        df_trace_filtered['cfg_idx'] = cfg_idx
        df_trace_filtered['kernel_name'] = kernel_name
        
        # Add operator-specific columns
        config_columns = self.get_config_columns(config)
        for col_name, col_value in config_columns.items():
            df_trace_filtered[col_name] = col_value
        
        # Merge counters into trace on Dispatch_Id if both have it
        if 'Dispatch_Id' in df_trace_filtered.columns and 'Dispatch_Id' in df_counters_filtered.columns:
            # Get counter columns (exclude those already in trace or to be dropped)
            trace_cols = set(df_trace_filtered.columns)
            counter_cols = ['Dispatch_Id'] + [
                c for c in df_counters_filtered.columns 
                if c not in trace_cols and c != 'Kernel_Name' and c not in COLUMNS_TO_DROP
            ]
            
            df_merged = df_trace_filtered.merge(
                df_counters_filtered[counter_cols],
                on='Dispatch_Id',
                how='left',
                suffixes=('', '_counter')
            )
        else:
            df_merged = df_trace_filtered
        
        # Drop unwanted columns
        cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df_merged.columns]
        df_merged = df_merged.drop(columns=cols_to_drop)
        
        # Reorder columns: cfg_idx, kernel_name, execution_time_us, operator columns, then rest
        common_cols = ['cfg_idx', 'kernel_name', 'execution_time_us']
        operator_cols = list(config_columns.keys())
        
        existing_common = [c for c in common_cols if c in df_merged.columns]
        existing_operator = [c for c in operator_cols if c in df_merged.columns]
        other_cols = [c for c in df_merged.columns if c not in existing_common + existing_operator]
        
        col_order = existing_common + existing_operator + other_cols
        df_merged = df_merged[col_order]
        
        return ProcessedResult(
            success=True,
            message=f"Processed {len(df_merged)} dispatches",
            cfg_idx=cfg_idx,
            df=df_merged,
            num_dispatches=len(df_merged),
        )
    
    def process_and_append(
        self,
        kernel_dir: Path,
        config: ScriptConfig,
        combined_file: Path,
        fallback_idx: int,
        cleanup: bool = True,
    ) -> ProcessedResult:
        """
        Process single kernel and append results to combined CSV.
        
        This method enables incremental processing as each kernel completes,
        rather than waiting for all kernels to finish.
        
        Args:
            kernel_dir: Directory containing counters.csv and trace_kernel_trace.csv
            config: The script configuration
            combined_file: Path to combined results CSV (created if doesn't exist)
            fallback_idx: Fallback index if operator doesn't provide cfg_idx
            cleanup: If True, remove kernel_dir after processing
            
        Returns:
            ProcessedResult with processing status
        """
        result = self.process_single(kernel_dir, config, fallback_idx)
        
        if result.success and result.df is not None:
            combined_file = Path(combined_file)
            combined_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Append to CSV (create with header if first write)
            write_header = not combined_file.exists()
            result.df.to_csv(
                combined_file, 
                mode='a', 
                header=write_header, 
                index=False
            )
            
            # Cleanup intermediate files
            if cleanup:
                shutil.rmtree(kernel_dir)
        
        return result
    
    def combine_results(
        self,
        output_dir: Path,
        configs: List[ScriptConfig],
        cleanup: bool = True,
    ) -> Tuple[pd.DataFrame, List[ProcessedResult]]:
        """
        Combine profiling results from multiple kernels.
        
        Processes each kernel directory, combines into single DataFrame,
        and optionally cleans up intermediate files.
        
        Args:
            output_dir: Directory containing per-kernel subdirectories
            configs: List of script configurations (in order they were profiled)
            cleanup: If True, remove per-kernel directories after processing
            
        Returns:
            Tuple of (combined DataFrame, list of ProcessedResults)
        """
        output_dir = Path(output_dir)
        results = []
        dfs = []
        
        for idx, config in enumerate(configs):
            kernel_dir = output_dir / config.unique_id
            
            result = self.process_single(kernel_dir, config, fallback_idx=idx)
            results.append(result)
            
            if result.success and result.df is not None:
                dfs.append(result.df)
                
                # Cleanup if requested
                if cleanup:
                    shutil.rmtree(kernel_dir)
        
        # Combine all DataFrames
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
        else:
            combined_df = pd.DataFrame()
        
        return combined_df, results
