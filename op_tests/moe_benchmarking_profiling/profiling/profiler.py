#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Generic rocprofv3-based kernel profiler.

This module provides the Profiler class which uses rocprofv3 to collect
hardware performance counters and kernel traces for any Python script.
"""

import subprocess
import sys
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
from datetime import datetime

import pandas as pd

from .gpu_utils import get_gfx_arch, validate_arch


@dataclass
class ProfileResult:
    """Result from profiling a script."""
    success: bool
    message: str
    script_path: Optional[Path] = None
    output_dir: Optional[Path] = None
    counters_file: Optional[Path] = None
    trace_file: Optional[Path] = None
    duration_seconds: float = 0.0


class Profiler:
    """
    Generic rocprofv3-based profiler for GPU kernels.
    
    Runs rocprofv3 with hardware counter collection and kernel tracing
    for any Python script. Outputs counters.csv and trace_kernel_trace.csv.
    
    Usage:
        profiler = Profiler(arch='MI300X')
        result = profiler.profile(script_path, output_dir)
    """
    
    def __init__(
        self,
        arch: str = 'MI300X',
        timeout: int = 600,
        verbose: bool = False,
    ):
        """
        Initialize the Profiler.
        
        Args:
            arch: GPU architecture (MI250X, MI300A, MI300X, MI355X)
            timeout: Timeout in seconds for profiling a single script
            verbose: If True, print detailed progress
        """
        self.arch = validate_arch(arch)
        self.gfx_arch = get_gfx_arch(self.arch)
        self.timeout = timeout
        self.verbose = verbose
        
        # Locate counter file in package
        self.counters_dir = Path(__file__).parent / "counters"
        self.counter_file = self.counters_dir / f"roof-counters-{self.gfx_arch}.txt"
        
        if not self.counter_file.exists():
            raise FileNotFoundError(
                f"Counter file not found: {self.counter_file}. "
                f"Available: {list(self.counters_dir.glob('roof-counters-*.txt'))}"
            )
    
    def _run_command(
        self,
        cmd: List[str],
        cwd: Optional[Path] = None,
        timeout: Optional[int] = None,
    ) -> subprocess.CompletedProcess:
        """Run a command with timeout and capture output."""
        return subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout or self.timeout,
        )
    
    def _run_rocprofv3_with_retry(
        self,
        cmd_with_csv: List[str],
        cmd_without_csv: List[str],
        cwd: Optional[Path] = None,
    ) -> subprocess.CompletedProcess:
        """
        Run rocprofv3 with -f csv flag, retry without if not supported.
        
        Pre-ROCm 7 doesn't support -f csv flag.
        """
        result = self._run_command(cmd_with_csv, cwd=cwd)
        
        # Check if -f flag not recognized (pre-ROCm 7)
        if result.returncode != 0 and result.stderr:
            stderr_lower = result.stderr.lower()
            flag_issues = [
                'unrecognized option', 'invalid option', 'unknown option',
                'unrecognized argument', '-f'
            ]
            if any(phrase in stderr_lower for phrase in flag_issues):
                if self.verbose:
                    print("  Note: -f csv flag not recognized, retrying without it...")
                result = self._run_command(cmd_without_csv, cwd=cwd)
        
        return result
    
    def _convert_counters(self, input_dir: Path, output_file: Path) -> None:
        """
        Convert rocprofv3 counter collection output to unified CSV.
        
        Finds all counter_collection.csv files in pmc_* subdirectories,
        concatenates them, and pivots Counter_Name to columns.
        
        Args:
            input_dir: Directory containing pmc_* subdirectories
            output_file: Path to write the combined counters.csv
        """
        # Find all counter_collection.csv files in pmc_* directories
        counter_files = []
        for pmc_dir in input_dir.glob('pmc_*'):
            if pmc_dir.is_dir():
                for csv_file in pmc_dir.glob('*counter_collection.csv'):
                    counter_files.append(csv_file)
        
        if not counter_files:
            raise ValueError(f"No counter_collection.csv files found in {input_dir}")
        
        # Read and concatenate all files
        dfs = []
        for f in counter_files:
            df = pd.read_csv(f)
            dfs.append(df)
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Column groupings for pivot
        index_columns = [
            "Dispatch_Id",
            "Agent_Id",
            "Grid_Size",
            "Kernel_Name",
            "LDS_Block_Size",
            "Queue_Id",
            "SGPR_Count",
            "Scratch_Size",
            "VGPR_Count",
            "Workgroup_Size",
        ]
        
        # Filter to only columns that exist
        index_columns = [c for c in index_columns if c in combined_df.columns]
        
        # Drop duplicates (same counter collected multiple times)
        if 'Counter_Name' in combined_df.columns:
            combined_df.drop_duplicates(
                subset=index_columns + ["Counter_Name"],
                keep="first",
                inplace=True
            )
            
            # Pivot: Counter_Name values become columns
            pivoted = combined_df.pivot_table(
                index=index_columns,
                columns="Counter_Name",
                values="Counter_Value",
                aggfunc="sum"
            ).reset_index()
        else:
            # Already in wide format
            pivoted = combined_df
        
        # Write output
        output_file.parent.mkdir(parents=True, exist_ok=True)
        pivoted.to_csv(output_file, index=False)
    
    def profile(
        self,
        script_path: Path,
        output_dir: Path,
    ) -> ProfileResult:
        """
        Profile a Python script using rocprofv3.
        
        Runs:
        1. rocprofv3 with counter collection (4 runs using counter file)
        2. Converts counter format
        3. rocprofv3 with kernel tracing (1 run)
        
        Args:
            script_path: Path to Python script to profile
            output_dir: Directory to write profiling results
            
        Returns:
            ProfileResult with paths to counters.csv and trace_kernel_trace.csv
        """
        script_path = Path(script_path).absolute()
        output_dir = Path(output_dir).absolute()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = datetime.now()
        
        if not script_path.exists():
            return ProfileResult(
                success=False,
                message=f"Script not found: {script_path}",
                script_path=script_path,
            )
        
        try:
            # Step 1: Run rocprofv3 with counter collection
            if self.verbose:
                print(f"  [1/3] Running rocprofv3 counter collection...")
            
            counter_cmd_with_csv = [
                'rocprofv3',
                '-i', str(self.counter_file),
                '-o', 'counters',
                '-f', 'csv',
                '--',
                sys.executable, str(script_path)
            ]
            counter_cmd_without_csv = [
                'rocprofv3',
                '-i', str(self.counter_file),
                '-o', 'counters',
                '--',
                sys.executable, str(script_path)
            ]
            
            result = self._run_rocprofv3_with_retry(
                counter_cmd_with_csv,
                counter_cmd_without_csv,
                cwd=output_dir,
            )
            
            if result.returncode != 0:
                return ProfileResult(
                    success=False,
                    message=f"Counter collection failed: {result.stderr[:500] if result.stderr else 'Unknown error'}",
                    script_path=script_path,
                    output_dir=output_dir,
                )
            
            # Step 2: Convert counter collection format
            if self.verbose:
                print(f"  [2/3] Converting counter format...")
            
            counters_csv = output_dir / 'counters.csv'
            try:
                self._convert_counters(output_dir, counters_csv)
            except Exception as e:
                return ProfileResult(
                    success=False,
                    message=f"Counter conversion failed: {str(e)[:500]}",
                    script_path=script_path,
                    output_dir=output_dir,
                )
            
            # Step 3: Run rocprofv3 with kernel tracing
            if self.verbose:
                print(f"  [3/3] Running rocprofv3 kernel tracing...")
            
            trace_cmd_with_csv = [
                'rocprofv3',
                '--kernel-trace',
                '-o', 'trace',
                '-f', 'csv',
                '--',
                sys.executable, str(script_path)
            ]
            trace_cmd_without_csv = [
                'rocprofv3',
                '--kernel-trace',
                '-o', 'trace',
                '--',
                sys.executable, str(script_path)
            ]
            
            result = self._run_rocprofv3_with_retry(
                trace_cmd_with_csv,
                trace_cmd_without_csv,
                cwd=output_dir,
            )
            
            if result.returncode != 0:
                # Check for HIP errors
                if result.stderr and ('HIP error' in result.stderr or 'illegal memory' in result.stderr):
                    return ProfileResult(
                        success=False,
                        message="HIP memory error (kernel bug)",
                        script_path=script_path,
                        output_dir=output_dir,
                    )
                return ProfileResult(
                    success=False,
                    message=f"Kernel tracing failed: {result.stderr[:500] if result.stderr else 'Unknown error'}",
                    script_path=script_path,
                    output_dir=output_dir,
                )
            
            # Verify output files exist
            counters_file = output_dir / "counters.csv"
            trace_file = output_dir / "trace_kernel_trace.csv"
            
            if not counters_file.exists():
                return ProfileResult(
                    success=False,
                    message="Counter file not generated",
                    script_path=script_path,
                    output_dir=output_dir,
                )
            
            if not trace_file.exists():
                return ProfileResult(
                    success=False,
                    message="Trace file not generated",
                    script_path=script_path,
                    output_dir=output_dir,
                )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return ProfileResult(
                success=True,
                message="Profiling completed successfully",
                script_path=script_path,
                output_dir=output_dir,
                counters_file=counters_file,
                trace_file=trace_file,
                duration_seconds=duration,
            )
            
        except subprocess.TimeoutExpired:
            return ProfileResult(
                success=False,
                message=f"Timeout after {self.timeout}s",
                script_path=script_path,
                output_dir=output_dir,
            )
        except FileNotFoundError as e:
            if 'rocprofv3' in str(e):
                return ProfileResult(
                    success=False,
                    message="rocprofv3 not found. Ensure ROCm is installed and in PATH.",
                    script_path=script_path,
                )
            raise
        except Exception as e:
            return ProfileResult(
                success=False,
                message=f"Exception: {e}",
                script_path=script_path,
                output_dir=output_dir,
            )
    
    def cleanup_intermediate_files(self, output_dir: Path) -> None:
        """
        Clean up intermediate rocprofv3 files, keeping only final outputs.
        
        Removes pmc_* directories and other temp files.
        
        Args:
            output_dir: Directory to clean
        """
        output_dir = Path(output_dir)
        
        # Remove pmc_* directories
        for pmc_dir in output_dir.glob('pmc_*'):
            if pmc_dir.is_dir():
                shutil.rmtree(pmc_dir)
        
        # Remove counters directory (old rocprofv3 format)
        counters_subdir = output_dir / 'counters'
        if counters_subdir.is_dir():
            shutil.rmtree(counters_subdir)
        
        # Remove .rocprofv3 directory
        rocprof_dir = output_dir / '.rocprofv3'
        if rocprof_dir.is_dir():
            shutil.rmtree(rocprof_dir)
