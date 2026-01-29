#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Profiling runner for single and multi-GPU execution.

Provides a unified Runner class that automatically uses:
- Sequential execution for single GPU (no multiprocessing overhead)
- Parallel execution for multiple GPUs
"""

import os
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any, Union

from .profiler import Profiler, ProfileResult
from .gpu_utils import get_num_gpus


@dataclass
class WorkItem:
    """A single profiling work item."""
    script_path: Path
    output_dir: Path
    script_id: str  # Unique identifier for this script


@dataclass
class RunResult:
    """Result from profiling run."""
    total: int
    successful: int
    failed: int
    results: List[ProfileResult]


def _worker_profile(args: tuple) -> Dict[str, Any]:
    """
    Worker function for parallel GPU execution.
    
    Each worker runs on a dedicated GPU.
    """
    gpu_id, script_path, output_dir, script_id, arch, timeout, verbose = args
    
    # Set GPU for this worker process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['HIP_VISIBLE_DEVICES'] = str(gpu_id)
    
    try:
        profiler = Profiler(arch=arch, timeout=timeout, verbose=verbose)
        result = profiler.profile(Path(script_path), Path(output_dir))
        
        return {
            'script_id': script_id,
            'gpu_id': gpu_id,
            'success': result.success,
            'message': result.message,
            'output_dir': str(result.output_dir) if result.output_dir else None,
            'counters_file': str(result.counters_file) if result.counters_file else None,
            'trace_file': str(result.trace_file) if result.trace_file else None,
            'duration_seconds': result.duration_seconds,
        }
    except Exception as e:
        return {
            'script_id': script_id,
            'gpu_id': gpu_id,
            'success': False,
            'message': f"Worker exception: {e}",
            'output_dir': None,
            'counters_file': None,
            'trace_file': None,
            'duration_seconds': 0.0,
        }


class Runner:
    """
    Unified profiling runner for single and multi-GPU execution.
    
    Automatically uses sequential execution for single GPU (avoiding
    multiprocessing overhead) and parallel execution for multiple GPUs.
    
    Usage:
        # Use all available GPUs (default)
        runner = Runner(arch='MI300X')
        
        # Use specific number of GPUs
        runner = Runner(num_gpus=4, arch='MI300X')
        
        # Force single GPU
        runner = Runner(num_gpus=1, arch='MI300X')
        
        results = runner.run(work_items, progress_callback=print_progress)
    """
    
    def __init__(
        self,
        num_gpus: Optional[int] = None,
        arch: str = 'MI300X',
        timeout: int = 600,
        verbose: bool = False,
    ):
        """
        Initialize the Runner.
        
        Args:
            num_gpus: Number of GPUs to use (None = auto-detect all available)
            arch: GPU architecture
            timeout: Timeout in seconds per script
            verbose: If True, print detailed progress
        """
        self.num_gpus = num_gpus if num_gpus is not None else get_num_gpus()
        if self.num_gpus < 1:
            self.num_gpus = 1
        self.arch = arch
        self.timeout = timeout
        self.verbose = verbose
        
        # Create profiler for sequential mode (reused across runs)
        self._profiler: Optional[Profiler] = None
    
    @property
    def is_parallel(self) -> bool:
        """Return True if running in parallel mode."""
        return self.num_gpus > 1
    
    def _get_profiler(self) -> Profiler:
        """Get or create profiler for sequential mode."""
        if self._profiler is None:
            self._profiler = Profiler(
                arch=self.arch,
                timeout=self.timeout,
                verbose=self.verbose,
            )
        return self._profiler
    
    def run(
        self,
        work_items: List[WorkItem],
        progress_callback: Optional[Callable[[int, int, Union[Dict, ProfileResult]], None]] = None,
    ) -> RunResult:
        """
        Profile all work items.
        
        Uses sequential execution for single GPU, parallel for multiple.
        
        Args:
            work_items: List of WorkItem objects to profile
            progress_callback: Optional callback(idx, total, result) for progress.
                              Result is Dict for parallel mode, ProfileResult for sequential.
            
        Returns:
            RunResult with all profiling results
        """
        if not work_items:
            return RunResult(total=0, successful=0, failed=0, results=[])
        
        if self.num_gpus == 1:
            return self._run_sequential(work_items, progress_callback)
        else:
            return self._run_parallel(work_items, progress_callback)
    
    def _run_sequential(
        self,
        work_items: List[WorkItem],
        progress_callback: Optional[Callable],
    ) -> RunResult:
        """Run profiling sequentially on single GPU."""
        profiler = self._get_profiler()
        results = []
        successful = 0
        failed = 0
        
        for idx, item in enumerate(work_items, 1):
            result = profiler.profile(item.script_path, item.output_dir)
            results.append(result)
            
            if result.success:
                successful += 1
            else:
                failed += 1
            
            if progress_callback:
                progress_callback(idx, len(work_items), result)
        
        return RunResult(
            total=len(work_items),
            successful=successful,
            failed=failed,
            results=results,
        )
    
    def _run_parallel(
        self,
        work_items: List[WorkItem],
        progress_callback: Optional[Callable],
    ) -> RunResult:
        """Run profiling in parallel across multiple GPUs."""
        # Prepare worker arguments
        worker_args = []
        for idx, item in enumerate(work_items):
            gpu_id = idx % self.num_gpus
            worker_args.append((
                gpu_id,
                str(item.script_path),
                str(item.output_dir),
                item.script_id,
                self.arch,
                self.timeout,
                self.verbose,
            ))
        
        results = []
        successful = 0
        failed = 0
        
        with Pool(processes=self.num_gpus) as pool:
            for idx, result_dict in enumerate(pool.imap(_worker_profile, worker_args), 1):
                # Convert dict back to ProfileResult
                result = ProfileResult(
                    success=result_dict['success'],
                    message=result_dict['message'],
                    script_path=Path(work_items[idx-1].script_path),
                    output_dir=Path(result_dict['output_dir']) if result_dict['output_dir'] else None,
                    counters_file=Path(result_dict['counters_file']) if result_dict['counters_file'] else None,
                    trace_file=Path(result_dict['trace_file']) if result_dict['trace_file'] else None,
                    duration_seconds=result_dict['duration_seconds'],
                )
                results.append(result)
                
                if result.success:
                    successful += 1
                else:
                    failed += 1
                
                if progress_callback:
                    progress_callback(idx, len(work_items), result_dict)
        
        return RunResult(
            total=len(work_items),
            successful=successful,
            failed=failed,
            results=results,
        )
