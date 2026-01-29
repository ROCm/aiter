#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Profiling session orchestrator.

Provides a high-level ProfilingSession class that coordinates:
- Script generation from configurations
- Running the profiler (single or multi-GPU)
- Incremental post-processing with cleanup
- Combined results generation

This keeps operator-specific CLIs thin and moves shared logic here.
"""

import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Union, Dict, Any

from .runner import Runner, RunResult, WorkItem
from .profiler import ProfileResult
from .script_generators.base import ScriptGenerator, ScriptConfig
from .post_processors.base import PostProcessor, ProcessedResult


@dataclass
class SessionResult:
    """Result from a profiling session."""
    # Profiling results
    total_kernels: int
    successful_profiling: int
    failed_profiling: int
    profiling_results: List[ProfileResult]
    
    # Post-processing results
    total_processed: int
    successful_processing: int
    failed_processing: int
    processing_results: List[ProcessedResult]
    
    # Output paths
    output_dir: Path
    combined_file: Optional[Path] = None
    
    # Timing
    duration_seconds: float = 0.0
    
    @property
    def total_dispatches(self) -> int:
        """Total number of kernel dispatches across all processed kernels."""
        return sum(r.num_dispatches for r in self.processing_results if r.success)


class ProfilingSession:
    """
    High-level orchestrator for profiling sessions.
    
    Coordinates script generation, profiling, and post-processing into
    a single workflow. Handles incremental result combination and cleanup.
    
    Usage:
        from profiling.script_generators import MoeScriptGenerator
        from profiling.post_processors import MoePostProcessor
        
        session = ProfilingSession(
            generator=MoeScriptGenerator(),
            processor=MoePostProcessor(),
            output_dir=Path("results"),
        )
        
        configs = session.generator.load_configs("kernels.csv")
        result = session.run(configs)
        
        print(f"Profiled {result.successful_profiling} kernels")
        print(f"Combined results: {result.combined_file}")
    """
    
    def __init__(
        self,
        generator: ScriptGenerator,
        processor: PostProcessor,
        output_dir: Path,
        num_gpus: Optional[int] = None,
        arch: str = 'MI300X',
        timeout: int = 600,
        verbose: bool = False,
        keep_scripts: bool = False,
        keep_intermediate: bool = False,
        skip_combine: bool = False,
    ):
        """
        Initialize a profiling session.
        
        Args:
            generator: Script generator for the operator
            processor: Post-processor for combining results
            output_dir: Directory for all output files
            num_gpus: Number of GPUs (None = auto-detect all)
            arch: GPU architecture
            timeout: Timeout per kernel in seconds
            verbose: Print detailed progress
            keep_scripts: Keep generated scripts after completion
            keep_intermediate: Keep per-kernel directories after combining
            skip_combine: Skip post-processing entirely
        """
        self.generator = generator
        self.processor = processor
        self.output_dir = Path(output_dir)
        self.scripts_dir = self.output_dir / "scripts"
        self.keep_scripts = keep_scripts
        self.keep_intermediate = keep_intermediate
        self.skip_combine = skip_combine
        
        self.runner = Runner(
            num_gpus=num_gpus,
            arch=arch,
            timeout=timeout,
            verbose=verbose,
        )
        
        # Progress callback
        self._progress_callback: Optional[Callable] = None
    
    @property
    def num_gpus(self) -> int:
        """Number of GPUs being used."""
        return self.runner.num_gpus
    
    @property
    def is_parallel(self) -> bool:
        """Whether running in parallel mode."""
        return self.runner.is_parallel
    
    def run(
        self,
        configs: List[ScriptConfig],
        resume: bool = False,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> SessionResult:
        """
        Run a complete profiling session.
        
        Generates scripts, runs profiling, and combines results incrementally.
        
        Args:
            configs: List of kernel configurations to profile
            resume: If True, append to existing combined file
            progress_callback: Optional callback(message) for progress updates
            
        Returns:
            SessionResult with all profiling and processing results
        """
        self._progress_callback = progress_callback
        start_time = datetime.now()
        
        # Setup directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup combined file
        combined_file = self.output_dir / "combined_results.csv" if not self.skip_combine else None
        if combined_file and combined_file.exists() and not resume:
            combined_file.unlink()
        
        # Generate scripts and work items
        self._log("Generating profiling scripts...")
        work_items = []
        for config in configs:
            script_path = self.generator.write_script(config, self.scripts_dir)
            kernel_output_dir = self.output_dir / config.unique_id
            work_items.append(WorkItem(
                script_path=script_path,
                output_dir=kernel_output_dir,
                script_id=config.unique_id,
            ))
        self._log(f"Generated {len(work_items)} scripts")
        
        # Track processing results
        process_results: List[ProcessedResult] = []
        
        # Create progress callback with incremental post-processing
        def internal_callback(idx: int, total: int, result: Union[Dict, ProfileResult]) -> None:
            # Extract success/message from result
            if isinstance(result, dict):
                success = result['success']
                msg = result['message']
                gpu_id = result.get('gpu_id', 0)
                script_id = result['script_id']
                prefix = f"GPU{gpu_id} " if self.is_parallel else ""
                self._log(f"[{idx}/{total}] {prefix}{script_id}: {'OK' if success else 'FAIL'} - {msg}")
            else:
                success = result.success
                msg = result.message
                script_id = work_items[idx-1].script_id
                self._log(f"[{idx}/{total}] {script_id}: {'OK' if success else 'FAIL'} - {msg}")
            
            # Incremental post-processing
            if success and not self.skip_combine and combined_file:
                config = configs[idx-1]
                kernel_dir = self.output_dir / config.unique_id
                
                proc_result = self.processor.process_and_append(
                    kernel_dir=kernel_dir,
                    config=config,
                    combined_file=combined_file,
                    fallback_idx=idx-1,
                    cleanup=not self.keep_intermediate,
                )
                process_results.append(proc_result)
                
                if proc_result.success:
                    self._log(f"    -> Processed {proc_result.num_dispatches} dispatch(es)")
                else:
                    self._log(f"    -> Post-processing failed: {proc_result.message}")
        
        # Run profiling
        if self.is_parallel:
            self._log(f"Running parallel profiling with {self.num_gpus} GPUs...")
        else:
            self._log("Running sequential profiling...")
        
        run_result = self.runner.run(work_items, progress_callback=internal_callback)
        
        # Cleanup scripts
        if not self.keep_scripts and self.scripts_dir.exists():
            self._log("Cleaning up generated scripts...")
            shutil.rmtree(self.scripts_dir)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return SessionResult(
            total_kernels=run_result.total,
            successful_profiling=run_result.successful,
            failed_profiling=run_result.failed,
            profiling_results=run_result.results,
            total_processed=len(process_results),
            successful_processing=sum(1 for r in process_results if r.success),
            failed_processing=sum(1 for r in process_results if not r.success),
            processing_results=process_results,
            output_dir=self.output_dir,
            combined_file=combined_file if combined_file and combined_file.exists() else None,
            duration_seconds=duration,
        )
    
    def _log(self, message: str) -> None:
        """Log a message via callback or print."""
        if self._progress_callback:
            self._progress_callback(message)
        else:
            print(message)
