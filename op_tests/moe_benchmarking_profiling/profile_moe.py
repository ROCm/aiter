#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Profile MOE kernels using rocprofv3.

Thin CLI wrapper that uses the profiling library's ProfilingSession
with MOE-specific script generator and post-processor.

Usage:
    python profile_moe.py -i best_kernels.csv -o results/profiling
    python profile_moe.py -i best_kernels.csv --num-gpus 4  # Multi-GPU
    python profile_moe.py -i best_kernels.csv --keep-scripts  # Keep generated scripts
"""

import argparse
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from profiling import ProfilingSession
from profiling.gpu_utils import detect_gpu, get_num_gpus, SUPPORTED_ARCHS
from profiling.script_generators import MoeScriptGenerator
from profiling.post_processors import MoePostProcessor


def print_header(title: str, char: str = '=', width: int = 80) -> None:
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(title)
    print(f"{char * width}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Profile MOE kernels using rocprofv3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile all kernels from CSV (uses all GPUs by default)
  python profile_moe.py -i best_kernels.csv

  # Profile with specific number of GPUs
  python profile_moe.py -i best_kernels.csv --num-gpus 4

  # Resume interrupted profiling
  python profile_moe.py -i best_kernels.csv --resume

  # Keep generated scripts for debugging
  python profile_moe.py -i best_kernels.csv --keep-scripts
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input CSV file with kernel configurations'
    )
    parser.add_argument(
        '-o', '--output-dir',
        default='results/profiling',
        help='Output directory for results (default: results/profiling)'
    )
    parser.add_argument(
        '--arch',
        default=None,
        choices=SUPPORTED_ARCHS,
        help='GPU architecture (default: auto-detect)'
    )
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=None,
        help='Number of GPUs to use (default: all available)'
    )
    parser.add_argument(
        '--keep-scripts',
        action='store_true',
        help='Keep generated kernel execution scripts'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume profiling (skip already-profiled kernels)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=600,
        help='Timeout per kernel in seconds (default: 600)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--keep-intermediate',
        action='store_true',
        help='Keep intermediate per-kernel directories after combining results'
    )
    parser.add_argument(
        '--skip-combine',
        action='store_true',
        help='Skip post-processing (do not combine results into single CSV)'
    )
    
    args = parser.parse_args()
    
    # Validate input
    input_csv = Path(args.input)
    if not input_csv.exists():
        print(f"Error: Input file not found: {input_csv}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    
    # Detect architecture
    arch = args.arch or detect_gpu()
    if not arch:
        print("Warning: Could not auto-detect GPU. Defaulting to MI300X.")
        arch = 'MI300X'
    
    # Number of GPUs (None = auto-detect all)
    num_gpus = args.num_gpus or get_num_gpus()
    
    # Print header
    print_header("MOE KERNEL PROFILING")
    print(f"Input: {input_csv}")
    print(f"Output: {output_dir.absolute()}")
    print(f"Architecture: {arch}")
    print(f"GPUs: {num_gpus}")
    print(f"Timeout: {args.timeout}s per kernel")
    print(f"Keep scripts: {args.keep_scripts}")
    print(f"Keep intermediate: {args.keep_intermediate}")
    print(f"Resume mode: {args.resume}")
    
    # Load configurations
    print("\nLoading kernel configurations...")
    generator = MoeScriptGenerator()
    
    try:
        configs = generator.load_configs(
            input_csv,
            skip_failed=True,
            resume_from=output_dir if args.resume else None,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    if not configs:
        print("No kernels to profile (all skipped or already profiled)")
        sys.exit(0)
    
    print(f"Loaded {len(configs)} kernel configurations")
    
    # Create profiling session
    session = ProfilingSession(
        generator=generator,
        processor=MoePostProcessor(),
        output_dir=output_dir,
        num_gpus=num_gpus,
        arch=arch,
        timeout=args.timeout,
        verbose=args.verbose,
        keep_scripts=args.keep_scripts,
        keep_intermediate=args.keep_intermediate,
        skip_combine=args.skip_combine,
    )
    
    # Run profiling
    print_header("PROFILING KERNELS")
    result = session.run(configs, resume=args.resume)
    
    # Print summary
    print_header("PROFILING SUMMARY")
    print(f"Total kernels: {result.total_kernels}")
    print(f"Successful profiling: {result.successful_profiling}")
    print(f"Failed profiling: {result.failed_profiling}")
    print(f"Duration: {result.duration_seconds:.1f}s ({result.duration_seconds/60:.1f} minutes)")
    print(f"\nOutput directory: {result.output_dir.absolute()}")
    
    if result.combined_file:
        print(f"\nCombined results: {result.combined_file}")
        print(f"  Kernels processed: {result.successful_processing}")
        print(f"  Total dispatches: {result.total_dispatches}")
        
        if result.failed_processing > 0:
            print(f"\nFailed to post-process {result.failed_processing} kernels:")
            for r in result.processing_results:
                if not r.success:
                    print(f"  - cfg_idx {r.cfg_idx}: {r.message}")
    elif args.skip_combine:
        print(f"\nPer-kernel results in: {output_dir}/<kernel_id>/")
        print(f"  - counters.csv: Hardware performance counters")
        print(f"  - trace_kernel_trace.csv: Kernel execution traces")
    
    if result.failed_profiling > 0:
        print("\nFailed kernels:")
        for r in result.profiling_results:
            if not r.success:
                print(f"  - {r.script_path.stem if r.script_path else 'unknown'}: {r.message}")


if __name__ == "__main__":
    main()
