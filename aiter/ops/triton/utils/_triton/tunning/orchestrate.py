#!/usr/bin/env python3
"""Top-level orchestrator for the Triton GEMM tuning pipeline.

Usage:
    python orchestrate.py baseline --kernels all --gpus 0-7
    python orchestrate.py tune --kernels all --gpus 0-7
    python orchestrate.py validate --kernels all --gpus 0-7
    python orchestrate.py full --kernels all --gpus 0-7
"""

import argparse
import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

ALL_KERNELS = [
    "a16w16", "a16w16_agnostic", "a16w16_atomic", "a16w16_gated",
    "a16w8_blockscale", "a16wfp4",
    "a8w8", "a8w8_blockscale", "a8w8_per_token_scale",
    "a8wfp4", "afp4wfp4", "afp4wfp4_pre_quant_atomic",
]


def parse_gpus(gpu_str):
    """Parse GPU argument: '0-7' -> [0..7], '0,2,4' -> [0,2,4], '0' -> [0]."""
    if "-" in gpu_str and "," not in gpu_str:
        parts = gpu_str.split("-")
        return list(range(int(parts[0]), int(parts[1]) + 1))
    elif "," in gpu_str:
        return [int(x) for x in gpu_str.split(",")]
    else:
        return [int(gpu_str)]


def parse_kernels(kernel_str):
    """Parse kernel argument: 'all' -> ALL_KERNELS, 'a8w8,a16w16' -> list."""
    if kernel_str == "all":
        return ALL_KERNELS[:]
    kernels = [k.strip() for k in kernel_str.split(",")]
    for k in kernels:
        if k not in ALL_KERNELS:
            print(f"Error: unknown kernel '{k}'", file=sys.stderr)
            print(f"Available: {', '.join(ALL_KERNELS)}", file=sys.stderr)
            sys.exit(1)
    return kernels


def run_cmd(cmd, dry_run=False, cwd=None):
    """Run a command, printing it first."""
    print(f"  $ {' '.join(cmd)}")
    if dry_run:
        return 0
    result = subprocess.run(cmd, cwd=cwd or SCRIPT_DIR)
    return result.returncode


def do_baseline(kernels, gpus, dry_run=False):
    """Phase 1: Collect shapes and baseline timings on current Triton."""
    shapes_dir = os.path.join(RESULTS_DIR, "shapes")
    baseline_dir = os.path.join(RESULTS_DIR, "baseline")
    os.makedirs(shapes_dir, exist_ok=True)
    os.makedirs(baseline_dir, exist_ok=True)

    print("\n=== Phase 1: Baseline Collection ===\n")

    # Step 1: Collect shapes for all kernels
    print("Step 1: Collecting shapes...")
    for kernel in kernels:
        run_cmd([
            sys.executable, "collect_shapes.py",
            "--kernel", kernel,
            "--output-dir", shapes_dir,
        ], dry_run=dry_run)

    # Step 2: Collect baseline timings (parallel across GPUs)
    print("\nStep 2: Collecting baseline timings...")
    if dry_run:
        for i, kernel in enumerate(kernels):
            gpu = gpus[i % len(gpus)]
            shapes_file = os.path.join(shapes_dir, f"shapes_gemm_{kernel}.json")
            run_cmd([
                sys.executable, "collect_baseline.py",
                "--kernel", kernel,
                "--shapes-file", shapes_file,
                "--output-dir", baseline_dir,
                "--gpu", str(gpu),
            ], dry_run=True)
    else:
        import time
        active = {}  # gpu_id -> (process, kernel_name)
        queue = []
        for kernel in kernels:
            shapes_file = os.path.join(shapes_dir, f"shapes_gemm_{kernel}.json")
            if not os.path.isfile(shapes_file):
                print(f"  [SKIP] No shapes file for {kernel}")
                continue
            queue.append((kernel, shapes_file))

        while queue or active:
            # Check completed
            for gpu_id in list(active.keys()):
                proc, kname = active[gpu_id]
                if proc.poll() is not None:
                    print(f"  [GPU {gpu_id}] Finished {kname} (rc={proc.returncode})")
                    del active[gpu_id]
            # Launch new
            for gpu_id in gpus:
                if gpu_id not in active and queue:
                    kernel, shapes_file = queue.pop(0)
                    cmd = [
                        sys.executable, "collect_baseline.py",
                        "--kernel", kernel,
                        "--shapes-file", shapes_file,
                        "--output-dir", baseline_dir,
                        "--gpu", str(gpu_id),
                    ]
                    print(f"  [GPU {gpu_id}] Starting {kernel}")
                    env = os.environ.copy()
                    proc = subprocess.Popen(cmd, cwd=SCRIPT_DIR, env=env)
                    active[gpu_id] = (proc, kernel)
            if active:
                time.sleep(2)

    # Step 3: Merge per-kernel baselines into one file
    if not dry_run:
        merged = {}
        for kernel in kernels:
            bl_file = os.path.join(baseline_dir, f"baseline_{kernel}.json")
            if os.path.isfile(bl_file):
                with open(bl_file) as f:
                    merged[f"gemm_{kernel}"] = json.load(f)
        merged_path = os.path.join(baseline_dir, "baseline_triton34.json")
        with open(merged_path, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"\nBaseline saved to {merged_path}")


def do_tune(kernels, gpus, dry_run=False):
    """Phase 2: Run LDS-aware tuning on latest Triton."""
    shapes_dir = os.path.join(RESULTS_DIR, "shapes")
    tuning_dir = os.path.join(RESULTS_DIR, "tuning")
    configs_dir = os.path.join(RESULTS_DIR, "configs")
    os.makedirs(shapes_dir, exist_ok=True)
    os.makedirs(tuning_dir, exist_ok=True)
    os.makedirs(configs_dir, exist_ok=True)

    print("\n=== Phase 2: Tuning ===\n")

    for kernel in kernels:
        # Ensure shapes exist
        shapes_file = os.path.join(shapes_dir, f"shapes_gemm_{kernel}.json")
        if not os.path.isfile(shapes_file) and not dry_run:
            print(f"  Collecting shapes for {kernel}...")
            run_cmd([
                sys.executable, "collect_shapes.py",
                "--kernel", kernel,
                "--output-dir", shapes_dir,
            ])

        if not dry_run and not os.path.isfile(shapes_file):
            print(f"  [SKIP] No shapes for {kernel}")
            continue

        # Get LDS-filtered args for each num_stages
        print(f"\n--- Tuning {kernel} ---")
        if not dry_run:
            lds_result = subprocess.run(
                [sys.executable, "lds_filter.py", "--kernel", kernel,
                 "--num-stages", "2", "3", "--print-cli"],
                capture_output=True, text=True, cwd=SCRIPT_DIR,
            )
            # Parse all non-comment lines (one per num_stages)
            lds_lines = [l.strip() for l in lds_result.stdout.strip().split("\n")
                         if l.strip() and not l.startswith("#")]
        else:
            lds_lines = [
                "--block-size-m-range 16 32 64 128 --block-size-n-range 32 64 128 --block-size-k-range 128 256 --num-stages-range 2",
                "--block-size-m-range 16 32 64 --block-size-n-range 32 64 --block-size-k-range 128 --num-stages-range 3",
            ]

        gpu_str = ",".join(str(g) for g in gpus)
        # Run tuning for each num_stages configuration
        for lds_args in lds_lines:
            print(f"  LDS args: {lds_args}")
            run_cmd([
                sys.executable, "run_tuning.py",
                "--kernel", kernel,
                "--shapes-file", shapes_file,
                "--gpus", gpu_str,
                "--output-dir", configs_dir,
                "--lds-args", lds_args,
            ] + (["--dry-run"] if dry_run else []), dry_run=False)


def do_validate(kernels, gpus, dry_run=False):
    """Phase 3: Re-benchmark with new configs and compare."""
    shapes_dir = os.path.join(RESULTS_DIR, "shapes")
    validation_dir = os.path.join(RESULTS_DIR, "validation")
    baseline_path = os.path.join(RESULTS_DIR, "baseline", "baseline_triton34.json")
    os.makedirs(validation_dir, exist_ok=True)

    print("\n=== Phase 3: Validation ===\n")

    # Step 1: Collect new timings (parallel across GPUs)
    print("Step 1: Collecting new timings...")
    if dry_run:
        for i, kernel in enumerate(kernels):
            gpu = gpus[i % len(gpus)]
            shapes_file = os.path.join(shapes_dir, f"shapes_gemm_{kernel}.json")
            run_cmd([
                sys.executable, "collect_baseline.py",
                "--kernel", kernel,
                "--shapes-file", shapes_file,
                "--output-dir", validation_dir,
                "--gpu", str(gpu),
            ], dry_run=True)
    else:
        import time
        active = {}
        queue = []
        for kernel in kernels:
            shapes_file = os.path.join(shapes_dir, f"shapes_gemm_{kernel}.json")
            if not os.path.isfile(shapes_file):
                print(f"  [SKIP] No shapes file for {kernel}")
                continue
            queue.append((kernel, shapes_file))

        while queue or active:
            for gpu_id in list(active.keys()):
                proc, kname = active[gpu_id]
                if proc.poll() is not None:
                    print(f"  [GPU {gpu_id}] Finished {kname} (rc={proc.returncode})")
                    del active[gpu_id]
            for gpu_id in gpus:
                if gpu_id not in active and queue:
                    kernel, shapes_file = queue.pop(0)
                    cmd = [
                        sys.executable, "collect_baseline.py",
                        "--kernel", kernel,
                        "--shapes-file", shapes_file,
                        "--output-dir", validation_dir,
                        "--gpu", str(gpu_id),
                    ]
                    print(f"  [GPU {gpu_id}] Starting {kernel}")
                    proc = subprocess.Popen(cmd, cwd=SCRIPT_DIR)
                    active[gpu_id] = (proc, kernel)
            if active:
                time.sleep(2)

    # Step 2: Merge validation results
    if not dry_run:
        merged = {}
        for kernel in kernels:
            val_file = os.path.join(validation_dir, f"baseline_{kernel}.json")
            if os.path.isfile(val_file):
                merged[f"gemm_{kernel}"] = json.load(open(val_file))
        val_path = os.path.join(validation_dir, "validation_latest.json")
        with open(val_path, "w") as f:
            json.dump(merged, f, indent=2)

    # Step 3: Compare
    print("\nStep 2: Comparing results...")
    report_path = os.path.join(RESULTS_DIR, "report.txt")
    val_path = os.path.join(validation_dir, "validation_latest.json")
    run_cmd([
        sys.executable, "compare_results.py",
        "--baseline", baseline_path,
        "--new", val_path,
        "--output", report_path,
    ], dry_run=dry_run)

    if not dry_run:
        print(f"\nReport saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Triton GEMM tuning pipeline orchestrator"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common args for all subcommands
    for name in ["baseline", "tune", "validate", "full"]:
        sp = subparsers.add_parser(name)
        sp.add_argument("--kernels", type=str, default="all",
                        help='Kernel names (comma-separated) or "all"')
        sp.add_argument("--gpus", type=str, default="0",
                        help='GPU IDs: "0-7", "0,2,4", or "0"')
        sp.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
        if name == "full":
            sp.add_argument("--skip-baseline", action="store_true",
                            help="Skip baseline collection (use existing)")

    args = parser.parse_args()
    kernels = parse_kernels(args.kernels)
    gpus = parse_gpus(args.gpus)
    dry_run = args.dry_run

    print(f"Kernels: {kernels}")
    print(f"GPUs: {gpus}")
    if dry_run:
        print("*** DRY RUN MODE ***")

    if args.command == "baseline":
        do_baseline(kernels, gpus, dry_run)
    elif args.command == "tune":
        do_tune(kernels, gpus, dry_run)
    elif args.command == "validate":
        do_validate(kernels, gpus, dry_run)
    elif args.command == "full":
        if not getattr(args, "skip_baseline", False):
            do_baseline(kernels, gpus, dry_run)
        do_tune(kernels, gpus, dry_run)
        do_validate(kernels, gpus, dry_run)


if __name__ == "__main__":
    main()
