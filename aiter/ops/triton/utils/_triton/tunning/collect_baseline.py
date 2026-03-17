#!/usr/bin/env python3
"""Collect baseline kernel timings using rocprofv3.

Runs rocprofv3 benchmarks for a given kernel and set of shapes, parses
the profiler output, and stores kernel timing results as JSON.

Usage:
    python collect_baseline.py --kernel a8w8 --shapes-file shapes.json \
        --output-dir results/baseline/ --gpu 0
"""

import argparse
import glob
import json
import logging
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

KERNEL_MAP = {
    "a16w16": {
        "bench": "bench_gemm_a16w16.py",
        "ut": "ut_a16w16_gemm.py",
        "pattern": ["_gemm_a16w16_kernel"],
    },
    "a16w16_agnostic": {
        "bench": None,
        "ut": "ut_a16w16_gemm_agnostic.py",
        "pattern": ["_gemm_a16_w16_kernel"],
    },
    "a16w16_atomic": {
        "bench": None,
        "ut": "ut_a16w16_gemm_atomic.py",
        "pattern": ["_gemm_a16w16_atomic"],
    },
    "a16w16_gated": {
        "bench": "bench_gemm_a16w16.py",
        "ut": "ut_a16w16_gemm_gated.py",
        "pattern": ["_gemm_a16w16_gated"],
    },
    "a16w8_blockscale": {
        "bench": None,
        "ut": "ut_a16w8_gemm_blockscale.py",
        "pattern": ["_gemm_a16w8_blockscale"],
    },
    "a16wfp4": {
        "bench": None,
        "ut": "ut_a16wfp4_gemm.py",
        "pattern": ["_gemm_a16wfp4"],
    },
    "a8w8": {
        "bench": "bench_gemm_a8w8.py",
        "ut": "ut_a8w8_gemm.py",
        "pattern": ["_gemm_a8w8_kernel"],
    },
    "a8w8_blockscale": {
        "bench": "bench_gemm_a8w8_blockscale.py",
        "ut": "ut_a8w8_gemm_blockscale.py",
        "pattern": ["_gemm_a8w8_blockscale"],
    },
    "a8w8_per_token_scale": {
        "bench": "bench_gemm_a8w8_per_token_scale.py",
        "ut": "ut_a8w8_gemm_per_token_scale.py",
        "pattern": ["_gemm_a8w8_per_token"],
    },
    "a8wfp4": {
        "bench": "bench_gemm_a8wfp4.py",
        "ut": "ut_a8wfp4_gemm.py",
        "pattern": ["_gemm_a8wfp4"],
    },
    "afp4wfp4": {
        "bench": "bench_gemm_afp4wfp4.py",
        "ut": "ut_afp4wfp4_gemm.py",
        "pattern": ["_gemm_afp4wfp4", "gemm_afp4_wfp4"],
    },
    "afp4wfp4_pre_quant_atomic": {
        "bench": None,
        "ut": "ut_afp4wfp4_gemm_pre_quant_atomic.py",
        "pattern": ["_gemm_a16wfp4"],
    },
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "..", "..", "..", ".."))
BENCH_DIR = os.path.join(REPO_ROOT, "op_tests", "op_benchmarks", "triton")
ROCPROF_TIMEOUT = 600  # seconds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect baseline kernel timings using rocprofv3."
    )
    parser.add_argument(
        "--kernel",
        type=str,
        required=True,
        choices=sorted(KERNEL_MAP.keys()),
        help="Kernel name to benchmark.",
    )
    parser.add_argument(
        "--shapes-file",
        type=str,
        required=True,
        help="Path to JSON file containing array of {M, N, K} shapes.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write profiler traces and baseline JSON.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU device index for HIP_VISIBLE_DEVICES (default: 0).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=ROCPROF_TIMEOUT,
        help=f"Timeout in seconds for each rocprofv3 run (default: {ROCPROF_TIMEOUT}).",
    )
    parser.add_argument(
        "--use-bench",
        action="store_true",
        default=False,
        help="Prefer bench scripts over ut scripts when available.",
    )
    return parser.parse_args()


def build_command(kernel_name, kernel_info, M, N, K, trace_path, use_bench):
    """Build the rocprofv3 command for a given kernel and shape.

    Returns the command as a list of strings, or None if no script is available.
    """
    rocprof_prefix = [
        "rocprofv3", "--kernel-trace", "-f", "csv", "-o", trace_path, "--",
    ]

    if use_bench and kernel_info["bench"] is not None:
        script_path = os.path.join(BENCH_DIR, kernel_info["bench"])
        cmd = rocprof_prefix + [
            "python", script_path,
            "--shape", str(M), str(N), str(K),
            "--metric", "time",
            "--layout", "TN",
        ]
    elif kernel_info["ut"] is not None:
        script_path = os.path.join(SCRIPT_DIR, kernel_info["ut"])
        cmd = rocprof_prefix + [
            "python", script_path, str(M), str(N), str(K),
        ]
    else:
        logger.warning("No script available for kernel %s", kernel_name)
        return None

    return cmd


def find_trace_csv(trace_path):
    """Locate the kernel trace CSV produced by rocprofv3.

    rocprofv3 appends suffixes like '_kernel_trace.csv' to the output path.
    """
    # Try the most common suffix first
    candidate = trace_path + "_kernel_trace.csv"
    if os.path.isfile(candidate):
        return candidate

    # Try globbing for any CSV with the trace_path prefix
    pattern = trace_path + "*kernel_trace*.csv"
    matches = glob.glob(pattern)
    if matches:
        return matches[0]

    # Try looking inside a directory that rocprofv3 sometimes creates
    if os.path.isdir(trace_path):
        inner_matches = glob.glob(os.path.join(trace_path, "*kernel_trace*.csv"))
        if inner_matches:
            return inner_matches[0]

    return None


def parse_ut_trace(csv_path, patterns):
    """Parse a kernel trace CSV from a ut script run.

    The ut scripts run the kernel 250 times with a split_dummy separator.
    We group runs by split_dummy boundaries, sum matching kernel durations
    within each run, and return the median total duration in nanoseconds
    along with the matched kernel name.

    Returns:
        (median_time_ns, kernel_name) or (None, None) on failure.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.warning("Failed to read CSV %s: %s", csv_path, e)
        return None, None

    if "Kernel_Name" not in df.columns:
        logger.warning("CSV %s missing Kernel_Name column", csv_path)
        return None, None

    # Find split_dummy indices to delimit runs
    split_dummy_idx = (df["Kernel_Name"] == "split_dummy").to_numpy().nonzero()[0]

    if len(split_dummy_idx) > 0:
        # Insert 0 at the beginning so we capture the first run
        split_dummy_idx = np.insert(split_dummy_idx, 0, 0)
        runs = []
        for i in range(len(split_dummy_idx) - 1):
            runs.append(df.iloc[split_dummy_idx[i]:split_dummy_idx[i + 1]])
    else:
        runs = [df]

    # Find target kernel names matching the patterns
    all_kernel_names = df["Kernel_Name"].unique()
    target_kernels = []
    for name in all_kernel_names:
        for pat in patterns:
            if pat in name:
                target_kernels.append(name)
                break

    if not target_kernels:
        logger.warning(
            "No kernel matching patterns %s found in %s. Available: %s",
            patterns, csv_path, list(all_kernel_names),
        )
        return None, None

    # For each run, sum durations of all matching kernels
    run_totals = []
    for run_df in runs:
        total = 0.0
        for kname in target_kernels:
            kdf = run_df[run_df["Kernel_Name"] == kname]
            if len(kdf) > 0:
                durations = (kdf["End_Timestamp"] - kdf["Start_Timestamp"]).to_numpy()
                total += durations.sum()
        if total > 0:
            run_totals.append(total)

    if not run_totals:
        logger.warning("No valid runs found in %s", csv_path)
        return None, None

    run_totals = np.array(run_totals)
    median_ns = float(np.median(run_totals))
    matched_name = ", ".join(target_kernels)

    return median_ns, matched_name


def parse_bench_trace(csv_path, patterns):
    """Parse a kernel trace CSV from a bench script run.

    Bench scripts may not use split_dummy. We match kernel names by pattern,
    compute per-invocation totals, and return the median.

    Returns:
        (median_time_ns, kernel_name) or (None, None) on failure.
    """
    # Bench script traces follow the same structure; reuse ut parsing logic
    # which handles both split_dummy-delimited and non-delimited cases.
    return parse_ut_trace(csv_path, patterns)


def run_rocprofv3(cmd, env, timeout):
    """Execute a rocprofv3 command.

    Returns True on success, False on failure.
    """
    logger.info("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            env=env,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning(
                "rocprofv3 failed (rc=%d) for command: %s\nstderr: %s",
                result.returncode, " ".join(cmd), result.stderr[:2000],
            )
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.warning("rocprofv3 timed out after %ds for: %s", timeout, " ".join(cmd))
        return False
    except FileNotFoundError:
        logger.error("rocprofv3 not found. Is it installed and on PATH?")
        return False


def cleanup_trace_files(trace_path):
    """Remove intermediate rocprofv3 trace files and directories."""
    # Remove files matching the trace path prefix
    for f in glob.glob(trace_path + "*"):
        try:
            if os.path.isfile(f):
                os.remove(f)
            elif os.path.isdir(f):
                shutil.rmtree(f)
        except OSError:
            pass

    # Also try removing the parent directory entry if it was created
    trace_dir = os.path.dirname(trace_path)
    if trace_dir and os.path.isdir(trace_dir):
        try:
            # Only remove if empty
            if not os.listdir(trace_dir):
                os.rmdir(trace_dir)
        except OSError:
            pass


def collect_baseline(kernel_name, shapes, output_dir, gpu, timeout, use_bench):
    """Run rocprofv3 for each shape and collect timing results.

    Returns a dict mapping shape keys to timing results.
    """
    kernel_info = KERNEL_MAP[kernel_name]
    patterns = kernel_info["pattern"]
    results = {}

    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu)

    os.makedirs(output_dir, exist_ok=True)

    for shape in shapes:
        M, N, K = shape["M"], shape["N"], shape["K"]
        shape_key = f"{M}-{N}-{K}"
        trace_name = f"trace_{kernel_name}_{M}_{N}_{K}"
        trace_path = os.path.join(output_dir, trace_name)

        cmd = build_command(kernel_name, kernel_info, M, N, K, trace_path, use_bench)
        if cmd is None:
            logger.warning("Skipping shape %s: no script available", shape_key)
            continue

        # Clean up any pre-existing trace files
        cleanup_trace_files(trace_path)

        success = run_rocprofv3(cmd, env, timeout)
        if not success:
            logger.warning("Skipping shape %s: rocprofv3 failed", shape_key)
            cleanup_trace_files(trace_path)
            continue

        csv_path = find_trace_csv(trace_path)
        if csv_path is None:
            logger.warning(
                "Skipping shape %s: could not find trace CSV (tried %s*)",
                shape_key, trace_path,
            )
            cleanup_trace_files(trace_path)
            continue

        if use_bench and kernel_info["bench"] is not None:
            time_ns, matched_kernel = parse_bench_trace(csv_path, patterns)
        else:
            time_ns, matched_kernel = parse_ut_trace(csv_path, patterns)

        if time_ns is None:
            logger.warning("Skipping shape %s: could not extract timing", shape_key)
            cleanup_trace_files(trace_path)
            continue

        results[shape_key] = {
            "time_ns": time_ns,
            "kernel_name": matched_kernel,
        }
        logger.info("Shape %s: %.1f ns (%s)", shape_key, time_ns, matched_kernel)

        # Clean up intermediate files
        cleanup_trace_files(trace_path)

    return results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_args()

    if args.kernel not in KERNEL_MAP:
        logger.error("Unknown kernel: %s", args.kernel)
        sys.exit(1)

    # Read shapes
    with open(args.shapes_file, "r") as f:
        shapes = json.load(f)

    if not isinstance(shapes, list):
        logger.error("Shapes file must contain a JSON array")
        sys.exit(1)

    logger.info(
        "Collecting baseline for kernel=%s with %d shapes on GPU %s",
        args.kernel, len(shapes), args.gpu,
    )

    results = collect_baseline(
        kernel_name=args.kernel,
        shapes=shapes,
        output_dir=args.output_dir,
        gpu=args.gpu,
        timeout=args.timeout,
        use_bench=args.use_bench,
    )

    # Write results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"baseline_{args.kernel}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")

    logger.info(
        "Wrote %d/%d results to %s", len(results), len(shapes), output_path
    )


if __name__ == "__main__":
    main()
