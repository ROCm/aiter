#!/usr/bin/env python3
"""Dispatch tuning runs across multiple GPUs and generate JSON config files.

Usage examples:
    python run_tuning.py --kernel a8w8 --shapes-file shapes.json --gpus 0,1,2,3 \
        --output-dir results/tuning/ \
        --lds-args "--block-size-m-range 16 32 64 128 --block-size-n-range 32 64 128 --block-size-k-range 128 256"
    python run_tuning.py --kernel a8w8 --shapes-file shapes.json --gpus 0-7 --output-dir results/tuning/
    python run_tuning.py --kernel a8w8 --shapes-file shapes.json --gpus 0-7 --output-dir results/tuning/ --dry-run
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

KERNEL_UT_MAP = {
    "a16w16": "ut_a16w16_gemm.py",
    "a16w16_agnostic": "ut_a16w16_gemm_agnostic.py",
    "a16w16_atomic": "ut_a16w16_gemm_atomic.py",
    "a16w16_gated": "ut_a16w16_gemm_gated.py",
    "a16w8_blockscale": "ut_a16w8_gemm_blockscale.py",
    "a16wfp4": "ut_a16wfp4_gemm.py",
    "a8w8": "ut_a8w8_gemm.py",
    "a8w8_blockscale": "ut_a8w8_gemm_blockscale.py",
    "a8w8_per_token_scale": "ut_a8w8_gemm_per_token_scale.py",
    "a8wfp4": "ut_a8wfp4_gemm.py",
    "afp4wfp4": "ut_afp4wfp4_gemm.py",
    "afp4wfp4_pre_quant_atomic": "ut_afp4wfp4_gemm_pre_quant_atomic.py",
}


def parse_gpus(gpu_str):
    """Parse GPU specification string into a list of GPU IDs.

    Supports formats:
        '0-7'   -> [0, 1, 2, 3, 4, 5, 6, 7]
        '0,2,4' -> [0, 2, 4]
        '0'     -> [0]
    """
    gpu_str = gpu_str.strip()
    if "-" in gpu_str and "," not in gpu_str:
        start, end = gpu_str.split("-", 1)
        return list(range(int(start), int(end) + 1))
    elif "," in gpu_str:
        return [int(g.strip()) for g in gpu_str.split(",")]
    else:
        return [int(gpu_str)]


def shape_key(m, n, k):
    """Return a string key for a shape tuple."""
    return f"{m}_{n}_{k}"


def load_progress(progress_file):
    """Load set of completed shape keys from progress file."""
    if os.path.isfile(progress_file):
        with open(progress_file, "r") as f:
            return set(json.load(f))
    return set()


def save_progress(progress_file, completed):
    """Save set of completed shape keys to progress file."""
    with open(progress_file, "w") as f:
        json.dump(sorted(completed), f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dispatch Triton tuning runs across multiple GPUs."
    )
    parser.add_argument(
        "--kernel",
        type=str,
        required=True,
        choices=sorted(KERNEL_UT_MAP.keys()),
        help="Kernel name to tune.",
    )
    parser.add_argument(
        "--shapes-file",
        type=str,
        required=True,
        help="Path to JSON file containing shapes (list of {M, N, K} dicts).",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        required=True,
        help="GPU IDs to use. Formats: '0-7', '0,2,4', '0'.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to store generated JSON config files.",
    )
    parser.add_argument(
        "--lds-args",
        type=str,
        default="",
        help="Additional arguments passed to screen.py (e.g. block size ranges).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print commands without executing.",
    )
    return parser.parse_args()


def build_screen_cmd(ut_script, m, n, k, lds_args):
    """Build the screen.py command for a given shape.

    GPU ID is always 0 because HIP_VISIBLE_DEVICES remaps the device.
    """
    cmd = f"python screen.py {m} {n} {k} 0 {ut_script}"
    if lds_args:
        cmd += f" {lds_args}"
    cmd += " --num-stages-range 2 3"
    return cmd


def run_tuning(args):
    """Phase 1: Run tuning sweep across GPUs."""
    kernel = args.kernel
    ut_script = KERNEL_UT_MAP[kernel]
    gpu_ids = parse_gpus(args.gpus)
    lds_args = args.lds_args.strip()
    dry_run = args.dry_run

    # Resolve the tunning directory (where screen.py lives)
    tunning_dir = os.path.dirname(os.path.abspath(__file__))

    # Load shapes
    with open(args.shapes_file, "r") as f:
        shapes = json.load(f)

    work_queue = [(s["M"], s["N"], s["K"]) for s in shapes]
    print(f"Kernel: {kernel} -> {ut_script}")
    print(f"GPUs: {gpu_ids}")
    print(f"Total shapes: {len(work_queue)}")

    if dry_run:
        print("\n=== DRY RUN: Work queue and commands ===\n")
        for i, (m, n, k) in enumerate(work_queue):
            cmd = build_screen_cmd(ut_script, m, n, k, lds_args)
            gpu = gpu_ids[i % len(gpu_ids)]
            print(f"[GPU {gpu}] HIP_VISIBLE_DEVICES={gpu} {cmd}")
        return work_queue

    # Progress tracking
    progress_file = os.path.join(
        tunning_dir, f"progress_tuning_{kernel}.json"
    )
    completed = load_progress(progress_file)

    # Filter out already-completed shapes
    remaining = [
        (m, n, k)
        for m, n, k in work_queue
        if shape_key(m, n, k) not in completed
    ]
    if len(remaining) < len(work_queue):
        print(
            f"Resuming: {len(work_queue) - len(remaining)} shapes already completed, "
            f"{len(remaining)} remaining."
        )
    work_queue = remaining

    if not work_queue:
        print("All shapes already completed.")
        return [(s["M"], s["N"], s["K"]) for s in shapes]

    # Process pool
    active = {}  # gpu_id -> (process, key_str)
    total = len(work_queue)
    done_count = 0

    while work_queue or active:
        # Check for completed processes
        for gpu_id in list(active.keys()):
            proc, key_str = active[gpu_id]
            if proc.poll() is not None:
                rc = proc.returncode
                done_count += 1
                if rc == 0:
                    print(
                        f"[{done_count}/{total}] Completed {key_str} on GPU {gpu_id}"
                    )
                    completed.add(key_str)
                    save_progress(progress_file, completed)
                else:
                    print(
                        f"[{done_count}/{total}] FAILED {key_str} on GPU {gpu_id} "
                        f"(exit code {rc})"
                    )
                del active[gpu_id]

        # Launch new work on free GPUs
        for gpu_id in gpu_ids:
            if work_queue and gpu_id not in active:
                m, n, k = work_queue.pop(0)
                key_str = shape_key(m, n, k)
                cmd = build_screen_cmd(ut_script, m, n, k, lds_args)
                env = os.environ.copy()
                env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
                print(f"Launching {key_str} on GPU {gpu_id}: {cmd}")
                proc = subprocess.Popen(
                    cmd,
                    shell=True,
                    cwd=tunning_dir,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                active[gpu_id] = (proc, key_str)

        if active:
            time.sleep(1)

    print(f"\nTuning phase complete. {done_count} shapes processed.")
    return [(s["M"], s["N"], s["K"]) for s in shapes]


def run_config_generation(args, all_shapes):
    """Phase 2: Run view-screen.py to generate JSON config files."""
    kernel = args.kernel
    ut_script = KERNEL_UT_MAP[kernel]
    output_dir = args.output_dir
    dry_run = args.dry_run
    tunning_dir = os.path.dirname(os.path.abspath(__file__))

    # Collect unique (N, K) pairs preserving order
    seen = set()
    nk_pairs = []
    for m, n, k in all_shapes:
        if (n, k) not in seen:
            seen.add((n, k))
            nk_pairs.append((n, k))

    if not nk_pairs:
        print("No (N, K) pairs to generate configs for.")
        return

    # Build paired N and K lists for view-screen.py
    n_list = [str(n) for n, k in nk_pairs]
    k_list = [str(k) for n, k in nk_pairs]

    cmd = (
        f"python view-screen.py {ut_script} "
        f"--n-list {' '.join(n_list)} "
        f"--k-list {' '.join(k_list)}"
    )

    print(f"\n=== Config Generation ===")
    print(f"Unique (N, K) pairs: {len(nk_pairs)}")
    print(f"Command: {cmd}")

    if dry_run:
        print("(dry run, skipping execution)")
        return

    result = subprocess.run(
        cmd,
        shell=True,
        cwd=tunning_dir,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print(
            f"view-screen.py failed with exit code {result.returncode}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Move generated JSON files to output directory
    os.makedirs(output_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(tunning_dir, "*.json"))
    # Only move config JSONs (not progress files)
    progress_prefix = "progress_tuning_"
    moved = 0
    for json_file in json_files:
        basename = os.path.basename(json_file)
        if basename.startswith(progress_prefix):
            continue
        if "N=" in basename and "K=" in basename:
            dest = os.path.join(output_dir, basename)
            shutil.move(json_file, dest)
            print(f"Moved {basename} -> {dest}")
            moved += 1

    print(f"Moved {moved} JSON config files to {output_dir}")


def main():
    args = parse_args()

    # Phase 1: Tuning sweep
    print("=" * 60)
    print("Phase 1: Tuning Sweep")
    print("=" * 60)
    all_shapes = run_tuning(args)

    # Phase 2: Config generation
    print("\n" + "=" * 60)
    print("Phase 2: Config Generation")
    print("=" * 60)
    run_config_generation(args, all_shapes)


if __name__ == "__main__":
    sys.exit(main())
