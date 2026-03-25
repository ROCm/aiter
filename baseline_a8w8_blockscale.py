#!/usr/bin/env python3
"""
Phase 1: Baseline collection for gemm_a8w8_blockscale (both variants).
Run on main branch + Triton 3.4.
Parallelizes across 8 GPUs, each running sequentially.
"""
import subprocess
import json
import os
import sys
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

RESULTS_DIR = "/app/aiter/baseline_results_a8w8_blockscale"
os.makedirs(RESULTS_DIR, exist_ok=True)

M_VALUES = [8, 16, 32, 64, 128, 256, 512, 8192]
NUM_GPUS = 8

NK_PAIRS_NONPRESHUFFLE = [
    (512, 7168), (1024, 8192), (2112, 7168), (3072, 1536),
    (4096, 7168), (4608, 7168), (7168, 256), (7168, 2048),
    (7168, 16384), (7168, 18432), (8192, 1024), (8192, 8192),
    (8192, 32768), (16384, 1536), (24576, 1536), (32768, 512),
    (32768, 8192), (36864, 7168),
]


def run_baseline_on_gpu(work_items, gpu_id):
    """Run all work items sequentially on one GPU, each in its own temp dir."""
    results = []
    for M, N, K, variant in work_items:
        # Create unique working directory to avoid rocprof file conflicts
        work_dir = f"/tmp/baseline_gpu{gpu_id}"
        os.makedirs(work_dir, exist_ok=True)

        env = os.environ.copy()
        env["HIP_VISIBLE_DEVICES"] = str(gpu_id)

        bench_script = "/app/aiter/op_tests/op_benchmarks/triton/bench_gemm_a8w8_blockscale.py"

        cmd = [
            "rocprof", "--stats",
            "python", bench_script,
            "--shape", str(M), str(N), str(K),
            "--metric", "time", "--layout", "TN",
        ]

        avg_ns = None
        kernel_name = None
        status = "unknown"

        try:
            result = subprocess.run(
                cmd, env=env, capture_output=True, text=True,
                timeout=300, cwd=work_dir
            )

            stats_file = os.path.join(work_dir, "results.stats.csv")
            if os.path.exists(stats_file):
                with open(stats_file) as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 4 and "gemm_a8w8_blockscale" in row[0]:
                            avg_ns = float(row[3])
                            kernel_name = row[0].strip('"')
                            status = "ok"
                            break
                    else:
                        status = "kernel_not_found"

                # Clean up rocprof output files
                for fname in os.listdir(work_dir):
                    if fname.startswith("results"):
                        os.remove(os.path.join(work_dir, fname))
            else:
                status = "no_stats_file"

        except subprocess.TimeoutExpired:
            status = "timeout"
        except Exception as e:
            status = f"error: {e}"

        avg_str = f"{avg_ns:.0f}ns" if avg_ns else status
        print(f"  GPU{gpu_id}: M={M} N={N} K={K} -> {avg_str}", flush=True)
        results.append({
            "M": M, "N": N, "K": K, "variant": variant,
            "avg_ns": avg_ns, "kernel_name": kernel_name,
            "status": status,
        })

    return results


def main():
    print("=== Phase 1: Baseline Collection for gemm_a8w8_blockscale ===")
    import triton
    print(f"Triton version: {triton.__version__}")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd="/app/aiter").decode().strip()
    print(f"Branch: {branch}")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Build work queue for non-preshuffle
    work_items = []
    for N, K in NK_PAIRS_NONPRESHUFFLE:
        for M in M_VALUES:
            work_items.append((M, N, K, "nonpreshuffle"))

    print(f"Total benchmarks: {len(work_items)} ({len(NK_PAIRS_NONPRESHUFFLE)} NK pairs x {len(M_VALUES)} M values)")
    print(f"Using {NUM_GPUS} GPUs")
    print()

    # Distribute round-robin across GPUs
    gpu_work = [[] for _ in range(NUM_GPUS)]
    for i, item in enumerate(work_items):
        gpu_work[i % NUM_GPUS].append(item)

    for g in range(NUM_GPUS):
        print(f"  GPU{g}: {len(gpu_work[g])} items")

    print()

    # Run in parallel across GPUs
    all_results = []
    with ProcessPoolExecutor(max_workers=NUM_GPUS) as executor:
        futures = {}
        for gpu_id in range(NUM_GPUS):
            if gpu_work[gpu_id]:
                f = executor.submit(run_baseline_on_gpu, gpu_work[gpu_id], gpu_id)
                futures[f] = gpu_id

        for f in as_completed(futures):
            gpu_id = futures[f]
            try:
                results = f.result()
                all_results.extend(results)
            except Exception as e:
                print(f"GPU{gpu_id} failed: {e}")

    # Save results as JSON
    baseline = {}
    for r in all_results:
        if r["avg_ns"] is not None:
            key = f"{r['M']}-{r['N']}-{r['K']}"
            baseline[key] = {
                "time_ns": r["avg_ns"],
                "kernel_name": r["kernel_name"],
            }

    output_file = os.path.join(RESULTS_DIR, "baseline_triton34_nonpreshuffle.json")
    with open(output_file, "w") as f:
        json.dump(baseline, f, indent=2, sort_keys=True)
    print(f"\nResults saved to {output_file}")

    # Print summary table
    print("\n=== Summary ===")
    ok_count = sum(1 for r in all_results if r["status"] == "ok")
    fail_count = sum(1 for r in all_results if r["status"] != "ok")
    print(f"OK: {ok_count}, Failed: {fail_count}")

    if fail_count > 0:
        print("\nFailed items:")
        for r in sorted(all_results, key=lambda x: (x["N"], x["K"], x["M"])):
            if r["status"] != "ok":
                print(f"  M={r['M']} N={r['N']} K={r['K']} - {r['status']}")

    # Print table sorted by N,K,M
    print("\n=== Baseline Timings (non-preshuffle) ===")
    print(f"{'M':>6} {'N':>6} {'K':>6} {'AverageNs':>12}")
    print("-" * 36)
    for r in sorted(all_results, key=lambda x: (x["N"], x["K"], x["M"])):
        if r["avg_ns"] is not None:
            print(f"{r['M']:>6} {r['N']:>6} {r['K']:>6} {r['avg_ns']:>12.0f}")


if __name__ == "__main__":
    main()
