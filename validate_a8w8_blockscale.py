#!/usr/bin/env python3
"""
Phase 3: Validation for gemm_a8w8_blockscale (non-preshuffle).
Apples-to-apples: rocprof --stats on Triton 3.6 with new configs vs Triton 3.4 baseline.
Parallelized across 8 GPUs.
"""
import subprocess
import json
import os
import csv
import time
import math
from concurrent.futures import ProcessPoolExecutor, as_completed

RESULTS_DIR = "/app/aiter/baseline_results_a8w8_blockscale"
M_VALUES = [8, 16, 32, 64, 128, 256, 512, 8192]
NUM_GPUS = 8

NK_PAIRS = [
    (512, 7168), (1024, 8192), (2112, 7168), (3072, 1536),
    (4096, 7168), (4608, 7168), (7168, 256), (7168, 2048),
    (7168, 16384), (7168, 18432), (8192, 1024), (8192, 8192),
    (8192, 32768), (16384, 1536), (24576, 1536), (32768, 512),
    (32768, 8192), (36864, 7168),
]


def run_validation_on_gpu(work_items, gpu_id):
    results = []
    for M, N, K in work_items:
        work_dir = f"/tmp/validate_gpu{gpu_id}"
        os.makedirs(work_dir, exist_ok=True)

        env = os.environ.copy()
        env["HIP_VISIBLE_DEVICES"] = str(gpu_id)

        cmd = [
            "rocprof", "--stats",
            "python", "/app/aiter/op_tests/op_benchmarks/triton/bench_gemm_a8w8_blockscale.py",
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
            "M": M, "N": N, "K": K,
            "avg_ns": avg_ns, "kernel_name": kernel_name,
            "status": status,
        })

    return results


def main():
    print("=== Phase 3: Validation for gemm_a8w8_blockscale (non-preshuffle) ===")
    import triton
    print(f"Triton version: {triton.__version__}")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd="/app/aiter").decode().strip()
    print(f"Branch: {branch}")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    work_items = []
    for N, K in NK_PAIRS:
        for M in M_VALUES:
            work_items.append((M, N, K))

    print(f"Total benchmarks: {len(work_items)}")
    print(f"Using {NUM_GPUS} GPUs")
    print()

    gpu_work = [[] for _ in range(NUM_GPUS)]
    for i, item in enumerate(work_items):
        gpu_work[i % NUM_GPUS].append(item)

    all_results = []
    with ProcessPoolExecutor(max_workers=NUM_GPUS) as executor:
        futures = {}
        for gpu_id in range(NUM_GPUS):
            if gpu_work[gpu_id]:
                f = executor.submit(run_validation_on_gpu, gpu_work[gpu_id], gpu_id)
                futures[f] = gpu_id

        for f in as_completed(futures):
            gpu_id = futures[f]
            try:
                results = f.result()
                all_results.extend(results)
            except Exception as e:
                print(f"GPU{gpu_id} failed: {e}")

    # Save validation results
    validation = {}
    for r in all_results:
        if r["avg_ns"] is not None:
            key = f"{r['M']}-{r['N']}-{r['K']}"
            validation[key] = {
                "time_ns": r["avg_ns"],
                "kernel_name": r["kernel_name"],
            }

    output_file = os.path.join(RESULTS_DIR, "validation_triton36_nonpreshuffle.json")
    with open(output_file, "w") as f:
        json.dump(validation, f, indent=2, sort_keys=True)
    print(f"\nResults saved to {output_file}")

    # Load baseline and compare
    baseline = json.load(open(os.path.join(RESULTS_DIR, "baseline_triton34_nonpreshuffle.json")))

    ok_count = sum(1 for r in all_results if r["status"] == "ok")
    fail_count = sum(1 for r in all_results if r["status"] != "ok")
    print(f"\nBenchmarks: OK={ok_count}, Failed={fail_count}")

    # Per-(N,K) report
    print(f"\n{'M':>6} {'N':>6} {'K':>6} {'3.4(ns)':>10} {'3.6(ns)':>10} {'Delta':>8} {'Status':>8}")
    print("=" * 60)

    ratios = []
    regressions = []
    per_nk_ratios = {}

    for r in sorted(all_results, key=lambda x: (x["N"], x["K"], x["M"])):
        if r["avg_ns"] is None:
            continue
        key = f"{r['M']}-{r['N']}-{r['K']}"
        nk_key = f"{r['N']}-{r['K']}"
        if key not in baseline:
            continue

        old = baseline[key]["time_ns"]
        new = r["avg_ns"]
        delta = (new - old) / old * 100
        status = "OK" if delta <= 3 else "REGRESS"
        ratio = old / new
        ratios.append(ratio)

        if nk_key not in per_nk_ratios:
            per_nk_ratios[nk_key] = []
        per_nk_ratios[nk_key].append(ratio)

        if delta > 3:
            regressions.append((r["M"], r["N"], r["K"], delta))

        print(f"{r['M']:>6} {r['N']:>6} {r['K']:>6} {old:>10.0f} {new:>10.0f} {delta:>+7.1f}% {status:>8}")

    # Overall summary
    if ratios:
        geomean = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        print(f"\n{'=' * 60}")
        print(f"Overall geomean speedup: {geomean:.3f}x ({len(ratios)} shapes)")
        print(f"Regressions (>3%): {len(regressions)}/{len(ratios)}")

        if regressions:
            print(f"\nRegression details:")
            for M, N, K, delta in sorted(regressions):
                print(f"  M={M} N={N} K={K}: +{delta:.1f}%")

        # Per-(N,K) geomean
        print(f"\nPer-(N,K) geomean:")
        for nk_key in sorted(per_nk_ratios.keys(), key=lambda x: tuple(int(v) for v in x.split('-'))):
            rs = per_nk_ratios[nk_key]
            gm = math.exp(sum(math.log(r) for r in rs) / len(rs))
            status = "PASS" if gm >= 1.0 else "FAIL"
            N, K = nk_key.split("-")
            print(f"  N={N:>6} K={K:>6}: {gm:.3f}x ({status})")


if __name__ == "__main__":
    main()
