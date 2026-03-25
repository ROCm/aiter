#!/usr/bin/env python3
"""
Phase 3: Validation for gemm_a8w8_blockscale_preshuffle.
Apples-to-apples: rocprof --stats on Triton 3.6 with new configs vs Triton 3.4 baseline.
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
    (2112, 7168), (3072, 1536), (4096, 512), (4096, 7168),
    (4608, 7168), (7168, 2048), (7168, 2304), (7168, 16384),
    (7168, 18432), (8192, 8192), (24576, 1536), (32768, 512),
    (36864, 7168),
]


def run_validation_on_gpu(work_items, gpu_id):
    results = []
    for M, N, K in work_items:
        work_dir = f"/tmp/validate_preshuffle_gpu{gpu_id}"
        os.makedirs(work_dir, exist_ok=True)
        for fname in os.listdir(work_dir):
            if fname.startswith("results"):
                os.remove(os.path.join(work_dir, fname))

        env = os.environ.copy()
        env["HIP_VISIBLE_DEVICES"] = str(gpu_id)

        cmd = [
            "rocprof", "--stats",
            "python", "/app/aiter/op_tests/op_benchmarks/triton/bench_gemm_a8w8_blockscale.py",
            "--shape", str(M), str(N), str(K),
            "--metric", "time", "--layout", "TN",
            "-preshuffle",
        ]

        avg_ns = None
        status = "unknown"

        try:
            subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300, cwd=work_dir)
            stats_file = os.path.join(work_dir, "results.stats.csv")
            if os.path.exists(stats_file):
                with open(stats_file) as f:
                    for row in csv.reader(f):
                        if len(row) >= 4 and "gemm_a8w8_blockscale" in row[0]:
                            avg_ns = float(row[3])
                            status = "ok"
                            break
                    else:
                        status = "kernel_not_found"
            else:
                status = "no_stats_file"
        except subprocess.TimeoutExpired:
            status = "timeout"
        except Exception as e:
            status = f"error: {e}"

        avg_str = f"{avg_ns:.0f}ns" if avg_ns else status
        print(f"  GPU{gpu_id}: M={M} N={N} K={K} -> {avg_str}", flush=True)
        results.append({"M": M, "N": N, "K": K, "avg_ns": avg_ns, "status": status})

    return results


def main():
    print("=== Phase 3: Validation for gemm_a8w8_blockscale_preshuffle ===")
    import triton
    print(f"Triton version: {triton.__version__}")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    work_items = [(M, N, K) for N, K in NK_PAIRS for M in M_VALUES]
    print(f"Total benchmarks: {len(work_items)}")

    gpu_work = [[] for _ in range(NUM_GPUS)]
    for i, item in enumerate(work_items):
        gpu_work[i % NUM_GPUS].append(item)

    all_results = []
    with ProcessPoolExecutor(max_workers=NUM_GPUS) as executor:
        futures = {executor.submit(run_validation_on_gpu, gpu_work[g], g): g
                   for g in range(NUM_GPUS) if gpu_work[g]}
        for f in as_completed(futures):
            try:
                all_results.extend(f.result())
            except Exception as e:
                print(f"GPU{futures[f]} failed: {e}")

    # Save
    validation = {}
    for r in all_results:
        if r["avg_ns"]:
            validation[f"{r['M']}-{r['N']}-{r['K']}"] = {"time_ns": r["avg_ns"]}
    with open(os.path.join(RESULTS_DIR, "validation_triton36_preshuffle.json"), "w") as f:
        json.dump(validation, f, indent=2, sort_keys=True)

    # Compare
    baseline = json.load(open(os.path.join(RESULTS_DIR, "baseline_triton34_preshuffle.json")))

    print(f"\n{'M':>6} {'N':>6} {'K':>6} {'3.4 (ns)':>12} {'3.6 (ns)':>12} {'Delta':>8} {'Status':>8}")
    print("=" * 68)

    ratios = []
    regressions = []
    per_nk = {}
    prev_nk = None

    for r in sorted(all_results, key=lambda x: (x["N"], x["K"], x["M"])):
        if not r["avg_ns"]:
            continue
        key = f"{r['M']}-{r['N']}-{r['K']}"
        nk = f"{r['N']}-{r['K']}"
        if key not in baseline:
            continue
        old = baseline[key]["time_ns"]
        new = r["avg_ns"]
        delta = (new - old) / old * 100
        status = "OK" if delta <= 3 else "REGRESS"
        ratio = old / new
        ratios.append(ratio)
        per_nk.setdefault(nk, []).append(ratio)
        if delta > 3:
            regressions.append((r["M"], r["N"], r["K"], old, new, delta))
        if prev_nk and prev_nk != nk:
            print()
        prev_nk = nk
        print(f"{r['M']:>6} {r['N']:>6} {r['K']:>6} {old:>12.0f} {new:>12.0f} {delta:>+7.1f}% {status:>8}")

    if ratios:
        geomean = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        print(f"\n{'=' * 68}")
        print(f"Overall geomean speedup: {geomean:.3f}x ({len(ratios)} shapes)")
        print(f"Regressions (>3%): {len(regressions)}/{len(ratios)}")

        if regressions:
            print(f"\nRegression details:")
            for M, N, K, old, new, delta in sorted(regressions):
                print(f"  M={M:>5} N={N:>5} K={K:>5}: {old:>10.0f} -> {new:>10.0f}  {delta:>+6.1f}%")

        print(f"\nPer-(N,K) geomean:")
        for nk in sorted(per_nk, key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1]))):
            rs = per_nk[nk]
            gm = math.exp(sum(math.log(r) for r in rs) / len(rs))
            N, K = nk.split("-")
            print(f"  N={N:>6} K={K:>6}: {gm:.3f}x ({'PASS' if gm >= 1.0 else 'FAIL'})")


if __name__ == "__main__":
    main()
