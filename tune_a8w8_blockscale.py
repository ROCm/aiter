#!/usr/bin/env python3
"""
Phase 2: Tuning gemm_a8w8_blockscale (non-preshuffle) on Triton 3.6.
Dispatches screen.py across 8 GPUs with M-dependent block size ranges.
BK=128 only (kernel constraint: GROUP_K == BLOCK_SIZE_K, and scale block is 128).
"""
import subprocess
import os
import sys
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

TUNNING_DIR = "/app/aiter/aiter/ops/triton/utils/_triton/tunning"
NUM_GPUS = 8

M_VALUES = [8, 16, 32, 64, 128, 256, 512, 8192]

NK_PAIRS = [
    (512, 7168), (1024, 8192), (2112, 7168), (3072, 1536),
    (4096, 7168), (4608, 7168), (7168, 256), (7168, 2048),
    (7168, 16384), (7168, 18432), (8192, 1024), (8192, 8192),
    (8192, 32768), (16384, 1536), (24576, 1536), (32768, 512),
    (32768, 8192), (36864, 7168),
]

# M-dependent block size ranges (from learnings doc)
def get_block_ranges(M):
    if M <= 16:
        return {
            "bm": [4, 8, 16],
            "bn": [16, 32],
            "bk": [128],
        }
    elif M <= 64:
        return {
            "bm": [16, 32, 64],
            "bn": [32, 64],
            "bk": [128],
        }
    elif M <= 512:
        return {
            "bm": [64, 128, 256],
            "bn": [64, 128],
            "bk": [128],
        }
    else:  # M >= 1024
        return {
            "bm": [128, 256, 512],
            "bn": [128, 256],
            "bk": [128],
        }


def build_work_queue():
    """Build list of (M, N, K) tuples to tune."""
    work = []
    for N, K in NK_PAIRS:
        for M in M_VALUES:
            work.append((M, N, K))
    return work


def run_screen(gpu_id, M, N, K):
    """Run screen.py for one shape on one GPU."""
    ranges = get_block_ranges(M)
    bm_str = " ".join(str(x) for x in ranges["bm"])
    bn_str = " ".join(str(x) for x in ranges["bn"])
    bk_str = " ".join(str(x) for x in ranges["bk"])

    cmd = (
        f"python screen.py {M} {N} {K} {gpu_id} ut_a8w8_gemm_blockscale.py "
        f"--block-size-m-range {bm_str} "
        f"--block-size-n-range {bn_str} "
        f"--block-size-k-range {bk_str} "
        f"--num-stages-range 2 3 "
        f"--overwrite"
    )

    try:
        result = subprocess.run(
            cmd.split(), capture_output=True, text=True,
            timeout=3600, cwd=TUNNING_DIR
        )
        # Check if screen log was created
        log_file = f"screen-ut_a8w8_gemm_blockscale.py-{M}-{N}-{K}.log"
        log_path = os.path.join(TUNNING_DIR, log_file)
        if os.path.exists(log_path):
            return {"M": M, "N": N, "K": K, "status": "ok", "log": log_file}
        else:
            return {"M": M, "N": N, "K": K, "status": "no_log",
                    "stderr": result.stderr[-500:] if result.stderr else ""}
    except subprocess.TimeoutExpired:
        return {"M": M, "N": N, "K": K, "status": "timeout"}
    except Exception as e:
        return {"M": M, "N": N, "K": K, "status": f"error: {e}"}


def run_gpu_work(work_items, gpu_id):
    """Run all work items sequentially on one GPU."""
    results = []
    for M, N, K in work_items:
        t0 = time.time()
        r = run_screen(gpu_id, M, N, K)
        elapsed = time.time() - t0
        status = r["status"]
        print(f"  GPU{gpu_id}: M={M} N={N} K={K} -> {status} ({elapsed:.1f}s)", flush=True)
        results.append(r)
    return results


def main():
    print("=== Phase 2: Tuning gemm_a8w8_blockscale (non-preshuffle) ===")
    import triton
    print(f"Triton version: {triton.__version__}")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd="/app/aiter").decode().strip()
    print(f"Branch: {branch}")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    work = build_work_queue()
    print(f"Total shapes to tune: {len(work)}")
    print(f"Using {NUM_GPUS} GPUs")
    print()

    # Distribute round-robin
    gpu_work = [[] for _ in range(NUM_GPUS)]
    for i, item in enumerate(work):
        gpu_work[i % NUM_GPUS].append(item)

    for g in range(NUM_GPUS):
        print(f"  GPU{g}: {len(gpu_work[g])} items")
    print()

    # Run in parallel
    all_results = []
    with ProcessPoolExecutor(max_workers=NUM_GPUS) as executor:
        futures = {}
        for gpu_id in range(NUM_GPUS):
            if gpu_work[gpu_id]:
                f = executor.submit(run_gpu_work, gpu_work[gpu_id], gpu_id)
                futures[f] = gpu_id

        for f in as_completed(futures):
            gpu_id = futures[f]
            try:
                results = f.result()
                all_results.extend(results)
            except Exception as e:
                print(f"GPU{gpu_id} failed: {e}")

    # Summary
    ok = sum(1 for r in all_results if r["status"] == "ok")
    fail = sum(1 for r in all_results if r["status"] != "ok")
    print(f"\n=== Tuning Summary ===")
    print(f"OK: {ok}, Failed: {fail}")
    if fail > 0:
        print("\nFailed:")
        for r in all_results:
            if r["status"] != "ok":
                print(f"  M={r['M']} N={r['N']} K={r['K']} - {r['status']}")
                if "stderr" in r:
                    print(f"    {r['stderr'][:200]}")

    # Now run view-screen.py to generate configs
    print("\n=== Generating configs with view-screen.py ===")
    # Collect unique (N,K) pairs
    nk_pairs = sorted(set((r["N"], r["K"]) for r in all_results if r["status"] == "ok"))
    n_list = [str(n) for n, k in nk_pairs]
    k_list = [str(k) for n, k in nk_pairs]

    if nk_pairs:
        cmd = (
            f"python view-screen.py ut_a8w8_gemm_blockscale.py "
            f"--n-list {' '.join(n_list)} "
            f"--k-list {' '.join(k_list)}"
        )
        print(f"Running: {cmd}")
        result = subprocess.run(
            cmd.split(), capture_output=True, text=True,
            cwd=TUNNING_DIR
        )
        print(result.stdout[-2000:] if result.stdout else "")
        if result.stderr:
            print("STDERR:", result.stderr[-500:])

        # List generated config files
        print("\n=== Generated config files ===")
        for f in sorted(os.listdir(TUNNING_DIR)):
            if f.startswith("gfx950-GEMM-") and f.endswith(".json"):
                print(f"  {f}")


if __name__ == "__main__":
    main()
