#!/usr/bin/env python3
"""
Phase 2: Tuning gemm_a8w8_blockscale_preshuffle on Triton 3.6.
Dispatches screen.py across 8 GPUs with M-dependent block size ranges.
BK=128 only. Handles per-shape valid NUM_KSPLIT ranges.
"""
import subprocess
import os
import time
import math
from concurrent.futures import ProcessPoolExecutor, as_completed

TUNNING_DIR = "/app/aiter/aiter/ops/triton/utils/_triton/tunning"
NUM_GPUS = 8

M_VALUES = [8, 16, 32, 64, 128, 256, 512, 8192]

NK_PAIRS = [
    (2112, 7168), (3072, 1536), (4096, 512), (4096, 7168),
    (4608, 7168), (7168, 2048), (7168, 2304), (7168, 16384),
    (7168, 18432), (8192, 8192), (24576, 1536), (32768, 512),
    (36864, 7168),
]


def get_block_ranges(M):
    if M <= 16:
        return {"bm": [4, 8, 16], "bn": [16, 32], "bk": [128]}
    elif M <= 64:
        return {"bm": [16, 32, 64], "bn": [32, 64], "bk": [128]}
    elif M <= 512:
        return {"bm": [64, 128, 256], "bn": [64, 128], "bk": [128]}
    else:
        return {"bm": [128, 256, 512], "bn": [128, 256], "bk": [128]}


def get_valid_splitk(K, M):
    """Compute valid NUM_KSPLIT values for given K."""
    BK = 128
    default_spk = [3, 4, 7, 8, 14, 16, 28]
    valid = [1]
    for spk in default_spk:
        sbk = math.ceil(K / spk)
        if K % spk == 0 and sbk >= BK:
            valid.append(spk)
    # For M=8192, skip split-K (too slow to tune)
    if M >= 1024:
        valid = [1]
    return valid


def run_screen(gpu_id, M, N, K):
    ranges = get_block_ranges(M)
    bm_str = " ".join(str(x) for x in ranges["bm"])
    bn_str = " ".join(str(x) for x in ranges["bn"])
    bk_str = " ".join(str(x) for x in ranges["bk"])
    spk = get_valid_splitk(K, M)
    spk_str = " ".join(str(x) for x in spk)

    cmd = (
        f"python screen.py {M} {N} {K} {gpu_id} ut_a8w8_gemm_blockscale_preshuffle.py "
        f"--block-size-m-range {bm_str} "
        f"--block-size-n-range {bn_str} "
        f"--block-size-k-range {bk_str} "
        f"--num-ksplit-range {spk_str} "
        f"--num-stages-range 2 3 "
        f"--overwrite"
    )

    try:
        result = subprocess.run(
            cmd.split(), capture_output=True, text=True,
            timeout=3600, cwd=TUNNING_DIR
        )
        log_file = f"screen-ut_a8w8_gemm_blockscale_preshuffle.py-{M}-{N}-{K}.log"
        log_path = os.path.join(TUNNING_DIR, log_file)
        if os.path.exists(log_path):
            with open(log_path) as f:
                content = f.read()
            complete = "Screen complete" in content
            cases = content.count("screencase")
            return {"M": M, "N": N, "K": K, "status": "ok" if complete else "partial",
                    "cases": cases}
        return {"M": M, "N": N, "K": K, "status": "no_log", "cases": 0}
    except subprocess.TimeoutExpired:
        return {"M": M, "N": N, "K": K, "status": "timeout", "cases": 0}
    except Exception as e:
        return {"M": M, "N": N, "K": K, "status": f"error: {e}", "cases": 0}


def run_gpu_work(work_items, gpu_id):
    results = []
    for M, N, K in work_items:
        t0 = time.time()
        r = run_screen(gpu_id, M, N, K)
        elapsed = time.time() - t0
        print(f"  GPU{gpu_id}: M={M} N={N} K={K} -> {r['status']} ({r['cases']} results, {elapsed:.0f}s)", flush=True)
        results.append(r)
    return results


def main():
    print("=== Phase 2: Tuning gemm_a8w8_blockscale_preshuffle ===")
    import triton
    print(f"Triton version: {triton.__version__}")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd="/app/aiter").decode().strip()
    print(f"Branch: {branch}")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    work = []
    for N, K in NK_PAIRS:
        for M in M_VALUES:
            work.append((M, N, K))

    print(f"Total shapes to tune: {len(work)} ({len(NK_PAIRS)} NK pairs x {len(M_VALUES)} M values)")
    print(f"Using {NUM_GPUS} GPUs")
    print()

    gpu_work = [[] for _ in range(NUM_GPUS)]
    for i, item in enumerate(work):
        gpu_work[i % NUM_GPUS].append(item)

    for g in range(NUM_GPUS):
        print(f"  GPU{g}: {len(gpu_work[g])} items")
    print()

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
    no_results = sum(1 for r in all_results if r["cases"] == 0)
    print(f"\n=== Tuning Summary ===")
    print(f"Complete: {ok}, Failed: {fail}, No results: {no_results}")
    if fail > 0:
        print("\nFailed/incomplete:")
        for r in all_results:
            if r["status"] != "ok":
                print(f"  M={r['M']} N={r['N']} K={r['K']} - {r['status']} ({r['cases']} results)")

    # Generate configs with view-screen.py
    print("\n=== Generating configs with view-screen.py ===")
    nk_pairs = sorted(set((r["N"], r["K"]) for r in all_results if r["status"] == "ok" and r["cases"] > 0))
    if nk_pairs:
        n_list = " ".join(str(n) for n, k in nk_pairs)
        k_list = " ".join(str(k) for n, k in nk_pairs)
        cmd = f"python view-screen.py ut_a8w8_gemm_blockscale_preshuffle.py --n-list {n_list} --k-list {k_list}"
        print(f"Running: {cmd}")
        result = subprocess.run(cmd.split(), capture_output=True, text=True, cwd=TUNNING_DIR)
        # Show config generation results
        for line in result.stdout.split("\n"):
            if "created" in line or "Warning" in line or "No file" in line or line.startswith("M\t"):
                print(line)
            elif any(line.strip().startswith(str(m)) for m in M_VALUES):
                print(line)

        print("\n=== Generated config files ===")
        for f in sorted(os.listdir(TUNNING_DIR)):
            if f.startswith("gfx950-GEMM-A8W8_BLOCKSCALE_PRESHUFFLED") and f.endswith(".json"):
                print(f"  {f}")


if __name__ == "__main__":
    main()
