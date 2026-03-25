#!/usr/bin/env python3
"""
Clean baseline collection for gemm_a8w8_blockscale (both variants).
Sequential on GPU 0 — no parallel conflicts.
"""
import subprocess, csv, json, os, time, math

RESULTS_DIR = "/app/aiter/baseline_results_a8w8_blockscale"
os.makedirs(RESULTS_DIR, exist_ok=True)

M_VALUES = [8, 16, 32, 64, 128, 256, 512, 8192]

NK_NONPRESHUFFLE = [
    (512, 7168), (1024, 8192), (2112, 7168), (3072, 1536),
    (4096, 7168), (4608, 7168), (7168, 256), (7168, 2048),
    (7168, 16384), (7168, 18432), (8192, 1024), (8192, 8192),
    (8192, 32768), (16384, 1536), (24576, 1536), (32768, 512),
    (32768, 8192), (36864, 7168),
]

NK_PRESHUFFLE = [
    (2112, 7168), (3072, 1536), (4096, 512), (4096, 7168),
    (4608, 7168), (7168, 2048), (7168, 2304), (7168, 16384),
    (7168, 18432), (8192, 8192), (24576, 1536), (32768, 512),
    (36864, 7168),
]


def rocprof_bench(M, N, K, preshuffle=False):
    work_dir = "/tmp/baseline_clean"
    os.makedirs(work_dir, exist_ok=True)
    for f in os.listdir(work_dir):
        if f.startswith("results"):
            os.remove(os.path.join(work_dir, f))

    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = "0"

    cmd = [
        "rocprof", "--stats",
        "python", "/app/aiter/op_tests/op_benchmarks/triton/bench_gemm_a8w8_blockscale.py",
        "--shape", str(M), str(N), str(K),
        "--metric", "time", "--layout", "TN",
    ]
    if preshuffle:
        cmd.append("-preshuffle")

    try:
        subprocess.run(cmd, env=env, capture_output=True, timeout=300, cwd=work_dir)
        stats = os.path.join(work_dir, "results.stats.csv")
        if os.path.exists(stats):
            with open(stats) as f:
                for row in csv.reader(f):
                    if len(row) >= 4 and "gemm_a8w8_blockscale" in row[0]:
                        return float(row[3])
    except Exception as e:
        return None
    return None


def collect(nk_pairs, preshuffle, output_file):
    variant = "preshuffle" if preshuffle else "nonpreshuffle"
    total = len(nk_pairs) * len(M_VALUES)
    print(f"\n=== Collecting {variant} baselines ({total} shapes, sequential GPU 0) ===")

    results = {}
    count = 0
    for N, K in nk_pairs:
        for M in M_VALUES:
            count += 1
            ns = rocprof_bench(M, N, K, preshuffle)
            key = f"{M}-{N}-{K}"
            if ns:
                results[key] = {"time_ns": ns}
                print(f"  [{count}/{total}] M={M} N={N} K={K}: {ns:.0f}ns", flush=True)
            else:
                print(f"  [{count}/{total}] M={M} N={N} K={K}: FAILED", flush=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"Saved {len(results)} results to {output_file}")


def main():
    import triton
    print(f"Triton: {triton.__version__}")
    print(f"Branch: {subprocess.check_output(['git', 'branch', '--show-current'], cwd='/app/aiter').decode().strip()}")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    collect(NK_NONPRESHUFFLE, False,
            os.path.join(RESULTS_DIR, "baseline_clean_nonpreshuffle.json"))
    collect(NK_PRESHUFFLE, True,
            os.path.join(RESULTS_DIR, "baseline_clean_preshuffle.json"))


if __name__ == "__main__":
    main()
