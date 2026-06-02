"""Benchmark + autotune harness for preshuffle_gemm_blockscaled (handcraft).

Reproduces the recorded performance and runs a tile/block-swizzle sweep for a
given (M, N, K). Designed to be run ONE CONFIG PER PROCESS for the sweep (the
flydsl AST rewriter can corrupt across many in-process compiles), so the sweep
driver shells out to this script per config.

Single-config bench / correctness:
    CUDA_VISIBLE_DEVICES=6 HIP_VISIBLE_DEVICES=6 \
      PYTHONPATH=$PWD:$PYTHONPATH /opt/venv/bin/python \
      preshuffle_gemm_blockscaled_perf/bench_and_tune.py \
        --M 32768 --N 12288 --K 2048 --tm 128 --tn 128 --tk 128 --bsw 2 \
        --scale fp32_post_mfma

Full sweep for a shape (shells out per config):
    ... bench_and_tune.py --M 32768 --N 12288 --K 2048 --scale fp32_post_mfma --sweep
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _bench_one(M, N, K, tm, tn, tk, bsw, scale, iters=120, warmup=1500, reps=6):
    import torch
    from aiter.ops.flydsl.kernels.preshuffle_gemm_blockscaled import (
        compile_preshuffle_gemm_blockscaled as C,
    )
    from op_tests.test_preshuffle_gemm_blockscaled import (
        build_inputs, build_inputs_fp32,
    )
    torch.cuda.init()
    build = build_inputs if scale == "ue8m0" else build_inputs_fp32
    a, b, sa, sb, ref = build(M, N, K)
    c = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    try:
        launch = C(M=M, N=N, K=K, tile_m=tm, tile_n=tn, tile_k=tk,
                   block_swizzle_n=bsw, scale_format=scale)
    except Exception as e:
        print(f"FAIL {str(e)[:70]}")
        return
    s = torch.cuda.current_stream().cuda_stream
    for _ in range(warmup):
        launch(c, a, b, sa, sb, M, s)
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    best = 1e9
    for _ in range(reps):
        e0.record()
        for _ in range(iters):
            launch(c, a, b, sa, sb, M, s)
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) / iters)
    tf = 2.0 * M * N * K / (best * 1e-3) / 1e12
    # tolerance: 0.15 abs covers the benign +-1 bf16-ULP at large magnitudes.
    nf = int(((c.float() - ref.float()).abs() > 0.15).sum())
    print(f"OK {tf:.1f} {best * 1e3:.1f} nf={nf}")


def _sweep(M, N, K, scale):
    here = os.path.abspath(__file__)
    results = []
    for tm in (64, 128, 256):
        for tn in (128, 256):
            for tk in (128, 256):
                if N % tn or K % tk:
                    continue
                gy = N // tn
                for bsw in (0, 1, 2, 4, 8):
                    if bsw and gy % bsw:
                        continue
                    env = dict(os.environ)
                    out = subprocess.run(
                        [sys.executable, here,
                         "--M", str(M), "--N", str(N), "--K", str(K),
                         "--tm", str(tm), "--tn", str(tn), "--tk", str(tk),
                         "--bsw", str(bsw), "--scale", scale],
                        capture_output=True, text=True, env=env, timeout=300,
                    )
                    line = ""
                    for ln in out.stdout.splitlines():
                        if ln.startswith(("OK", "FAIL")):
                            line = ln
                    tag = f"t{tm}x{tn}x{tk} bsw{bsw}"
                    print(f"  {tag:22s}: {line}")
                    if line.startswith("OK"):
                        tf = float(line.split()[1])
                        results.append((tf, tag))
    results.sort(reverse=True)
    if results:
        print(f"\nWINNER: {results[0][1]} = {results[0][0]:.0f} TF")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--tm", type=int, default=128)
    ap.add_argument("--tn", type=int, default=128)
    ap.add_argument("--tk", type=int, default=128)
    ap.add_argument("--bsw", type=int, default=0)
    ap.add_argument("--scale", default="fp32_post_mfma",
                    choices=["ue8m0", "fp32", "fp32_post_mfma"])
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    if args.sweep:
        print(f"sweep {args.M}x{args.N}x{args.K} scale={args.scale}")
        _sweep(args.M, args.N, args.K, args.scale)
    else:
        _bench_one(args.M, args.N, args.K, args.tm, args.tn, args.tk,
                   args.bsw, args.scale)


if __name__ == "__main__":
    main()
