# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Robust micro-tuner for the bpreshuffle CKTile FP8 blockscale GEMM that sweeps
splitK and y_is_zeroed independently using many iterations and median timing
for noise robustness.

Two modes:

1. Spot benchmark (default when --shape is given): sweep splitK and
   y_is_zeroed for a handful of kernelIds on one shape and print all rows.

2. Tune mode (when --untuned-csv is given): sweep all kernelIds x splitK
   for each shape in the input CSV, pick the median-best (kernelId, splitK)
   per shape, and write a tuned CSV in the same format the regular tune
   script produces (gfx,cu_num,M,N,K,libtype,kernelId,splitK,us,kernelName,
   tflops,bw,errRatio).  The chosen rows correspond to the producer-fused
   demo target (y_is_zeroed=True, kernel skips Y.zero_()).

Run examples:

    # Spot bench
    PYTHONPATH=/home/AMD/samremes/dev/aiter python op_tests/bench_splitk_yzeroed.py \
        --shape 1,2048,4096 --kernelIds 0,1,13,16 --iters 200 --warmup 20

    # Robust tune
    PYTHONPATH=/home/AMD/samremes/dev/aiter python op_tests/bench_splitk_yzeroed.py \
        --untuned-csv aiter/configs/zero_init_demo/qwen3_next_80b_a3b_per1x128_untuned.csv \
        --output-csv  aiter/configs/zero_init_demo/qwen3_next_80b_a3b_per1x128_cktile_splitk_gfx950.csv \
        --iters 100 --warmup 10 --maxSplitK 3
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys

import torch

import aiter
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight

# Make the codegen-side cktile instance dict importable so we can enumerate
# all kernel candidates.
_AITER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_AITER_ROOT, "csrc", "ck_gemm_a8w8_blockscale"))
from gemm_a8w8_blockscale_cktile_instance import (  # noqa: E402
    candidate_kernels_cktile_dict,
)


def _parse_shape(s: str) -> tuple[int, int, int]:
    parts = s.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"shape must be M,N,K, got {s!r}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def _gen_inputs(M: int, N: int, K: int, seed: int = 12345, device: str = "cuda"):
    """Mirror generate_data() from gemm_a8w8_blockscale_tune.py."""
    torch.manual_seed(seed)
    block_n, block_k = 128, 128
    scale_n = (N + block_n - 1) // block_n
    scale_k = (K + block_k - 1) // block_k
    x = (torch.rand((M, K), dtype=dtypes.fp16, device=device) / 10).to(dtypes.fp8)
    weight = (torch.rand((N, K), dtype=dtypes.fp16, device=device) / 10).to(dtypes.fp8)
    x_scale = torch.rand([M, scale_k], dtype=dtypes.fp32, device=device)
    w_scale = torch.rand([scale_n, scale_k], dtype=dtypes.fp32, device=device)
    weight_shuffle = shuffle_weight(weight, layout=(16, 16))
    x_scale_t = x_scale.transpose(0, 1).contiguous().view(*x_scale.shape)
    out = torch.zeros(M, N, dtype=dtypes.bf16, device=device)
    return x, weight_shuffle, x_scale_t, w_scale, out


def _bench_one(
    *,
    x,
    weight_shuffle,
    x_scale_t,
    w_scale,
    out,
    kernelId: int,
    splitK: int,
    y_is_zeroed: bool,
    iters: int,
    warmup: int,
    rezero_between_iters: bool,
) -> dict:
    """Time one (kernelId, splitK, y_is_zeroed) configuration.

    When ``rezero_between_iters`` is True, ``out`` is zeroed *outside* the
    cudaEvent-timed region so atomic-add accumulation doesn't pollute the
    splitK>0 timing across iterations.
    """
    samples_us: list[float] = []
    for _ in range(warmup):
        if rezero_between_iters:
            out.zero_()
        try:
            aiter.gemm_a8w8_blockscale_bpreshuffle_cktile_tune(
                x, weight_shuffle, x_scale_t, w_scale, out,
                kernelId, splitK, True, y_is_zeroed,
            )
        except RuntimeError as e:
            return {"error": str(e), "median_us": float("inf")}
    torch.cuda.synchronize()

    for _ in range(iters):
        if rezero_between_iters:
            out.zero_()
            torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            aiter.gemm_a8w8_blockscale_bpreshuffle_cktile_tune(
                x, weight_shuffle, x_scale_t, w_scale, out,
                kernelId, splitK, True, y_is_zeroed,
            )
        except RuntimeError as e:
            return {"error": str(e), "median_us": float("inf")}
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0)

    samples_us.sort()
    return {
        "median_us": statistics.median(samples_us),
        "min_us": min(samples_us),
        "p10_us": samples_us[max(0, int(0.10 * len(samples_us)) - 1)],
        "p90_us": samples_us[min(len(samples_us) - 1, int(0.90 * len(samples_us)))],
        "mean_us": statistics.mean(samples_us),
        "stdev_us": statistics.stdev(samples_us) if len(samples_us) > 1 else 0.0,
        "n": len(samples_us),
    }


def _spot_bench(args) -> int:
    M, N, K = args.shape
    print(f"# shape: M={M}, N={N}, K={K}")
    print(f"# iters={args.iters} warmup={args.warmup}")
    print(f"# device: {torch.cuda.get_device_name(0)}")
    print()

    x, weight_shuffle, x_scale_t, w_scale, out = _gen_inputs(
        M, N, K, seed=args.seed
    )

    cols = ("kernelId", "splitK", "KBatch", "y_is_zeroed",
            "median_us", "min_us", "p10_us", "p90_us", "stdev_us")
    print("\t".join(cols))

    for kid in args.kernelIds:
        for splitK in range(args.maxSplitK + 1):
            kbatch = 2 ** splitK
            for y_is_zeroed in (False, True):
                rezero = (kbatch > 1) and y_is_zeroed
                stats = _bench_one(
                    x=x, weight_shuffle=weight_shuffle, x_scale_t=x_scale_t,
                    w_scale=w_scale, out=out,
                    kernelId=kid, splitK=splitK, y_is_zeroed=y_is_zeroed,
                    iters=args.iters, warmup=args.warmup,
                    rezero_between_iters=rezero,
                )
                if "error" in stats:
                    print(f"{kid}\t{splitK}\t{kbatch}\t{y_is_zeroed}\tERROR")
                    continue
                print(
                    f"{kid}\t{splitK}\t{kbatch}\t{y_is_zeroed}\t"
                    f"{stats['median_us']:.3f}\t{stats['min_us']:.3f}\t"
                    f"{stats['p10_us']:.3f}\t{stats['p90_us']:.3f}\t"
                    f"{stats['stdev_us']:.3f}"
                )
    return 0


def _gfx_string() -> str:
    props = torch.cuda.get_device_properties(0)
    return getattr(props, "gcnArchName", "gfx950").split(":")[0]


def _cu_count() -> int:
    return torch.cuda.get_device_properties(0).multi_processor_count


def _tune_robust(args) -> int:
    """Drive the full untuned CSV with high-iter median timing."""
    in_path = args.untuned_csv
    out_path = args.output_csv
    if not os.path.exists(in_path):
        print(f"ERROR: untuned CSV not found: {in_path}", file=sys.stderr)
        return 1

    gfx = _gfx_string()
    cu_num = _cu_count()
    print(f"# device: {torch.cuda.get_device_name(0)}  gfx={gfx}  cu_num={cu_num}")
    print(f"# iters={args.iters} warmup={args.warmup} maxSplitK={args.maxSplitK}")
    print(f"# y_is_zeroed=True (modeling producer-fused zero-init)")
    print(f"# input:  {in_path}")
    print(f"# output: {out_path}")
    print()

    shapes: list[tuple[int, int, int]] = []
    with open(in_path, newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            shapes.append((int(row["M"]), int(row["N"]), int(row["K"])))

    kernel_ids = sorted(candidate_kernels_cktile_dict.keys())
    print(f"# {len(shapes)} shapes x {len(kernel_ids)} kernelIds x "
          f"{args.maxSplitK + 1} splitK = "
          f"{len(shapes) * len(kernel_ids) * (args.maxSplitK + 1)} configs")
    print()

    fieldnames = [
        "gfx", "cu_num", "M", "N", "K", "libtype", "kernelId", "splitK",
        "us", "kernelName", "tflops", "bw", "errRatio",
    ]
    out_rows = []
    for shape_idx, (M, N, K) in enumerate(shapes):
        x, ws, xs, ws_, out = _gen_inputs(M, N, K, seed=12345 + shape_idx)
        best = None
        for kid in kernel_ids:
            for splitK in range(args.maxSplitK + 1):
                kbatch = 2 ** splitK
                rezero = kbatch > 1
                stats = _bench_one(
                    x=x, weight_shuffle=ws, x_scale_t=xs, w_scale=ws_, out=out,
                    kernelId=kid, splitK=splitK, y_is_zeroed=True,
                    iters=args.iters, warmup=args.warmup,
                    rezero_between_iters=rezero,
                )
                if "error" in stats:
                    continue
                if best is None or stats["median_us"] < best["median_us"]:
                    best = {
                        "kernelId": kid,
                        "splitK": splitK,
                        "median_us": stats["median_us"],
                        "min_us": stats["min_us"],
                    }
        if best is None:
            print(f"  shape ({M},{N},{K}): NO valid kernel — using kernelId=0 splitK=0")
            best = {"kernelId": 0, "splitK": 0, "median_us": float("inf"),
                    "min_us": float("inf")}
        kernel_obj = candidate_kernels_cktile_dict[best["kernelId"]]
        kname = kernel_obj.name
        flops = 2.0 * M * N * K
        bytes_ = M * K + N * K + M * N * 2  # rough fp8 in, bf16 out
        us = best["median_us"]
        tflops = flops / (us * 1e-6) / 1e12 if us > 0 else 0.0
        bw = bytes_ / (us * 1e-6) / 1e9 if us > 0 else 0.0
        out_rows.append({
            "gfx": gfx, "cu_num": cu_num, "M": M, "N": N, "K": K,
            "libtype": "cktile",
            "kernelId": best["kernelId"], "splitK": best["splitK"],
            "us": f"{us:.4f}",
            "kernelName": kname,
            "tflops": f"{tflops:.2f}",
            "bw": f"{bw:.2f}",
            "errRatio": "0.0",
        })
        print(
            f"  shape ({M},{N},{K}): kernelId={best['kernelId']} "
            f"splitK={best['splitK']} kbatch={2**best['splitK']} "
            f"median_us={us:.3f} (min={best['min_us']:.3f}) "
            f"-> {tflops:.2f} TFLOPS"
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(out_rows)

    sk_dist: dict[int, int] = {}
    for r in out_rows:
        sk = int(r["splitK"])
        sk_dist[sk] = sk_dist.get(sk, 0) + 1
    print()
    print(f"Wrote {len(out_rows)} rows to {out_path}")
    print(f"splitK distribution: {dict(sorted(sk_dist.items()))}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", type=_parse_shape, default=None,
                    help="Spot-bench mode: M,N,K shape to benchmark.")
    ap.add_argument("--kernelIds", type=_parse_int_list,
                    default=[0, 1, 13, 16],
                    help="Spot-bench: comma-separated kernel IDs to sweep.")
    ap.add_argument("--maxSplitK", type=int, default=3,
                    help="Sweep splitK in [0, maxSplitK]; KBatch=2^splitK "
                    "(default 3).")
    ap.add_argument("--iters", type=int, default=100,
                    help="Timed iterations per config (default: 100).")
    ap.add_argument("--warmup", type=int, default=10,
                    help="Warmup iterations per config (default: 10).")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--untuned-csv", default=None,
                    help="Tune mode: input CSV with M,N,K columns to tune.")
    ap.add_argument("--output-csv", default=None,
                    help="Tune mode: output tuned CSV path.")
    args = ap.parse_args()

    if args.untuned_csv is not None:
        if args.output_csv is None:
            print("ERROR: --output-csv required with --untuned-csv", file=sys.stderr)
            return 1
        return _tune_robust(args)

    if args.shape is None:
        args.shape = (1, 2048, 4096)
    return _spot_bench(args)


if __name__ == "__main__":
    raise SystemExit(main())
