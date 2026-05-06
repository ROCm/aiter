# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Three-config end-to-end demo for the SplitK zero-init fusion mechanism.

Mimics the per_1x128 path of ``LinearBase.forward`` in ATOM
(``atom/model_ops/linear.py``), without pulling in distributed/ATOM.
For each Qwen3-Next decode shape, this runs the two-step "producer
quant -> bpreshuffle CKTile GEMM" pipeline under one of three modes:

  none          : baseline. Producer is plain per_group_quant_hip; GEMM
                  reads splitK=0 from the tuned CSV; no Y pre-alloc.

  splitk        : pre-allocate Y; producer is plain per_group_quant_hip;
                  GEMM reads splitK>0 from the tuned CSV and zeros Y
                  itself before the atomic-add. The trivial fill kernel
                  is visible in the trace just before the GEMM.

  splitk_fused  : pre-allocate Y; producer is per_group_quant_hip with
                  gemm_out_zero_init=Y so it absorbs the zero-fill;
                  GEMM is invoked with y_is_zeroed=True and skips its
                  own Y.zero_(). No fill kernel in the trace.

Usage (run once per mode, with the matching CSV; remember to delete
the bpreshuffle CKTile .so when switching CSVs because the kernel
manifest is built at compile time):

    python op_tests/bench_zero_init_splitk_demo.py \
        --mode none \
        --tuned-csv aiter/configs/zero_init_demo/robust/qwen3_next_80b_a3b_per1x128_cktile_nosplitk_gfx950.csv \
        --shapes-csv aiter/configs/zero_init_demo/qwen3_next_80b_a3b_per1x128_untuned.csv \
        --iters 200 --warmup 30 --out demo_none.csv

    python op_tests/bench_zero_init_splitk_demo.py \
        --mode splitk \
        --tuned-csv aiter/configs/zero_init_demo/robust/qwen3_next_80b_a3b_per1x128_cktile_splitk_yz_gfx950.csv \
        --shapes-csv aiter/configs/zero_init_demo/qwen3_next_80b_a3b_per1x128_untuned.csv \
        --iters 200 --warmup 30 --out demo_splitk.csv

    python op_tests/bench_zero_init_splitk_demo.py \
        --mode splitk_fused \
        --tuned-csv aiter/configs/zero_init_demo/robust/qwen3_next_80b_a3b_per1x128_cktile_splitk_yz_gfx950.csv \
        --shapes-csv aiter/configs/zero_init_demo/qwen3_next_80b_a3b_per1x128_untuned.csv \
        --iters 200 --warmup 30 --out demo_splitk_fused.csv

Optional ``--trace-dir DIR`` enables a torch-profiler capture of one
"warm" iteration per shape (writes one chrome trace per shape).
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
from aiter.ops.quant import per_group_quant_hip
from aiter.ops.gemm_op_a8w8 import (
    gemm_a8w8_blockscale_bpreshuffle,
    get_CKGEMM_config,
)
from aiter.ops.shuffle import shuffle_weight


VALID_MODES = ("none", "splitk", "splitk_fused")


def _shape_splitk(M: int, N: int, K: int, tuned_file: str) -> int:
    """Return the splitK column from the tuned CSV for (M, N, K), or 0.

    Mirrors the per-shape lookup ATOM's LinearBase.forward does so the
    demo only stages the producer-side zero-init when the tuner picked
    splitK > 0 (otherwise the kernel does no atomic-add pass and the
    pre-zero is wasted bandwidth).
    """
    try:
        cfg = get_CKGEMM_config(M, N, K, tuned_file=tuned_file)
    except Exception:  # noqa: BLE001
        return 0
    if cfg is None:
        return 0
    try:
        return int(cfg.get("splitK", 0))
    except (TypeError, ValueError):
        return 0


def _parse_shape(s: str) -> tuple[int, int, int]:
    parts = s.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"shape must be M,N,K, got {s!r}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _read_shapes_csv(path: str) -> list[tuple[int, int, int]]:
    shapes: list[tuple[int, int, int]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            shapes.append((int(row["M"]), int(row["N"]), int(row["K"])))
    return shapes


def _gen_inputs(M: int, N: int, K: int, seed: int = 12345, device: str = "cuda"):
    """Allocate BF16 input + preshuffled FP8 weight + scales.

    Mirrors ``aiter/op_tests/bench_splitk_yzeroed.py::_gen_inputs`` but
    keeps the activation in BF16 because the producer (per_group_quant_hip)
    does the BF16->FP8 quant.
    """
    torch.manual_seed(seed)
    block_n, block_k = 128, 128
    scale_n = (N + block_n - 1) // block_n
    scale_k = (K + block_k - 1) // block_k
    x_bf16 = (
        torch.rand((M, K), dtype=dtypes.bf16, device=device) / 10
    )
    weight = (
        torch.rand((N, K), dtype=dtypes.fp16, device=device) / 10
    ).to(dtypes.fp8)
    weight_shuffle = shuffle_weight(weight, layout=(16, 16))
    w_scale = torch.rand([scale_n, scale_k], dtype=dtypes.fp32, device=device)
    return x_bf16, weight_shuffle, w_scale


def _producer_then_gemm(
    *,
    x_bf16: torch.Tensor,
    weight_shuffle: torch.Tensor,
    w_scale: torch.Tensor,
    mode: str,
    tuned_file: str,
    M: int,
    N: int,
    out_buf: torch.Tensor | None,
    do_fused_zero_init: bool,
):
    """One LinearBase-equivalent forward pass under ``mode``.

    ``out_buf`` must be supplied (and reused across iterations) for the
    splitk/splitk_fused modes so we measure the production-style code
    path where the caller owns Y.

    ``do_fused_zero_init`` controls whether the producer absorbs the
    zero-fill for this iteration. It only flips on when ``mode`` is
    ``splitk_fused`` AND the tuned CSV picked splitK > 0 for the shape;
    callers compute it once per (mode, shape) via :func:`_shape_splitk`.
    """
    if mode == "none":
        x_q, x_scale = per_group_quant_hip(
            x_bf16,
            quant_dtype=dtypes.fp8,
            group_size=128,
            transpose_scale=True,
        )
        return gemm_a8w8_blockscale_bpreshuffle(
            x_q, weight_shuffle, x_scale, w_scale,
            dtype=dtypes.bf16,
            tuned_file=tuned_file,
        )

    assert out_buf is not None, "splitk modes need a pre-allocated out buffer"

    zero_init = out_buf if do_fused_zero_init else None
    x_q, x_scale = per_group_quant_hip(
        x_bf16,
        quant_dtype=dtypes.fp8,
        group_size=128,
        transpose_scale=True,
        gemm_out_zero_init=zero_init,
    )
    return gemm_a8w8_blockscale_bpreshuffle(
        x_q, weight_shuffle, x_scale, w_scale,
        dtype=dtypes.bf16,
        out=out_buf,
        y_is_zeroed=do_fused_zero_init,
        tuned_file=tuned_file,
    )


def _bench_shape(
    *,
    M: int,
    N: int,
    K: int,
    mode: str,
    tuned_file: str,
    iters: int,
    warmup: int,
    seed: int,
) -> dict:
    x_bf16, weight_shuffle, w_scale = _gen_inputs(M, N, K, seed=seed)
    out_buf: torch.Tensor | None = None
    if mode != "none":
        out_buf = torch.empty(M, N, dtype=dtypes.bf16, device="cuda")

    do_fused_zero_init = (
        mode == "splitk_fused" and _shape_splitk(M, N, K, tuned_file) > 0
    )

    for _ in range(warmup):
        _producer_then_gemm(
            x_bf16=x_bf16,
            weight_shuffle=weight_shuffle,
            w_scale=w_scale,
            mode=mode,
            tuned_file=tuned_file,
            M=M,
            N=N,
            out_buf=out_buf,
            do_fused_zero_init=do_fused_zero_init,
        )
    torch.cuda.synchronize()

    samples_us: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _producer_then_gemm(
            x_bf16=x_bf16,
            weight_shuffle=weight_shuffle,
            w_scale=w_scale,
            mode=mode,
            tuned_file=tuned_file,
            M=M,
            N=N,
            out_buf=out_buf,
            do_fused_zero_init=do_fused_zero_init,
        )
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0)

    samples_us.sort()
    return {
        "median_us": statistics.median(samples_us),
        "min_us": min(samples_us),
        "p10_us": samples_us[max(0, int(0.10 * len(samples_us)) - 1)],
        "p90_us": samples_us[min(len(samples_us) - 1, int(0.90 * len(samples_us)))],
        "stdev_us": statistics.stdev(samples_us) if len(samples_us) > 1 else 0.0,
        "n": len(samples_us),
        "do_fused_zero_init": int(do_fused_zero_init),
    }


def _maybe_capture_trace(
    *,
    M: int,
    N: int,
    K: int,
    mode: str,
    tuned_file: str,
    trace_dir: str,
    seed: int,
) -> None:
    """Capture a single iteration with torch.profiler for trace inspection."""
    os.makedirs(trace_dir, exist_ok=True)
    x_bf16, weight_shuffle, w_scale = _gen_inputs(M, N, K, seed=seed)
    out_buf = (
        torch.empty(M, N, dtype=dtypes.bf16, device="cuda")
        if mode != "none"
        else None
    )
    do_fused_zero_init = (
        mode == "splitk_fused" and _shape_splitk(M, N, K, tuned_file) > 0
    )
    for _ in range(3):
        _producer_then_gemm(
            x_bf16=x_bf16,
            weight_shuffle=weight_shuffle,
            w_scale=w_scale,
            mode=mode,
            tuned_file=tuned_file,
            M=M,
            N=N,
            out_buf=out_buf,
            do_fused_zero_init=do_fused_zero_init,
        )
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
    ) as prof:
        for _ in range(5):
            _producer_then_gemm(
                x_bf16=x_bf16,
                weight_shuffle=weight_shuffle,
                w_scale=w_scale,
                mode=mode,
                tuned_file=tuned_file,
                M=M,
                N=N,
                out_buf=out_buf,
                do_fused_zero_init=do_fused_zero_init,
            )
        torch.cuda.synchronize()
    out_file = os.path.join(trace_dir, f"trace_{mode}_M{M}_N{N}_K{K}.json")
    prof.export_chrome_trace(out_file)
    print(f"# trace -> {out_file}", file=sys.stderr)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=VALID_MODES)
    p.add_argument(
        "--tuned-csv",
        required=True,
        help="Path to the tuned CSV passed to gemm_a8w8_blockscale_bpreshuffle "
        "via tuned_file=. MUST be consistent with the .so build (delete "
        "module_gemm_a8w8_blockscale_bpreshuffle_cktile.so when switching "
        "between CSVs that select different kernel instances).",
    )
    p.add_argument(
        "--shapes-csv",
        required=True,
        help="CSV listing {M,N,K} shapes to sweep (header required).",
    )
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument(
        "--out",
        default=None,
        help="Optional output CSV path. Always also prints to stdout.",
    )
    p.add_argument(
        "--trace-dir",
        default=None,
        help="If set, capture one torch.profiler chrome trace per shape "
        "into this directory.",
    )
    p.add_argument(
        "--shape",
        type=_parse_shape,
        default=None,
        help="If set, only run this single M,N,K shape (overrides --shapes-csv).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 1
    if not os.path.exists(args.tuned_csv):
        print(f"ERROR: tuned CSV not found: {args.tuned_csv}", file=sys.stderr)
        return 1

    if args.shape is not None:
        shapes = [args.shape]
    else:
        if not os.path.exists(args.shapes_csv):
            print(f"ERROR: shapes CSV not found: {args.shapes_csv}", file=sys.stderr)
            return 1
        shapes = _read_shapes_csv(args.shapes_csv)

    print(
        f"# mode={args.mode} tuned_csv={args.tuned_csv} "
        f"iters={args.iters} warmup={args.warmup} "
        f"device={torch.cuda.get_device_name(0)}"
    )
    cols = (
        "mode", "M", "N", "K",
        "median_us", "min_us", "p10_us", "p90_us", "stdev_us", "n",
        "do_fused_zero_init",
    )
    print("\t".join(cols))

    rows: list[dict] = []
    for (M, N, K) in shapes:
        try:
            stats = _bench_shape(
                M=M, N=N, K=K,
                mode=args.mode, tuned_file=args.tuned_csv,
                iters=args.iters, warmup=args.warmup, seed=args.seed,
            )
        except Exception as e:  # noqa: BLE001
            print(f"# {args.mode} M={M} N={N} K={K}: ERROR {e}", file=sys.stderr)
            continue
        row = {
            "mode": args.mode, "M": M, "N": N, "K": K, **stats,
        }
        rows.append(row)
        print(
            f"{args.mode}\t{M}\t{N}\t{K}\t"
            f"{stats['median_us']:.3f}\t{stats['min_us']:.3f}\t"
            f"{stats['p10_us']:.3f}\t{stats['p90_us']:.3f}\t"
            f"{stats['stdev_us']:.3f}\t{stats['n']}\t"
            f"{stats['do_fused_zero_init']}"
        )

        if args.trace_dir is not None:
            _maybe_capture_trace(
                M=M, N=N, K=K,
                mode=args.mode, tuned_file=args.tuned_csv,
                trace_dir=args.trace_dir, seed=args.seed,
            )

    if args.out is not None:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cols))
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in cols})
        print(f"# results -> {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
