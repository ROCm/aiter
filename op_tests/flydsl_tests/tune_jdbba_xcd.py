#!/usr/bin/env python3
"""Tier-A live-knob tuner for jdbba: sweep XCD chiplet remap (xcd_c x xcd_w).

No kernel source edits -- drives the public ``jagged_dense_bmm`` entry with
explicit xcd_c/xcd_w for both regimes. The production dispatch forces the remap
OFF under skew (uniform_seqlen=False -> xcd_c/xcd_w=None); this tuner re-derives
whether that gate is still optimal on the CURRENT arch by trying the remap ON
under skew too.

For each headline shape x regime:
  - enumerate xcd_c grid x xcd_w grid (+ the remap-off baseline xcd_c=1)
  - cos-gate every point (>=0.999 vs torch eager); skip failures
  - time with triton.testing.do_bench (cold-L2 headline)
  - print a TSV row per config, then the per-(shape,regime) winner

Run inside the container:
  docker exec -e PYTHONPATH=/home/anguyenh/aiter:/home/anguyenh/generative-recommenders \\
    -w /home/anguyenh/aiter <container> python3 op_tests/flydsl_tests/tune_jdbba_xcd.py
"""
import argparse
import itertools

import torch
import triton

import flydsl.compiler as flyc
from aiter.ops.flydsl.kernels.jagged_dense_bmm_gen import jagged_dense_bmm, BLOCK_M as _BLOCK_M
from bench_jagged_dense_bmm_perf import (
    _make_inputs,
    _torch_reference,
    default_benchmark_configs,
)

# Grids. xcd_c=1 disables the remap (the skew default). 8 XCDs on MI3xx ->
# windows of 4/8 are the meaningful ones; chunk C spans 1..n_groups.
XCD_C_GRID = [1, 16, 32, 60, 120, 240]
XCD_W_GRID = [4, 8]


def _flydsl_call(tC, tA, dense_tall, bias_flat, seq_offsets, B, Mi, xcd_c, xcd_w, uniform):
    jagged_dense_bmm(
        tC, tA, dense_tall, bias_flat, seq_offsets, B, Mi,
        stream=torch.cuda.current_stream(),
        xcd_c=xcd_c, xcd_w=xcd_w, use_mfma_k32=None, uniform_seqlen=uniform,
    )


def tune_shape(B, D, Kout, Mi, regime, warmup, rep):
    uniform = regime == "uniform"
    jagged, dense, bias, seq_offsets, L, N, K = _make_inputs(B, D, Kout, Mi, regime=regime)
    ref = _torch_reference(jagged, dense, bias, seq_offsets, N)

    dense_tall = dense.transpose(1, 2).reshape(B * N, K).contiguous()
    bias_flat = bias.reshape(B * N).contiguous()
    out = torch.zeros(L + _BLOCK_M, N, dtype=torch.bfloat16, device="cuda")
    tA = flyc.from_dlpack(jagged).mark_layout_dynamic(leading_dim=1, divisibility=8)
    tC = flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=1, divisibility=8)

    grid = [(1, XCD_W_GRID[0])] + list(itertools.product(XCD_C_GRID[1:], XCD_W_GRID))
    best = None
    rows = []
    for xcd_c, xcd_w in grid:
        try:
            out.zero_()
            _flydsl_call(tC, tA, dense_tall, bias_flat, seq_offsets, B, Mi, xcd_c, xcd_w, uniform)
            torch.cuda.synchronize()
            cos = torch.nn.functional.cosine_similarity(
                ref.float().flatten(), out[:L].float().flatten(), dim=0
            ).item()
        except Exception as e:  # noqa: BLE001
            rows.append((xcd_c, xcd_w, float("nan"), float("nan"), f"CRASH {type(e).__name__}"))
            continue
        if cos <= 0.999:
            rows.append((xcd_c, xcd_w, cos, float("nan"), "COS_FAIL"))
            continue
        ms = triton.testing.do_bench(
            lambda: _flydsl_call(tC, tA, dense_tall, bias_flat, seq_offsets, B, Mi, xcd_c, xcd_w, uniform),
            warmup=warmup, rep=rep,
        )
        rows.append((xcd_c, xcd_w, cos, ms, "ok"))
        if best is None or ms < best[3]:
            best = (xcd_c, xcd_w, cos, ms)

    print(f"\n=== {regime} B{B}_D{D}_K{Kout} (Mi={Mi}, L={L}) ===")
    print("xcd_c\txcd_w\tcos\tms\tstatus")
    for xcd_c, xcd_w, cos, ms, st in rows:
        print(f"{xcd_c}\t{xcd_w}\t{cos:.4f}\t{ms:.4f}\t{st}")
    if best:
        print(f"WINNER B{B}_D{D}_K{Kout} {regime}: xcd_c={best[0]} xcd_w={best[1]} -> {best[3]:.4f} ms")
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regime", choices=["uniform", "skew", "both"], default="both")
    p.add_argument("-mi", type=int, default=7680)
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--rep", type=int, default=100)
    p.add_argument("-b", type=int, default=None)
    p.add_argument("-d", type=int, default=None)
    p.add_argument("-kout", type=int, default=None)
    args = p.parse_args()

    if args.b is not None:
        shapes = [(args.b, args.d, args.kout)]
    else:
        shapes = default_benchmark_configs()
    regimes = ["uniform", "skew"] if args.regime == "both" else [args.regime]

    summary = {}
    for regime in regimes:
        for B, D, Kout in shapes:
            best = tune_shape(B, D, Kout, args.mi, regime, args.warmup, args.rep)
            summary[(regime, B, D, Kout)] = best

    print("\n=== WINNERS SUMMARY ===")
    for (regime, B, D, Kout), best in summary.items():
        if best:
            print(f"{regime}\tB{B}_D{D}_K{Kout}\txcd_c={best[0]}\txcd_w={best[1]}\t{best[3]:.4f} ms")
        else:
            print(f"{regime}\tB{B}_D{D}_K{Kout}\tNO_VALID_CONFIG")


if __name__ == "__main__":
    main()
