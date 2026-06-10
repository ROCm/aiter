#!/usr/bin/env python3
"""Tier-B tuner for jdbba: sweep output-tile size (BLOCK_M x BLOCK_N).

Drives ``_build_launcher`` DIRECTLY with per-config tile constants (the public
entry only ever passes the module defaults 128x128). No kernel source edit:
BLOCK_M/BLOCK_N/BLOCK_K/STAGES_A/THREADS are already launcher parameters.

The amortization lever (#3): bigger tiles -> fewer/fatter M-tiles -> fewer
pipeline fills + epilogues, which matters because this short-K kernel spends a
large fraction of its time in fixed per-block cost. The gfx950 opportunity is
that a 256x256 tile needs BLOCK_M*BLOCK_N*2 = 128KB of epilogue LDS, which fits
CDNA4's 160KB but NOT CDNA3's 64KB -- so it is a genuinely arch-new config.

Legality:
  - N % BLOCK_N == 0 (N_BLOCKS integer)
  - BLOCK_M*BLOCK_N % THREADS == 0 (C_FRAG_LEN integer)
  - epilogue LDS = max(BLOCK_M*block_k*STAGES_A, BLOCK_M*BLOCK_N) * 2 <= LDS_CAP
  - block_k is shape-derived (128 if K<=256 else 64), matching the public entry.

For each headline shape x regime: cos-gate (>=0.999) then do_bench cold-L2.
"""
import argparse
import itertools

import torch
import triton

import flydsl.compiler as flyc
from aiter.ops.flydsl.kernels.jagged_dense_bmm_gen import _build_launcher, STAGES_A, THREADS
from bench_jagged_dense_bmm_perf import (
    _make_inputs,
    _torch_reference,
    default_benchmark_configs,
)

LDS_CAP = 160 * 1024  # gfx950 / MI355X
BLOCK_M_GRID = [128, 256]
BLOCK_N_GRID = [128, 256, 512]


def _legal(BLOCK_M, BLOCK_N, block_k, N):
    if N % BLOCK_N != 0:
        return False, "N%BN"
    if (BLOCK_M * BLOCK_N) % THREADS != 0:
        return False, "frag"
    lds = max(BLOCK_M * block_k * STAGES_A, BLOCK_M * BLOCK_N) * 2
    if lds > LDS_CAP:
        return False, f"LDS={lds//1024}K"
    return True, "ok"


def tune_shape(B, D, Kout, Mi, regime, warmup, rep):
    N, K = Kout, D
    block_k = 128 if K <= 256 else 64
    jagged, dense, bias, seq_offsets, L, _, _ = _make_inputs(B, D, Kout, Mi, regime=regime)
    ref = _torch_reference(jagged, dense, bias, seq_offsets, N)
    dense_tall = dense.transpose(1, 2).reshape(B * N, K).contiguous()
    bias_flat = bias.reshape(B * N).contiguous()

    best = None
    rows = []
    for BLOCK_M, BLOCK_N in itertools.product(BLOCK_M_GRID, BLOCK_N_GRID):
        ok, why = _legal(BLOCK_M, BLOCK_N, block_k, N)
        if not ok:
            rows.append((BLOCK_M, BLOCK_N, float("nan"), float("nan"), f"skip:{why}"))
            continue
        bm = (Mi + BLOCK_M - 1) // BLOCK_M
        out = torch.zeros(L + BLOCK_M, N, dtype=torch.bfloat16, device="cuda")
        tA = flyc.from_dlpack(jagged).mark_layout_dynamic(leading_dim=1, divisibility=8)
        tC = flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=1, divisibility=8)
        try:
            launch = _build_launcher(N, K, BLOCK_M, BLOCK_N, block_k, STAGES_A, THREADS, bm, B, 1, 1, True)
            out.zero_()
            launch(tC, tA, dense_tall, bias_flat, seq_offsets, B, Mi, stream=torch.cuda.current_stream())
            torch.cuda.synchronize()
            cos = torch.nn.functional.cosine_similarity(
                ref.float().flatten(), out[:L].float().flatten(), dim=0
            ).item()
        except Exception as e:  # noqa: BLE001
            rows.append((BLOCK_M, BLOCK_N, float("nan"), float("nan"), f"CRASH {type(e).__name__}"))
            continue
        if cos <= 0.999:
            rows.append((BLOCK_M, BLOCK_N, cos, float("nan"), "COS_FAIL"))
            continue
        ms = triton.testing.do_bench(
            lambda: launch(tC, tA, dense_tall, bias_flat, seq_offsets, B, Mi, stream=torch.cuda.current_stream()),
            warmup=warmup, rep=rep,
        )
        rows.append((BLOCK_M, BLOCK_N, cos, ms, "ok"))
        if best is None or ms < best[3]:
            best = (BLOCK_M, BLOCK_N, cos, ms)

    print(f"\n=== {regime} B{B}_D{D}_K{Kout} (Mi={Mi}, L={L}, block_k={block_k}) ===")
    print("BM\tBN\tcos\tms\tstatus")
    for BLOCK_M, BLOCK_N, cos, ms, st in rows:
        print(f"{BLOCK_M}\t{BLOCK_N}\t{cos:.4f}\t{ms:.4f}\t{st}")
    if best:
        print(f"WINNER B{B}_D{D}_K{Kout} {regime}: BM={best[0]} BN={best[1]} -> {best[3]:.4f} ms")
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

    shapes = [(args.b, args.d, args.kout)] if args.b is not None else default_benchmark_configs()
    regimes = ["uniform", "skew"] if args.regime == "both" else [args.regime]

    summary = {}
    for regime in regimes:
        for B, D, Kout in shapes:
            summary[(regime, B, D, Kout)] = tune_shape(B, D, Kout, args.mi, regime, args.warmup, args.rep)

    print("\n=== WINNERS SUMMARY (tiles) ===")
    for (regime, B, D, Kout), best in summary.items():
        if best:
            print(f"{regime}\tB{B}_D{D}_K{Kout}\tBM={best[0]}\tBN={best[1]}\t{best[3]:.4f} ms")
        else:
            print(f"{regime}\tB{B}_D{D}_K{Kout}\tNO_VALID_CONFIG")


if __name__ == "__main__":
    main()
