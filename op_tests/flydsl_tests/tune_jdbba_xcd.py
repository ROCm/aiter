# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Tier-A live-knob autotuner for jagged_dense_bmm (jdbba): sweeps the chiplet
# (XCD) remap knobs xcd_c x xcd_w over the 4 headline shapes in the UNIFORM
# regime (the only regime where the remap is active -- skew forces remap off in
# the dispatch). No kernel-source edits: each (shape, xcd_c, xcd_w) memoizes its
# own compiled kernel via the public jagged_dense_bmm_dispatched override args.
#
# For every (shape, config): correctness-gate (cos >= 0.999 vs torch eager),
# then time with triton.testing.do_bench (cold-L2, the headline metric). Prints
# a TSV row per config and the chosen min-ms winner per shape, plus the speedup
# vs the kernel auto-default (xcd_c=None -> 32 for D<=256, 120 otherwise; xcd_w=8).
#
# Run inside the devcontainer:
#   PYTHONPATH=/workspaces/meta/aiter \
#     python3 op_tests/flydsl_tests/tune_jdbba_xcd.py
#
# Emits a candidate JSON (--out) the autoresearch loop reviews before promoting
# winners into by_arch.gfx942.winners of jagged_dense_bmm_dispatch_v2.json.

from __future__ import annotations

import argparse
import json
import sys

import torch
import triton

import flydsl.compiler as flyc
from aiter.ops.flydsl.jagged_dense_bmm_dispatch_v2 import (
    jagged_dense_bmm_dispatched,
    shape_id,
)
from aiter.ops.flydsl.kernels.jagged_dense_bmm_gen import BLOCK_M as _BLOCK_M

# reuse the bench's input builders so tuner + bench measure identical tensors
from bench_jagged_dense_bmm_perf import (
    _make_inputs,
    _torch_reference,
    default_benchmark_configs,
)


# grid: xcd_c spans 1 (remap off) up to large clustering; xcd_w the window.
# 1 is the remap-off control; 8/16/32/60/120 cover the chiplet-cluster sizes;
# both kernel auto defaults (32 for D<=256, 120 for D>256) are in the grid.
DEFAULT_XCD_C = [1, 8, 16, 32, 60, 120, 240]
DEFAULT_XCD_W = [4, 8, 16]


def _auto_xcd_c(reduction_k: int) -> int:
    return 32 if reduction_k <= 256 else 120


def _time_one(jagged, dense, bias, seq_offsets, B, Mi, N, K, xcd_c, xcd_w,
              warmup, rep, do_test):
    dense_tall = dense.transpose(1, 2).reshape(B * N, K).contiguous()
    bias_flat = bias.reshape(B * N).contiguous()
    L = jagged.shape[0]
    out = torch.zeros(L + _BLOCK_M, N, dtype=torch.bfloat16, device="cuda")
    tA = flyc.from_dlpack(jagged).mark_layout_dynamic(leading_dim=1, divisibility=8)
    tC = flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=1, divisibility=8)

    def fn():
        jagged_dense_bmm_dispatched(
            tC, tA, dense_tall, bias_flat, seq_offsets, B, Mi,
            stream=torch.cuda.current_stream(), uniform_seqlen=True,
            xcd_c=xcd_c, xcd_w=xcd_w,
        )

    cos = None
    if do_test:
        ref = _torch_reference(jagged, dense, bias, seq_offsets, N)
        fn()
        torch.cuda.synchronize()
        cos = torch.nn.functional.cosine_similarity(
            ref.float().flatten(), out[:L].float().flatten(), dim=0
        ).item()
        if cos <= 0.999:
            return None, cos  # failed correctness; skip timing

    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms, cos


def main(argv=None):
    p = argparse.ArgumentParser(prog="tune_jdbba_xcd (tier-A XCD remap sweep)")
    p.add_argument("-mi", type=int, default=7680)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--rep", type=int, default=100)
    p.add_argument("--xcd-c", type=int, nargs="+", default=DEFAULT_XCD_C)
    p.add_argument("--xcd-w", type=int, nargs="+", default=DEFAULT_XCD_W)
    p.add_argument("--out", default="jdbba_xcd_candidates.json")
    args = p.parse_args(argv)

    shapes = default_benchmark_configs()
    print("# tier-A XCD sweep (uniform regime), Mi=%d" % args.mi)
    print("shape\txcd_c\txcd_w\tms\tcos\tnote")

    winners = {}
    for (B, D, Kout) in shapes:
        jagged, dense, bias, seq_offsets, L, N, K = _make_inputs(
            B, D, Kout, args.mi, regime="uniform", seed=args.seed
        )
        sid = shape_id(n_groups=B, reduction_k=D, output_n=Kout, max_seq_len=args.mi)
        auto_c = _auto_xcd_c(D)

        best = None  # (ms, xcd_c, xcd_w)
        auto_ms = None
        for xcd_c in args.xcd_c:
            for xcd_w in args.xcd_w:
                # xcd_c=1 means remap off; xcd_w is forced to 1 internally then
                eff_w = 1 if xcd_c == 1 else xcd_w
                if xcd_c == 1 and xcd_w != args.xcd_w[0]:
                    continue  # only time remap-off once
                try:
                    ms, cos = _time_one(
                        jagged, dense, bias, seq_offsets, B, args.mi, N, K,
                        xcd_c, eff_w, args.warmup, args.rep, do_test=True,
                    )
                except Exception as exc:
                    print(f"{sid}\t{xcd_c}\t{eff_w}\tCRASH\t-\t{type(exc).__name__}")
                    continue
                if ms is None:
                    print(f"{sid}\t{xcd_c}\t{eff_w}\tFAIL\t{cos:.4f}\tcos<0.999")
                    continue
                note = []
                if xcd_c == auto_c and eff_w == 8:
                    note.append("AUTO")
                    auto_ms = ms
                print(f"{sid}\t{xcd_c}\t{eff_w}\t{ms:.4f}\t{cos:.4f}\t{','.join(note)}")
                if best is None or ms < best[0]:
                    best = (ms, xcd_c, eff_w)

        if best is not None:
            speedup = (auto_ms / best[0]) if auto_ms else float("nan")
            tag = "WIN" if (auto_ms and best[0] < auto_ms) else "tie/auto"
            print(f"#  -> {sid} best: xcd_c={best[1]} xcd_w={best[2]} "
                  f"{best[0]:.4f}ms (auto={auto_ms}, {speedup:.3f}x) [{tag}]")
            winners[sid] = {
                "xcd_c": best[1], "xcd_w": best[2],
                "use_mfma_k32": False,
                "_ms": round(best[0], 4),
                "_auto_ms": round(auto_ms, 4) if auto_ms else None,
                "_speedup": round(speedup, 4) if auto_ms else None,
            }

    with open(args.out, "w") as fh:
        json.dump({"regime": "uniform", "mi": args.mi, "winners": winners}, fh, indent=2)
    print(f"\n# wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
