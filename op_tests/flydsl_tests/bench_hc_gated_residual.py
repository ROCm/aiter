# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Latency benchmark for the two-stage HC Gated-Residual op (SILOTIGER-1042).

Times the shipped fused two-stage ``combine_and_mix`` (fold_w=True, pre-folded
weight -- the production configuration) across a token sweep, and validates each
timed shape against the float32 oracle. Self-contained: no external baseline
(the vLLM/Triton comparison that gated the ticket lives in the legacy bench).

    HIP_VISIBLE_DEVICES=2 python op_tests/flydsl_tests/bench_hc_gated_residual.py
    HIP_VISIBLE_DEVICES=2 python op_tests/flydsl_tests/bench_hc_gated_residual.py --tokens 512 4096 8192
"""

import argparse
import statistics
import sys

import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.hyper_connection_gated_residual import (
    flydsl_gr_two_stage_combine_and_mix,
    fold_norm_weight,
    gr_combine_and_mix,
    merge_gr_two_stage_weight,
)

HC, HS, LOWRANK = 4, 2560, 320
HIDDEN = HC * HS
EPS = 1e-6


def make_inputs(tokens, seed=0):
    torch.manual_seed(seed)

    def _rand(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda") * scale).bfloat16()

    inp = {
        "residual": _rand(tokens, HIDDEN),
        "block_output": _rand(tokens, HS),
        "injection": _rand(tokens, HC),
        "norm_weight": _rand(HS, scale=0.1),
        "w_down": _rand(LOWRANK, HIDDEN, scale=HIDDEN**-0.5),
        "w_up": _rand(HIDDEN, LOWRANK, scale=LOWRANK**-0.5),
        "w_inject": _rand(HC, HIDDEN, scale=HIDDEN**-0.5),
    }
    merged = merge_gr_two_stage_weight(inp["w_down"], inp["w_inject"], HC)
    inp["_w_folded"] = fold_norm_weight(merged, inp["norm_weight"], HC)
    return inp


def _run(inp):
    return flydsl_gr_two_stage_combine_and_mix(
        inp["residual"], inp["block_output"], inp["injection"], inp["norm_weight"],
        inp["w_down"], inp["w_up"], inp["w_inject"], HC, EPS,
        w_down_merged=inp["_w_folded"], fold_w=True,
    )


def _validate(inp, tokens):
    r2, x, inj = _run(inp)
    torch.cuda.synchronize()
    r2o, xo, injo = gr_combine_and_mix(
        inp["residual"], inp["block_output"], inp["injection"], inp["norm_weight"],
        inp["w_down"], inp["w_up"], inp["w_inject"], HC, EPS, round_bf16=True,
    )
    ex = (x.float() - xo).abs().max().item()
    if x.isnan().any() or ex > 0.2:
        print(f"  [WARN] M={tokens}: x max-abs-err {ex:.3f} (nan={x.isnan().any().item()})")


def _time_us(inp, iters, warmup_s):
    import time
    torch.cuda.synchronize()
    deadline = time.perf_counter() + warmup_s
    while time.perf_counter() < deadline:
        _run(inp)
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        s.record()
        _run(inp)
        e.record()
        e.synchronize()
        samples.append(s.elapsed_time(e) * 1.0e3)  # ms -> us
    return statistics.median(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+",
                    default=[1, 8, 32, 128, 512, 2048, 4096, 8192, 16384, 32768])
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--warmup-s", type=float, default=1.0)
    args = ap.parse_args()

    arch = get_gfx()
    if arch not in ("gfx950", "gfx942"):
        print(f"[skip] requires gfx950/gfx942 FlyDSL, got {arch}")
        sys.exit(0)

    print(f"device={torch.cuda.get_device_name(0)} ({arch})  "
          f"HIDDEN={HIDDEN} LOWRANK={LOWRANK} HC={HC}  "
          f"iters={args.iters} warmup={args.warmup_s}s  fused two-stage (fold_w=True)")
    print(f"  {'tokens':>7} {'fk2+fold':>10}")
    print("  " + "-" * 20)
    for tokens in args.tokens:
        inp = make_inputs(tokens)
        _validate(inp, tokens)
        us = _time_us(inp, args.iters, args.warmup_s)
        print(f"  {tokens:7d} {us:9.1f}u")


if __name__ == "__main__":
    main()
