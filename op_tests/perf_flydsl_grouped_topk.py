#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Performance comparison for FlyDSL grouped_topk.

Two parts:
  1. Standalone routing kernel latency (FlyDSL vs HIP) across a range of token
     counts.
  2. End-to-end MoE (routing -> moe_sorting -> fused_moe) latency, swapping only
     the routing kernel, to see whether the FlyDSL router moves the full-pipeline
     number at all (routing is normally a tiny fraction of MoE GEMM time).

Usage:
    python3 op_tests/perf_flydsl_grouped_topk.py
    python3 op_tests/perf_flydsl_grouped_topk.py --no-e2e
"""

from __future__ import annotations

import argparse

import torch

import aiter
from aiter import dtypes
from aiter.test_common import run_perftest
from aiter.ops.flydsl import flydsl_grouped_topk

# (num_tokens, num_experts, num_expert_group, topk_group, topk)
ROUTING_CASES = [
    {"token": 1, "expert": 256, "group": 8, "topk_group": 4, "topk": 8},
    {"token": 32, "expert": 256, "group": 8, "topk_group": 4, "topk": 8},
    {"token": 64, "expert": 256, "group": 8, "topk_group": 4, "topk": 8},
    {"token": 256, "expert": 256, "group": 8, "topk_group": 4, "topk": 8},
    {"token": 1024, "expert": 256, "group": 8, "topk_group": 4, "topk": 8},
    {"token": 4096, "expert": 256, "group": 8, "topk_group": 4, "topk": 8},
    {"token": 8192, "expert": 256, "group": 8, "topk_group": 4, "topk": 8},
]


def _alloc_out(token, topk):
    w = torch.empty_strided((token, topk), (topk + 10, 1), dtype=dtypes.fp32, device="cuda")
    i = torch.empty_strided((token, topk), (topk + 10, 1), dtype=dtypes.i32, device="cuda")
    return w, i


def bench_routing(dtype, scoring_func="softmax", need_renorm=True, scale=1.0):
    is_softmax = scoring_func == "softmax"
    print(
        f"\n=== routing latency  dtype={str(dtype).split('.')[-1]} "
        f"scoring={scoring_func} renorm={need_renorm} ==="
    )
    print(f"{'shape':<34}{'HIP (us)':>12}{'FlyDSL (us)':>14}{'speedup':>10}")
    rows = []
    for c in ROUTING_CASES:
        token, E, G, TG, K = c["token"], c["expert"], c["group"], c["topk_group"], c["topk"]
        gating = torch.randn((token, E), dtype=dtype, device="cuda")

        w_hip, id_hip = _alloc_out(token, K)
        _, us_hip = run_perftest(
            aiter.grouped_topk,
            gating, w_hip, id_hip, G, TG, need_renorm, is_softmax, scale,
            num_iters=100, num_warmup=10,
        )

        w_fly, id_fly = _alloc_out(token, K)
        _, us_fly = run_perftest(
            flydsl_grouped_topk,
            gating, w_fly, id_fly, G, TG, need_renorm, is_softmax, scale,
            num_iters=100, num_warmup=10,
        )
        speedup = us_hip / us_fly if us_fly > 0 else float("nan")
        label = f"t{token}_e{E}_g{G}_tg{TG}_k{K}"
        print(f"{label:<34}{us_hip:>12.3f}{us_fly:>14.3f}{speedup:>9.2f}x")
        rows.append((label, us_hip, us_fly, speedup))
    return rows


def build_moe_weights(E, model_dim, inter_dim, device, q="fp8"):
    """Build pre-shuffled fp8 g1u1 weights for aiter.fused_moe (block-quant off)."""
    from aiter import QuantType
    from aiter.utility import shuffle_weight

    w1 = torch.randn((E, 2 * inter_dim, model_dim), dtype=dtypes.bf16, device=device) / 10
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtypes.bf16, device=device) / 10
    # per-tensor fp8 quant
    w1_q, w1_s = aiter.pertoken_quant(w1.view(E, -1), quant_dtype=dtypes.fp8) if False else (None, None)
    return w1, w2


def _time_cuda(fn, reps=50, blocks=20, warmup=30):
    """Per-call latency (us) via CUDA events, with inner-loop batching.

    Each timed *block* runs ``reps`` back-to-back calls under a single event
    pair, so the ~7-10us per-call event/sync overhead is amortised (divided by
    ``reps``) instead of being added to every sample -- essential for resolving
    a few-us kernel delta. Returns (mean_us, std_us, p50_us) over ``blocks``.
    """
    import statistics

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(blocks):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps):
            fn()
        e.record()
        e.synchronize()
        samples.append(s.elapsed_time(e) * 1000.0 / reps)  # ms -> us, per call
    samples.sort()
    mean = sum(samples) / len(samples)
    std = statistics.pstdev(samples)
    p50 = samples[len(samples) // 2]
    return mean, std, p50


def bench_e2e(dtype):
    """End-to-end MoE latency with component decomposition.

    Measures, in the same pipeline and with identical weights/inputs, three
    timed sections per router (HIP vs FlyDSL):
      * route : just the grouped_topk kernel
      * moe   : just fused_moe(hidden, w1, w2, w, i)   (router-independent)
      * e2e   : route + moe back-to-back

    Reporting mean +/- std (CUDA-event timed, 300 iters) lets the ~10us routing
    delta surface above GEMM noise: e2e_delta should match route_delta within
    the e2e std.
    """
    from aiter.fused_moe import fused_moe

    print(f"\n=== end-to-end MoE latency (decomposed)  dtype={str(dtype).split('.')[-1]} ===")

    cfgs = [
        # decode-like (small token -> routing is a larger fraction)
        {"token": 8, "expert": 256, "group": 8, "topk_group": 4, "topk": 8,
         "model_dim": 2048, "inter_dim": 768},
        {"token": 64, "expert": 256, "group": 8, "topk_group": 4, "topk": 8,
         "model_dim": 2048, "inter_dim": 768},
        {"token": 1024, "expert": 256, "group": 8, "topk_group": 4, "topk": 8,
         "model_dim": 2048, "inter_dim": 768},
    ]
    dev = "cuda"
    for c in cfgs:
        token, E, G, TG, K = c["token"], c["expert"], c["group"], c["topk_group"], c["topk"]
        model_dim, inter_dim = c["model_dim"], c["inter_dim"]
        hidden = torch.randn((token, model_dim), dtype=dtypes.bf16, device=dev) / 10
        gating = torch.randn((token, E), dtype=dtype, device=dev)
        w1 = torch.randn((E, 2 * inter_dim, model_dim), dtype=dtypes.bf16, device=dev) / 10
        w2 = torch.randn((E, model_dim, inter_dim), dtype=dtypes.bf16, device=dev) / 10
        w = torch.empty((token, K), dtype=dtypes.fp32, device=dev)
        i = torch.empty((token, K), dtype=dtypes.i32, device=dev)

        def route_hip():
            aiter.grouped_topk(gating, w, i, G, TG, True, True, 1.0)

        def route_fly():
            flydsl_grouped_topk(gating, w, i, G, TG, True, True, 1.0)

        def moe_only():
            # use a fixed valid routing (compute once outside timing)
            return fused_moe(hidden, w1, w2, w, i)

        def e2e_hip():
            aiter.grouped_topk(gating, w, i, G, TG, True, True, 1.0)
            return fused_moe(hidden, w1, w2, w, i)

        def e2e_fly():
            flydsl_grouped_topk(gating, w, i, G, TG, True, True, 1.0)
            return fused_moe(hidden, w1, w2, w, i)

        try:
            # prime routing so `w`/`i` are valid before timing moe_only
            route_hip()
            torch.cuda.synchronize()
            r_hip = _time_cuda(route_hip)
            r_fly = _time_cuda(route_fly)
            m_only = _time_cuda(moe_only)
            t_hip = _time_cuda(e2e_hip)
            t_fly = _time_cuda(e2e_fly)
        except Exception as exc:  # pragma: no cover
            print(f"  [t{token}_e{E}] e2e skipped: {type(exc).__name__}: {exc}")
            continue

        label = f"t{token}_e{E}_md{model_dim}_id{inter_dim}_k{K}"
        print(f"\n  {label}")
        print(f"    route  HIP  : {r_hip[0]:8.2f} +/- {r_hip[1]:5.2f} us   (p50 {r_hip[2]:7.2f})")
        print(f"    route  FlyDSL: {r_fly[0]:8.2f} +/- {r_fly[1]:5.2f} us   (p50 {r_fly[2]:7.2f})")
        print(f"    moe    only  : {m_only[0]:8.2f} +/- {m_only[1]:5.2f} us   (p50 {m_only[2]:7.2f})")
        print(f"    e2e    HIP   : {t_hip[0]:8.2f} +/- {t_hip[1]:5.2f} us   (p50 {t_hip[2]:7.2f})")
        print(f"    e2e    FlyDSL: {t_fly[0]:8.2f} +/- {t_fly[1]:5.2f} us   (p50 {t_fly[2]:7.2f})")
        print(f"    --> route delta (HIP-FlyDSL): {r_hip[0]-r_fly[0]:+6.2f} us")
        print(f"    --> e2e   delta (HIP-FlyDSL): {t_hip[0]-t_fly[0]:+6.2f} us "
              f"(e2e speedup {t_hip[0]/t_fly[0] if t_fly[0]>0 else float('nan'):.3f}x)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-e2e", action="store_true", help="skip end-to-end MoE bench")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()
    dmap = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dmap[args.dtype]

    if not torch.cuda.is_available():
        print("CUDA not available; abort.")
        return

    bench_routing(dtype, scoring_func="softmax", need_renorm=True)
    bench_routing(dtype, scoring_func="sigmoid", need_renorm=True)

    if not args.no_e2e:
        bench_e2e(dtype)


if __name__ == "__main__":
    main()
