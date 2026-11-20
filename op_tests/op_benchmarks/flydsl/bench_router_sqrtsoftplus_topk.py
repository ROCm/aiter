# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CUDA-graph replay time of ``flydsl_router_sqrtsoftplus_topk`` against the
router implementations it replaces, per decode row count.

DeepSeek-V4.1-Flash MoE router: x bf16 [M, 5120], gate bf16 [E, 5120], fp32
bias, sqrt-softplus top-k, renormalized x 1.5, for the MoE layers (E = 384,
top-6) and the DSpark draft layers (E = 128, top-3). Candidates:

- vLLM's router (when vLLM is importable): ``fp32_router_gemm_dispatch``
  (its gfx950 low-M kernel or hipBLASLt) + ``topk_hash_softplus_sqrt``;
- the same GEMM + AITER's HIP ``topk_gating`` (sqrtsoftplus);
- ``torch.mm`` (hipBLASLt, fp32 out) + AITER's ``topk_gating``;
- the FlyDSL op, row-major gate and load-time shuffled gate.

Each graph replays the router on 100 distinct inputs (more bytes than the LLC,
so every call starts cold, as in a decode step); a row is the median of 5
rounds of 5 replays, candidates timed in A B ... B A order, us per call.

Run: python3 op_tests/op_benchmarks/flydsl/bench_router_sqrtsoftplus_topk.py
     [--experts 384 128] [--rows 16 192]
"""

import argparse
import importlib
import statistics

import torch

ROWS = (1, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
NIN = 100
SCALE = 1.5
# Routed experts -> top-k.
SHAPES = {384: 6, 128: 3}


def _graph(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    return g


def _timed(g):
    ts = []
    for _ in range(5):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(5):
            g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / (5 * NIN))
    return statistics.median(ts)


def _abba(fns):
    graphs = {n: _graph(f) for n, f in fns.items()}
    names = list(graphs)
    res = {n: [] for n in names}
    for n in names + names[::-1]:
        res[n].append(_timed(graphs[n]))
    return {n: statistics.median(v) for n, v in res.items()}


def _print(title, rows):
    names = list(rows[0][1])
    print(f"\n{title} (us per call, graph replay)")
    print("| rows | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 1) + "|")
    for m, t in rows:
        print(f"| {m} | " + " | ".join(f"{t[n]:.2f}" for n in names) + " |")


def _vllm():
    try:
        import vllm._custom_ops as ops

        return ops
    except ImportError:
        return None


def _router_row(m, vops, w, ws, bias):
    topk = SHAPES[w.shape[0]]
    from aiter.ops.flydsl import flydsl_router_sqrtsoftplus_topk
    from aiter.ops.topk import topk_gating

    torch.manual_seed(m)
    xs = [torch.randn(m, 5120, device="cuda").bfloat16() for _ in range(NIN)]
    bufs = [
        (
            torch.empty(m, topk, device="cuda"),
            torch.empty(m, topk, device="cuda", dtype=torch.int32),
            torch.empty(m, topk, device="cuda", dtype=torch.int32),
        )
        for _ in range(NIN)
    ]

    def flydsl():
        for x, (tw, ti, _) in zip(xs, bufs):
            flydsl_router_sqrtsoftplus_topk(
                x, w, bias, SCALE, topk_weights=tw, topk_ids=ti
            )

    def flydsl_shuffled():
        for x, (tw, ti, _) in zip(xs, bufs):
            flydsl_router_sqrtsoftplus_topk(
                x, ws, bias, SCALE, topk_weights=tw, topk_ids=ti, weight_shuffled=True
            )

    fns = {}
    if vops is not None:
        # Registers torch.ops.vllm.fp32_router_gemm_dispatch.
        importlib.import_module(
            "vllm.model_executor.layers.fused_moe.router.gate_linear"
        )

        gemm = torch.ops.vllm.fp32_router_gemm_dispatch

        def vllm_pair():
            for x, (tw, ti, te) in zip(xs, bufs):
                lg = gemm(x, w, False)
                vops.topk_hash_softplus_sqrt(
                    tw, ti, te, lg, True, SCALE, bias, None, None, None, None, 0
                )

        def vllm_gemm_aiter_topk():
            for x, (tw, ti, _) in zip(xs, bufs):
                lg = gemm(x, w, False)
                topk_gating(tw, ti, lg, bias, True, SCALE, "sqrtsoftplus")

        fns["vLLM router GEMM + topk_softplus_sqrt"] = vllm_pair
        fns["vLLM router GEMM + aiter topk_gating"] = vllm_gemm_aiter_topk

    def mm_aiter_topk():
        for x, (tw, ti, _) in zip(xs, bufs):
            lg = torch.mm(x, w.t(), out_dtype=torch.float32)
            topk_gating(tw, ti, lg, bias, True, SCALE, "sqrtsoftplus")

    fns["torch.mm + aiter topk_gating"] = mm_aiter_topk
    fns["FlyDSL (this PR)"] = flydsl
    fns["FlyDSL, shuffled gate (this PR)"] = flydsl_shuffled
    return _abba(fns)


def bench_router(rows, vops, experts):
    from aiter.ops.flydsl import shuffle_router_gate

    torch.manual_seed(0)
    w = (torch.randn(experts, 5120, device="cuda") * 0.02).bfloat16()
    ws = shuffle_router_gate(w)
    bias = torch.randn(experts, device="cuda") * 0.1
    out = [(m, _router_row(m, vops, w, ws, bias)) for m in rows]
    _print(f"router GEMM 5120x{experts} + sqrtsoftplus top-{SHAPES[experts]}", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, nargs="*", default=list(ROWS))
    parser.add_argument(
        "--experts", type=int, nargs="*", default=list(SHAPES), choices=list(SHAPES)
    )
    args = parser.parse_args()
    print(torch.cuda.get_device_name(), torch.version.hip)
    vops = _vllm()
    if vops is None:
        print("vLLM is not importable: timing the AITER candidates only")
    for experts in args.experts:
        bench_router(args.rows, vops, experts)


if __name__ == "__main__":
    main()
