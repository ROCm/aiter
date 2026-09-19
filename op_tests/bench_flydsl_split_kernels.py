# SPDX-License-Identifier: MIT
"""Paired, end-to-end HIP graph timings for the opt-in split implementations.

Softmax baseline source: ROCm/FlyDSL kernels/norm/softmax_kernel.py.
Run with --flydsl-root pointing at that checkout. Compilation and workspaces
are outside timing; both split softmax stages and GEMM conversion are inside.
"""

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import torch

from aiter.ops.flydsl import (
    build_gemm_bf16_split_fp32,
    build_softmax_split,
    flydsl_hgemm,
)


def capture(fn, out, count):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(count):
            fn()
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(out).all(), "empty graph or nonfinite output"
    return graph


def validate(out, ref, op):
    delta = (out.float() - ref).abs()
    if op == "softmax":
        assert (delta / (ref.abs() + ref.amax(-1, keepdim=True))).max() <= 0.02
        assert (out.float().sum(-1) - 1).abs().max() <= 0.01
    else:
        rms = ref.square().mean().sqrt().item()
        assert torch.all(delta <= 0.02 * ref.abs() + 0.01 * max(rms, 1e-6))
        assert delta.square().mean().sqrt().item() / max(rms, 1e-20) <= 0.01


def measure(op, shape, baseline, optimized, outputs, ref, *, iters, rounds, rng):
    functions = {"baseline": baseline, "optimized": optimized}
    graphs = {name: capture(fn, outputs[name], iters) for name, fn in functions.items()}
    for name in graphs:
        validate(outputs[name], ref, op)
    samples = {name: [] for name in graphs}
    for _ in range(20):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    for _ in range(rounds):
        order = list(graphs)
        rng.shuffle(order)
        for name in order:
            graphs[name].replay()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / iters)
    for name in graphs:
        validate(outputs[name], ref, op)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    return {
        "op": op,
        "shape": shape,
        "samples_us": samples,
        "median_us": medians,
        "speedup": medians["baseline"] / medians["optimized"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flydsl-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=15)
    args = parser.parse_args()
    if args.iters < 2 or args.rounds < 1:
        parser.error("iters >= 2 and rounds >= 1 required")
    sys.path.insert(0, str(args.flydsl_root.resolve()))
    from kernels.norm.softmax_kernel import build_softmax_module

    torch.backends.cuda.matmul.allow_tf32 = False
    seed = 20260908
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rng = random.Random(947)
    rows = []
    for n in (131072, 89999):
        x = torch.randn((1, n), device="cuda", generator=gen).to(torch.bfloat16)
        outputs = {name: torch.empty_like(x) for name in ("baseline", "optimized")}
        stock = build_softmax_module(1, n, "bf16")
        split = build_softmax_split(1, n)

        def baseline(stock=stock, x=x, outputs=outputs):
            return stock(x, outputs["baseline"], 1, stream=torch.cuda.current_stream())

        def optimized(split=split, x=x, outputs=outputs):
            return split(x, outputs["optimized"])

        rows.append(
            measure(
                "softmax",
                (1, n),
                baseline,
                optimized,
                outputs,
                x.float().softmax(-1),
                iters=args.iters,
                rounds=args.rounds,
                rng=rng,
            )
        )
    a = torch.randn((32, 7168), device="cuda", generator=gen).to(torch.bfloat16)
    b = torch.randn((384, 7168), device="cuda", generator=gen).to(torch.bfloat16)
    outputs = {
        name: torch.empty((32, 384), device="cuda", dtype=torch.bfloat16)
        for name in ("baseline", "optimized")
    }
    stock_config = {
        "block_m": 16,
        "block_n": 16,
        "block_k": 256,
        "stages": 4,
        "split_k": 1,
        "m_waves": 1,
        "n_waves": 1,
        "k_waves": 2,
        "group_m": 0,
        "policy": "ft",
    }
    split = build_gemm_bf16_split_fp32(32, 384)
    rows.append(
        measure(
            "a16w16",
            (32, 384, 7168),
            lambda: flydsl_hgemm(a, b, out=outputs["baseline"], **stock_config),
            lambda: split(a, b, outputs["optimized"]),
            outputs,
            a.float() @ b.float().T,
            iters=args.iters,
            rounds=args.rounds,
            rng=rng,
        )
    )
    result = {
        "seed": seed,
        "iters": args.iters,
        "rounds": args.rounds,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
