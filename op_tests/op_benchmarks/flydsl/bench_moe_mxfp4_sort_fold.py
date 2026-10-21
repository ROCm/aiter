# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Decode benchmark for the sort-folded MXFP4 MoE GEMM1 (GLM-5.2 TP8, gfx950).

Times ``aiter.fused_moe`` for the GLM-5.2 TP8 routed MoE (hidden 6144, 256
inter per rank, 256 routed + 1 fused shared expert, top-9, per-1x32 MXFP4
preshuffled weights, separated gate/up) the way a decode step runs it:

* CUDA-graph replay of ``--calls`` back-to-back MoE calls, each on its own
  expert-permuted weight set (``--weight-sets`` >= 16), so the weights are
  cold in L2/MALL like consecutive layers;
* two arms in one process and one graph pool: ``main`` keeps the separate
  BM16 sort launch (``fused_moe._mxfp4_sort_fold_disabled = True``), ``fold``
  is the default dispatch; rounds alternate the arm order;
* ``--processes`` fresh processes; the table reports the median over
  processes of the per-process median, and the min-max range of the
  per-process speedups.

On a build without the fold only the ``main`` arm runs, so the same command
measures upstream main for a cross-build comparison.

Usage:
    python op_tests/op_benchmarks/flydsl/bench_moe_mxfp4_sort_fold.py
    python op_tests/op_benchmarks/flydsl/bench_moe_mxfp4_sort_fold.py -m 5 6 12 48 \\
        --processes 3 -o fold.json
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile

MODEL_DIM = 6144
INTER_DIM = 256
EXPERTS = 257
TOPK = 9


def _build_weight_sets(n_sets, seed=7):
    import torch

    import aiter
    from aiter import QuantType, dtypes
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility import fp4_utils

    gen = torch.Generator(device="cuda").manual_seed(seed)
    torch_quant = aiter.get_torch_quant(QuantType.per_1x32)
    w1 = torch.randn(
        (EXPERTS, 2 * INTER_DIM, MODEL_DIM), generator=gen, device="cuda"
    ).to(dtypes.bf16)
    w2 = torch.randn((EXPERTS, MODEL_DIM, INTER_DIM), generator=gen, device="cuda")
    w2 = w2.to(dtypes.bf16)
    w1_qt, w1_scale = torch_quant(w1, quant_dtype=dtypes.fp4x2)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=dtypes.fp4x2)
    del w1, w2
    w1_qt = shuffle_weight(w1_qt.view(EXPERTS, 2 * INTER_DIM, -1), layout=(16, 16))
    w2_qt = shuffle_weight(w2_qt.view(EXPERTS, MODEL_DIM, -1), layout=(16, 16))
    s1 = fp4_utils.e8m0_shuffle(w1_scale).view(EXPERTS, 2 * INTER_DIM, -1)
    s2 = fp4_utils.e8m0_shuffle(w2_scale).view(EXPERTS, MODEL_DIM, -1)
    gen_cpu = torch.Generator().manual_seed(seed + 1)
    sets = []
    for k in range(n_sets):
        perm = (
            torch.arange(EXPERTS)
            if k == 0
            else torch.randperm(EXPERTS, generator=gen_cpu)
        ).cuda()

        def pick(t, perm=perm):
            u8 = t.view(torch.uint8).index_select(0, perm).contiguous()
            return u8.view(t.dtype)

        a, b = pick(w1_qt), pick(w2_qt)
        a.is_shuffled = True
        b.is_shuffled = True
        sets.append((a, b, pick(s1), pick(s2)))
    return sets


def _inputs(M, seed):
    import torch

    from aiter import dtypes

    gen = torch.Generator(device="cuda").manual_seed(seed)
    hidden = torch.randn((M, MODEL_DIM), generator=gen, device="cuda").to(dtypes.bf16)
    ids = torch.argsort(
        torch.rand((M, EXPERTS - 1), generator=gen, device="cuda"), dim=-1
    )[:, : TOPK - 1]
    ids = torch.cat([ids, torch.full((M, 1), EXPERTS - 1, device="cuda")], -1)
    weights = torch.softmax(
        torch.randn((M, TOPK - 1), generator=gen, device="cuda"), dim=-1
    )
    weights = torch.cat([weights, torch.ones((M, 1), device="cuda")], -1)
    return hidden, weights.float().contiguous(), ids.to(dtypes.i32).contiguous()


def _child(args):
    import torch

    import aiter.fused_moe as fm
    from aiter import ActivationType, QuantType

    torch.cuda.set_device(0)
    has_fold = hasattr(fm, "_mxfp4_sort_fold_disabled")
    arms = ["main", "fold"] if has_fold else ["main"]
    sets = _build_weight_sets(args.weight_sets)

    def call(arm, hidden, weights, ids, ws):
        if has_fold:
            fm._mxfp4_sort_fold_disabled = arm == "main"
        w1, w2, s1, s2 = ws
        return fm.fused_moe(
            hidden,
            w1,
            w2,
            weights,
            ids,
            quant_type=QuantType.per_1x32,
            activation=ActivationType.Silu,
            w1_scale=s1,
            w2_scale=s2,
            gate_mode="separated",
            moe_sorting_dispatch_policy=2,
        )

    def capture(arm, slots):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                for hidden, weights, ids, k in slots:
                    call(arm, hidden, weights, ids, sets[k])
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outs = [call(arm, h, w, i, sets[k]) for h, w, i, k in slots]
        return graph, outs

    def time_graph(graph, n):
        start = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        end = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        for i in range(n):
            start[i].record()
            graph.replay()
            end[i].record()
        torch.cuda.synchronize()
        return [s.elapsed_time(e) * 1e3 for s, e in zip(start, end)]

    result = {"arms": arms, "device": torch.cuda.get_device_name(0), "rows": []}
    for M in args.m:
        slots = [
            (*_inputs(M, 1000 * M + i + args.seed), i % len(sets))
            for i in range(args.calls)
        ]
        refs = [call("main", h, w, i, sets[k]) for h, w, i, k in slots]
        torch.cuda.synchronize()
        row = {"M": M}
        graphs = {}
        for arm in arms:
            graph, outs = capture(arm, slots)
            graph.replay()
            torch.cuda.synchronize()
            err = max(
                float((o.float() - r.float()).norm() / r.float().norm())
                for o, r in zip(outs, refs)
            )
            finite = all(bool(torch.isfinite(o).all()) for o in outs)
            row[arm] = {"rel_err_vs_main_eager": err, "finite": finite}
            graphs[arm] = graph
            time_graph(graph, 10)
        samples = {arm: [] for arm in arms}
        for r in range(args.rounds):
            for arm in arms if r % 2 == 0 else arms[::-1]:
                samples[arm] += [
                    t / args.calls for t in time_graph(graphs[arm], args.replays)
                ]
        for arm in arms:
            row[arm]["us"] = statistics.median(samples[arm])
        result["rows"].append(row)
        del graphs
        torch.cuda.empty_cache()
    if has_fold:
        fm._mxfp4_sort_fold_disabled = False
    with open(args.child_out, "w") as f:
        json.dump(result, f)


def _summarize(runs):
    arms = runs[0]["arms"]
    table = []
    for i, row in enumerate(runs[0]["rows"]):
        rows = [run["rows"][i] for run in runs]
        entry = {"M": row["M"]}
        for arm in arms:
            entry[f"{arm}_us"] = statistics.median(r[arm]["us"] for r in rows)
            entry[f"{arm}_max_rel_err"] = max(
                r[arm]["rel_err_vs_main_eager"] for r in rows
            )
            entry[f"{arm}_finite"] = all(r[arm]["finite"] for r in rows)
        if "fold" in arms:
            speedups = [r["main"]["us"] / r["fold"]["us"] for r in rows]
            entry["speedup"] = entry["main_us"] / entry["fold_us"]
            entry["speedup_min"] = min(speedups)
            entry["speedup_max"] = max(speedups)
        table.append(entry)
    return table


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-m", type=int, nargs="+", default=list(range(1, 65)))
    parser.add_argument("--weight-sets", type=int, default=16)
    parser.add_argument("--calls", type=int, default=16, help="MoE calls per graph")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--replays", type=int, default=30, help="per arm and round")
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-o", "--output", help="write the JSON summary here")
    parser.add_argument("--child-out", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.weight_sets < 16:
        parser.error("--weight-sets must be >= 16 to keep the weights cold")
    if args.child_out:
        _child(args)
        return

    runs = []
    with tempfile.TemporaryDirectory() as tmp:
        for p in range(args.processes):
            out = os.path.join(tmp, f"run{p}.json")
            cmd = [sys.executable, os.path.abspath(__file__), "--child-out", out]
            cmd += ["-m", *map(str, args.m)]
            cmd += ["--weight-sets", str(args.weight_sets), "--calls", str(args.calls)]
            cmd += ["--rounds", str(args.rounds), "--replays", str(args.replays)]
            cmd += ["--seed", str(args.seed + 100 * p)]
            subprocess.run(cmd, check=True)
            with open(out) as f:
                runs.append(json.load(f))
    table = _summarize(runs)
    arms = runs[0]["arms"]
    print(f"{runs[0]['device']}, {args.processes} processes, us per MoE call")
    if "fold" in arms:
        print("| M | main us | fold us | speedup | range | fold max rel err |")
        print("|---|---|---|---|---|---|")
        for e in table:
            print(
                f"| {e['M']} | {e['main_us']:.2f} | {e['fold_us']:.2f} | "
                f"{e['speedup']:.3f} | {e['speedup_min']:.3f}-{e['speedup_max']:.3f}"
                f" | {e['fold_max_rel_err']:.1e} |"
            )
    else:
        print("| M | main us |")
        print("|---|---|")
        for e in table:
            print(f"| {e['M']} | {e['main_us']:.2f} |")
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"args": vars(args), "runs": runs, "table": table}, f, indent=1)


if __name__ == "__main__":
    main()
