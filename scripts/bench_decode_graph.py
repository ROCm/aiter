# Benchmark graph-replayed shuffled-cache decode against eager launch timing.
"""Run from the checkout root with HIP_VISIBLE_DEVICES and ENABLE_CK=0 set.

--trace executes one warmed call per backend in named NVTX ranges for rocprofv3
--kernel-trace --marker-trace. All times are microseconds per attention call.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
import aiter.ops.unified_attention as ua
from scripts.bench_decode_mixed_shuffled import (
    build_shuffled_full,
    ref_paged_attn,
    time_us,
)

BACKENDS = ("flydsl", "triton", "gluon", "default")
BATCHES = (8, 16, 24, 32, 48, 56, 64)
CONTEXTS = (1024, 2048, 4096, 8192, 16384)
CALLS_PER_GRAPH = 20
REPLAYS = 50


def invoke(kw, backend):
    return ua.unified_attention(**kw, backend=None if backend == "default" else backend)


def check_output(kw, output, want, backend):
    invoke(kw, backend)
    of = output.float()
    err = (of - want).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        of.flatten(), want.flatten(), dim=0
    ).item()
    if not (err < 1e-1 and cos > 0.99):
        raise AssertionError(
            f"{backend}: correctness failed max_err={err:.6f} cos={cos:.6f}"
        )


def graph_us(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(CALLS_PER_GRAPH):
            fn()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(REPLAYS):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / (REPLAYS * CALLS_PER_GRAPH)


def bench_cell(b, ctx, trace=False):
    kw, ref_kw, output = build_shuffled_full([1] * b, [ctx] * b)
    want = ref_paged_attn(**ref_kw)
    row = {"b": b, "ctx": ctx}
    for backend in BACKENDS:
        fn = lambda backend=backend: invoke(kw, backend)
        check_output(kw, output, want, backend)
        if trace:
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(f"decode_b{b}_ctx{ctx}_{backend}")
            fn()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            continue
        row[f"{backend}_eager_us"] = time_us(fn)
        try:
            row[f"{backend}_graph_us"] = graph_us(fn)
        except Exception as exc:  # noqa: BLE001 - record any capture failure per cell
            row[f"{backend}_graph_us"] = f"capture-failed: {type(exc).__name__}: {exc}"
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cell", nargs=2, type=int, metavar=("B", "CTX"))
    p.add_argument("--quick", action="store_true")
    p.add_argument(
        "--trace", action="store_true", help="one warmed, NVTX-marked call per backend"
    )
    a = p.parse_args()
    import aiter

    print(
        f"aiter={aiter.__file__} HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')} device={torch.cuda.get_device_name()}",
        flush=True,
    )
    if a.cell:
        cells = [tuple(a.cell)]
    elif a.trace:
        cells = [(8, 4096), (16, 8192), (56, 16384), (64, 1024)]
    elif a.quick:
        cells = [(8, 4096), (16, 8192), (64, 1024)]
    else:
        cells = [(b, ctx) for b in BATCHES for ctx in CONTEXTS]
    fields = ["b", "ctx"] + [
        f"{backend}_{kind}_us" for backend in BACKENDS for kind in ("graph", "eager")
    ]
    if not a.trace:
        writer = csv.DictWriter(sys.stdout, fieldnames=fields)
        writer.writeheader()
    for b, ctx in cells:
        row = bench_cell(b, ctx, trace=a.trace)
        if a.trace:
            print(f"traced b={b} ctx={ctx}", flush=True)
        else:
            writer.writerow(row)
            sys.stdout.flush()


if __name__ == "__main__":
    main()
