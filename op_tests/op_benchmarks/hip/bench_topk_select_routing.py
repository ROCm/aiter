#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Device time of topk_select and top_k_per_row_prefill per call over a grid of
shapes, with the backend each call ran: the measurement the routing rules in
aiter/ops/topk_select.py and aiter/ops/topk.py rest on.

Each call is captured ten times into a CUDA graph and replayed, so the time is
the device's and not the launch path's, and every answer is checked against
torch.topk. To compare two trees, run this on each on the same GPU, alternating
the two twice with a fresh process each time, then compare the files:

    python bench_topk_select_routing.py --grid k2048 -o head.csv
    python bench_topk_select_routing.py --compare base.csv head.csv --markdown

    # Time one backend wherever it serves, to refit a rule
    python bench_topk_select_routing.py --grid k2048 --force-backend decode -o d.csv
"""

from __future__ import annotations

import argparse
import csv
import functools
import math
from collections import defaultdict

import torch

GRIDS = {
    "k2048": {
        "op": "select",
        "k": [2048],
        "rows": [1, 8, 64, 256, 1024, 4096],
        "widths": [2048, 2050, 2056, 4096, 8192, 16384, 32768, 65536, 131072],
    },
    "select": {
        "op": "select",
        "k": [128, 512, 1024, 4096],
        "rows": [1, 8, 64, 256, 1024, 2048, 4096, 8192],
        "widths": [16384, 32768, 65536, 131072, 262144, 524288, 1048576],
    },
    "prefill": {
        "op": "prefill",
        "k": [128, 512, 1024, 2048, 4096, 8192],
        "rows": [1, 64, 128, 256, 1024, 4096],
        "widths": [32768, 65536, 131072, 163840, 196608, 262144],
    },
}
FIELDS = ["op", "k", "pattern", "rows", "width", "backend", "us", "ok"]


def _ends(pattern, rows, width, k, gen):
    """Row ends for a pattern: None for dense rows."""
    dev = "cuda"
    if pattern == "dense":
        return None
    if pattern == "full":
        return torch.full((rows,), width, dtype=torch.int32, device=dev)
    if pattern == "causal":
        ends = torch.arange(width - rows + 1, width + 1, dtype=torch.int32, device=dev)
        return ends.clamp_(min=k)
    if pattern == "tiny":
        low, high = k, min(width, 4 * k)
    else:
        low, high = (k if pattern == "short" else max(k, width // 2)), width
    return torch.randint(
        low, high + 1, (rows,), dtype=torch.int32, device=dev, generator=gen
    )


def _graph_us(fn, calls):
    """Device time per call: `calls` calls captured once, replayed, best of five."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        for _ in range(calls):
            fn()
    torch.cuda.synchronize()
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    best = math.inf
    for _ in range(5):
        start.record()
        for _ in range(10):
            graph.replay()
        stop.record()
        stop.synchronize()
        best = min(best, start.elapsed_time(stop) / (10 * calls) * 1e3)
    return best


def _correct(x, idx, ends, k):
    """Indices in range, distinct live columns, the same values as torch.topk."""
    rows, width = x.shape
    lens = (
        torch.full((rows,), width, device=x.device) if ends is None else ends
    ).long()
    col = torch.arange(width, device=x.device).unsqueeze(0)
    for lo in range(0, rows, 512):
        sl = slice(lo, lo + 512)
        got, live = idx[sl].long(), lens[sl].unsqueeze(1)
        if bool((got < 0).any()) or bool((got >= live).any()):
            return False
        ref = x[sl].masked_fill(col >= live, float("-inf")).topk(k, dim=1).values
        if not torch.equal(
            x[sl].gather(1, got).sort(dim=1).values, ref.sort(dim=1).values
        ):
            return False
    return True


def _select_backend(rows, width, k, ragged, dev):
    """The backend topk_select names for the call, or "" on a tree that names
    it differently."""
    import aiter.ops.topk_select as ts
    from aiter.ops.flydsl.kernels.tensor_shim import wave_size_of

    args = (rows, width, width, k, wave_size_of(dev), ragged, None, False, True)
    try:
        return ts._choose.__wrapped__(*args, dev, True)
    except TypeError:
        return ""


def _prefill_backend(rows, width, k, dev):
    from aiter.ops import topk

    use = getattr(topk, "_should_use_sampled_prefill", None)
    return "sampled" if use and use(rows, width, 1, k, False, dev) else "per_row"


def run(args):
    import aiter  # noqa: F401
    import aiter.ops.topk_select as ts
    from aiter.ops.topk import top_k_per_row_prefill
    from aiter.ops.topk_select import topk_select

    grid = GRIDS[args.grid]
    ks = args.k or grid["k"]
    rows_list = args.rows or grid["rows"]
    widths = args.widths or grid["widths"]
    dev = torch.cuda.current_device()
    if args.force_backend:
        if grid["op"] != "select":
            raise SystemExit("--force-backend applies to topk_select grids only")
        ts._BACKENDS_BY_TIE = dict(ts._BACKENDS_BY_TIE)
        ts._BACKENDS_BY_TIE[None] = (args.force_backend,)
        ts._choose.cache_clear()
    results = []
    for k in ks:
        for pattern in args.patterns:
            for width in widths:
                for rows in rows_list:
                    if width < k or rows * width > args.max_elems:
                        continue
                    gen = torch.Generator(device="cuda").manual_seed(
                        rows * 13 + width + k
                    )
                    x = torch.randn(rows, width, device="cuda", generator=gen)
                    ends = _ends(pattern, rows, width, k, gen)
                    idx = torch.empty(rows, k, dtype=torch.int32, device="cuda")
                    if grid["op"] == "select":
                        try:
                            backend = _select_backend(
                                rows, width, k, ends is not None, dev
                            )
                        except RuntimeError:
                            continue  # no backend serves it, as when forced

                        fn = functools.partial(
                            topk_select, x, k, end=ends, output_idx=idx
                        )

                    else:
                        backend = _prefill_backend(rows, width, k, dev)
                        starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
                        stop = (
                            torch.full((rows,), width, dtype=torch.int32, device="cuda")
                            if ends is None
                            else ends
                        )

                        fn = functools.partial(
                            top_k_per_row_prefill,
                            x,
                            starts,
                            stop,
                            idx,
                            None,
                            rows,
                            width,
                            1,
                            k,
                        )

                    us = _graph_us(fn, args.calls)
                    ok = _correct(x, idx, ends, k)
                    results.append(
                        {
                            "op": grid["op"],
                            "k": k,
                            "pattern": pattern,
                            "rows": rows,
                            "width": width,
                            "backend": backend,
                            "us": f"{us:.3f}",
                            "ok": int(ok),
                        }
                    )
                    print(
                        f"{grid['op']} k={k} {pattern} {rows}x{width} "
                        f"{backend or '?'} {us:.2f} us{'' if ok else ' WRONG'}"
                    )
                    del x, ends, idx
                    torch.cuda.empty_cache()
    if args.o:
        with open(args.o, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(results)
    wrong = sum(1 for r in results if not r["ok"])
    print(f"{len(results)} cells, {wrong} wrong")
    return 1 if wrong else 0


def _load(path):
    with open(path) as f:
        return {
            (r["op"], int(r["k"]), r["pattern"], int(r["rows"]), int(r["width"])): r
            for r in csv.DictReader(f)
        }


def compare(args):
    base, head = (_load(p) for p in args.compare)
    groups = defaultdict(list)
    for key in sorted(base.keys() & head.keys()):
        b, h = base[key], head[key]
        groups[key[:2]].append((key, float(h["us"]) / float(b["us"]), b, h))
    for (op, k), cells in groups.items():
        ratios = [r for _, r, _, _ in cells]
        geo = math.exp(sum(map(math.log, ratios)) / len(ratios))
        moved = [
            c for c in cells if c[2]["backend"] and c[2]["backend"] != c[3]["backend"]
        ]
        slow = sorted((c for c in cells if c[1] > args.threshold), key=lambda c: -c[1])
        wrong = sum(1 for c in cells if not int(c[3]["ok"]) or not int(c[2]["ok"]))
        print(
            f"{op} k={k}: {len(cells)} cells, head/base geomean {geo:.3f}, "
            f"best {min(ratios):.2f}, worst {max(ratios):.3f}; backend changed on "
            f"{len(moved)}; {len(slow)} above {args.threshold}; {wrong} wrong"
        )
        for key, r, b, h in slow:
            print(
                f"  {key[2]} {key[3]}x{key[4]}: {b['us']} -> {h['us']} us "
                f"({b['backend'] or '?'} -> {h['backend'] or '?'}) {r:.3f}"
            )
        if args.markdown:
            for pattern in sorted({c[0][2] for c in cells}):
                sub = {(c[0][3], c[0][4]): c for c in cells if c[0][2] == pattern}
                rows = sorted({r for r, _ in sub})
                widths = sorted({w for _, w in sub})
                print(f"\n**{op} k={k}, {pattern} rows** -- head/base (backend)\n")
                print("| rows \\ width | " + " | ".join(map(str, widths)) + " |")
                print("|---|" + "---|" * len(widths))
                for r in rows:
                    cells_md = []
                    for w in widths:
                        c = sub.get((r, w))
                        if c is None:
                            cells_md.append("")
                            continue
                        b, h = c[2]["backend"], c[3]["backend"]
                        tag = h if b in (h, "") else f"{b}->{h or '?'}"
                        cells_md.append(f"{c[1]:.2f} {tag}")
                    print(f"| {r} | " + " | ".join(cells_md) + " |")
            print()
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--grid", choices=sorted(GRIDS), default="k2048")
    parser.add_argument("--k", type=int, nargs="+", help="override the grid's k")
    parser.add_argument("--rows", type=int, nargs="+", help="override its rows")
    parser.add_argument("--widths", type=int, nargs="+", help="override its widths")
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["dense", "short"],
        choices=["dense", "short", "half", "causal", "full", "tiny"],
        help="row ends: none, uniform in [k, width], uniform in [width/2, "
        "width], causal, the full width, or uniform in [k, 4k] -- rows far "
        "short of a preallocated pitch",
    )
    parser.add_argument(
        "--force-backend", help="time this topk_select backend wherever it serves"
    )
    parser.add_argument("--calls", type=int, default=10, help="calls per graph")
    parser.add_argument(
        "--max-elems", type=int, default=1 << 31, help="skip larger inputs"
    )
    parser.add_argument("-o", help="write the cells to this CSV")
    parser.add_argument(
        "--compare", nargs=2, metavar=("BASE", "HEAD"), help="compare two CSVs"
    )
    parser.add_argument("--threshold", type=float, default=1.02)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()
    raise SystemExit(compare(args) if args.compare else run(args))


if __name__ == "__main__":
    main()
