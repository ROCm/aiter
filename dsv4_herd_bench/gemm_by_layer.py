"""Routed GEMM time by layer index within a decode step.

    python gemm_by_layer.py <trace_a.json.gz> <trace_b.json.gz> [--layers 43]

Layer index is the GEMM1 launch's position between two consecutive launches of
--step-kernel, a kernel launched once per decode step. Prints, per layer index,
the median GEMM1/GEMM2 time of each trace and the difference (b - a).
"""

import argparse
import collections
import gzip
import json
import re
import statistics

GEMM1 = r"a8w4_tdm.*K4096|_moe_gemm_a8w4.*SWIGLU_1"
GEMM2 = r"a8w4_tdm.*K2048|_moe_gemm_a8w4.*SWIGLU_0"


def load(path):
    with gzip.open(path, "rt") as f:
        ev = [e for e in json.load(f)["traceEvents"] if e.get("ph") == "X" and e.get("cat") == "kernel"]
    ev.sort(key=lambda e: e["ts"])
    return ev


def once_per_step_candidates(ev, layers):
    counts = collections.Counter(e["name"] for e in ev)
    g1 = sum(1 for e in ev if re.search(GEMM1, e["name"]))
    steps = g1 / layers
    return [n for n, c in counts.items() if abs(c - steps) <= 1], steps


def by_layer(ev, step_kernel, layers):
    out = collections.defaultdict(lambda: ([], []))
    idx = None
    pending2 = None
    for e in ev:
        if step_kernel in e["name"]:
            idx = 0
            continue
        if idx is None:
            continue
        if re.search(GEMM1, e["name"]):
            if idx < layers:
                out[idx][0].append(e["dur"])
                pending2 = idx
            idx += 1
        elif re.search(GEMM2, e["name"]) and pending2 is not None:
            out[pending2][1].append(e["dur"])
            pending2 = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--layers", type=int, default=43)
    ap.add_argument("--step-kernel", default=None)
    args = ap.parse_args()

    ea, eb = load(args.a), load(args.b)
    if args.step_kernel is None:
        ca, _ = once_per_step_candidates(ea, args.layers)
        cb, _ = once_per_step_candidates(eb, args.layers)
        common = sorted(set(ca) & set(cb))
        if not common:
            raise SystemExit(f"no common once-per-step kernel; candidates a={ca[:5]} b={cb[:5]}")
        args.step_kernel = common[0]
    print(f"step boundary kernel: {args.step_kernel[:90]}")
    la, lb = by_layer(ea, args.step_kernel, args.layers), by_layer(eb, args.step_kernel, args.layers)
    med = lambda v: statistics.median(v) if v else float("nan")
    print("\nlayer |  GEMM1 a    b     b-a  |  GEMM2 a    b     b-a  | n")
    rows = []
    for i in range(args.layers):
        g1a, g1b = med(la[i][0]), med(lb[i][0])
        g2a, g2b = med(la[i][1]), med(lb[i][1])
        rows.append((i, g1b - g1a, g2b - g2a))
        print(f"{i:5d} | {g1a:6.2f} {g1b:6.2f} {g1b - g1a:+6.2f} | {g2a:6.2f} {g2b:6.2f} {g2b - g2a:+6.2f} | "
              f"{len(la[i][0])}/{len(lb[i][0])}")
    hash_rows, rest = rows[:3], rows[3:]
    print(f"\nlayers 0-2  mean diff: GEMM1 {statistics.mean(r[1] for r in hash_rows):+.2f}  "
          f"GEMM2 {statistics.mean(r[2] for r in hash_rows):+.2f}")
    print(f"layers 3-{args.layers - 1} mean diff: GEMM1 {statistics.mean(r[1] for r in rest):+.2f}  "
          f"GEMM2 {statistics.mean(r[2] for r in rest):+.2f}  "
          f"(GEMM1 slower in {sum(r[1] > 0 for r in rest)}/{len(rest)} layers)")


if __name__ == "__main__":
    main()
