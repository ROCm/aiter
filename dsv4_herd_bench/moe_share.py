"""Routed-MoE GPU kernel time per decode step, split hash layers vs the rest.

    python moe_share.py <trace_a.json.gz> <trace_b.json.gz> [--layers 43 --hash-layers 3]

Every routed-MoE kernel is charged to a layer: routing / quant kernels to the
layer whose GEMM1 comes next, GEMM1 to its own layer, GEMM2 / reduce to the
layer whose GEMM1 came last. Steps are delimited by --step-kernel.
Per-step numbers are medians over complete steps. "GEMM" rows count only the
routed GEMM1 + GEMM2 kernels.
"""

import argparse
import collections
import gzip
import json
import re
import statistics

PRE = r"topk_gating|^_topk_|_keepk|moe_route|psum|_combined_routing|_sum_bitmatrix|" \
      r"FillFunctor<int>|multidest_quant|_downcast_to_mxfp|_hash_topk"
GEMM1 = r"a8w4_tdm.*K4096|_moe_gemm_a8w4.*SWIGLU_1"
GEMM2 = r"a8w4_tdm.*K2048|_moe_gemm_a8w4.*SWIGLU_0"
POST = r"moe_gather_reduce|reduce_grouped"


def steps(path, step_kernel, layers):
    with gzip.open(path, "rt") as f:
        ev = [e for e in json.load(f)["traceEvents"] if e.get("ph") == "X" and e.get("cat") == "kernel"]
    ev.sort(key=lambda e: e["ts"])
    out, cur, gemm, count, total = [], None, None, 0, 0.0
    for e in ev:
        n = e["name"]
        if step_kernel in n:
            if cur is not None and count == layers:
                out.append((cur, gemm, total))
            cur, gemm = collections.defaultdict(float), collections.defaultdict(float)
            count, total = 0, 0.0
            continue
        if cur is None:
            continue
        total += e["dur"]
        if re.search(GEMM1, n):
            cur[count] += e["dur"]
            gemm[count] += e["dur"]
            count += 1
        elif re.search(GEMM2, n):
            cur[count - 1] += e["dur"]
            gemm[count - 1] += e["dur"]
        elif re.search(PRE, n):
            cur[count] += e["dur"]
        elif re.search(POST, n):
            cur[count - 1] += e["dur"]
    return out


def summarize(path, args):
    rows = steps(path, args.step_kernel, args.layers)
    med = statistics.median
    h = args.hash_layers

    def split(idx):
        all_ = med(sum(r[idx].values()) for r in rows)
        hashl = med(sum(v for k, v in r[idx].items() if k < h) for r in rows)
        rest = med(sum(v for k, v in r[idx].items() if k >= h) for r in rows)
        return all_, hashl, rest

    return len(rows), split(0), split(1), med(r[2] for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--layers", type=int, default=43)
    ap.add_argument("--hash-layers", type=int, default=3)
    ap.add_argument("--step-kernel", default="_fill_deferred_decode_ids_kernel")
    args = ap.parse_args()
    na, moe_a, gemm_a, ta = summarize(args.a, args)
    nb, moe_b, gemm_b, tb = summarize(args.b, args)
    pct = lambda x, y: (y - x) / x * 100
    h, L = args.hash_layers, args.layers
    print(f"complete steps: a={na} b={nb}")
    print(f"{'':34s} {'a (us)':>9s} {'b (us)':>9s} {'b-a':>8s} {'b/a-1':>7s}")
    rows = (
        (f"routed MoE, all {L} layers", moe_a[0], moe_b[0]),
        (f"routed MoE, layers 0-{h - 1} (hash)", moe_a[1], moe_b[1]),
        (f"routed MoE, layers {h}-{L - 1}", moe_a[2], moe_b[2]),
        (f"GEMM1+GEMM2, all {L} layers", gemm_a[0], gemm_b[0]),
        (f"GEMM1+GEMM2, layers 0-{h - 1} (hash)", gemm_a[1], gemm_b[1]),
        (f"GEMM1+GEMM2, layers {h}-{L - 1}", gemm_a[2], gemm_b[2]),
        ("all GPU kernels in the step", ta, tb),
    )
    for label, x, y in rows:
        print(f"{label:34s} {x:9.1f} {y:9.1f} {y - x:+8.1f} {pct(x, y):+6.1f}%")
    print(f"routed MoE share of step kernel time: a {moe_a[0] / ta:.1%}  b {moe_b[0] / tb:.1%}")


if __name__ == "__main__":
    main()
