"""Per-decode-step GPU kernel time from an ATOM torch-profiler trace.

    python analyze_trace.py <trace.json.gz> [<trace.json.gz> ...] [--top N]

Steps are counted from a kernel every layer launches once (--layer-kernel,
43 layers), so the numbers are GPU kernel time per decode step, independent of
launch gaps (the classic graph path used for tracing spreads launches out).
With two traces, prints a side-by-side diff (second minus first).
"""

import argparse
import collections
import gzip
import json
import re

CATEGORIES = (
    ("moe_select", r"topk_gating|^_topk_|_keepk|herd_fused|herd_finalize"),
    ("moe_sort", r"moe_route|psum|_combined_routing|_sum_bitmatrix|FillFunctor<int>"),
    ("moe_quant", r"multidest_quant|_downcast_to_mxfp|_static_per_tensor_quant"),
    ("moe_gemm1", r"a8w4_tdm.*K4096|_moe_gemm_a8w4.*SWIGLU_1|gugu"),
    ("moe_gemm2", r"a8w4_tdm.*K2048|_moe_gemm_a8w4.*SWIGLU_0"),
    ("moe_reduce", r"moe_gather_reduce|reduce_grouped"),
    ("hash_route", r"hash_topk"),
)


def category(name):
    for cat, pattern in CATEGORIES:
        if re.search(pattern, name):
            return cat
    return "non_moe"


def per_step(path, layer_kernel, layers):
    with gzip.open(path, "rt") as f:
        events = json.load(f)["traceEvents"]
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel"]
    steps = sum(1 for k in kernels if layer_kernel in k["name"]) / layers
    us = collections.defaultdict(float)
    n = collections.Counter()
    for k in kernels:
        us[k["name"]] += k["dur"]
        n[k["name"]] += 1
    return steps, {name: (us[name] / steps, n[name] / steps) for name in us}


def by_category(rows):
    cats = collections.defaultdict(lambda: [0.0, 0.0])
    for name, (us, cnt) in rows.items():
        cats[category(name)][0] += us
        cats[category(name)][1] += cnt
    return cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--layer-kernel", default="qk_norm_rope")
    ap.add_argument("--layers", type=int, default=43)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    data = [per_step(t, args.layer_kernel, args.layers) for t in args.traces]
    order = [c for c, _ in CATEGORIES] + ["non_moe"]
    for path, (steps, rows) in zip(args.traces, data):
        cats = by_category(rows)
        total = sum(v[0] for v in cats.values())
        print(f"\n{path}\n  steps={steps:.1f}  GPU kernel time/step={total:.1f} us")
        for c in order:
            us, cnt = cats.get(c, (0.0, 0.0))
            print(f"  {c:11s} {us:9.1f} us  {cnt:7.1f} launches  ({us / args.layers:6.2f} us/layer)")
        print(f"  top {args.top} kernels (us/step, launches/step, us/launch):")
        for name, (us, cnt) in sorted(rows.items(), key=lambda kv: -kv[1][0])[: args.top]:
            print(f"    {us:8.1f} {cnt:6.1f} {us / cnt:7.2f}  [{category(name)}] {name[:110]}")

    if len(data) == 2:
        (_, a), (_, b) = data
        ca, cb = by_category(a), by_category(b)
        print("\ncategory diff (second - first), us/step:")
        for c in order:
            d = cb.get(c, [0.0])[0] - ca.get(c, [0.0])[0]
            print(f"  {c:11s} {ca.get(c, [0.0])[0]:9.1f} -> {cb.get(c, [0.0])[0]:9.1f}  ({d:+8.1f})")
        ta = sum(v[0] for v in ca.values())
        tb = sum(v[0] for v in cb.values())
        print(f"  {'total':11s} {ta:9.1f} -> {tb:9.1f}  ({tb - ta:+8.1f})")
        print("\nlargest non-MoE kernel diffs (us/step):")
        names = set(a) | set(b)
        diffs = sorted(
            ((b.get(n, (0, 0))[0] - a.get(n, (0, 0))[0], n) for n in names
             if category(n) == "non_moe"),
            key=lambda x: -abs(x[0]),
        )
        for d, n in diffs[:15]:
            print(f"  {d:+8.1f}  {a.get(n, (0, 0))[0]:8.1f} -> {b.get(n, (0, 0))[0]:8.1f}  {n[:100]}")


if __name__ == "__main__":
    main()
