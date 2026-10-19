"""GPU wall vs busy time per decode step, plus non-kernel GPU ops.

    python idle_trace.py <trace.json.gz> [--layer-kernel qk_norm_rope --layers 43]

busy = union of all GPU op intervals (any stream); wall = first-to-last GPU op;
idle = wall - busy. Also lists the largest inter-kernel gaps on the main stream
by the kernel that follows the gap, so launch-side stalls show up by name.
"""

import argparse
import collections
import gzip
import json
import re
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--layer-kernel", default="qk_norm_rope")
ap.add_argument("--layers", type=int, default=43)
ap.add_argument("--top", type=int, default=15)
args = ap.parse_args()

with gzip.open(args.trace, "rt") as f:
    events = json.load(f)["traceEvents"]
gpu = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memset", "gpu_memcpy")]
gpu.sort(key=lambda e: e["ts"])
steps = sum(1 for e in gpu if e.get("cat") == "kernel" and args.layer_kernel in e["name"]) / args.layers

wall = gpu[-1]["ts"] + gpu[-1]["dur"] - gpu[0]["ts"]
busy, cur_s, cur_e = 0.0, gpu[0]["ts"], gpu[0]["ts"] + gpu[0]["dur"]
for e in gpu[1:]:
    s, t = e["ts"], e["ts"] + e["dur"]
    if s > cur_e:
        busy += cur_e - cur_s
        cur_s, cur_e = s, t
    else:
        cur_e = max(cur_e, t)
busy += cur_e - cur_s
print(f"steps={steps:.1f}  wall/step={wall / steps:8.1f} us  busy/step={busy / steps:8.1f} us  "
      f"idle/step={(wall - busy) / steps:8.1f} us")

cats = collections.Counter(e["cat"] for e in gpu)
print("GPU ops per step by category:", {k: round(v / steps, 1) for k, v in cats.items()})
for cat in ("gpu_memset", "gpu_memcpy"):
    names = collections.Counter(e["name"] for e in gpu if e["cat"] == cat)
    for n, c in names.most_common(5):
        print(f"  {cat}: {c / steps:6.1f}/step  {n[:80]}")

main = [e for e in gpu if e["tid"] == 0]
gap_by_next = collections.defaultdict(list)
for prev, nxt in zip(main, main[1:]):
    gap = nxt["ts"] - (prev["ts"] + prev["dur"])
    key = re.sub(r"_\d+$", "", nxt["name"][:70])
    gap_by_next[key].append(gap)
rows = sorted(((sum(v) / steps, statistics.median(v), len(v) / steps, k)
               for k, v in gap_by_next.items()), reverse=True)
print(f"\nmain-stream gap before kernel (us/step, median us, count/step), top {args.top}:")
for tot, med, cnt, k in rows[: args.top]:
    print(f"  {tot:8.1f} {med:7.2f} {cnt:6.1f}  {k}")
