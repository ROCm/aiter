"""Stream placement and cross-stream overlap of the MoE-adjacent kernels.

    python overlap_trace.py <trace.json.gz>
"""

import collections
import gzip
import json
import re
import sys

WATCH = {
    "gemm1": r"a8w4_tdm.*K4096|_moe_gemm_a8w4.*SWIGLU_1",
    "gemm2": r"a8w4_tdm.*K2048|_moe_gemm_a8w4.*SWIGLU_0",
    "a8w8_nb3_512": r"gemm_a8w8_mx128_t32x32x512_mw2_nw2_nb3_sk1",
    "a8w8_nb3_1024": r"gemm_a8w8_mx128_t32x32x1024_mw2_nw2_nb3_sk1",
    "act_and_mul": r"act_and_mul_kernel",
}

with gzip.open(sys.argv[1], "rt") as f:
    ev = [e for e in json.load(f)["traceEvents"] if e.get("ph") == "X" and e.get("cat") == "kernel"]
ev.sort(key=lambda e: e["ts"])

streams = collections.defaultdict(collections.Counter)
for e in ev:
    streams[e["tid"]]["_all"] += 1
    for tag, pat in WATCH.items():
        if re.search(pat, e["name"]):
            streams[e["tid"]][tag] += 1
print("kernels per stream (tid):")
for tid, c in sorted(streams.items(), key=lambda kv: -kv[1]["_all"]):
    print(f"  tid={tid}: {dict(c)}")


def tagged(tag):
    return [e for e in ev if re.search(WATCH[tag], e["name"])]


def overlap_us(a, b):
    """Total time each kernel in `a` overlaps any kernel of `b` on another stream."""
    out = []
    j = 0
    for x in a:
        xs, xe = x["ts"], x["ts"] + x["dur"]
        ov = 0.0
        for y in b:
            if y["tid"] == x["tid"]:
                continue
            ys, ye = y["ts"], y["ts"] + y["dur"]
            if ye < xs or ys > xe:
                continue
            ov += min(xe, ye) - max(xs, ys)
        out.append((x["dur"], ov))
    return out

for tag in ("a8w8_nb3_512", "a8w8_nb3_1024", "act_and_mul"):
    for other in ("gemm1", "gemm2"):
        rows = overlap_us(tagged(tag), tagged(other))
        if not rows:
            continue
        n_ov = sum(1 for d, o in rows if o > 0)
        mean_d = sum(d for d, _ in rows) / len(rows)
        mean_d_ov = (sum(d for d, o in rows if o > 0) / n_ov) if n_ov else 0
        mean_d_no = (sum(d for d, o in rows if o == 0) / (len(rows) - n_ov)) if len(rows) > n_ov else 0
        print(f"{tag:14s} vs {other}: {n_ov}/{len(rows)} launches overlap; "
              f"mean dur {mean_d:.2f} us (overlapping {mean_d_ov:.2f}, alone {mean_d_no:.2f})")
