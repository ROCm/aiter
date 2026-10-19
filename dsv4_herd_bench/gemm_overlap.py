"""Routed GEMM duration split by whether another stream runs concurrently.

    python gemm_overlap.py <trace.json.gz>
"""

import gzip
import json
import re
import statistics
import sys

PATS = {
    "gemm1": r"a8w4_tdm.*K4096|_moe_gemm_a8w4.*SWIGLU_1",
    "gemm2": r"a8w4_tdm.*K2048|_moe_gemm_a8w4.*SWIGLU_0",
}

with gzip.open(sys.argv[1], "rt") as f:
    ev = [e for e in json.load(f)["traceEvents"] if e.get("ph") == "X" and e.get("cat") == "kernel"]
ev.sort(key=lambda e: e["ts"])
others = [e for e in ev if e["tid"] != 0]

for tag, pat in PATS.items():
    alone, shared, frac = [], [], []
    for x in (e for e in ev if re.search(pat, e["name"])):
        xs, xe = x["ts"], x["ts"] + x["dur"]
        ov = sum(
            max(0.0, min(xe, y["ts"] + y["dur"]) - max(xs, y["ts"]))
            for y in others
            if y["ts"] < xe and y["ts"] + y["dur"] > xs and y["tid"] != x["tid"]
        )
        (shared if ov > 0 else alone).append(x["dur"])
        frac.append(ov / x["dur"])
    med = lambda v: statistics.median(v) if v else float("nan")
    print(f"{tag}: n={len(alone) + len(shared)} alone n={len(alone)} median {med(alone):.2f} us | "
          f"overlapped n={len(shared)} median {med(shared):.2f} us | "
          f"mean overlapped fraction {statistics.mean(frac):.2%}")
