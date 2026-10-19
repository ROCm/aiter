"""Print the kernel timeline around one routed GEMM1 launch.

    python timeline_trace.py <trace.json.gz> [index] [before] [after]
"""

import gzip
import json
import re
import sys

path = sys.argv[1]
idx = int(sys.argv[2]) if len(sys.argv) > 2 else 400
before = int(sys.argv[3]) if len(sys.argv) > 3 else 14
after = int(sys.argv[4]) if len(sys.argv) > 4 else 8

with gzip.open(path, "rt") as f:
    ev = [e for e in json.load(f)["traceEvents"] if e.get("ph") == "X" and e.get("cat") == "kernel"]
ev.sort(key=lambda e: e["ts"])
g1 = [i for i, e in enumerate(ev) if re.search(r"a8w4_tdm.*K4096|_moe_gemm_a8w4.*SWIGLU_1", e["name"])]
c = g1[idx]
t0 = ev[c]["ts"]
for e in ev[c - before : c + after]:
    mark = "<== GEMM1" if e is ev[c] else ""
    print(f"  tid={e['tid']:>3} start {e['ts'] - t0:+9.1f} end {e['ts'] + e['dur'] - t0:+9.1f} "
          f"dur {e['dur']:7.2f}  {e['name'][:80]} {mark}")
