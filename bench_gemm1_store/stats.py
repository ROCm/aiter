#!/usr/bin/env python3
"""Load-only bandwidth for the gemm1 store on/off sweep.  Usage: stats.py <log>"""
import re, sys, statistics as st, math

E, N1, TOKENS, K, INTER, TOPK = 96, 6144, 512, 7168, 3072, 6
# Load only: w1 + activations, each +1/16 for the e8m0 block scales.
# The output store is deliberately NOT in the numerator.
LOAD = 17 / 16 * (E * N1 * K * 0.5 + TOKENS * K * 0.5)
# gemm1 has stage1_quant_out=1 (kernel name carries _q1r4): the output is
# quantized to fp4 in the epilogue, 0.5 B/elem, not bf16.
STORE = (TOKENS * TOPK) * INTER * 0.5 + (TOKENS * TOPK) * INTER / 32

d = {}
for path in sys.argv[1:]:
    for line in open(path):
        m = re.search(r"iter=(\d+) store(ON|OFF).*?[\d,]+\.\d\s+([\d.]+)\s+CUDA", line)
        if m:
            d.setdefault(m.group(2), {})[int(m.group(1))] = float(m.group(3))

print(f"load  = {LOAD:,.0f} B   (bandwidth numerator; same for both cases)")
print(f"store = {STORE:,.0f} B   ({100 * STORE / LOAD:.3f}% of load; NOT counted)\n")
for k in ("ON", "OFF"):
    if k not in d:
        continue
    v = sorted(d[k].values()); n = len(v); med = st.median(v)
    iqr = v[int(.75 * (n - 1))] - v[int(.25 * (n - 1))]
    print(f"  store {k:<3}  n={n:2d}  us med {med:7.2f}  IQR {iqr:4.2f}  {LOAD / (med * 1e6):.3f} TB/s")

if "ON" in d and "OFF" in d:
    rs = [100 * (d["ON"][i] / d["OFF"][i] - 1) for i in d["ON"] if i in d["OFF"]]
    m, s = st.mean(rs), st.stdev(rs)
    half = 1.96 * s / math.sqrt(len(rs))
    print(f"\n  removing the store: mean {m:+.2f}%  median {st.median(rs):+.2f}%  "
          f"sd {s:.2f}%  95%CI {m - half:+.2f}..{m + half:+.2f}%  "
          f"faster {sum(1 for r in rs if r > 0)}/{len(rs)}")
    print("  per-round: " + " ".join(f"{r:+.2f}" for r in rs))
