"""One traced wave of CTA BID as a timeline: consecutive instructions of one source line merged
(us after the layer's t0, the stamps' clock), from att.sh output (VLLM_M3_GLUON_TRACE=3).
    python3 att_wave.py RUN DISPATCH LAYER BID [--simd 0 --slot 0] [--from T --to T] [--min 0.03]"""
import argparse, glob, json, os, re, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from att_cta import parse_chain

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("dispatch"); ap.add_argument("layer", type=int); ap.add_argument("bid", type=int)
ap.add_argument("--simd", type=int, default=0); ap.add_argument("--slot", type=int, default=0)
ap.add_argument("--from", dest="t_from", type=float, default=-1e9); ap.add_argument("--to", type=float, default=1e9)
ap.add_argument("--min", type=float, default=0.03)
a = ap.parse_args()
dd = glob.glob(f"{a.run}/prof/ui_output_agent_*_dispatch_{a.dispatch}")[0]
tl = torch.load(glob.glob(f"{a.run}/tl_gluon_*[0-9].pt")[0])
hw = torch.load(glob.glob(f"{a.run}/tl_gluon_*_hw.pt")[0])
L = a.layer
t0 = tl[L, :, 0][tl[L, :, 0] > 0].min().item()
h = int(hw[L, a.bid, 0]); x = int(hw[L, a.bid, 1]) & 15
se, cu = x * 4 + ((h >> 13) & 7), (h >> 8) & 15
code = json.load(open(f"{dd}/code.json"))["code"]
rt = json.load(open(f"{dd}/realtime.json"))
(c0, r0), (c1, r1) = rt[f"SE{se}"][0], rt[f"SE{se}"][-1]
us = lambda c: ((r0 + (c - c0) * (r1 - r0) / (c1 - c0)) - t0) / 100.0
w = None
for f in os.listdir(dd):
    m = re.match(rf"se{se}_sm(\d+)_sl(\d+)_wv(\d+)\.json", f)
    if m:
        ww = json.load(open(f"{dd}/{f}"))["wave"]
        if ww["cu"] == cu and ww["simd"] == a.simd and ww["slot"] == a.slot:
            w = ww
assert w, f"CTA {a.bid} (SE {se} CU {cu}) not traced"
stamps = {k: (tl[L, a.bid, k].item() - t0) / 100.0 for k in range(64) if tl[L, a.bid, k] > 0}
print(f"CTA {a.bid} SE {se} CU {cu} simd {a.simd} slot {a.slot}; stamps " +
      " ".join(f"p{k}={v:.2f}" for k, v in sorted(stamps.items(), key=lambda kv: kv[1])))
ins = w["instructions"]
runs = []
for i, it in enumerate(ins):
    t = us(it[0]); tn = us(ins[i + 1][0]) if i + 1 < len(ins) else us(w["end"])
    if tn < a.t_from or t > a.to:
        continue
    ch = parse_chain(code[it[4]][3]) if code[it[4]][3] else []
    own = next((c for c in ch if c[0] == "kernel.py"), ch[0] if ch else ("?", 0, "?"))
    inner = ch[0] if ch else ("?", 0, "?")
    key = (own[2], own[1], inner[2] if inner != own else "")
    op = code[it[4]][0].split()[0]
    if runs and runs[-1][0] == key:
        r = runs[-1]; r[2] = tn; r[3] += 1
        if tn - t > r[5]: r[4], r[5] = op, tn - t
    else:
        runs.append([key, t, tn, 1, op, tn - t])
for (fn, ln, helper), t, tn, n, op, mx in runs:
    if tn - t >= a.min:
        print(f"{t:7.2f} .. {tn:7.2f} ({tn - t:5.2f})  {fn}:{ln} {helper:<14s} x{n:<4d} longest {op} {mx:.2f}")
