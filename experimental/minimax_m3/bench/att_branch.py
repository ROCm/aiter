"""Code-fetch stalls: time attributed to branch instructions (issue -> next issue) per traced CTA,
all waves, plus the top offending branch sites.  python3 att_branch.py RUN DISPATCH LAYER [--min 0.05]"""
import argparse, collections, glob, json, os, re, sys
import torch
sys.path.insert(0, "/home/vigokhal/m3_monokernel/bench")
from att_cta import parse_chain

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("dispatch"); ap.add_argument("layer", type=int)
ap.add_argument("--min", type=float, default=0.05)
a = ap.parse_args()
dd = glob.glob(f"{a.run}/prof/ui_output_agent_*_dispatch_{a.dispatch}")[0]
tl = torch.load(glob.glob(f"{a.run}/tl_gluon_*[0-9].pt")[0])
hw = torch.load(glob.glob(f"{a.run}/tl_gluon_*_hw.pt")[0])
L = a.layer
h = hw[L, :, 0].int(); x = hw[L, :, 1].int() & 15
key2bid = {(int(x[b]) * 4 + int((h[b] >> 13) & 7), int((h[b] >> 8) & 15)): b for b in range(h.shape[0])}
code = json.load(open(f"{dd}/code.json"))["code"]
rt = json.load(open(f"{dd}/realtime.json"))
sites = collections.Counter(); cnt = collections.Counter()
per_cta = collections.defaultdict(lambda: [0.0, 0, 0])
for f in os.listdir(dd):
    m = re.match(r"se(\d+)_sm(\d+)_sl(\d+)_wv(\d+)\.json", f)
    if not m:
        continue
    se = int(m.group(1)); w = json.load(open(f"{dd}/{f}"))["wave"]
    bid = key2bid.get((se, w["cu"]), -1)
    (c0, r0), (c1, r1) = rt[f"SE{se}"][0], rt[f"SE{se}"][-1]
    us = lambda c: (r0 + (c - c0) * (r1 - r0) / (c1 - c0)) / 100.0
    ins = w["instructions"]
    for i, it in enumerate(ins):
        op = code[it[4]][0].split()[0] if it[4] < len(code) else "?"
        if not op.startswith("s_branch") and not op.startswith("s_cbranch") and op != "s_setpc_b64":
            continue
        t = us(it[0]); tn = us(ins[i + 1][0]) if i + 1 < len(ins) else us(w["end"])
        d = tn - t
        per_cta[bid][1] += 1
        if d >= a.min:
            per_cta[bid][0] += d; per_cta[bid][2] += 1
            ch = parse_chain(code[it[4]][3]) if code[it[4]][3] else []
            key = (f"{ch[0][0]}:{ch[0][1]} {ch[0][2]}" if ch else "?", op)
            sites[key] += d; cnt[key] += 1
nw = 8
print(f"branch stalls >= {a.min} us per CTA (all waves summed / 8 = per wave):")
for bid in sorted(per_cta):
    s, n, k = per_cta[bid]
    print(f"  CTA {bid:3d}: {s / nw:5.2f} us a wave over {k / nw:5.1f} stalled branches ({n / nw:.0f} branches)")
print("top sites (us summed over all traced waves, count):")
for k, v in sites.most_common(25):
    print(f"  {v:7.2f} x{cnt[k]:4d}  {k[0]:<40s} {k[1]}")
