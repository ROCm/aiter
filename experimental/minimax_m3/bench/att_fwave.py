"""One traced wave of ANY kernel (e.g. FlyDSL's, no source lines) as an opcode timeline:
consecutive instructions of one opcode merged, us after the dispatch's earliest traced wave
start (rocprofv3 --att ui output).

    python3 att_fwave.py RUN DISPATCH [--se N --cu N --simd N --slot N] [--from T --to T] [--min 0.05]
                         [--list]   (every traced wave: se cu simd slot start end n_instr)
                         [--top 15] (the wave's top opcodes by summed issue-to-next time)"""
import argparse
import collections
import glob
import json
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("dispatch")
ap.add_argument("--se", type=int, default=-1); ap.add_argument("--cu", type=int, default=0)
ap.add_argument("--simd", type=int, default=0); ap.add_argument("--slot", type=int, default=0)
ap.add_argument("--from", dest="t_from", type=float, default=-1e9); ap.add_argument("--to", type=float, default=1e9)
ap.add_argument("--min", type=float, default=0.05); ap.add_argument("--list", action="store_true")
ap.add_argument("--top", type=int, default=15)
a = ap.parse_args()
dd = glob.glob(f"{a.run}/prof/ui_output_agent_*_dispatch_{a.dispatch}")[0]
code = json.load(open(f"{dd}/code.json"))["code"]
rt = json.load(open(f"{dd}/realtime.json"))


def mk_us(se):
    (c0, r0), (c1, r1) = rt[f"SE{se}"][0], rt[f"SE{se}"][-1]
    return lambda c: (r0 + (c - c0) * (r1 - r0) / (c1 - c0)) / 100.0


waves = []
for f in os.listdir(dd):
    m = re.match(r"se(\d+)_sm(\d+)_sl(\d+)_wv(\d+)\.json", f)
    if m:
        w = json.load(open(f"{dd}/{f}"))["wave"]
        us = mk_us(int(m.group(1)))
        ins = w["instructions"]
        if not ins:
            continue
        waves.append((int(m.group(1)), w["cu"], w["simd"], w["slot"], us(ins[0][0]), us(w["end"]), len(ins), w, us))
t0 = min(v[4] for v in waves)
if a.list:
    for se, cu, simd, slot, st, en, n, _, _ in sorted(waves):
        print(f"se {se:2d} cu {cu:2d} simd {simd} slot {slot}  {st - t0:7.2f} .. {en - t0:7.2f} us  {n} instr")
    raise SystemExit
sel = [v for v in waves if (a.se < 0 or v[0] == a.se) and v[1] == a.cu and v[2] == a.simd and v[3] == a.slot]
assert sel, "no such wave (use --list)"
se, cu, simd, slot, st, en, n, w, us = sel[0]
ins = w["instructions"]
print(f"SE {se} CU {cu} simd {simd} slot {slot}: {n} instructions, {st - t0:.2f} .. {en - t0:.2f} us (t0 = earliest wave)")
runs = []
tot = collections.Counter()
for i, it in enumerate(ins):
    t = us(it[0]) - t0
    tn = (us(ins[i + 1][0]) if i + 1 < len(ins) else us(w["end"])) - t0
    op = code[it[4]][0].split()[0]
    tot[op] += tn - t
    if tn < a.t_from or t > a.to:
        continue
    if runs and runs[-1][0] == op:
        r = runs[-1]; r[2] = tn; r[3] += 1; r[4] = max(r[4], tn - t)
    else:
        runs.append([op, t, tn, 1, tn - t])
for op, t, tn, k, mx in runs:
    if tn - t >= a.min:
        print(f"{t:7.2f} .. {tn:7.2f} ({tn - t:5.2f})  {op:<34s} x{k:<4d} longest {mx:.2f}")
print("top opcodes by time (whole wave):")
for op, v in tot.most_common(a.top):
    print(f"  {v:7.2f} us  {op}")
