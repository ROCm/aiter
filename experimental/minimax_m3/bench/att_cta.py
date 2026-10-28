"""Per-CTA execution from a rocprofv3 thread trace (att.sh) of the Gluon M3 kernel.

    python3 att_cta.py RUN_DIR DISPATCH LAYER [--bids 0,224,..] [--top 12] [--wave 0]

RUN_DIR: bench/logs/att/NAME (prof/ui_output_agent_*_dispatch_DISPATCH, tl_gluon_*.pt and
tl_gluon_*_hw.pt from the same run, VLLM_M3_GLUON_TRACE=3). LAYER: the traced dispatch's layer
(its hw placement maps each traced wave to its CTA; its stamps give t0). Times are us after the
layer's first CTA start (the timeline's t0), on the stamps' s_memrealtime clock.

Per traced CTA: its stage segments (the function layer_body called, from the source-line chain of
every instruction) and its longest gaps (an instruction's issue to the next one's: waits, polls,
barriers), by source line."""
import argparse
import bisect
import collections
import glob
import json
import os
import re

import torch

KERNEL = os.environ.get("ATT_KERNEL", "/home/vigokhal/m3_monokernel/m3mono/gluon/kernel.py")
PRIMS = "/home/vigokhal/m3_monokernel/m3mono/gluon/prims.py"


def def_index(path):
    """Sorted (line, function) of a file's top-level defs."""
    out = []
    for i, ln in enumerate(open(path), 1):
        m = re.match(r"def (\w+)\(", ln)
        if m:
            out.append((i, m.group(1)))
    return out


DEFS = {"kernel.py": def_index(KERNEL), "prims.py": def_index(PRIMS)}


def func_of(fname, line):
    d = DEFS.get(fname)
    if not d:
        return fname
    i = bisect.bisect_right([a for a, _ in d], line) - 1
    return d[i][1] if i >= 0 else "?"


def parse_chain(src):
    """'a.py:12 -> b.py:30 -> ..' (innermost first) -> [(file, line, func), ..]."""
    out = []
    for part in src.split(" -> "):
        m = re.search(r"([\w.]+):(\d+)$", part.strip())
        if m:
            f, l = m.group(1), int(m.group(2))
            out.append((f, l, func_of(f, l)))
    return out


def stage_of(chain):
    """The function layer_body called (else the outermost kernel.py function)."""
    funcs = [c[2] for c in chain]
    if "layer_body" in funcs:
        i = funcs.index("layer_body")
        return funcs[i - 1] if i > 0 else "layer_body"
    return funcs[-1] if funcs else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("dispatch")
    ap.add_argument("layer", type=int)
    ap.add_argument("--bids", default="")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--wave", type=int, default=-1, help="wave slot to show (-1: the CTA's latest-ending wave)")
    ap.add_argument("--window", default="", help="pA,pB: gaps only between the CTA's stamps A and B, every wave")
    a = ap.parse_args()
    dd = glob.glob(f"{a.run}/prof/ui_output_agent_*_dispatch_{a.dispatch}")[0]
    tl = torch.load(glob.glob(f"{a.run}/tl_gluon_*[0-9].pt")[0])
    hw = torch.load(glob.glob(f"{a.run}/tl_gluon_*_hw.pt")[0])
    L = a.layer
    t0 = tl[L, :, 0][tl[L, :, 0] > 0].min().item()
    h = hw[L, :, 0].int()
    x = hw[L, :, 1].int() & 15
    key2bid = {}
    for b in range(h.shape[0]):
        k = (int(x[b]) * 4 + int((h[b] >> 13) & 7), int((h[b] >> 8) & 15))
        assert k not in key2bid, f"placement key {k} not unique"
        key2bid[k] = b
    code = json.load(open(f"{dd}/code.json"))["code"]
    chains = [parse_chain(c[3]) if c[3] else [] for c in code]
    rt = json.load(open(f"{dd}/realtime.json"))

    def to_us(se, cyc):
        (c0, r0), (c1, r1) = rt[f"SE{se}"][0], rt[f"SE{se}"][-1]
        return (r0 + (cyc - c0) * (r1 - r0) / (c1 - c0) - t0) / 100.0

    ctas = collections.defaultdict(list)
    for f in os.listdir(dd):
        m = re.match(r"se(\d+)_sm(\d+)_sl(\d+)_wv(\d+)\.json", f)
        if not m:
            continue
        w = json.load(open(f"{dd}/{f}"))["wave"]
        se = int(m.group(1))
        bid = key2bid.get((se, w["cu"]), -1)
        ctas[bid].append((se, w))
    want = [int(b) for b in a.bids.split(",")] if a.bids else sorted(ctas)
    for bid in want:
        if bid not in ctas:
            print(f"CTA {bid}: not traced (traced: {sorted(ctas)})")
            continue
        waves = ctas[bid]
        stamps = {k: (tl[L, bid, k].item() - t0) / 100.0 for k in range(64) if tl[L, bid, k] > 0}
        ends = [(to_us(se, w["end"]), i) for i, (se, w) in enumerate(waves)]
        se, w = waves[max(ends)[1]] if a.wave < 0 else next(v for v in waves if v[1]["slot"] * 4 + v[1]["simd"] == a.wave)
        ins = w["instructions"]
        print(f"=== CTA {bid} (SE {se} CU {w['cu']}; {len(waves)} waves; shown: simd {w['simd']} slot {w['slot']}, "
              f"{len(ins)} instructions, {to_us(se, w['begin']):.2f} .. {to_us(se, w['end']):.2f} us) stamps: "
              + " ".join(f"p{k}={v:.2f}" for k, v in sorted(stamps.items(), key=lambda kv: kv[1])))
        segs = []
        gaps = collections.Counter()
        cnt = collections.Counter()
        for i, it in enumerate(ins):
            t = to_us(se, it[0])
            tn = to_us(se, ins[i + 1][0]) if i + 1 < len(ins) else to_us(se, w["end"])
            ch = chains[it[4]] if it[4] < len(chains) else []
            st = stage_of(ch)
            if not segs or segs[-1][0] != st:
                segs.append([st, t, tn])
            else:
                segs[-1][2] = tn
            inner = ch[0] if ch else ("?", 0, "?")
            k = (st, inner[2], f"{inner[0]}:{inner[1]}", code[it[4]][0].split()[0] if it[4] < len(code) else "?")
            gaps[k] += tn - t
            cnt[k] += 1
        merged = []
        for s in segs:  # drop sub-0.05 us blips into their neighbours
            if merged and (s[0] == merged[-1][0] or s[2] - s[1] < 0.05):
                merged[-1][2] = s[2]
            else:
                merged.append(list(s))
        print("  stages: " + " | ".join(f"{s} {b:.2f}-{e:.2f}" for s, b, e in merged))
        if a.window:
            pa, pb = [int(v.strip("p")) for v in a.window.split(",")]
            if pa not in stamps or pb not in stamps:
                print(f"  window: stamps p{pa} / p{pb} missing")
                continue
            ta, tb = stamps[pa], stamps[pb]
            print(f"  window p{pa}={ta:.2f} .. p{pb}={tb:.2f} ({tb - ta:.2f} us), per wave (simd, slot): ", end="")
            wg = collections.Counter()
            wc = collections.Counter()
            for se_, w_ in sorted(waves, key=lambda v: (v[1]["slot"], v[1]["simd"])):
                ins_ = w_["instructions"]
                first = last = None
                for i, it in enumerate(ins_):
                    t = to_us(se_, it[0])
                    if t < ta or t > tb:
                        continue
                    tn = min(to_us(se_, ins_[i + 1][0]) if i + 1 < len(ins_) else tb, tb)
                    first = t if first is None else first
                    last = tn
                    ch = chains[it[4]] if it[4] < len(chains) else []
                    inner = ch[0] if ch else ("?", 0, "?")
                    k = (inner[2], f"{inner[0]}:{inner[1]}", code[it[4]][0].split()[0])
                    wg[k] += (tn - t) / len(waves)
                    wc[k] += 1
                print(f"[{w_['simd']},{w_['slot']}] {first if first is not None else float('nan'):.2f}-"
                      f"{last if last is not None else float('nan'):.2f}", end=" ")
            print()
            for k, v in wg.most_common(a.top):
                print(f"    {v:6.2f} us/wave  x{wc[k]:<5d} {k[0]:<16s} {k[1]:<16s} {k[2]}")
            continue
        print(f"  top time by (stage, function, line, opcode):")
        for k, v in gaps.most_common(a.top):
            print(f"    {v:6.2f} us  x{cnt[k]:<5d} {k[0]:>14s} / {k[1]:<16s} {k[2]:<16s} {k[3]}")


if __name__ == "__main__":
    main()
