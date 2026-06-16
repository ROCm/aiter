"""Attribute LOAD / ADDR (and any) phase cycles to source lines + ISA.

    python -m att_analysis.breakdown <run_dir> [--simd 0] [--iters 4]
                                     [--phases load,addr]

For each co-resident wave, over the SAME window the report uses, roll up the
instructions of the requested phases by (source file:line) and by ISA mnemonic,
reporting busy (exec+stall) and stall cycles and instruction counts. This is the
"where do the brown/purple bars come from" view: is it the buffer_load issue
itself, or the address math (magic_div / coordinate_transform / offset refresh)?
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

from .model import Trace, find_ui_dir
from .window import pick_window
from .phases import PhaseTagger


def _mnemonic(isa: str) -> str:
    s = isa.strip()
    return s.split()[0] if s else "?"


def _src_short(src: str) -> str:
    if not src:
        return "(no source)"
    path, _, line = src.rpartition(":")
    return f"{os.path.basename(path)}:{line}"


def breakdown_wave(trace, tagger, wave, lo, hi, want_phases):
    by_src = defaultdict(lambda: [0, 0, 0])   # src -> [busy, stall, ninst]
    by_mn = defaultdict(lambda: [0, 0, 0])    # mnemonic -> [busy, stall, ninst]
    by_phase = defaultdict(lambda: [0, 0, 0])
    for i in wave.insts_in_window(lo, hi):
        ph = tagger.phase_of_lineno(i.lineno)
        by_phase[ph][0] += i.total
        by_phase[ph][1] += i.stall
        by_phase[ph][2] += 1
        if ph not in want_phases:
            continue
        c = trace.code_line(i.lineno)
        isa = c.isa if c else ""
        src = c.source if c else ""
        key = f"{ph:5s} {_src_short(src)}"
        by_src[key][0] += i.total
        by_src[key][1] += i.stall
        by_src[key][2] += 1
        mk = f"{ph:5s} {_mnemonic(isa)}"
        by_mn[mk][0] += i.total
        by_mn[mk][1] += i.stall
        by_mn[mk][2] += 1
    return by_src, by_mn, by_phase


def _dump(title, d):
    print(f"\n== {title} ==")
    print(f"{'key':42s} {'busy':>7s} {'stall':>7s} {'ninst':>6s}")
    for k in sorted(d, key=lambda k: -d[k][0]):
        busy, stall, n = d[k]
        print(f"{k:42s} {busy:7d} {stall:7d} {n:6d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--simd", type=int, default=0)
    ap.add_argument("--se", type=int, default=0)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--phases", default="load,addr")
    args = ap.parse_args()

    want = set(p.strip() for p in args.phases.split(","))
    ui = find_ui_dir(args.run_dir)
    trace = Trace(ui)
    tagger = PhaseTagger(trace)
    w0, w1 = trace.coresident_pair(se=args.se, simd=args.simd)
    lo, hi, _ = pick_window(trace, w0, n_iters=args.iters)
    print(f"window [{lo}, {hi}]  ({hi-lo} cyc, {args.iters} iters)  phases={sorted(want)}")

    for tag, w in (("WG0 (V-loader)", w0), ("WG1 (K-loader)", w1)):
        by_src, by_mn, by_phase = breakdown_wave(trace, tagger, w, lo, hi, want)
        print(f"\n########## {tag}  slot{w.slot} wv{w.wid} ##########")
        total_busy = sum(v[0] for v in by_phase.values())
        pj = "  ".join(f"{p}={by_phase[p][0]}" for p in sorted(by_phase, key=lambda p: -by_phase[p][0]))
        print(f"phase busy (cyc): {pj}   [sum={total_busy}]")
        _dump(f"{tag}: by source line", by_src)
        _dump(f"{tag}: by ISA mnemonic", by_mn)


if __name__ == "__main__":
    main()
