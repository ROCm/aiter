"""Diagnostic: identify the SOFTMAX-classified ('orange') instructions that
appear at the START of a MATRIX phase block in the FA4 overlap timeline.

For the coresident pair on (se,simd) we pick the same window the overlay uses,
tile each wave into phase runs, and for every MATRIX run we dump the leading
instructions and any SOFTMAX-classified instructions inside it (cycle, state,
mnemonic, source file:line, ISA text) so we can see what they really are.
"""
from __future__ import annotations

import argparse
import os

from .model import Trace, find_ui_dir, STATE_NAME
from .phases import PhaseTagger, MATRIX, SOFTMAX
from .window import pick_window


def state_at(wave, cycle):
    for s0, s1, st in wave.state_segments():
        if s0 <= cycle < s1:
            return STATE_NAME.get(st, "?")
    return "?"


def phase_runs(trace, tagger, wave, lo, hi):
    insts = wave.insts_in_window(lo, hi)
    runs = []  # (phase, start_cycle, end_cycle, [insts])
    for i in insts:
        ph = tagger.phase_of_lineno(i.lineno)
        if runs and runs[-1][0] == ph:
            runs[-1][3].append(i)
        else:
            runs.append([ph, i.cycle, i.cycle, [i]])
        runs[-1][2] = i.cycle
    return runs


def short_src(trace, lineno):
    c = trace.code_line(lineno)
    if c is None:
        return "?", "?"
    src = c.source or ""
    if "/" in src:
        src = ".../" + "/".join(src.rsplit("/", 1)[-1:])
    return src, (c.isa.strip()[:70] if c else "?")


def dump_runs_compact(trace, tagger, wave, lo, hi, label):
    runs = phase_runs(trace, tagger, wave, lo, hi)
    print(f"\n==== {label}  ({wave})  COMPACT RUNS  window [{lo},{hi}] ====")
    for idx, (ph, c0, c1, insts) in enumerate(runs):
        width = max(c1 - c0, 1)
        # mark the leading instruction's source so we see *what* the run is
        src0, isa0 = short_src(trace, insts[0].lineno)
        bar = "#" * min(width // 8, 60)
        print(f"  [{idx:>3}] {ph:>7} cyc[{c0}..{c1}] w={width:>4} n={len(insts):>3} "
              f"{src0:<40} | {bar}")


def slot_analysis(trace, tagger, wave, lo, hi, label):
    """Segment the window by s_barrier into slots; classify each slot by its
    dominant (cycle-weighted) phase; then list every instruction whose phase is
    the *minority* type inside that slot (e.g. softmax inside a matrix slot)."""
    bar_linenos = {c.lineno for c in trace.code
                   if (m := c.mnemonic) and m.startswith("s_barrier")}
    insts = wave.insts_in_window(lo, hi)
    # split into slots at barriers
    slots, cur = [], []
    for i in insts:
        if i.lineno in bar_linenos and cur:
            slots.append(cur); cur = []
        cur.append(i)
    if cur:
        slots.append(cur)
    print(f"\n##### {label}: per-slot phase mix + minority instrs #####")
    for si, slot in enumerate(slots):
        cyc = {}
        for k, i in enumerate(slot):
            ph = tagger.phase_of_lineno(i.lineno)
            nxt = slot[k + 1].cycle if k + 1 < len(slot) else i.cycle + max(i.total, 1)
            cyc[ph] = cyc.get(ph, 0) + max(nxt - i.cycle, 1)
        if not cyc:
            continue
        dom = max(cyc, key=cyc.get)
        tot = sum(cyc.values())
        mix = "  ".join(f"{p}:{c}({100*c//tot}%)" for p, c in
                        sorted(cyc.items(), key=lambda kv: -kv[1]))
        print(f"\n  slot {si}: dom={dom} span={slot[0].cycle}..{slot[-1].cycle}  {mix}")
        if dom == MATRIX and SOFTMAX in cyc:
            print(f"    -- SOFTMAX instrs inside this MATRIX slot:")
            for k, i in enumerate(slot):
                if tagger.phase_of_lineno(i.lineno) == SOFTMAX:
                    src, isa = short_src(trace, i.lineno)
                    print(f"      {i.cycle:>9} {state_at(wave,i.cycle):>5} "
                          f"tot={i.total:>4}  {src:<42} {isa}")


def dump_wave(trace, tagger, wave, lo, hi, label):
    runs = phase_runs(trace, tagger, wave, lo, hi)
    print(f"\n==== {label}  ({wave})  window [{lo},{hi}] ====")
    for idx, (ph, c0, c1, insts) in enumerate(runs):
        if ph != MATRIX:
            continue
        # The MATRIX run; look at the previous run (often barrier/softmax) and
        # any softmax inside the leading part of this matrix block.
        prev = runs[idx - 1] if idx > 0 else None
        prev_ph = prev[0] if prev else "-"
        print(f"\n  MATRIX run #{idx}: cyc[{c0}..{c1}] {len(insts)} insts "
              f"(prev run = {prev_ph})")
        # dump first 30 insts of the matrix block, flag softmax ones
        for i in insts[:30]:
            sub = tagger.phase_of_lineno(i.lineno)
            src, isa = short_src(trace, i.lineno)
            flag = "  <== SOFTMAX-in-MATRIX" if sub == SOFTMAX else ""
            print(f"    {i.cycle:>9} {state_at(wave,i.cycle):>5} {sub:>7} "
                  f"tot={i.total:>4}  {src:<42} {isa}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--simd", type=int, default=0)
    ap.add_argument("--se", type=int, default=0)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--center", type=float, default=0.78)
    args = ap.parse_args()

    ui = find_ui_dir(args.run_dir)
    trace = Trace(ui)
    w0, w1 = trace.coresident_pair(se=args.se, simd=args.simd)
    tagger = PhaseTagger(trace)
    lo, hi, bounds = pick_window(trace, w0, n_iters=args.iters, center_frac=args.center)
    dump_runs_compact(trace, tagger, w0, lo, hi, "WG0 sl0")
    dump_runs_compact(trace, tagger, w1, lo, hi, "WG1 sl1")
    slot_analysis(trace, tagger, w0, lo, hi, "WG0 sl0")
    slot_analysis(trace, tagger, w1, lo, hi, "WG1 sl1")


if __name__ == "__main__":
    main()
