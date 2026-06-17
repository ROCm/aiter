"""Measure the *work* duration of each FA4 pipeline phase, excluding the
barrier wait — i.e. how long the phase's compute (incl. internal lgkm/vmem/addr
stalls) actually takes before the wave reaches the next block barrier.

There are 4 high-level phases (matrix includes the next-tile prefetch):
    WG0 matrix, WG0 softmax, WG1 softmax, WG1 matrix
WG0 and WG1 run the same matrix‖softmax pipeline staggered by one phase, so
between any two consecutive block barriers one WG is in matrix and the other in
softmax. The block barrier (s_barrier) syncs all 8 waves, so whichever WG
finishes its slot first idles at the barrier (barrier_wait) until the slower one
arrives — that idle is the bubble we care about.

Per slot we report:
    work  = barrier[i+1].arrival - barrier[i].release   (phase compute time)
    wait  = barrier[i+1].stall                           (idle at the barrier)
Slots are labelled matrix/softmax by the presence of v_mfma in the slot.

    python -m att_analysis.phase_durations <run_dir> [--simd 0] [--se 0] [--iters 6]
"""
from __future__ import annotations

import argparse
import statistics
from collections import defaultdict

from .model import Trace, find_ui_dir
from .window import pick_window
from .phases import PhaseTagger


def _is_barrier(c) -> bool:
    return bool(c) and c.mnemonic.startswith("s_barrier")


def _slots(trace: Trace, wave, lo: int, hi: int):
    """Yield (label, work_cyc, wait_cyc, start_cyc) for each slot between
    consecutive s_barriers in [lo, hi]."""
    insts = wave.insts_in_window(lo, hi)
    # locate barriers (index into insts)
    bar = [k for k, i in enumerate(insts) if _is_barrier(trace.code_line(i.lineno))]
    for a, b in zip(bar, bar[1:]):
        ba, bb = insts[a], insts[b]
        release = ba.cycle + max(ba.total, 1)   # when this barrier lets the wave go
        arrival = bb.cycle                        # when wave reaches next barrier
        work = arrival - release
        wait = bb.stall                           # idle spent at the next barrier
        if work <= 0:
            continue
        has_mfma = any(
            (trace.code_line(insts[k].lineno) or None)
            and trace.code_line(insts[k].lineno).mnemonic.startswith(("v_mfma", "v_smfmac"))
            for k in range(a + 1, b)
        )
        yield ("matrix" if has_mfma else "softmax"), work, wait, release


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--simd", type=int, default=0)
    ap.add_argument("--se", type=int, default=0)
    ap.add_argument("--iters", type=int, default=6)
    args = ap.parse_args()

    trace = Trace(find_ui_dir(args.run_dir))
    w0, w1 = trace.coresident_pair(se=args.se, simd=args.simd)
    lo, hi, _ = pick_window(trace, w0, n_iters=args.iters)

    agg: dict[tuple[str, str], dict[str, list]] = defaultdict(
        lambda: {"work": [], "wait": []})
    per_wave = {}
    for wg, w in (("WG0", w0), ("WG1", w1)):
        rows = list(_slots(trace, w, lo, hi))
        per_wave[wg] = rows
        for label, work, wait, _ in rows:
            agg[(wg, label)]["work"].append(work)
            agg[(wg, label)]["wait"].append(wait)

    def med(xs):
        return statistics.median(xs) if xs else 0

    print(f"window [{lo}..{hi}] ({hi-lo} cyc)  simd={args.simd} se={args.se}\n")
    print(f"{'phase':14} {'n':>3} {'work(med)':>10} {'wait(med)':>10} {'work(all)'}")
    order = [("WG0", "matrix"), ("WG1", "softmax"),
             ("WG0", "softmax"), ("WG1", "matrix")]
    for key in order:
        wk = agg[key]["work"]
        wt = agg[key]["wait"]
        allw = ",".join(str(x) for x in wk)
        print(f"{key[0]+' '+key[1]:14} {len(wk):>3} {med(wk):>10.0f} {med(wt):>10.0f} [{allw}]")

    mx0, mx1 = med(agg[("WG0", "matrix")]["work"]), med(agg[("WG1", "matrix")]["work"])
    sm0, sm1 = med(agg[("WG0", "softmax")]["work"]), med(agg[("WG1", "softmax")]["work"])
    print("\noverlap pairs (same two barriers; faster side waits):")
    print(f"  WG0 matrix  {mx0:>5.0f}  ‖  WG1 softmax {sm1:>5.0f}   -> bubble {abs(mx0-sm1):>4.0f} "
          f"({'WG1 softmax' if sm1>mx0 else 'WG0 matrix'} is longer)")
    print(f"  WG0 softmax {sm0:>5.0f}  ‖  WG1 matrix  {mx1:>5.0f}   -> bubble {abs(sm0-mx1):>4.0f} "
          f"({'WG1 matrix' if mx1>sm0 else 'WG0 softmax'} is longer)")
    print(f"\n  WG0 matrix vs WG1 matrix: {mx0:.0f} vs {mx1:.0f}  (Δ {mx1-mx0:+.0f})")


if __name__ == "__main__":
    main()
