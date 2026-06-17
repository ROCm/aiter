"""Break down the OTHER phase: which source lines / mnemonics land in it.

    python -m att_analysis.diag_other <run_dir> [--simd 0] [--iters 3]
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

from .model import Trace, find_ui_dir, STATE_NAME
from .window import pick_window
from .aggregate import summarize  # noqa
from .phases import PhaseTagger, OTHER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--simd", type=int, default=0)
    ap.add_argument("--se", type=int, default=0)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--topn", type=int, default=25)
    args = ap.parse_args()

    ui = find_ui_dir(args.run_dir)
    trace = Trace(ui)
    tagger = PhaseTagger(trace)
    w0, w1 = trace.coresident_pair(se=args.se, simd=args.simd)
    lo, hi, _ = pick_window(trace, w0, n_iters=args.iters)

    for tag, w in (("WG0", w0), ("WG1", w1)):
        by_src = defaultdict(lambda: [0, 0, 0, 0])  # busy, stall, total, hits
        by_mnem = defaultdict(lambda: [0, 0, 0, 0])
        other_total = 0
        all_total = 0
        for inst in w.insts_in_window(lo, hi):
            ph = tagger.phase_of_lineno(inst.lineno)
            all_total += inst.total
            if ph != OTHER:
                continue
            other_total += inst.total
            c = trace.code_line(inst.lineno)
            src = c.source if c and c.source else "<no-source>"
            isa = c.isa.strip() if c else ""
            mnem = c.mnemonic if c else ""
            for d, key in ((by_src, src), (by_mnem, mnem)):
                d[key][0] += inst.exec
                d[key][1] += inst.stall
                d[key][2] += inst.total
                d[key][3] += 1
            # keep an example ISA per source line
            by_src[src].append(isa) if len(by_src[src]) == 4 else None

        print(f"\n===== {tag} {w} =====")
        print(f"OTHER total={other_total} cyc  ({100*other_total/max(all_total,1):.1f}% of window busy+stall)")
        print(f"\n-- top OTHER by source line (busy/stall/total/hits) --")
        for src, v in sorted(by_src.items(), key=lambda kv: -kv[1][2])[:args.topn]:
            ex = v[4] if len(v) > 4 else ""
            print(f"  {v[2]:>7} cyc  busy={v[0]:>6} stall={v[1]:>6} hits={v[3]:>5}  {src}")
            if ex:
                print(f"            e.g. {ex[:90]}")
        print(f"\n-- top OTHER by mnemonic --")
        for mnem, v in sorted(by_mnem.items(), key=lambda kv: -kv[1][2])[:args.topn]:
            print(f"  {v[2]:>7} cyc  busy={v[0]:>6} stall={v[1]:>6} hits={v[3]:>5}  {mnem}")


if __name__ == "__main__":
    main()
