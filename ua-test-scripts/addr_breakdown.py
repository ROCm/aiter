#!/usr/bin/env python3
"""Attribute the ATT `addr` phase to specific source lines.

Sums per-instruction aggregate latency (busy) and stall cycles from a captured
ATT trace, classifies each instruction with the same PhaseTagger used by the
overlap chart, and ranks the ADDR-phase instructions by source file:line so we
can see what the `addr` cycles actually are (offset refresh vs kernel setup vs
mask vs store vs coordinate transform).
"""
import sys
from collections import defaultdict

from att_analysis.model import Trace, find_ui_dir
from att_analysis import phases

run = sys.argv[1] if len(sys.argv) > 1 else \
    "rocprof_analysis/runs/att_linetables_b16_sq10000"
ui = find_ui_dir(run)
tr = Trace(ui)
tagger = phases.PhaseTagger(tr)

# Sum latency(busy)+stall per (phase, source-file:line, isa) over all code lines.
by_phase = defaultdict(lambda: [0, 0])           # phase -> [busy, stall]
addr_by_src = defaultdict(lambda: [0, 0, set()])  # "file:line" -> [busy, stall, {isa}]
addr_by_file = defaultdict(lambda: [0, 0])

for c in tr.code:
    if c.is_comment:
        continue
    ph = tagger.phase_of_lineno(c.lineno)
    by_phase[ph][0] += c.latency
    by_phase[ph][1] += c.stall
    if ph == phases.ADDR:
        src = c.source or "<no-source>"
        addr_by_src[src][0] += c.latency
        addr_by_src[src][1] += c.stall
        addr_by_src[src][2].add(c.isa.strip()[:60])
        f = src.rsplit(":", 1)[0] if ":" in src else src
        addr_by_file[f][0] += c.latency
        addr_by_file[f][1] += c.stall

tot_busy = sum(v[0] for v in by_phase.values()) or 1
print("=== phase totals (aggregate latency over whole trace) ===")
for ph, (b, s) in sorted(by_phase.items(), key=lambda kv: -kv[1][0]):
    print(f"  {ph:9s} busy={b:>12,}  ({100*b/tot_busy:5.1f}%)  stall={s:>12,}")

addr_busy = by_phase[phases.ADDR][0] or 1
print(f"\n=== ADDR phase by source FILE  (addr busy={addr_busy:,}) ===")
for f, (b, s) in sorted(addr_by_file.items(), key=lambda kv: -kv[1][0]):
    base = f.rsplit("/", 1)[-1]
    print(f"  {100*b/addr_busy:5.1f}%  busy={b:>11,}  stall={s:>11,}  {base}")

print("\n=== ADDR phase top source LINES ===")
rows = sorted(addr_by_src.items(), key=lambda kv: -kv[1][0])[:25]
for src, (b, s, isas) in rows:
    base = src.rsplit("/", 1)[-1]
    ex = next(iter(isas)) if isas else ""
    print(f"  {100*b/addr_busy:5.1f}%  busy={b:>10,}  stall={s:>10,}  {base:42s} {ex}")
