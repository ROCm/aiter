"""Attribute stall (esp. MEMWAIT / s_waitcnt) cycles to ISA and back to source.

code.json carries, per ISA program counter, aggregate {hit, latency, stall}
summed over the whole dispatch (all waves, all iterations). That is exactly the
right granularity to ask "where do the mem-waits come from":

  * MEMWAIT stall = cycles a wave sits blocked at an ``s_waitcnt``. The waitcnt's
    *counter operand* says WHAT it waits on:
        vmcnt   -> outstanding VMEM (global/DRAM buffer_load/store) responses
        lgkmcnt -> outstanding LDS (ds_*) + scalar/constant loads
        expcnt  -> outstanding exports / GDS
  * LOAD latency = the memory access latency of buffer_load/global_load PCs.

We rank both, and resolve each PC to its ``file:line`` so the waits map to
specific kernel code (pipeline lambdas, gemm, softmax, ...).

Usage:  python -m att_analysis.memwait <run_or_ui_dir> [--top N]
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict

from .model import Trace, find_ui_dir
from .phases import PhaseTagger, PHASE_ORDER

# s_waitcnt operand decode, e.g. "s_waitcnt vmcnt(0) lgkmcnt(1)" -> {'vmcnt':0,...}
_CNT_RE = re.compile(r"(vmcnt|lgkmcnt|expcnt|loadcnt|storecnt|dscnt|kmcnt)\((\d+)\)")


def waitcnt_kinds(isa: str) -> dict[str, int]:
    return {k: int(v) for k, v in _CNT_RE.findall(isa)}


def _waitcnt_label(isa: str) -> str:
    """Coarse cause label from the counter operand (which counters it gates)."""
    k = waitcnt_kinds(isa)
    if not k:
        return "?"
    parts = []
    if any(c in k for c in ("vmcnt", "loadcnt", "storecnt")):
        parts.append("VMEM(DRAM)")
    if any(c in k for c in ("lgkmcnt", "dscnt", "kmcnt")):
        parts.append("LDS/scalar")
    if "expcnt" in k:
        parts.append("export")
    return "+".join(parts) or "?"


def _short_src(src: str) -> str:
    if not src:
        return "(no source)"
    f, _, ln = src.rpartition(":")
    return f"{f.rsplit('/', 1)[-1]}:{ln}"


def analyze(run_dir: str, top: int = 20) -> None:
    trace = Trace(find_ui_dir(run_dir))
    tagger = PhaseTagger(trace)
    code = [c for c in trace.code if not c.is_comment]

    # ---- 1. stall budget by phase ---------------------------------------
    by_phase_stall = defaultdict(int)
    by_phase_lat = defaultdict(int)
    total_stall = total_lat = 0
    for c in code:
        ph = tagger.phase_of_lineno(c.lineno)
        by_phase_stall[ph] += c.stall
        by_phase_lat[ph] += c.latency
        total_stall += c.stall
        total_lat += c.latency

    print(f"# memwait attribution  ({_short_src(trace.dir)})")
    print(f"phase mode: {tagger.mode}   total stall={total_stall:,} cyc   "
          f"total latency={total_lat:,} cyc\n")
    print("## stall budget by phase (aggregate over all waves/iters)")
    print(f"{'phase':<9} {'stall cyc':>12} {'stall%':>7}   {'latency cyc':>12}")
    for ph in PHASE_ORDER:
        s = by_phase_stall.get(ph, 0)
        if not s and not by_phase_lat.get(ph, 0):
            continue
        pct = 100.0 * s / total_stall if total_stall else 0
        print(f"{ph:<9} {s:>12,} {pct:>6.1f}%   {by_phase_lat.get(ph,0):>12,}")

    # ---- 2. hottest s_waitcnt PCs (the actual blocking points) ----------
    waits = [c for c in code if c.mnemonic.startswith("s_waitcnt")]
    waits.sort(key=lambda c: c.stall, reverse=True)
    print(f"\n## top {top} s_waitcnt by stall  (where the wave blocks + what it waits on)")
    print(f"{'stall cyc':>10} {'%mw':>5} {'hit':>6} {'avg':>6}  {'waits-on':<12} {'source':<34} isa")
    mw_total = by_phase_stall.get("memwait", 0)
    for c in waits[:top]:
        pct = 100.0 * c.stall / mw_total if mw_total else 0
        avg = c.stall / c.hit if c.hit else 0
        print(f"{c.stall:>10,} {pct:>4.0f}% {c.hit:>6} {avg:>6.0f}  "
              f"{_waitcnt_label(c.isa):<12} {_short_src(c.source):<34} {c.isa.strip()}")

    # ---- 3. hottest memory ops by latency (what the waitcnt waits for) --
    loads = [c for c in code
             if re.match(r"^(buffer|global|flat)_(load|store|atomic)|^ds_(read|write|load|store)",
                         c.mnemonic)]
    loads.sort(key=lambda c: c.latency, reverse=True)
    print(f"\n## top {top} memory ops by latency  (the loads/stores the waits gate)")
    print(f"{'latency cyc':>12} {'stall cyc':>10} {'hit':>6} {'avg lat':>8}  {'source':<34} isa")
    for c in loads[:top]:
        avg = c.latency / c.hit if c.hit else 0
        print(f"{c.latency:>12,} {c.stall:>10,} {c.hit:>6} {avg:>8.0f}  "
              f"{_short_src(c.source):<34} {c.isa.strip()}")

    # ---- 4. memwait stall grouped by source line ------------------------
    by_src = defaultdict(lambda: [0, 0, 0])  # src -> [stall, hit, n_pc]
    for c in waits:
        e = by_src[_short_src(c.source)]
        e[0] += c.stall
        e[1] += c.hit
        e[2] += 1
    rows = sorted(by_src.items(), key=lambda kv: kv[1][0], reverse=True)
    print(f"\n## memwait stall grouped by source line")
    print(f"{'stall cyc':>10} {'%mw':>5} {'hit':>7} {'#pc':>4}  source")
    for src, (s, h, n) in rows[:top]:
        pct = 100.0 * s / mw_total if mw_total else 0
        print(f"{s:>10,} {pct:>4.0f}% {h:>7} {n:>4}  {src}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()
    analyze(args.run_dir, args.top)


if __name__ == "__main__":
    main()
