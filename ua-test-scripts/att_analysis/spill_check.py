"""Spill -> prefetch-stall detector for FA4 ATT traces.

Background (kv128 finding, 2026-06-12)
--------------------------------------
On gfx950 `vmcnt` is ONE counter shared by every VMEM op -- async global
prefetch loads (`buffer_load`/`global_load_lds`) AND register spill reloads
(`scratch_load`). When the kernel is over the 256-VGPR budget the allocator
spills loop-invariant address bases (k/v_thread_n_pos) into scratch and reloads
them inside `refresh_{k,v}_offsets`. The reload's consumer forces `s_waitcnt
vmcnt(0)`, which -- because the counter is shared -- also drains the in-flight
prefetch, turning a ~L2 spill reload into a full ~1200-cyc DRAM stall.

This detector flags exactly that pattern so we never have to rediscover it:
a big `s_waitcnt vmcnt(0)` whose look-back contains a `scratch_load` while an
async prefetch load is still outstanding => the spill is draining the prefetch.

Usage
-----
    python3 -m att_analysis.spill_check <run_or_ui_dir> [--stall 300]

Exit status is non-zero if any poisoned drains are found, so it can gate CI /
a rebuild-and-check loop.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from .model import Trace, find_ui_dir, Wave

# An s_waitcnt is "big" (worth attributing to memory latency) above this stall.
DEFAULT_STALL = 300
# How many instructions back to look for the scratch_load the wait covers.
LOOKBACK_INSTS = 16
# How many cycles back an async prefetch load counts as "still in flight".
INFLIGHT_CYC = 1600


def _is_vmcnt_wait(isa: str) -> bool:
    return "s_waitcnt" in isa and "vmcnt" in isa


def _is_scratch_load(isa: str) -> bool:
    return isa.lstrip().startswith("scratch_load")


def _is_async_prefetch(isa: str) -> bool:
    s = isa.lstrip()
    return s.startswith("buffer_load") or s.startswith("global_load")


@dataclass
class Poison:
    cycle: int
    stall: int
    source: str
    isa: str


def scan_wave(tr: Trace, w: Wave, stall_thresh: int) -> list[Poison]:
    insts = w.instructions
    out: list[Poison] = []
    for idx, i in enumerate(insts):
        cl = tr.code_line(i.lineno)
        if cl is None or not _is_vmcnt_wait(cl.isa) or i.stall < stall_thresh:
            continue
        # (1) does the wait cover a spill reload?
        saw_scratch = False
        for j in range(max(0, idx - LOOKBACK_INSTS), idx):
            jc = tr.code_line(insts[j].lineno)
            if jc and _is_scratch_load(jc.isa):
                saw_scratch = True
                break
        if not saw_scratch:
            continue
        # (2) was an async prefetch load issued recently (still in flight)?
        inflight = False
        k = idx - 1
        while k >= 0 and i.cycle - insts[k].cycle <= INFLIGHT_CYC:
            kc = tr.code_line(insts[k].lineno)
            if kc and _is_async_prefetch(kc.isa):
                inflight = True
                break
            k -= 1
        if not inflight:
            continue
        src = cl.source
        if src:
            f, _, ln = src.rpartition(":")
            src = f"{f.rsplit('/', 1)[-1]}:{ln}"
        out.append(Poison(i.cycle, i.stall, src or "(nosrc)", cl.isa.strip()))
    return out


def run(run_dir: str, stall_thresh: int = DEFAULT_STALL) -> int:
    tr = Trace(find_ui_dir(run_dir))
    if not tr.has_source:
        print("WARNING: trace has no source mapping; build with LINETABLES=1 for "
              "precise attribution.", file=sys.stderr)
    try:
        waves = list(tr.coresident_pair(0, 0))
        tags = ["WG0", "WG1"]
    except Exception:
        waves = tr.waves()[:2]
        tags = [str(w) for w in waves]

    total_cyc = 0
    total_n = 0
    print(f"# spill->prefetch-stall check: {run_dir}")
    print(f"#   (s_waitcnt vmcnt with stall>={stall_thresh} covering a scratch_load "
          f"while a prefetch is in flight)")
    for tag, w in zip(tags, waves):
        ps = scan_wave(tr, w, stall_thresh)
        cyc = sum(p.stall for p in ps)
        total_cyc += cyc
        total_n += len(ps)
        if ps:
            worst = max(ps, key=lambda p: p.stall)
            per_src: dict[str, list[int]] = {}
            for p in ps:
                per_src.setdefault(p.source, []).append(p.stall)
            print(f"\n{tag}: {len(ps)} poisoned drains, {cyc} stall cyc "
                  f"(worst {worst.stall} @ {worst.source})")
            for src, sts in sorted(per_src.items(), key=lambda kv: -sum(kv[1])):
                print(f"    {sum(sts):>8} cyc  x{len(sts):<4} {src}")
        else:
            print(f"\n{tag}: clean (no spill-poisoned prefetch drains)")
    print(f"\nTOTAL: {total_n} poisoned drains, {total_cyc} stall cyc")
    return 1 if total_n else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run dir, att/ subdir, or ui_output_* dir")
    ap.add_argument("--stall", type=int, default=DEFAULT_STALL,
                    help=f"min s_waitcnt stall to flag (default {DEFAULT_STALL})")
    args = ap.parse_args()
    sys.exit(run(args.run, args.stall))


if __name__ == "__main__":
    main()
