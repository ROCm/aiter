#!/usr/bin/env python3
"""Extract per-barrier stall latencies for the UA prefill pingpong from a
rocprofv3 ATT `stats_*.csv` and tabulate them by warp group / barrier / pi.

Why this exists
---------------
The 8-warp prefill_d128 pipeline is a two-warp-group producer/consumer
pingpong (W0-3 = core_loop(0), W4-7 = core_loop(1)). Each `iteration(pi)` has
three cross-group `s_barrier`s (B1/B2/B3), and the loop body is 2x-unrolled
(pi=0, pi=1). So each warp group emits 6 loop `s_barrier` opcodes, and the two
groups rendezvous pairwise by (barrier, pi).

The RCV GUI's "Latency Mean Wave" is a *single selected wave* and is noisy.
The `stats_*.csv` `Latency`/`Stall` columns are the SUM over all traced waves,
keyed by Vaddr (which already disambiguates the W0-3 vs W4-7 code paths since
core_loop(0) and core_loop(1) are separate template instantiations). Dividing
by Hitcount gives a robust mean stall per barrier execution.

At a barrier the side with the HIGHER mean stall arrived early and waited; the
LOWER side is the last arriver = the bottleneck phase at that rendezvous.

Mapping (verified against the kernel source + ISA context):
  * Two Vaddr clusters of 6 high-hitcount barriers each. The lower-address
    cluster is W0-3 (core_loop(0) is emitted first), the higher is W4-7.
  * Within a cluster, program/Vaddr order is:
        B1 pi0, B2 pi0, B3 pi0, B1 pi1, B2 pi1, B3 pi1
  * Phase pairing at each rendezvous (same for both pi):
        B1:  W0-3 compute (gemm0+alu1)   vs  W4-7 V-mem
        B2:  W0-3 K-mem+mask             vs  W4-7 compute
        B3:  W0-3 gemm1+D_upd            vs  W4-7 K-mem+mask

Usage
-----
  # Single run -> print the barrier table
  rocprof_barrier_latency.py <run_dir_or_stats_csv> [--label NAME]

  # Compare two runs (e.g. before/after a kernel change)
  rocprof_barrier_latency.py --compare <baseline> <variant>

A <run_dir> may be the TAG dir, its `att/` subdir, or the stats CSV itself.
"""

import argparse
import csv
import glob
import os
import sys

# Phase labels per (warp group, barrier). Index 0 = W0-3, 1 = W4-7.
PHASE = {
    ("W0-3", "B1"): "gemm0+alu1 (compute)",
    ("W0-3", "B2"): "K-mem+mask",
    ("W0-3", "B3"): "gemm1+D_upd",
    ("W4-7", "B1"): "V-mem",
    ("W4-7", "B2"): "gemm0+alu1 (compute)",
    ("W4-7", "B3"): "K-mem+mask",
}


def find_stats_csv(path):
    """Resolve a run dir / att dir / csv path to the stats CSV file."""
    if os.path.isfile(path):
        return path
    candidates = []
    for pat in ("stats_*.csv", "att/stats_*.csv", "*/stats_*.csv",
                "att/ui_output_*/stats_*.csv"):
        candidates += glob.glob(os.path.join(path, pat))
    # The per-dispatch stats CSV lives next to the att blobs; prefer the
    # largest (the real one, not an empty placeholder).
    candidates = [c for c in candidates if os.path.getsize(c) > 0]
    if not candidates:
        sys.exit(f"no stats_*.csv found under {path!r}")
    return max(candidates, key=os.path.getsize)


def load_barriers(csv_path):
    """Return list of (vaddr, hitcount, stall) for every s_barrier row."""
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["Instruction"].strip() == "s_barrier":
                hc = int(r["Hitcount"])
                rows.append((int(r["Vaddr"]), hc, int(r["Stall"])))
    return rows


def select_loop_barriers(barriers):
    """Keep the high-hitcount loop barriers (drop pre-stage/post one-shots).

    Loop barriers run once per (CTA, K-iteration); pre-stage barriers run once
    per CTA. The former dominate hitcount by >10x, so a half-of-max threshold
    cleanly separates them.
    """
    if not barriers:
        sys.exit("no s_barrier rows in stats CSV")
    max_hc = max(hc for _, hc, _ in barriers)
    loop = [b for b in barriers if b[1] >= max_hc * 0.5]
    loop.sort(key=lambda b: b[0])  # by Vaddr = program order
    if len(loop) != 12:
        print(f"WARN: expected 12 loop barriers, found {len(loop)} "
              f"(hitcounts: {[b[1] for b in loop]})", file=sys.stderr)
    return loop


def build_table(loop):
    """Map the 12 sorted loop barriers to (group, barrier, pi) -> mean stall."""
    half = len(loop) // 2
    groups = {"W0-3": loop[:half], "W4-7": loop[half:]}
    table = {}  # (group, barrier, pi) -> dict(vaddr, hc, stall, mean)
    bnames = ["B1", "B2", "B3"]
    for gname, items in groups.items():
        for idx, (vaddr, hc, stall) in enumerate(items):
            pi = idx // 3       # first 3 = pi0, next 3 = pi1
            bname = bnames[idx % 3]
            mean = stall / hc if hc else 0.0
            table[(gname, bname, pi)] = dict(
                vaddr=vaddr, hc=hc, stall=stall, mean=mean)
    return table


def print_table(table, label):
    print(f"\n=== UA prefill pingpong barrier stalls{f'  [{label}]' if label else ''} ===")
    print("mean = total stall cycles / hitcount (all traced waves). Higher mean "
          "= arrived early = faster phase;\nthe LOWER-mean side is the "
          "bottleneck at that rendezvous.\n")
    hdr = (f"{'rendezvous':<11} {'W0-3 phase':<22}{'W0-3 mean':>10}   "
           f"{'W4-7 phase':<22}{'W4-7 mean':>10}   {'imbalance':>9}  bottleneck")
    print(hdr)
    print("-" * len(hdr))
    for bname in ("B1", "B2", "B3"):
        for pi in (0, 1):
            a = table.get(("W0-3", bname, pi))
            b = table.get(("W4-7", bname, pi))
            if not a or not b:
                continue
            am, bm = a["mean"], b["mean"]
            imb = abs(am - bm)
            # lower mean = last arriver = bottleneck
            if am < bm:
                slow = f"W0-3 {PHASE[('W0-3', bname)]}"
            else:
                slow = f"W4-7 {PHASE[('W4-7', bname)]}"
            print(f"{bname} pi{pi:<7} {PHASE[('W0-3', bname)]:<22}{am:>10.1f}   "
                  f"{PHASE[('W4-7', bname)]:<22}{bm:>10.1f}   {imb:>9.1f}  {slow}")
    print()


def table_means(table):
    """Flatten to {(group,barrier,pi): mean} for diffing."""
    return {k: v["mean"] for k, v in table.items()}


def cmd_single(path, label):
    csv_path = find_stats_csv(path)
    print(f"stats: {csv_path}")
    table = build_table(select_loop_barriers(load_barriers(csv_path)))
    print_table(table, label or os.path.basename(os.path.normpath(path)))


def cmd_compare(base, var):
    bt = build_table(select_loop_barriers(load_barriers(find_stats_csv(base))))
    vt = build_table(select_loop_barriers(load_barriers(find_stats_csv(var))))
    bm, vm = table_means(bt), table_means(vt)
    print(f"\n=== before/after barrier-stall comparison ===")
    print(f"  baseline: {base}")
    print(f"  variant : {var}\n")
    hdr = (f"{'rendezvous':<11} {'phase pair':<40}{'base':>9}{'var':>9}"
           f"{'delta':>9}{'%':>8}")
    print(hdr)
    print("-" * len(hdr))
    for bname in ("B1", "B2", "B3"):
        for pi in (0, 1):
            for g in ("W0-3", "W4-7"):
                k = (g, bname, pi)
                if k not in bm or k not in vm:
                    continue
                b, v = bm[k], vm[k]
                d = v - b
                pct = (d / b * 100) if b else 0.0
                pair = f"{g} {PHASE[(g, bname)]}"
                print(f"{bname} pi{pi:<7} {pair:<40}{b:>9.1f}{v:>9.1f}"
                      f"{d:>+9.1f}{pct:>+7.1f}%")
    # totals
    tb = sum(bm.values())
    tv = sum(vm.values())
    print("-" * len(hdr))
    print(f"{'TOTAL':<11} {'sum of mean barrier stalls':<40}{tb:>9.1f}{tv:>9.1f}"
          f"{tv - tb:>+9.1f}{(tv - tb) / tb * 100 if tb else 0:>+7.1f}%\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+",
                    help="run dir / att dir / stats CSV (1 arg), or "
                         "two args with --compare")
    ap.add_argument("--label", default=None, help="label for single-run output")
    ap.add_argument("--compare", action="store_true",
                    help="compare two runs: <baseline> <variant>")
    args = ap.parse_args()

    if args.compare:
        if len(args.paths) != 2:
            sys.exit("--compare needs exactly two paths: <baseline> <variant>")
        cmd_compare(args.paths[0], args.paths[1])
    else:
        if len(args.paths) != 1:
            sys.exit("single-run mode takes exactly one path "
                     "(use --compare for two)")
        cmd_single(args.paths[0], args.label)


if __name__ == "__main__":
    main()
