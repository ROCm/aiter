#!/usr/bin/env python3
"""Per-warp-group analysis of rocprofv3 stochastic PC samples.

The 8-warp prefill pipeline is warp-specialized:
  - wave_in_grp 0..3 (warp_group 0) runs `core_loop(0)` — phase order:
      gemm0+alu1 / K_load+V_lds_load / gemm1 / V_load+K_lds_load
  - wave_in_grp 4..7 (warp_group 1) runs `core_loop(1)` — phase order:
      V_load+K_lds_load / gemm0+alu1 / K_load+V_lds_load / gemm1

Every `s_barrier` is a *cross-group rendezvous*: when one side issues
s_barrier and stalls, the other side is still busy with the work
*between the previous barrier and this one*. So the per-barrier-PC stall
count, split by warp_group, tells us which side is faster/slower at
each phase boundary.

This script reads `phase_c_pcsamp/pcsamp_results.json` and emits:

  1. Aggregate stall composition per warp_group (sanity: should be similar
     in *total* but skewed in *where* the stalls live).
  2. Top stalled instructions per warp_group, side by side.
  3. Per-PC ranking of `s_barrier` instances, with W0-3 vs W4-7 sample
     counts and an "imbalance ratio" (larger group / smaller group).
     This is the headline output — points the next optimization at the
     specific phase whose work needs balancing.

Usage:
  python3 rocprof_warpgroup_balance.py \
      runs/prefill_d128_fp8_b16_sq10000_sk10000 \
      [--out report.md]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict


def load_samples(run_dir: str):
    path = os.path.join(run_dir, "phase_c_pcsamp", "pcsamp_results.json")
    if not os.path.exists(path):
        sys.exit(f"missing: {path}")
    with open(path) as f:
        obj = json.load(f)
    top = obj["rocprofiler-sdk-tool"][0]
    samples = top["buffer_records"]["pc_sample_stochastic"]
    instrs  = top["strings"]["pc_sample_instructions"]
    return samples, instrs


def aggregate(samples, instrs):
    by_wg = [defaultdict(int), defaultdict(int)]                 # total samples per (wg, inst_text)
    issued_by_wg = [defaultdict(int), defaultdict(int)]
    stalled_by_wg = [defaultdict(int), defaultdict(int)]
    stall_reason_by_wg = [defaultdict(int), defaultdict(int)]
    inst_type_by_wg = [defaultdict(int), defaultdict(int)]
    pc_by_wg_for_barriers = [defaultdict(int), defaultdict(int)]  # for each s_barrier PC, samples per wg
    pc_by_wg_total = [defaultdict(int), defaultdict(int)]         # for ANY PC, samples per wg (small caveat: this can be huge but we only print top)
    wg_totals = [0, 0]
    wg_stalled = [0, 0]
    wg_issued = [0, 0]

    for s in samples:
        rec = s["record"]
        wig = rec["hw_id"]["wave_id"]  # NOT what we want — see below
        # Within-CTA wave index is `wave_in_grp`; the JSON spelling is `wave_in_grp` literal field.
        wig = rec.get("wave_in_grp", rec["hw_id"]["wave_id"])
        wg = 0 if wig < 4 else 1
        wg_totals[wg] += 1

        inst_idx  = s["inst_index"]
        inst_text = instrs[inst_idx] if inst_idx < len(instrs) else "(?)"
        issued    = bool(rec["wave_issued"])
        itype     = rec["inst_type"].replace("ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_", "")
        stall     = rec["snapshot"]["stall_reason"].replace(
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_", "")
        pc_off    = rec["pc"]["code_object_offset"]

        by_wg[wg][inst_text] += 1
        if issued:
            wg_issued[wg] += 1
            issued_by_wg[wg][inst_text] += 1
        else:
            wg_stalled[wg] += 1
            stalled_by_wg[wg][inst_text] += 1
            stall_reason_by_wg[wg][stall] += 1
        inst_type_by_wg[wg][itype] += 1

        if inst_text.strip() == "s_barrier":
            pc_by_wg_for_barriers[wg][pc_off] += 1
        pc_by_wg_total[wg][pc_off] += 1

    return dict(
        wg_totals=wg_totals,
        wg_issued=wg_issued,
        wg_stalled=wg_stalled,
        stalled_by_wg=stalled_by_wg,
        stall_reason_by_wg=stall_reason_by_wg,
        inst_type_by_wg=inst_type_by_wg,
        pc_by_wg_for_barriers=pc_by_wg_for_barriers,
        pc_by_wg_total=pc_by_wg_total,
    )


def fmt(n: int, denom: int) -> str:
    if denom == 0:
        return f"{n:>7d}   --  "
    return f"{n:>7d} ({100.0*n/denom:5.1f}%)"


def render(a: dict, top: int = 15) -> str:
    L = []
    wg0_n, wg1_n = a["wg_totals"]
    L += [
        "## per-warp-group sample totals",
        "",
        "| metric | warp_group 0 (W0-3, `core_loop(0)`) | warp_group 1 (W4-7, `core_loop(1)`) | wg0/wg1 |",
        "|---|---|---|---|",
        f"| total samples | {fmt(wg0_n, wg0_n+wg1_n)} | {fmt(wg1_n, wg0_n+wg1_n)} | {wg0_n/max(1,wg1_n):.3f} |",
        f"| issued | {fmt(a['wg_issued'][0], wg0_n)} | {fmt(a['wg_issued'][1], wg1_n)} | "
        f"{a['wg_issued'][0]/max(1,a['wg_issued'][1]):.3f} |",
        f"| stalled | {fmt(a['wg_stalled'][0], wg0_n)} | {fmt(a['wg_stalled'][1], wg1_n)} | "
        f"{a['wg_stalled'][0]/max(1,a['wg_stalled'][1]):.3f} |",
        "",
        "*If `wg0/wg1 ≈ 1.0` the two warp groups consume equal compute time "
        "(expected: they run concurrently). What matters is how the stalls "
        "**distribute** across phases — next sections.*",
        "",
    ]

    L += ["## stall composition per warp group", "",
          "| Stall_Reason | W0-3 | W4-7 | wg0/wg1 |",
          "|---|---:|---:|---:|"]
    all_reasons = set(a["stall_reason_by_wg"][0]) | set(a["stall_reason_by_wg"][1])
    rows = []
    for r in all_reasons:
        n0 = a["stall_reason_by_wg"][0][r]
        n1 = a["stall_reason_by_wg"][1][r]
        rows.append((r, n0, n1))
    rows.sort(key=lambda x: -(x[1] + x[2]))
    for r, n0, n1 in rows:
        ratio = f"{n0/max(1,n1):.2f}" if n1 > 0 else "∞"
        L.append(f"| {r} | {n0} ({100*n0/max(1,a['wg_stalled'][0]):.1f}%) | "
                 f"{n1} ({100*n1/max(1,a['wg_stalled'][1]):.1f}%) | {ratio} |")
    L.append("")

    L += ["## inst-type composition per warp group", "",
          "| Instruction_Type | W0-3 | W4-7 |",
          "|---|---:|---:|"]
    all_types = set(a["inst_type_by_wg"][0]) | set(a["inst_type_by_wg"][1])
    rows = [(t, a["inst_type_by_wg"][0][t], a["inst_type_by_wg"][1][t]) for t in all_types]
    rows.sort(key=lambda x: -(x[1] + x[2]))
    for t, n0, n1 in rows:
        L.append(f"| {t} | {n0} ({100*n0/max(1,a['wg_totals'][0]):.1f}%) | "
                 f"{n1} ({100*n1/max(1,a['wg_totals'][1]):.1f}%) |")
    L.append("")

    # -- The headline: per-`s_barrier`-PC sample counts split by warp group.
    L += [
        "## `s_barrier` rendezvous — per-PC sample counts by warp group",
        "",
        "Each row is one distinct `s_barrier` instruction in the compiled UA "
        "kernel. The warp group with the **higher** stall count at a given "
        "barrier arrived **first** and waited longer; the lower-count group "
        "was running the slower phase *between the previous barrier and "
        "this one*.",
        "",
        "| PC (offset) | W0-3 stalls | W4-7 stalls | imbalance (larger/smaller) | faster wg | slower wg |",
        "|---|---:|---:|---:|---|---|",
    ]
    barrier_pcs = set(a["pc_by_wg_for_barriers"][0]) | set(a["pc_by_wg_for_barriers"][1])
    rows = []
    for pc in barrier_pcs:
        n0 = a["pc_by_wg_for_barriers"][0][pc]
        n1 = a["pc_by_wg_for_barriers"][1][pc]
        rows.append((pc, n0, n1))
    rows.sort(key=lambda x: -(x[1] + x[2]))
    for pc, n0, n1 in rows:
        if n0 == 0 and n1 == 0:
            continue
        bigger = max(n0, n1)
        smaller = max(1, min(n0, n1))
        imbalance = bigger / smaller
        faster = ("W4-7" if n0 > n1 else "W0-3") if n0 != n1 else "tie"
        slower = ("W0-3" if n0 > n1 else "W4-7") if n0 != n1 else "tie"
        L.append(f"| `0x{pc:x}` | {n0} | {n1} | {imbalance:.2f}× | {faster} | {slower} |")
    L.append("")

    L += ["## top 15 stalled instructions per warp group (side-by-side)", "",
          "| W0-3 (samples / instruction) | W4-7 (samples / instruction) |",
          "|---|---|"]
    sw0 = sorted(a["stalled_by_wg"][0].items(), key=lambda kv: -kv[1])[:top]
    sw1 = sorted(a["stalled_by_wg"][1].items(), key=lambda kv: -kv[1])[:top]
    for i in range(top):
        c0 = f"{sw0[i][1]:>6}  `{sw0[i][0][:60]}`" if i < len(sw0) else ""
        c1 = f"{sw1[i][1]:>6}  `{sw1[i][0][:60]}`" if i < len(sw1) else ""
        L.append(f"| {c0} | {c1} |")
    L.append("")

    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    samples, instrs = load_samples(args.run_dir)
    a = aggregate(samples, instrs)
    title = f"# warp-group imbalance — `{args.run_dir}`\n"
    title += f"\n_{len(samples):,} total stochastic PC samples_\n\n"
    out = title + render(a, top=args.top)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out)
        print(f"wrote {args.out}")
    else:
        print(out)


if __name__ == "__main__":
    main()
