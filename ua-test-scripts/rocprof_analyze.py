#!/usr/bin/env python3
"""Aggregate rocprofv3 phase outputs for one or two UA kernel runs and
produce a side-by-side markdown report.

The four-phase layout matches `rocprof_prefill_d128.sh`:
  - phase_a_trace/trace_kernel_stats.csv  — per-kernel wall time
  - phase_b1_compute/pmc_counter_collection.csv  — instruction-mix counters
  - phase_b2_stalls/pmc_counter_collection.csv   — wait + memory busy counters
  - phase_c_pcsamp/pcsamp_pc_sampling_stochastic.csv (optional)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Optional


UA_PREFIX = "_ZN7ck_tile6kentry"      # mangled
UA_DEMANGLED = "ck_tile::kentry"      # rocprof sometimes emits demangled names


def _is_ua_kernel(name: str) -> bool:
    return UA_PREFIX in name or UA_DEMANGLED in name
WAVES_PER_CU = 8         # gfx950: max 8 waves per SIMD-quad (one CU)
NUM_CUS_MI355 = 256      # MI355X has 256 CUs (32 XCDs × 8 CUs)
GFX950_PEAK_CLOCK_GHZ = 2.1   # nominal boost clock; used for pct-of-peak rough math


def aggregate_kernel_pmc(path: str) -> Optional[dict]:
    """Sum counter values across all UA kernel dispatches in a PMC CSV.

    Returns dict counter_name -> total, plus `_num_dispatches` and per-counter
    `<name>_per_dispatch` averages.
    """
    if not os.path.exists(path):
        return None
    by_counter = defaultdict(float)
    dispatches = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            if not _is_ua_kernel(row["Kernel_Name"]):
                continue
            try:
                val = float(row["Counter_Value"])
            except ValueError:
                continue
            by_counter[row["Counter_Name"]] += val
            dispatches.add(row["Dispatch_Id"])
    n = len(dispatches)
    if n == 0:
        return None
    out = dict(by_counter)
    out["_num_dispatches"] = n
    for k, v in list(by_counter.items()):
        out[k + "_per_dispatch"] = v / n
    return out


def aggregate_kernel_trace(path: str) -> Optional[dict]:
    """Find the UA kernel row in trace_kernel_stats and return its timings."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for row in csv.DictReader(f):
            if _is_ua_kernel(row["Name"]):
                return dict(
                    calls=int(row["Calls"]),
                    avg_ms=float(row["AverageNs"]) / 1e6,
                    total_ms=float(row["TotalDurationNs"]) / 1e6,
                    min_ms=float(row["MinNs"]) / 1e6,
                    max_ms=float(row["MaxNs"]) / 1e6,
                )
    return None


def aggregate_pcsamp(path: str, top: int = 20) -> Optional[dict]:
    """Aggregate stochastic PC samples by issued/not-issued, by Instruction_Type,
    and by Stall_Reason. The stochastic-sample CSV does NOT include source
    file/line so we work at the instruction level.

    Returns a dict with:
      total              — total samples
      issued             — samples where Wave_Issued_Instruction=1
      not_issued         — samples where Wave_Issued_Instruction=0 (stalled)
      by_itype           — list of (Instruction_Type, total, issued, not_issued)
      by_stall_reason    — list of (Stall_Reason, count) over not-issued
      hot_instructions   — list of (instr_disasm, total, issued, not_issued)
    """
    if not os.path.exists(path):
        return None
    by_itype = defaultdict(lambda: [0, 0, 0])      # total, issued, stalled
    by_stall_reason = defaultdict(int)
    by_instr = defaultdict(lambda: [0, 0, 0])
    total = issued = not_issued = 0

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            itype = row.get("Instruction_Type", "")
            issu = (row.get("Wave_Issued_Instruction", "0").strip() in ("1", "True", "true"))
            instr = row.get("Instruction", "").replace("\n", " ").strip()[:80]
            by_itype[itype][0] += 1
            by_instr[instr][0] += 1
            if issu:
                issued += 1
                by_itype[itype][1] += 1
                by_instr[instr][1] += 1
            else:
                not_issued += 1
                by_itype[itype][2] += 1
                by_instr[instr][2] += 1
                stall = row.get("Stall_Reason", "")
                by_stall_reason[stall] += 1

    if total == 0:
        return None

    def srt(d, key=lambda kv: kv[1][0] if isinstance(kv[1], list) else kv[1], top_n=top):
        return sorted(d.items(), key=key, reverse=True)[:top_n]

    return dict(
        total=total, issued=issued, not_issued=not_issued,
        by_itype=srt(by_itype),
        by_stall_reason=srt(by_stall_reason, key=lambda kv: kv[1], top_n=top),
        hot_instructions=srt(by_instr, top_n=top),
    )


def fmt_pct(num, denom):
    if not denom or denom == 0:
        return "—"
    return f"{100.0*num/denom:6.2f}%"


def build_report(label: str, run_dir: str) -> dict:
    timing = aggregate_kernel_trace(os.path.join(run_dir, "phase_a_trace", "trace_kernel_stats.csv"))
    compute = aggregate_kernel_pmc(os.path.join(run_dir, "phase_b1_compute", "pmc_counter_collection.csv"))
    stalls = aggregate_kernel_pmc(os.path.join(run_dir, "phase_b2_stalls", "pmc_counter_collection.csv"))
    pcsamp = aggregate_pcsamp(os.path.join(run_dir, "phase_c_pcsamp", "pcsamp_pc_sampling_stochastic.csv"))
    return dict(label=label, run_dir=run_dir, timing=timing,
                compute=compute, stalls=stalls, pcsamp=pcsamp)


def emit_compute_table(reports: list[dict]) -> str:
    rows = [
        ("GUI_ACTIVE/dispatch (cycles)", "GRBM_GUI_ACTIVE_per_dispatch"),
        ("SQ_WAVES/dispatch",           "SQ_WAVES_per_dispatch"),
        ("MFMA/dispatch",               "SQ_INSTS_MFMA_per_dispatch"),
        ("VALU/dispatch",               "SQ_INSTS_VALU_per_dispatch"),
        ("VALU_CVT/dispatch",           "SQ_INSTS_VALU_CVT_per_dispatch"),
        ("SALU/dispatch",               "SQ_INSTS_SALU_per_dispatch"),
        ("VMEM/dispatch",               "SQ_INSTS_VMEM_per_dispatch"),
        ("LDS/dispatch",                "SQ_INSTS_LDS_per_dispatch"),
    ]
    out = ["| metric | " + " | ".join(r["label"] for r in reports) + " | ratio |", 
           "|---|" + "---|" * len(reports) + "---|"]
    for human, key in rows:
        cells = []
        vals = []
        for r in reports:
            v = r["compute"].get(key) if r["compute"] else None
            vals.append(v)
            cells.append(f"{v:,.0f}" if v is not None else "—")
        if len(vals) >= 2 and vals[0] and vals[1]:
            ratio = vals[0] / vals[1]
            cells.append(f"{ratio:.2f}x")
        else:
            cells.append("—")
        out.append(f"| {human} | " + " | ".join(cells) + " |")
    # Derived per-dispatch ratios (cycles-per-instr style)
    derived = [
        ("cycles / MFMA", "GRBM_GUI_ACTIVE_per_dispatch", "SQ_INSTS_MFMA_per_dispatch"),
        ("cycles / VALU", "GRBM_GUI_ACTIVE_per_dispatch", "SQ_INSTS_VALU_per_dispatch"),
        ("cycles / VMEM", "GRBM_GUI_ACTIVE_per_dispatch", "SQ_INSTS_VMEM_per_dispatch"),
        ("VALU_CVT / VALU", "SQ_INSTS_VALU_CVT_per_dispatch", "SQ_INSTS_VALU_per_dispatch"),
        ("VALU / MFMA",  "SQ_INSTS_VALU_per_dispatch", "SQ_INSTS_MFMA_per_dispatch"),
        ("VMEM / MFMA",  "SQ_INSTS_VMEM_per_dispatch", "SQ_INSTS_MFMA_per_dispatch"),
    ]
    out.append("| | | | |")
    for human, num_key, den_key in derived:
        cells = []
        vals = []
        for r in reports:
            num = r["compute"].get(num_key) if r["compute"] else None
            den = r["compute"].get(den_key) if r["compute"] else None
            if num is None or den is None or den == 0:
                cells.append("—")
                vals.append(None)
            else:
                ratio = num / den
                cells.append(f"{ratio:.2f}")
                vals.append(ratio)
        if len(vals) >= 2 and vals[0] and vals[1]:
            cells.append(f"{vals[0]/vals[1]:.2f}x")
        else:
            cells.append("—")
        out.append(f"| {human} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def emit_stall_table(reports: list[dict]) -> str:
    rows = [
        ("GUI_ACTIVE/dispatch",         "GRBM_GUI_ACTIVE_per_dispatch"),
        ("SQ_WAIT_ANY/dispatch",        "SQ_WAIT_ANY_per_dispatch"),
        ("SQ_WAIT_INST_ANY/dispatch",   "SQ_WAIT_INST_ANY_per_dispatch"),
        ("SQ_WAIT_INST_LDS/dispatch",   "SQ_WAIT_INST_LDS_per_dispatch"),
        ("SQC_TC_STALL/dispatch",       "SQC_TC_STALL_per_dispatch"),
        ("TA_BUSY_avr/dispatch",        "TA_BUSY_avr_per_dispatch"),
        ("TCC_BUSY_avr/dispatch",       "TCC_BUSY_avr_per_dispatch"),
        ("TCP_PENDING_STALL/dispatch",  "TCP_PENDING_STALL_CYCLES_sum_per_dispatch"),
    ]
    out = ["| metric | " + " | ".join(r["label"] for r in reports) + " | ratio |",
           "|---|" + "---|" * len(reports) + "---|"]
    for human, key in rows:
        cells = []
        vals = []
        for r in reports:
            v = r["stalls"].get(key) if r["stalls"] else None
            vals.append(v)
            cells.append(f"{v:,.0f}" if v is not None else "—")
        if len(vals) >= 2 and vals[0] and vals[1]:
            cells.append(f"{vals[0]/vals[1]:.2f}x")
        else:
            cells.append("—")
        out.append(f"| {human} | " + " | ".join(cells) + " |")
    # Derived: stall % of GUI_ACTIVE
    derived = [
        ("WAIT_ANY %",       "SQ_WAIT_ANY_per_dispatch"),
        ("WAIT_INST_ANY %",  "SQ_WAIT_INST_ANY_per_dispatch"),
        ("WAIT_INST_LDS %",  "SQ_WAIT_INST_LDS_per_dispatch"),
        ("TA_BUSY %",        "TA_BUSY_avr_per_dispatch"),
        ("TCC_BUSY %",       "TCC_BUSY_avr_per_dispatch"),
    ]
    out.append("| | | | |")
    for human, key in derived:
        cells = []
        for r in reports:
            v = r["stalls"].get(key) if r["stalls"] else None
            denom = r["stalls"].get("GRBM_GUI_ACTIVE_per_dispatch") if r["stalls"] else None
            if v is None or not denom:
                cells.append("—")
            else:
                cells.append(fmt_pct(v, denom))
        cells.append("—")
        out.append(f"| {human} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def emit_pcsamp_table(report: dict, top: int = 15) -> str:
    pc = report["pcsamp"]
    if not pc:
        return "_(no PC-sampling rows captured)_"
    total = pc["total"]
    issued = pc["issued"]
    not_issued = pc["not_issued"]
    out = [
        f"_{total} total samples — {issued} issued ({100*issued/total:.1f}%) /"
        f" {not_issued} stalled ({100*not_issued/total:.1f}%)_",
        "",
        "### by Instruction_Type",
        "",
        "| Instruction_Type | total | issued | stalled | stall% |",
        "|---|---:|---:|---:|---:|",
    ]
    for itype, (n, i, s) in pc["by_itype"]:
        if n == 0:
            continue
        short = itype.replace("ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_", "")
        out.append(f"| {short} | {n} ({100*n/total:.1f}%) | {i} | {s} | "
                   f"{100*s/n:.1f}% |")
    out += [
        "",
        "### top stall reasons (over stalled samples)",
        "",
        "| Stall_Reason | count | pct of stalled |",
        "|---|---:|---:|",
    ]
    for reason, n in pc["by_stall_reason"]:
        short = reason.replace("ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_", "")
        out.append(f"| {short} | {n} | {100*n/max(1,not_issued):.1f}% |")
    out += [
        "",
        "### top stalled instructions (highest sample count among non-issued PCs)",
        "",
        "| samples | issued | stalled | instruction |",
        "|---:|---:|---:|---|",
    ]
    # Re-sort hot_instructions by stalled count for this section
    hot_by_stall = sorted(pc["hot_instructions"], key=lambda kv: kv[1][2], reverse=True)[:top]
    for instr, (n, i, s) in hot_by_stall:
        if s == 0:
            break
        out.append(f"| {n} | {i} | {s} | `{instr}` |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="append", required=True,
                    help="label:rundir, e.g. fp8:runs/prefill_d128_fp8_b16_sq10000_sk10000")
    ap.add_argument("--out", default=None, help="output markdown path (default stdout)")
    ap.add_argument("--pc-top", type=int, default=20)
    args = ap.parse_args()

    reports = []
    for spec in args.label:
        label, rd = spec.split(":", 1)
        if not os.path.isdir(rd):
            sys.exit(f"missing run dir: {rd}")
        reports.append(build_report(label, rd))

    lines = []
    lines.append("# rocprofv3 UA kernel bottleneck analysis\n")
    lines.append("## kernel wall time\n")
    lines.append("| label | calls | avg ms | total ms | min | max | run dir |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for r in reports:
        t = r["timing"]
        if t:
            lines.append(f"| {r['label']} | {t['calls']} | {t['avg_ms']:.4f} | {t['total_ms']:.3f} "
                         f"| {t['min_ms']:.4f} | {t['max_ms']:.4f} | `{r['run_dir']}` |")

    lines.append("\n## compute / instruction mix (per dispatch)\n")
    lines.append(emit_compute_table(reports))

    lines.append("\n## stalls + memory pipeline (per dispatch)\n")
    lines.append(emit_stall_table(reports))

    for r in reports:
        lines.append(f"\n## PC-sampling top hotspots — {r['label']}\n")
        lines.append(emit_pcsamp_table(r, top=args.pc_top))

    out = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(out)
        print(f"wrote {args.out}")
    else:
        print(out)


if __name__ == "__main__":
    main()
