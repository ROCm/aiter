#!/usr/bin/env python3
"""Bucket FA4 prefill PC-samples into the two pipeline phases.

The FA4 core loop (unified_attention_pipeline.hpp `core_loop_fa4`) runs, per KV
tile, one MATRIX phase and one SOFTMAX phase per warp group, the two groups
offset by one phase (WG0 MATRIX ‖ WG1 SOFTMAX, then swap). rocprofv3 stochastic
PC samples don't carry a phase tag, but the *instruction* at each PC maps to a
phase by construction:

  MATRIX   : the PV + QK MFMAs, the V_lds/K_lds `ds_read`s that feed them, and
             the two `s_waitcnt lgkmcnt(0)` that guard those LDS reads.
  SOFTMAX  : mask + alu0(rowmax) + D_upd(rescale) + alu1(exp/rowsum/P-cvt),
             i.e. the transcendental / compare / convert / permute VALU.
  PREFETCH : the cooperative global->LDS `buffer_load ... lds` for tile k+1 and
             the page-table address math (refresh_*_offsets) that sets it up.
  SYNC     : `s_barrier` (phase rendezvous) + `s_waitcnt vmcnt(0)` (global-load
             wait at slot entry).

This is a semantic attribution (not a PC-range cut), but every bucket's
instructions live in exactly one phase in the source, so it is faithful for the
"where does the time go per phase" question. Unsymbolized NO_INST samples
(JIT code object not registered with the symbolizer) are reported separately.

Usage:
  python3 rocprof_phase_split.py runs/<tag> [--out report.md]
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict


def classify(instr: str) -> str:
    s = instr.strip()
    if not s:
        return "UNSYMBOLIZED"
    low = s.lower()

    # ---- MATRIX phase ----
    if "v_mfma" in low:
        return "MATRIX"
    if low.startswith("ds_read") or low.startswith("ds_load"):
        return "MATRIX"            # V_lds_load / K_lds_load LDS reads
    if "s_waitcnt" in low and "lgkmcnt" in low and "vmcnt" not in low:
        return "MATRIX"            # the two lgkmcnt(0) guarding the LDS reads

    # ---- SOFTMAX phase ----
    if any(k in low for k in (
        "v_exp", "v_max3", "v_max_f32", "v_fma_f32", "v_mul_f32", "v_add_f32",
        "v_sub_f32", "v_rcp", "v_cvt_", "v_pk_", "permlane", "v_perm", "ds_bpermute",
        "v_log", "v_fmac",
    )):
        return "SOFTMAX"

    # ---- PREFETCH (global->LDS load + page-table address math) ----
    if "buffer_load" in low and "lds" in low:
        return "PREFETCH"
    if any(k in low for k in (
        "v_lshlrev_b64", "v_ashrrev_i32", "v_add_u32", "v_mul_lo_u32",
        "v_lshl_add", "v_add_co", "v_addc", "v_lshlrev_b32", "v_mad_u",
    )):
        return "ADDR"              # address calc, feeds prefetch / page table

    # ---- SYNC ----
    if low.startswith("s_barrier"):
        return "BARRIER"
    if "s_waitcnt" in low and "vmcnt" in low:
        return "VMCNT_WAIT"        # global-load wait at slot entry

    if low.startswith("buffer_store"):
        return "EPILOGUE_STORE"
    if low.startswith("s_") or low.startswith("v_readfirstlane"):
        return "SCALAR_CTRL"
    return "OTHER"


PHASE_OF = {
    "MATRIX": "MATRIX",
    "SOFTMAX": "SOFTMAX",
    "PREFETCH": "PREFETCH/ADDR",
    "ADDR": "PREFETCH/ADDR",
    "BARRIER": "SYNC",
    "VMCNT_WAIT": "SYNC",
    "EPILOGUE_STORE": "EPILOGUE",
    "SCALAR_CTRL": "SCALAR_CTRL",
    "OTHER": "OTHER",
    "UNSYMBOLIZED": "UNSYMBOLIZED",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    csv_path = os.path.join(args.run_dir, "phase_c_pcsamp",
                            "pcsamp_pc_sampling_stochastic.csv")
    if not os.path.exists(csv_path):
        sys.exit(f"missing {csv_path}")

    cat_tot = defaultdict(int)
    cat_stall = defaultdict(int)
    phase_tot = defaultdict(int)
    phase_stall = defaultdict(int)
    instr_tot = defaultdict(int)         # (phase, instr) -> samples
    instr_stall = defaultdict(int)
    total = 0

    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            total += 1
            instr = row.get("Instruction", "")
            issued = row.get("Wave_Issued_Instruction", "0") == "1"
            cat = classify(instr)
            ph = PHASE_OF[cat]
            cat_tot[cat] += 1
            phase_tot[ph] += 1
            if not issued:
                cat_stall[cat] += 1
                phase_stall[ph] += 1
            if cat not in ("UNSYMBOLIZED",):
                # normalize register operands so identical opcodes group
                key = re.sub(r"\s+", " ", instr.strip())
                instr_tot[(ph, key)] += 1
                if not issued:
                    instr_stall[(ph, key)] += 1

    sym = total - cat_tot["UNSYMBOLIZED"]

    L = []
    def p(s=""): L.append(s)

    p(f"# FA4 phase split — `{os.path.basename(args.run_dir.rstrip('/'))}`\n")
    p(f"_total samples: {total:,}  |  symbolized: {sym:,} "
      f"({100*sym/total:.1f}%)  |  unsymbolized NO_INST: "
      f"{cat_tot['UNSYMBOLIZED']:,} ({100*cat_tot['UNSYMBOLIZED']/total:.1f}%)_\n")

    p("## phase share (of symbolized samples ≈ time)\n")
    p("| phase | samples | % symb | stalled | stall% |")
    p("|---|---:|---:|---:|---:|")
    order = ["MATRIX", "SOFTMAX", "SYNC", "PREFETCH/ADDR", "SCALAR_CTRL",
             "EPILOGUE", "OTHER"]
    for ph in order:
        t = phase_tot.get(ph, 0)
        if not t:
            continue
        st = phase_stall.get(ph, 0)
        p(f"| {ph} | {t:,} | {100*t/sym:.1f}% | {st:,} | {100*st/t:.1f}% |")

    p("\n## category detail (of symbolized)\n")
    p("| category | samples | % symb | stalled | stall% |")
    p("|---|---:|---:|---:|---:|")
    for cat in sorted(cat_tot, key=lambda c: -cat_tot[c]):
        if cat == "UNSYMBOLIZED":
            continue
        t = cat_tot[cat]
        st = cat_stall[cat]
        p(f"| {cat} | {t:,} | {100*t/sym:.1f}% | {st:,} | {100*st/t:.1f}% |")

    for ph in ["MATRIX", "SOFTMAX", "PREFETCH/ADDR", "SYNC"]:
        p(f"\n## top instructions — {ph}\n")
        p("| samples | stalled | stall% | instruction |")
        p("|---:|---:|---:|---|")
        keys = [k for k in instr_tot if k[0] == ph]
        keys.sort(key=lambda k: -instr_tot[k])
        for k in keys[:args.top]:
            t = instr_tot[k]
            st = instr_stall[k]
            p(f"| {t:,} | {st:,} | {100*st/t:.0f}% | `{k[1]}` |")

    out = "\n".join(L) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(out)
        print(f"wrote {args.out}")
    else:
        print(out)


if __name__ == "__main__":
    main()
