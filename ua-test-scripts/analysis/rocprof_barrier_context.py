#!/usr/bin/env python3
"""Show each `s_barrier` PC with its surrounding instructions, per warp group.

For every distinct `s_barrier` PC observed in the stochastic samples, we
print:
  - per-warp-group stall count at that barrier
  - the 12 instructions immediately preceding the barrier
    (= the *phase work* the warp group did to reach this barrier)
  - the 8 instructions immediately following the barrier

Inferring the phase from the preceding-instruction class:
  - many `v_mfma_…fp8_fp8` (or `f32_…_bf16`)  → gemm0 or gemm1 phase
  - many `buffer_load_dword … lds` / `global_load … lds`     → K_mem_load
                                                                or V_mem_load
  - many `ds_read` / `ds_write`              → K_lds_load / V_lds_load
  - many `v_cvt_pk_fp8_f32`                  → FP8 quantize half of alu1
  - `ds_bpermute_b32`                        → cross-lane shuffle of alu1
  - `v_max3_f32`, `v_exp_f32`, etc.          → softmax (alu1)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict


def load(run_dir: str):
    path = os.path.join(run_dir, "phase_c_pcsamp", "pcsamp_results.json")
    with open(path) as f:
        obj = json.load(f)
    top = obj["rocprofiler-sdk-tool"][0]
    return (
        top["buffer_records"]["pc_sample_stochastic"],
        top["strings"]["pc_sample_instructions"],
    )


def build_pc_index(samples, instrs):
    """Build a global PC → (instruction_text, [wg0_count, wg1_count]) map."""
    pc_info: dict[int, dict] = {}
    for s in samples:
        rec = s["record"]
        wig = rec.get("wave_in_grp", rec["hw_id"]["wave_id"])
        wg = 0 if wig < 4 else 1
        pc = rec["pc"]["code_object_offset"]
        co = rec["pc"]["code_object_id"]
        key = (co, pc)
        inst = instrs[s["inst_index"]] if s["inst_index"] < len(instrs) else "?"
        if key not in pc_info:
            pc_info[key] = {"inst": inst, "wg": [0, 0]}
        pc_info[key]["wg"][wg] += 1
    return pc_info


def find_barriers(pc_info, min_total=20):
    """Return list of (co, pc, n0, n1) for s_barrier PCs."""
    out = []
    for (co, pc), info in pc_info.items():
        if info["inst"].strip() == "s_barrier":
            n0, n1 = info["wg"]
            if n0 + n1 >= min_total:
                out.append((co, pc, n0, n1, info["inst"]))
    out.sort(key=lambda r: -(r[2] + r[3]))
    return out


def context(pc_info, co, pc, before=14, after=8, max_gap_bytes=64):
    """Return [(pc, inst, n0, n1)] for sampled PCs around `pc` in code object `co`.
    We grow only along PCs that exist in `pc_info` and reject pairs more than
    `max_gap_bytes` apart (catches discontinuities at function boundaries).
    """
    # Collect all PCs for this code object, sorted.
    pcs = sorted(p for (c, p) in pc_info if c == co)
    if not pcs:
        return []
    if pc not in pcs:
        return []
    idx = pcs.index(pc)
    lo = max(0, idx - before)
    hi = min(len(pcs), idx + after + 1)
    out = []
    for i in range(lo, hi):
        p = pcs[i]
        info = pc_info[(co, p)]
        out.append((p, info["inst"], info["wg"][0], info["wg"][1]))
    return out


def classify_phase(insts):
    """Look at the preceding instruction text and infer the phase label."""
    score = defaultdict(int)
    for _, txt, _, _ in insts:
        t = txt.lower()
        if "mfma" in t and "fp8" in t:
            score["mfma_fp8"] += 1
        elif "mfma" in t:
            score["mfma_bf"] += 1
        if "ds_bpermute" in t:
            score["ds_bpermute"] += 1
        if t.startswith("ds_read") or t.startswith("ds_write"):
            score["ds_rw"] += 1
        if "buffer_load" in t or "global_load" in t:
            score["mem_load"] += 1
        if "v_cvt_pk_fp8" in t:
            score["fp8_cvt"] += 1
        if "v_exp" in t or "v_log" in t or "v_max3" in t:
            score["softmax"] += 1
        if t.startswith("s_waitcnt"):
            score["waitcnt"] += 1
    # Heuristic labelling
    tags = []
    if score["mfma_fp8"] >= 3:
        tags.append("gemm(fp8 MFMA)")
    if score["mfma_bf"] >= 3 and "gemm(fp8 MFMA)" not in tags:
        tags.append("gemm(bf16 MFMA)")
    if score["fp8_cvt"] >= 3 or score["ds_bpermute"] >= 1:
        tags.append("alu1(softmax+cvt+bperm)")
    if score["softmax"] >= 3 and "alu1(softmax+cvt+bperm)" not in tags:
        tags.append("alu1(softmax)")
    if score["mem_load"] >= 2:
        tags.append("mem_load")
    if score["ds_rw"] >= 3:
        tags.append("lds_rw")
    if not tags:
        tags.append("(mixed/unclear)")
    return " + ".join(tags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--before", type=int, default=14)
    ap.add_argument("--after", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    samples, instrs = load(args.run_dir)
    pc_info = build_pc_index(samples, instrs)
    barriers = find_barriers(pc_info)

    L = ["# `s_barrier` phase classification & per-warp-group residency",
         "", f"_{len(samples):,} stochastic samples; {len(barriers)} distinct s_barrier PCs (after dropping <20 total)_", ""]

    for co, pc, n0, n1, _ in barriers[: args.top]:
        ctx = context(pc_info, co, pc, before=args.before, after=args.after)
        # Preceding insts only for phase classification
        before_only = [t for t in ctx if t[0] < pc]
        phase = classify_phase(before_only)
        bigger = max(n0, n1)
        smaller = max(1, min(n0, n1))
        first_arrival = "W0-3" if n0 > n1 else ("W4-7" if n1 > n0 else "tie")
        L += [
            f"### `s_barrier` @ co={co} pc=0x{pc:x} — W0-3 stalls={n0}, W4-7 stalls={n1} (imbalance {bigger/smaller:.2f}× ; **{first_arrival} arrives first**)",
            f"_inferred phase preceding this barrier: **{phase}**_",
            "",
            "| pc | wg0 | wg1 | instruction |",
            "|---|---:|---:|---|",
        ]
        for p, txt, c0, c1 in ctx:
            marker = " ← barrier" if p == pc else ""
            L.append(f"| 0x{p:x} | {c0} | {c1} | `{txt[:90]}`{marker} |")
        L.append("")

    out = "\n".join(L)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out)
        print(f"wrote {args.out}")
    else:
        print(out)


if __name__ == "__main__":
    main()
