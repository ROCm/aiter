#!/usr/bin/env python3
"""
Analyse a rocprofv3 PC-sampling JSON, bucket samples by PC offset
within the CK Unified Attention kernel, and characterise each hot
sample by its instruction category + neighbour context.

Usage:
  python3 pc_hotspots.py <pcsamp_results.json> [--kernel-pat PAT] [--top N]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict


CATEGORIES = [
    ("BARRIER", lambda op: op.startswith("s_barrier")),
    ("WAIT", lambda op: op.startswith("s_wait")),
    ("MFMA", lambda op: "mfma" in op),
    ("CVT", lambda op: op.startswith("v_cvt")),
    ("FMA", lambda op: op.startswith("v_fma") or op.startswith("v_pk_fma")),
    ("VEXP", lambda op: op == "v_exp_f32_e32" or op.startswith("v_exp_")),
    ("VMUL", lambda op: "mul_f" in op or op.startswith("v_pk_mul")),
    ("LDS_TR", lambda op: op.startswith("ds_read") and "tr" in op),
    ("LDS_OTHER", lambda op: op.startswith("ds_")),
    ("VMEM", lambda op: op.startswith("buffer_") or op.startswith("global_") or op.startswith("flat_")),
    ("VALU_OTHER", lambda op: op.startswith("v_")),
    ("SALU", lambda op: op.startswith("s_")),
]


def categorise(op: str) -> str:
    for cat, pred in CATEGORIES:
        if pred(op):
            return cat
    return "OTHER"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_file")
    ap.add_argument("--kernel-pat", default="UnifiedAttentionKernel")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--context", type=int, default=4,
                    help="show this many neighbour PCs around each hot PC")
    args = ap.parse_args()

    with open(args.json_file) as f:
        data = json.load(f)
    node = data["rocprofiler-sdk-tool"][0]
    samples = node["buffer_records"]["pc_sample_host_trap"]
    insts = node["strings"]["pc_sample_instructions"]
    ksyms = node["kernel_symbols"]
    disps = node["buffer_records"]["kernel_dispatch"]
    disp_to_kid = {
        d["dispatch_info"]["dispatch_id"]: d["dispatch_info"]["kernel_id"]
        for d in disps
    }
    kid_name = {k["kernel_id"]: k["kernel_name"] for k in ksyms}

    pc_hits: Counter = Counter()
    pc_inst: dict = {}
    for s in samples:
        rec = s["record"]
        idx = s["inst_index"]
        if idx < 0:
            continue
        did = rec["dispatch_id"]
        kid = disp_to_kid.get(did, 0)
        if args.kernel_pat not in kid_name.get(kid, ""):
            continue
        pc = rec["pc"]
        key = (pc["code_object_id"], pc["code_object_offset"])
        pc_hits[key] += 1
        pc_inst[key] = insts[idx]

    if not pc_hits:
        print("No samples in target kernel.")
        return

    total = sum(pc_hits.values())
    cat_hits: Counter = Counter()
    for k, n in pc_hits.items():
        cat_hits[categorise(pc_inst[k].split()[0])] += n

    print(f"Recognised samples in {args.kernel_pat!r}: {total}")
    print(f"Unique PCs: {len(pc_hits)}\n")
    print("Category breakdown (% of recognised samples):")
    for cat, n in cat_hits.most_common():
        print(f"  {cat:<10} {n:>6}  ({n/total*100:5.2f}%)")

    print(f"\nTop {args.top} hot PCs (with neighbour context, +/- {args.context}):")
    sorted_pcs = sorted(pc_hits.keys(), key=lambda k: -pc_hits[k])
    # Build sorted-by-offset map per code object for neighbour lookup
    per_cobj: dict[int, list[int]] = defaultdict(list)
    for (coid, off) in pc_hits:
        per_cobj[coid].append(off)
    for coid in per_cobj:
        per_cobj[coid].sort()

    for (coid, off) in sorted_pcs[:args.top]:
        n = pc_hits[(coid, off)]
        ordered = per_cobj[coid]
        i = ordered.index(off)
        lo = max(0, i - args.context)
        hi = min(len(ordered), i + args.context + 1)
        print(
            f"\n  HOT [cobj={coid} off=0x{off:08x}]  hits={n}  ({n/total*100:.2f}%)"
        )
        for j in range(lo, hi):
            n_off = ordered[j]
            mark = "==>" if n_off == off else "   "
            ins = pc_inst[(coid, n_off)]
            hits = pc_hits[(coid, n_off)]
            print(
                f"   {mark} off=0x{n_off:08x}  hits={hits:>4}  "
                f"({categorise(ins.split()[0]):<10}) {ins[:90]}"
            )


if __name__ == "__main__":
    main()
