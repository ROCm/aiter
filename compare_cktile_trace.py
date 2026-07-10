#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compare two CK-Tile fingerprint traces to localize where garbage begins.

Produce the traces by running the same workload twice with:

    AITER_CKTILE_TRACE=1 AITER_CKTILE_TRACE_FILE=/tmp/pass.jsonl  <run on pass commit>
    AITER_CKTILE_TRACE=1 AITER_CKTILE_TRACE_FILE=/tmp/fail.jsonl  <run on fail commit>

Then:

    python compare_cktile_trace.py /tmp/pass.jsonl /tmp/fail.jsonl

The script aligns the two traces by MoE-call sequence number and reports the
FIRST call whose numbers diverge and WHICH tensor diverged first:

    hidden      -> layer input (residual) already wrong  => problem is UPSTREAM
                   of this MoE (attention / previous layer / router feedback)
    topk_ids /
    topk_weight -> the router picked different experts    => routing is the origin
    stage1_out  -> gemm1 (up/gate + SwiGLU) is the origin
    stage2_out  -> gemm2 (down-proj + expert combine) is the origin

Because we compare the fail run against the pass run, the pass run IS the golden
reference -- no separate reference implementation is needed.
"""

import argparse
import json
import math


def load(path):
    """Return {seq: {phase-or-field: fingerprint}} keyed by MoE-call seq."""
    calls = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            seq = rec["seq"]
            entry = calls.setdefault(seq, {"M": None})
            phase = rec["phase"]
            if phase == "moe_in":
                entry["M"] = rec.get("M")
                entry["topk"] = rec.get("topk")
                entry["hidden"] = rec["hidden"]
                entry["topk_ids"] = rec["topk_ids"]
                entry["topk_weight"] = rec["topk_weight"]
            else:  # stage1_out / stage2_out
                entry[phase] = rec["fp"]
    return calls


def rel(a, b):
    """Relative difference between two scalars, robust to ~0 and NaN."""
    if a is None or b is None:
        return float("inf")
    if math.isnan(a) and math.isnan(b):
        return 0.0
    if math.isnan(a) != math.isnan(b):
        return float("inf")
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom


def fp_diff(fa, fb, rtol):
    """Return list of human-readable reasons two fingerprints differ."""
    if fa is None or fb is None:
        return ["missing in one run"]
    reasons = []
    if fa.get("shape") != fb.get("shape"):
        reasons.append(f"shape {fa.get('shape')} vs {fb.get('shape')}")
        return reasons  # shape mismatch => alignment lost, stop here
    if fa.get("nan", 0) != fb.get("nan", 0):
        reasons.append(f"nan {fa['nan']} vs {fb['nan']}")
    if fa.get("inf", 0) != fb.get("inf", 0):
        reasons.append(f"inf {fa['inf']} vs {fb['inf']}")
    for key in ("l2", "mean", "absmax", "sum"):
        r = rel(fa.get(key), fb.get(key))
        if r > rtol:
            reasons.append(
                f"{key} rel={r:.3g} ({fa.get(key):.6g} vs {fb.get(key):.6g})"
            )
    return reasons


# Order matters: report the earliest tensor in the dataflow that diverged.
FIELD_ORDER = [
    (
        "hidden",
        "layer input residual  => UPSTREAM of this MoE (attn/norm/embed/convert)",
    ),
    ("topk_ids", "router expert ids     => ROUTING"),
    ("topk_weight", "router weights        => ROUTING"),
    ("stage1_out", "gemm1 up/gate+SwiGLU  => STAGE 1 (CK decode only)"),
    ("stage2_out", "gemm2 down-proj       => STAGE 2 (CK decode only)"),
    ("moe_out", "MoE output            => MoE COMPUTE (prefill FlyDSL or decode CK)"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pass_trace", help="trace JSONL from the known-good commit")
    ap.add_argument("fail_trace", help="trace JSONL from the failing commit")
    ap.add_argument(
        "--rtol",
        type=float,
        default=5e-2,
        help="relative tolerance for fingerprint scalars (default 0.05)",
    )
    ap.add_argument(
        "--skip",
        type=int,
        default=0,
        help="skip the first N MoE calls (e.g. to ignore warmup)",
    )
    ap.add_argument(
        "--max-report",
        type=int,
        default=5,
        help="how many diverging calls to print (default 5)",
    )
    args = ap.parse_args()

    p = load(args.pass_trace)
    f = load(args.fail_trace)
    common = sorted(set(p) & set(f))
    print(f"pass calls={len(p)}  fail calls={len(f)}  common seqs={len(common)}")
    if not common:
        print("No overlapping seqs -- traces are not aligned.")
        return

    reported = 0
    first = None
    for seq in common:
        if seq <= args.skip:
            continue
        pe, fe = p[seq], f[seq]
        if pe.get("M") != fe.get("M"):
            print(
                f"[seq {seq}] M differs ({pe.get('M')} vs {fe.get('M')}); "
                "alignment likely lost from here."
            )
            break
        diverged = []
        for field, label in FIELD_ORDER:
            reasons = fp_diff(pe.get(field), fe.get(field), args.rtol)
            if reasons:
                diverged.append((field, label, reasons))
        if diverged:
            if first is None:
                first = (seq, diverged)
            if reported < args.max_report:
                m = pe.get("M")
                kind = "prefill" if (m or 0) > 1 else "decode"
                print(f"\n=== DIVERGENCE at seq {seq} (M={m}, {kind}) ===")
                for field, label, reasons in diverged:
                    print(f"  {field:<12} [{label}]")
                    for r in reasons:
                        print(f"       - {r}")
                reported += 1

    print("\n" + "=" * 60)
    if first is None:
        print("No divergence found within tolerance. Runs match.")
    else:
        seq, diverged = first
        earliest_field, earliest_label, _ = diverged[0]
        print(f"FIRST divergence: seq {seq}")
        print(f"Earliest diverging tensor: {earliest_field}")
        print(f"Interpretation: {earliest_label}")
        print(
            "\nRead as: at this MoE call the '%s' tensor is the first thing that\n"
            "differs between the good and bad run -- that is where garbage begins."
            % earliest_field
        )


if __name__ == "__main__":
    main()
