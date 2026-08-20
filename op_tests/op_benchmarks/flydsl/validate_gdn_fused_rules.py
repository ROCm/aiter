#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Validate the gfx942 GDN fused-variant selection rules against a sweep .md.

Both fused selection decisions are now closed-form (no tables):
  * VARIANT (BV/wave) = an ``H*N`` rule in ``select_fused_variant``
    (H*N<=32 -> bv16, <=64 -> bv32, >64 -> bv64w8);
  * ROUTING (fused vs separate) = an actual-fill threshold in
    ``should_use_fused_gfx942`` (fuse iff ceil(V/BV_run)*N*H/CU >= _FUSED_MIN_FILL).

This tool reads a full-sweep markdown file (as produced by
``bench_chunk_gdn_fwd.py``, per-shape ``### Shape N:`` tables), computes the
measured-best fused variant per ``(gate, H, _n_bucket(N), is_varlen)`` signature
(min-mean over the COMMON equal/ragged distributions), and checks that the H*N
rule still matches. Run it on a fresh sweep to confirm the rule holds (or learn
the thresholds need re-tuning). Nothing here is imported by the kernel at
runtime; the sweep .md files are gitignored.

Usage:
    python3 op_tests/op_benchmarks/flydsl/validate_gdn_fused_rules.py SWEEP.md
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

# Authoritative shape signatures come from the bench preset list, not regex.
_BENCH_DIR = str(Path(__file__).parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)
from bench_chunk_gdn_fwd import PRESET_SHAPES  # noqa: E402

FUSED_VARIANTS = ("bv16", "bv32", "bv64", "bv64w8")
_SEPARATE_IMPL = "K5_flydsl+K6_triton"
_FUSED_PREFIX = "K5K6_flydsl_fused:"

# Sequence-length distributions treated as "common" (realistic serving) for the
# fused-VARIANT pick. Adversarial skew/bimodal batches are excluded from the pick
# so the common case gets its optimal tile (e.g. bv64w8 on equal batches, which
# skew shapes would otherwise drag toward a smaller tile). The fuse-vs-separate
# DECISION still uses all distributions (conservative on whether to fuse at all).
_COMMON_SEQDISTS = frozenset({"equal", "ragged"})


def _n_bucket(N: int) -> int:
    """Must match chunk_gated_delta_h_gfx942._n_bucket exactly."""
    if N <= 1:
        return 1
    if N == 2:
        return 2
    if N <= 4:
        return 4
    return 8


def _signature(shape: tuple) -> tuple:
    """(gate, H, n_bucket, is_varlen) from a PRESET_SHAPES tuple.

    PRESET_SHAPES entry = (model_tag, H, Hg, T_flat, N, K, V, BT, gate, seqdist).
    is_varlen == (N > 1): the host builds cu_seqlens for every N>1 batch.
    """
    _, H, _Hg, _T, N, _K, _V, _BT, gate, _seq = shape
    return (gate, H, _n_bucket(N), N > 1)


def _parse_sweep(path: str) -> dict[int, dict[str, float]]:
    """{shape_index (1-based): {impl_name: time_us}} from per-shape tables."""
    txt = Path(path).read_text()
    secs = re.split(r"^### Shape (\d+):[^\n]*$", txt, flags=re.M)
    out: dict[int, dict[str, float]] = {}
    for i in range(1, len(secs), 2):
        idx = int(secs[i])
        rows: dict[str, float] = {}
        for line in secs[i + 1].splitlines():
            m = re.match(r"\|\s*([^|]+?)\s*\|\s*([\d.]+)\s*\|", line)
            if m:
                try:
                    rows[m.group(1).strip()] = float(m.group(2))
                except ValueError:
                    pass
        if rows:
            out[idx] = rows
    return out


def _best_fused(rows: dict[str, float]) -> tuple[str | None, float | None]:
    """(tag, time) of the fastest fused variant present in a shape's row set."""
    cand = {
        v: rows[_FUSED_PREFIX + v] for v in FUSED_VARIANTS if _FUSED_PREFIX + v in rows
    }
    if not cand:
        return None, None
    tag = min(cand, key=cand.get)
    return tag, cand[tag]


def _aggregate(sweep: dict[int, dict[str, float]]):
    """signature -> per-variant fused times (common-dist only + all) + separate.

    ``fused_common`` holds only equal/ragged shapes (drives the VARIANT pick);
    ``fused_all`` holds every distribution (used as a fallback when a signature
    has no common shapes). ``separate`` (all dists) drives the fuse DECISION.
    """
    agg: dict[tuple, dict] = defaultdict(
        lambda: {
            "fused_common": defaultdict(list),
            "fused_all": defaultdict(list),
            "separate": [],
        }
    )
    for idx, shape in enumerate(PRESET_SHAPES, 1):
        if idx not in sweep:
            continue
        sig = _signature(shape)
        seqdist = shape[9]
        rows = sweep[idx]
        for v in FUSED_VARIANTS:
            key = _FUSED_PREFIX + v
            if key in rows:
                agg[sig]["fused_all"][v].append(rows[key])
                if seqdist in _COMMON_SEQDISTS:
                    agg[sig]["fused_common"][v].append(rows[key])
        if _SEPARATE_IMPL in rows:
            agg[sig]["separate"].append(rows[_SEPARATE_IMPL])
    return agg


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def generate(sweep_path: str):
    sweep = _parse_sweep(sweep_path)
    agg = _aggregate(sweep)

    fused_rows = []  # (gate, H, n_bucket, is_varlen, tag)
    report = []
    for sig in sorted(agg, key=lambda s: (s[0], s[1], s[2], s[3])):
        gate, H, nb, varlen = sig
        d = agg[sig]
        # VARIANT pick: min-mean over COMMON (equal/ragged) shapes so the
        # realistic serving case gets its optimal tile; fall back to all-dist
        # when a signature has no common shapes (e.g. skew-only varlen sigs).
        pick_src = d["fused_common"] if d["fused_common"] else d["fused_all"]
        means = {v: _mean(t) for v, t in pick_src.items() if t}
        if not means:
            continue
        best_tag = min(means, key=means.get)
        best_fused_mean = means[best_tag]
        sep_mean = _mean(d["separate"])
        fused_rows.append((gate, H, nb, varlen, best_tag))
        report.append(
            f"#   {gate:>2} H={H:<3} nb={nb} varlen={int(varlen)}: "
            f"fused[{best_tag}]={best_fused_mean:7.1f}  "
            f"sep={sep_mean if sep_mean is None else round(sep_mean,1)!s:>8}"
        )

    # Validate the closed-form H*N rule against the measured best per signature.
    # (H*N <= 32 -> bv16, <= 64 -> bv32, > 64 -> bv64w8.) Uses the bucket's
    # representative N so it compares like-for-like with select_fused_variant.
    _BUCKET_N = {1: 1, 2: 2, 4: 4, 8: 8}

    def _hn_rule(H, N):
        hn = H * N
        return "bv16" if hn <= 32 else "bv32" if hn <= 64 else "bv64w8"

    mismatches = []
    for gate, H, nb, varlen, tag in fused_rows:
        N = _BUCKET_N[nb]
        pred = _hn_rule(H, N)
        if pred != tag:
            mismatches.append((gate, H, nb, tag, pred))

    print(f"# validate_gdn_fused_rules.py from {Path(sweep_path).name}")
    print("#")
    print("# Fused variant selection is a closed-form H*N rule in")
    print("# select_fused_variant (not a table); routing is an actual-fill")
    print("# threshold in should_use_fused_gfx942. This tool VALIDATES the H*N")
    print("# rule against a fresh sweep -- it no longer emits table literals.")
    print("#")
    print("# per-signature measured-best (min-mean over sharing shapes):")
    print("\n".join(report))
    print("#")
    if mismatches:
        print(f"# H*N-rule MISMATCHES ({len(mismatches)}): re-check the rule/thresholds")
        for gate, H, nb, tag, pred in mismatches:
            print(f"#   {gate} H={H} nb={nb}: measured={tag}  rule={pred}")
    else:
        print(f"# H*N rule AGREES with all {len(fused_rows)} measured signatures.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sweep", help="path to a full-sweep .md")
    args = ap.parse_args()
    generate(args.sweep)


if __name__ == "__main__":
    main()
