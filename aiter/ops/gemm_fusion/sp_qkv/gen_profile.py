# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Generate aiter/configs/gemm_fusion/minimax_m3_sp4_gfx950_v1.json from specs.py (signatures computed, never hand-typed).
Run: python -m aiter.ops.gemm_fusion.sp_qkv.gen_profile [out.json]   (pure Python; works without torch/GPU)
Rows: exact (sig, arch=gfx950, cu=256, world=4, m=T/4). Small cells here; large cells are added in S2-2b-iii."""

from __future__ import annotations

import json
import os
import sys

from . import specs

ARCH, CU, WORLD = "gfx950", 256, 4


def build_rows():
    rows = []
    for (mode, T), (vid, params) in specs.SMALL_CELLS.items():
        sp = specs.sp_qkv_spec(mode)
        rows.append({"sig": sp.signature(), "arch": ARCH, "cu": CU, "world": WORLD, "m": T // WORLD,
                     "vid": vid, "params": dict(params), "cell": f"{mode}_T{T}"})
    rows.extend(build_large_rows())
    return rows


def build_large_rows():
    rows = []
    for (mode, T), (vid, params) in specs.LARGE_CELLS.items():
        sp = specs.sp_qkv_spec(mode).prefix(1)  # GEMM + SPExchange only; HeadNormRopeCache runs natively in ATOM
        rows.append({"sig": sp.signature(), "arch": ARCH, "cu": CU, "world": WORLD, "m": T // WORLD,
                     "vid": vid, "params": dict(params), "cell": f"{mode}_T{T}"})
    return rows


def window_budget():
    """Window numbers of the two policies (see aiter.gemm_fusion.resources). uniform: every window registered on the shared comm gets window_bytes (max over all
    plans, 4 MiB multiple), and max_windows bounds how many are registered (2 per small owner, 3 per large owner; owners are keyed per (mode, params) small /
    (mode, M) large). arena: ONE shared S, R (max large plan) and C (control) used by every large plan (`shared`) plus each small owner's two windows
    (`small_window_sizes`); the arena is their sum."""
    from . import variants, variants_large
    from aiter.gemm_fusion.resources import round_up

    small = {}
    for (mode, T), (vid, params) in specs.SMALL_CELLS.items():
        b = variants.small_window_bytes(mode)
        small[(mode, tuple(sorted(params.items())))] = round_up(max(b["rx"], b["ctl"], 8192))
    large = {(mode, T // WORLD): variants_large.large_window_bytes(mode, T // WORLD) for (mode, T) in specs.LARGE_CELLS}
    wb = max(list(small.values()) + list(large.values()))
    from . import variants_large as vl

    small_sizes = sorted([s for s in small.values() for _ in range(2)], reverse=True)
    sr = max(large.values())
    shared = {vl.SHARE_S: sr, vl.SHARE_R: sr, vl.SHARE_C: round_up(vl.LG_CONTROL_MIN_BYTES)}
    return {"window_bytes": wb, "max_windows": 2 * len(small) + 3 * len(large), "small_owners": len(small), "large_owners": len(large),
            "small_window_sizes": small_sizes, "shared": shared}


WAIT_BUDGET_US = 5_000_000


def build_profile():
    return {
        "profile": "minimax_m3_sp4_gfx950_v1",
        "window_budget": window_budget(),
        # Large-M per-WAIT poll budget in microseconds (credit wait, arrival wait, group waits): a per-launch kernel ARGUMENT, BAKED into a captured graph's argument
        # block at capture time (resolve before capture). 5 s: the r11 kernel default of 50 ms is shorter than plausible cold-start / host skew between ranks in a serving
        # engine (a clean first call poisoned at the natural ~213 ms cold-start skew; Evidence 05055ac9); keep the PG timeout > ~3 x budget + recover; the cost of a large value is only failure-detection latency (a dead peer holds the stream for up to ~2-3 budgets). AITER_GEMM_FUSION_WAIT_BUDGET_US
        # overrides it for tuning only. Small-M uses an iteration count (20,000,000) instead and is not affected.
        "wait_budget_us": WAIT_BUDGET_US,
        "generated_by": {"tool": "aiter.ops.gemm_fusion.sp_qkv.gen_profile", "ledger": "sp4-opus-mori-playground records/branch-ledger.md r165 (small), r152 (large)",
                         "note": "m is local tokens per rank (T/4); exact rows only"},
        "rows": build_rows(),
    }


def main(argv=None):
    argv = argv or sys.argv[1:]
    out = argv[0] if argv else os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs", "gemm_fusion", "minimax_m3_sp4_gfx950_v1.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(build_profile(), f, indent=2, sort_keys=False)
        f.write("\n")
    print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    main()
