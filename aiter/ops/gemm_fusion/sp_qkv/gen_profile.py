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
    """Uniform-window policy numbers (see aiter.gemm_fusion.resources): every window registered on the shared comm gets
    window_bytes (max over all plans, 4 MiB multiple), and max_windows bounds how many are registered (2 per small owner,
    3 per large owner; owners are keyed per (mode, params) small / (mode, M) large)."""
    from . import variants, variants_large
    from aiter.gemm_fusion.resources import round_up

    small = {}
    for (mode, T), (vid, params) in specs.SMALL_CELLS.items():
        b = variants.small_window_bytes(mode)
        small[(mode, tuple(sorted(params.items())))] = round_up(max(b["rx"], b["ctl"], 8192))
    large = {(mode, T // WORLD): variants_large.large_window_bytes(mode, T // WORLD) for (mode, T) in specs.LARGE_CELLS}
    wb = max(list(small.values()) + list(large.values()))
    sizes = sorted([s for s in small.values() for _ in range(2)] + [s for s in large.values() for _ in range(3)], reverse=True)
    return {"window_bytes": wb, "max_windows": 2 * len(small) + 3 * len(large), "small_owners": len(small), "large_owners": len(large),
            "window_sizes": sizes}


def build_profile():
    return {
        "profile": "minimax_m3_sp4_gfx950_v1",
        "window_budget": window_budget(),
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
