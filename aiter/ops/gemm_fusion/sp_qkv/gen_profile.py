# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Generate aiter/configs/gemm_fusion/minimax_m3_sp4_gfx950_v1.json from specs.py (signatures computed, never hand-typed).
Run: python -m aiter.ops.gemm_fusion.sp_qkv.gen_profile [out.json]   (pure Python; works without torch/GPU)
Rows: exact (sig, arch=gfx950, cu=256, world=4, m): the 6 small cells (T64/T128, src r19) and 87 large cells, one per (mode, local m = 1024..8192 step 256) with that
cell's best (n, half_first) from the P1 sweep (p1_sweep.py, src r12).
Rules (P1): the "chunks mod 4 classes" fallback family (9 rules, see aiter.gemm_fusion.profile): consulted only where no exact row exists; none exists on the grid, so they
document the default for any other m % 256 == 0 in range.
"""

from __future__ import annotations

import json
import os
import sys

from aiter.gemm_fusion.errpage import GLOBAL_PAGE_BYTES, SHARE_GLOBAL

from . import p1_sweep, specs

ARCH, CU, WORLD = "gfx950", 256, 4
LARGE_PROFILE_SRC = (
    specs.LARGE_PROFILE_SRC
)  # the large rows and rules run the R12 kernels (ERR page, call_id tags, NaN-fill node); "r11" to A/B


SMALL_CAPS = frozenset({"mori_cco", "gfx950_sc_bypass"})


def _small_plan_valid(mode, params, m):
    """Would the small plan accept these params at local M = m? (the real constructor: M range of the source, push8 / b_rowsplit shape limits, rx_bypass capability, ...)"""
    from aiter.gemm_fusion.match import ContractError, MatchCtx

    from . import variants

    ctx = MatchCtx(ARCH, CU, WORLD, m, SMALL_CAPS)
    try:
        variants.SpQkvSmall(specs.sp_qkv_spec(mode), dict(params), ctx)
    except ContractError:
        return False
    return True


def build_rows():
    rows = []
    tuned = {(mode, T // WORLD) for (mode, T) in specs.SMALL_CELLS}
    for mode, m, vid, params in specs.small_exact_cells():
        assert _small_plan_valid(mode, params, m), (mode, m, params)
        sp = specs.sp_qkv_spec(mode)
        row = {
            "sig": sp.signature(),
            "arch": ARCH,
            "cu": CU,
            "world": WORLD,
            "m": m,
            "vid": vid,
            "params": dict(params),
            "cell": f"{mode}_T{WORLD * m}",
        }
        if (mode, m) not in tuned:
            row["note"] = {
                specs.SMALL_R20_SRC: "R13 MP64 (src r20)",
                specs.SMALL_R21_SRC: "R13 MP128 (src r21, 8 MiB Rx view)",
            }.get(params["src"], "R13 P2: copy of the mode's m=16 params")
        rows.append(row)
    rows.extend(build_large_rows())
    return rows


def build_small_rules():
    """The small range rules per mode (specs.small_range_rules: r19 over m 1..16 and 17..32, r20 over 33..64, r21 over 65..128; every m): consulted only where no exact row exists. Every
    covered m is validated against the real plan constructor (push8 above 16, b_rowsplit at or below 16 and an r20 build without the bypass would be refused).
    """
    rules = []
    for r in specs.small_range_rules():
        mode = r["mode"]
        for m in range(r["m_min"], r["m_max"] + 1, r["m_multiple"]):
            assert _small_plan_valid(mode, r["params"], m), (mode, m, r["params"])
        rules.append(
            {
                "sig": specs.sp_qkv_spec(mode).signature(),
                "arch": ARCH,
                "cu": CU,
                "world": WORLD,
                "m_min": r["m_min"],
                "m_max": r["m_max"],
                "m_multiple": r["m_multiple"],
                "vid": r["vid"],
                "params": dict(r["params"]),
                "name": r["name"],
                "note": r["note"],
            }
        )
    return rules


LARGE_VID = "sp_qkv.large.v1"
RULE_M_MIN, RULE_M_MAX, RULE_M_BASE = 1024, 8192, 256


def _plan_valid(mode, params, m):
    """Would the large plan accept these params at local M = m? (the real constructor: n in {1,2,4} with (M/256) % n == 0, n3q only at M=2048, ...)"""
    from aiter.gemm_fusion.match import ContractError, MatchCtx

    from . import variants_large

    ctx = MatchCtx(ARCH, CU, WORLD, m, frozenset({"mori_cco", "sdma"}))
    try:
        variants_large.SpQkvLarge(specs.sp_qkv_spec(mode).prefix(1), dict(params), ctx)
    except ContractError:
        return False
    return True


def build_large_rows():
    """One EXACT row per (mode, m) cell of the P1 sweep (p1_sweep.P1_BEST): local m = 1024..8192 step 256 x dense / indexer / reuse = 87 rows, each with that cell's best
    (n, half_first) on the r12 kernels. Compact m-range rules lose up to 4.6 % against this table (half_first at n=4 has no closed form), so the table IS the profile.
    The r11-era tuned rows (specs.LARGE_CELLS) are kept where the table agrees and replaced where it differs; a replaced row carries a `note` with the old params.
    """
    legacy = {
        (mode, T // WORLD): params
        for (mode, T), (_vid, params) in specs.LARGE_CELLS.items()
    }
    rows = []
    for mode in ("dense", "indexer", "reuse"):
        sp = specs.sp_qkv_spec(mode).prefix(
            1
        )  # GEMM + SPExchange only; HeadNormRopeCache runs natively in ATOM
        for m in p1_sweep.P1_GRID:
            n, half_first = p1_sweep.P1_BEST[mode][m]
            params = {
                "n": n,
                "half_first": half_first,
                "no_gridread": 1,
                "src": LARGE_PROFILE_SRC,
            }
            assert _plan_valid(mode, params, m), (
                mode,
                m,
                params,
            )  # the sweep only timed legal points; keep the generator honest
            row = {
                "sig": sp.signature(),
                "arch": ARCH,
                "cu": CU,
                "world": WORLD,
                "m": m,
                "vid": LARGE_VID,
                "params": params,
                "cell": f"{mode}_T{WORLD * m}",
            }
            old = legacy.get((mode, m))
            if old is not None and (old["n"], old["half_first"]) != (n, half_first):
                row["note"] = (
                    f"P1 sweep best; replaces the r11-era tuned (n={old['n']}, half_first={old['half_first']})"
                )
            rows.append(row)
    return rows


# The safe fallback for any m % 256 == 0 in range that is not on the sweep grid (none exist: the grid is every multiple of 256 in 1024..8192, so every rule below is shadowed
# by an exact row; they document the default): the "chunks mod 4 classes" family of the sweep, half_first 0. Layered by m_multiple (the most specific rule wins):
#   chunks % 4 == 0  (m % 1024 == 0)  -> n=4        chunks % 4 == 2  (m % 512 == 0)  -> n=2        odd chunks  -> n=1.
# Measured worst loss against the per-point best (Evidence): 4.6 % (dense) / 4.5 % (indexer) / 3.8 % (reuse), all at chunks % 4 == 0; the other classes lose <= 1.5 %.
FALLBACK_LAYERS = (
    (RULE_M_BASE, 1),
    (2 * RULE_M_BASE, 2),
    (4 * RULE_M_BASE, 4),
)  # (m_multiple, n)


def build_large_rules():
    """Range rules for local m = 1024..8192: the chunks-mod-4 fallback family, one layer per m_multiple and mode (9 rules). Every assignment is validated against the real
    plan constructor at every m it covers."""
    rules = []
    for mode in ("dense", "indexer", "reuse"):
        sp = specs.sp_qkv_spec(mode).prefix(1)  # GEMM + SPExchange only
        for step, n in FALLBACK_LAYERS:
            params = {
                "n": n,
                "half_first": 0,
                "no_gridread": 1,
                "src": LARGE_PROFILE_SRC,
            }
            for m in range(RULE_M_MIN, RULE_M_MAX + 1, step):
                assert _plan_valid(mode, params, m), (mode, step, m, params)
            rules.append(
                {
                    "sig": sp.signature(),
                    "arch": ARCH,
                    "cu": CU,
                    "world": WORLD,
                    "m_min": RULE_M_MIN,
                    "m_max": RULE_M_MAX,
                    "m_multiple": step,
                    "vid": LARGE_VID,
                    "params": params,
                    "name": f"{mode}_chunks_mod4_x{step}",
                    "note": "fallback default (chunks mod 4 classes, half_first 0); shadowed by the 87 exact P1 rows on the sweep grid",
                }
            )
    return rules


def window_budget():
    """Window numbers of the two policies (see aiter.gemm_fusion.resources). uniform: every window registered on the shared comm gets window_bytes (max over all
    plans, 4 MiB multiple), and max_windows bounds how many are registered (2 per small owner, 3 per large owner; owners are keyed per (mode, params) small /
    (mode, M) large). arena: ONE shared S, R (max large plan) and C (control) used by every large plan (`shared`) plus each small owner's two windows
    (`small_window_sizes`); the arena is their sum."""
    from aiter.gemm_fusion.resources import round_up

    from . import jit, variants, variants_large

    small = {}  # owner (mode + params) -> (Rx capacity, control capacity)
    small_cells = [
        (mode, params) for mode, _m, _vid, params in specs.small_exact_cells()
    ]
    small_cells += [(r["mode"], r["params"]) for r in specs.small_range_rules()]
    for mode, params in small_cells:
        small[(mode, tuple(sorted(params.items())))] = variants.small_window_caps(
            mode, jit.src_mp(params["src"])
        )
    large = {
        (mode, T // WORLD): variants_large.large_window_bytes(mode, T // WORLD)
        for (mode, T) in specs.LARGE_CELLS
    }
    wb = max([c for caps in small.values() for c in caps] + list(large.values()))
    from . import variants_large as vl

    small_sizes = sorted(
        [c for caps in small.values() for c in caps], reverse=True
    )  # each owner: its Rx and its control window (MP128: 8 MiB + 4 MiB)
    sr = max(large.values())
    shared = {
        vl.SHARE_S: sr,
        vl.SHARE_R: sr,
        vl.SHARE_C: round_up(vl.LG_CONTROL_MIN_BYTES),
        # R12: the per-comm ERR/STEP/DIAG page shared by every R12 plan (arena: one 4 MiB view; uniform: one more equal-size window, counted in max_windows)
        SHARE_GLOBAL: round_up(GLOBAL_PAGE_BYTES),
    }
    return {
        "window_bytes": wb,
        "max_windows": 2 * len(small) + 3 * len(large) + 1,
        "small_owners": len(small),
        "large_owners": len(large),
        "small_window_sizes": small_sizes,
        "shared": shared,
    }


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
        "generated_by": {
            "tool": "aiter.ops.gemm_fusion.sp_qkv.gen_profile",
            "ledger": "sp4-opus-mori-playground records/branch-ledger.md r165 (small), r152 (large)",
            "note": "m is local tokens per rank (T/4); exact rows plus P1 large range rules (placeholder params)",
        },
        # local m with NO fused path, routed native by design (informational: the semantics is "no row and no rule for any prefix")
        "native_m_ranges": [dict(r) for r in specs.NATIVE_M_RANGES],
        "rows": build_rows(),
        "rules": build_small_rules() + build_large_rules(),
    }


def main(argv=None):
    argv = argv or sys.argv[1:]
    out = (
        argv[0]
        if argv
        else os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "configs",
            "gemm_fusion",
            "minimax_m3_sp4_gfx950_v1.json",
        )
    )
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(build_profile(), f, indent=2, sort_keys=False)
        f.write("\n")
    print("wrote", os.path.abspath(out))


if __name__ == "__main__":
    main()
