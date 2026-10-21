# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Generate aiter/configs/gemm_fusion/minimax_m3_sp4_gfx950_v1.json from specs.py (signatures computed, never hand-typed).
Run: python -m aiter.ops.gemm_fusion.sp_qkv.gen_profile [out.json]   (pure Python; works without torch/GPU)
Rows: exact (sig, arch=gfx950, cu=256, world=4, m=T/4): the 6 small cells and the 9 large cells.
Rules (P1): large range rules for local m = 1024..8192 step 256 (see aiter.gemm_fusion.profile): consulted only where no exact row exists, the most specific
(largest m_multiple) rule wins. The params of the rules are PLACEHOLDERS (nearest exact row whose plan is valid at every covered m, else n=1); the final params come
from the tuning sweep.
"""

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
        rows.append(
            {
                "sig": sp.signature(),
                "arch": ARCH,
                "cu": CU,
                "world": WORLD,
                "m": T // WORLD,
                "vid": vid,
                "params": dict(params),
                "cell": f"{mode}_T{T}",
            }
        )
    rows.extend(build_large_rows())
    return rows


def build_large_rows():
    rows = []
    for (mode, T), (vid, params) in specs.LARGE_CELLS.items():
        sp = specs.sp_qkv_spec(mode).prefix(
            1
        )  # GEMM + SPExchange only; HeadNormRopeCache runs natively in ATOM
        rows.append(
            {
                "sig": sp.signature(),
                "arch": ARCH,
                "cu": CU,
                "world": WORLD,
                "m": T // WORLD,
                "vid": vid,
                "params": dict(params),
                "cell": f"{mode}_T{T}",
            }
        )
    return rows


RULE_M_MIN, RULE_M_MAX, RULE_M_BASE = 1024, 8192, 256
FALLBACK_PARAMS = {
    "n": 1,
    "half_first": 0,
    "no_gridread": 1,
}  # valid for every M multiple of 256 (n=1 needs no divisibility)


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


def build_large_rules():
    """P1 range rules for local m = 1024..8192, built in LAYERS by m_multiple (the most specific rule wins, see aiter.gemm_fusion.profile): the large plan needs
    (M/256) % n == 0, so params with n=2 are valid only for M multiples of 512 and n=4 only for multiples of 1024.
      layer 256   : the whole range, FALLBACK_PARAMS (n=1): valid everywhere;
      layer 512   : M multiples of 512, the nearest exact row with n <= 2;
      layer 1024  : M multiples of 1024, the nearest exact row with n <= 4.
    PLACEHOLDER params: nearest exact large row of the same mode (distance in m, ties to the smaller m; the M=2048-only quarter-first plan is never extended);
    every assignment is validated against the real plan constructor at every m it covers. Replace with the tuned params when the sweep is done (edit this
    function and regenerate: `python op_tests/gen_gemm_fusion_profile.py`)."""
    rules = []
    for mode in ("dense", "indexer", "reuse"):
        sp = specs.sp_qkv_spec(mode).prefix(1)  # GEMM + SPExchange only
        cells = sorted(
            (T // WORLD, vid, dict(params))
            for (md, T), (vid, params) in specs.LARGE_CELLS.items()
            if md == mode and params.get("half_first") != 2
        )
        vid0 = cells[0][1]
        for step in (RULE_M_BASE, 2 * RULE_M_BASE, 4 * RULE_M_BASE):
            runs = []  # [first_m, last_m, params, source]
            for m in range(RULE_M_MIN, RULE_M_MAX + 1, step):
                eligible = [
                    c
                    for c in cells
                    if step % (RULE_M_BASE * int(c[2].get("n", 1))) == 0
                    and _plan_valid(mode, c[2], m)
                ]
                if eligible:
                    src_m, _vid, params = min(
                        eligible, key=lambda c: (abs(c[0] - m), c[0])
                    )
                    source = f"nearest exact row (m={src_m})"
                else:
                    params, source = dict(FALLBACK_PARAMS), "fallback n=1"
                assert _plan_valid(mode, params, m), (mode, step, m, params)
                if runs and runs[-1][2] == params:
                    runs[-1][1] = m
                    if runs[-1][3] != source:
                        runs[-1][3] = "nearest exact rows"
                else:
                    runs.append([m, m, params, source])
            if (
                step > RULE_M_BASE
            ):  # a layer that only repeats the fallback adds nothing
                runs = [r for r in runs if r[2] != FALLBACK_PARAMS]
            for lo, hi, params, source in runs:
                rules.append(
                    {
                        "sig": sp.signature(),
                        "arch": ARCH,
                        "cu": CU,
                        "world": WORLD,
                        "m_min": lo,
                        "m_max": hi,
                        "m_multiple": step,
                        "vid": vid0,
                        "params": params,
                        "name": f"{mode}_m{lo}-{hi}_x{step}",
                        "note": f"PLACEHOLDER params: {source}; final params from the tuning sweep",
                    }
                )
    return rules


def window_budget():
    """Window numbers of the two policies (see aiter.gemm_fusion.resources). uniform: every window registered on the shared comm gets window_bytes (max over all
    plans, 4 MiB multiple), and max_windows bounds how many are registered (2 per small owner, 3 per large owner; owners are keyed per (mode, params) small /
    (mode, M) large). arena: ONE shared S, R (max large plan) and C (control) used by every large plan (`shared`) plus each small owner's two windows
    (`small_window_sizes`); the arena is their sum."""
    from aiter.gemm_fusion.resources import round_up

    from . import variants, variants_large

    small = {}
    for (mode, T), (vid, params) in specs.SMALL_CELLS.items():
        b = variants.small_window_bytes(mode)
        small[(mode, tuple(sorted(params.items())))] = round_up(
            max(b["rx"], b["ctl"], 8192)
        )
    large = {
        (mode, T // WORLD): variants_large.large_window_bytes(mode, T // WORLD)
        for (mode, T) in specs.LARGE_CELLS
    }
    wb = max(list(small.values()) + list(large.values()))
    from . import variants_large as vl

    small_sizes = sorted([s for s in small.values() for _ in range(2)], reverse=True)
    sr = max(large.values())
    shared = {
        vl.SHARE_S: sr,
        vl.SHARE_R: sr,
        vl.SHARE_C: round_up(vl.LG_CONTROL_MIN_BYTES),
    }
    return {
        "window_bytes": wb,
        "max_windows": 2 * len(small) + 3 * len(large),
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
        "rows": build_rows(),
        "rules": build_large_rules(),
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
