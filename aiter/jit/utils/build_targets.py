# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Pure-Python arch constants and env-driven build target resolution.
# No torch dependency — safe to import in build scripts, gen_instances, and tests
# that run without a GPU or a full PyTorch install.
import os

GFX_MAP = {
    0: "native",
    1: "gfx90a",
    2: "gfx908",
    3: "gfx940",
    4: "gfx941",
    5: "gfx942",
    6: "gfx945",
    7: "gfx1100",
    8: "gfx950",
    9: "gfx1101",
    10: "gfx1102",
    11: "gfx1103",
    12: "gfx1150",
    13: "gfx1151",
    14: "gfx1152",
    15: "gfx1153",
    16: "gfx1200",
    17: "gfx1201",
    18: "gfx1250",
}

# Maps gfx arch to the default (SPX / full-GPU) CU count used when no live GPU is
# present at build time (e.g. CI nodes with GPU_ARCHS set but no device visible).
# For live GPU builds, get_cu_num() is used instead and correctly reflects the
# actual visible CU count, including non-SPX partition modes (DPX / QPX / CPX)
# and binned variants (e.g. MI308X is gfx942 but has fewer CUs than MI300X).
# If building without a GPU for a binned or partitioned target, set CU_NUM
# explicitly alongside GPU_ARCHS to override the default here.
# Extend this table when adding support for new GPU targets.
GFX_CU_NUM_MAP = {
    "gfx942": 304,  # MI300X (SPX, full GPU); MI308X shares gfx942 — use CU_NUM override
    "gfx950": 256,  # MI350
}


def get_build_targets_env() -> list[tuple[str, int]]:
    """Resolve build targets from GPU_ARCHS env var only.  No live GPU detection.

    Raises RuntimeError if GPU_ARCHS is not set or contains an unknown arch.
    Intended for CI nodes, build scripts, and tests that run without a GPU.
    Use chip_info.get_build_targets() when live GPU fallback is also desired.
    """
    gfx_env = os.getenv("GPU_ARCHS")
    if not gfx_env:
        raise RuntimeError(
            "GPU_ARCHS is not set. "
            "Set GPU_ARCHS=gfx942 (or similar) to resolve build targets without a GPU."
        )
    targets = []
    for gfx in [g.strip() for g in gfx_env.split(";") if g.strip()]:
        if gfx not in GFX_CU_NUM_MAP:
            raise RuntimeError(
                f"Unknown gfx '{gfx}' in GPU_ARCHS — add it to "
                f"GFX_CU_NUM_MAP in build_targets.py. Known targets: "
                f"{list(GFX_CU_NUM_MAP.keys())}"
            )
        cu_num = int(os.getenv("CU_NUM", GFX_CU_NUM_MAP[gfx]))
        targets.append((gfx, cu_num))
    return targets


def filter_tune_df(tune_df, targets: list):
    """Return the subset of tune_df whose (gfx, cu_num) matches any entry in targets.

    Args:
        tune_df:  pandas DataFrame loaded from a tuning CSV (must have 'gfx' and
                  'cu_num' columns).
        targets:  list of (gfx, cu_num) tuples, as returned by get_build_targets()
                  or get_build_targets_env().

    Returns:
        Filtered DataFrame (original index preserved, no reset).
    """
    import pandas as pd

    mask = pd.Series([False] * len(tune_df), index=tune_df.index)
    for gfx, cu_num in targets:
        mask |= (tune_df["gfx"] == gfx) & (tune_df["cu_num"] == cu_num)
    return tune_df[mask]
