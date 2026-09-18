# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Pure-Python arch constants and env-driven build target resolution.
# No torch dependency — safe to import in build scripts, gen_instances, and tests
# that run without a GPU or a full PyTorch install.
import os
import re

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
    "gfx1250": 256,  # Gfx1250
}

# Valid arch names; those without a GFX_CU_NUM_MAP default must name their CU count.
KNOWN_GFX = frozenset(v for v in GFX_MAP.values() if v != "native")


def _parse_gpu_archs_env(gfx_env: str) -> list[str]:
    """Split a GPU_ARCHS string into a list of non-empty architecture names.

    Raises RuntimeError if no valid architecture names remain after splitting
    on ';' and stripping whitespace — e.g. GPU_ARCHS=" ; " would otherwise
    silently produce an empty target list and fall back to heuristic kernels.
    """
    archs = [g.strip().lower() for g in gfx_env.split(";") if g.strip()]
    if not archs:
        raise RuntimeError(
            f"GPU_ARCHS={gfx_env!r} contains no valid architecture names after splitting on ';'. "
            f"Known targets: {list(GFX_CU_NUM_MAP.keys())}"
        )
    return archs


def _cu_num_or_none(value: str) -> int | None:
    """The positive CU count in value, or None when it is not one."""
    try:
        cu_num = int(value)
    except (TypeError, ValueError):
        return None
    return cu_num if cu_num > 0 else None


def _parse_gpu_targets_env() -> list[tuple[str, int]] | None:
    """Parse AITER_GPU_TARGETS into (gfx, cu_num) targets, or None if it is unset.

    gfx950:128;gfx950:256  -> [("gfx950", 128), ("gfx950", 256)]
    gfx950                 -> [("gfx950", 256)]  # CU from GFX_CU_NUM_MAP
    gfx942;gfx950:128      -> [("gfx942", 304), ("gfx950", 128)]
    (unset or blank)       -> None

    Entries split on ';' or ','; arch names are case-folded and validated,
    CU counts must be positive integers.
    """
    targets_env = os.getenv("AITER_GPU_TARGETS")
    if not targets_env or not targets_env.strip():
        return None

    targets = []
    for entry in re.split(r"[;,]", targets_env):
        entry = entry.strip()
        if not entry:
            continue
        where = f"AITER_GPU_TARGETS entry {entry!r}"
        gfx, sep, cu = entry.partition(":")
        gfx = gfx.strip().lower()
        if gfx not in KNOWN_GFX:
            raise RuntimeError(
                f"{where}: unknown gfx {gfx!r}. Known targets: {sorted(KNOWN_GFX)}"
            )
        if sep:
            cu_num = _cu_num_or_none(cu.strip())
            if cu_num is None:
                raise RuntimeError(
                    f"{where}: CU count {cu.strip()!r} must be a positive integer."
                )
            targets.append((gfx, cu_num))
        elif gfx in GFX_CU_NUM_MAP:
            targets.append((gfx, GFX_CU_NUM_MAP[gfx]))
        else:
            raise RuntimeError(
                f"{where}: {gfx!r} has no default CU count — add it to "
                f"GFX_CU_NUM_MAP in build_targets.py, or name the count "
                f"explicitly as '{gfx}:<cu_num>'."
            )

    if not targets:
        raise RuntimeError(
            f"AITER_GPU_TARGETS={targets_env!r} names no targets. "
            f"Expected entries of the form "
            f"'gfx' or 'gfx:cu_num'."
        )

    # Preserve caller order, drop exact duplicates.
    return list(dict.fromkeys(targets))


def get_build_archs_env() -> list[str] | None:
    """Deduped arch names from AITER_GPU_TARGETS, or None if it is unset."""
    targets = _parse_gpu_targets_env()
    if targets is None:
        return None
    return list(dict.fromkeys(gfx for gfx, _ in targets))


def gpu_archs_env_names() -> list[str]:
    """Arch names GPU_ARCHS explicitly requests, empty when unset or native."""
    archs = _parse_gpu_archs_env(os.getenv("GPU_ARCHS") or "native")
    return [a for a in archs if a != "native"]


def get_build_targets_env() -> list[tuple[str, int]]:
    """Resolve build targets from env only. No live GPU detection.

    AITER_GPU_TARGETS, when set, overrides GPU_ARCHS + CU_NUM. Raises
    RuntimeError if neither is set or an arch is unknown. Use
    chip_info.get_build_targets() when live-GPU fallback is also desired.

    AITER_GPU_TARGETS=gfx950:128;gfx950:256 -> [("gfx950", 128), ("gfx950", 256)]
    GPU_ARCHS=gfx942;gfx950                 -> [("gfx942", 304), ("gfx950", 256)]
    GPU_ARCHS=gfx942 CU_NUM=80              -> [("gfx942", 80)]
    """
    targets = _parse_gpu_targets_env()
    if targets is not None:
        return targets

    gfx_env = os.getenv("GPU_ARCHS")
    if not gfx_env:
        raise RuntimeError(
            "Neither AITER_GPU_TARGETS nor GPU_ARCHS is set. "
            "Set GPU_ARCHS=gfx942 (or similar) to resolve build targets without a GPU."
        )
    cu_env = os.getenv("CU_NUM")
    cu_override = None
    if cu_env:
        cu_override = _cu_num_or_none(cu_env)
        if cu_override is None:
            raise RuntimeError(f"CU_NUM={cu_env!r} must be a positive integer.")

    targets = []
    for gfx in _parse_gpu_archs_env(gfx_env):
        if gfx not in GFX_CU_NUM_MAP:
            raise RuntimeError(
                f"Unknown gfx '{gfx}' in GPU_ARCHS — add it to "
                f"GFX_CU_NUM_MAP in build_targets.py. Known targets: "
                f"{list(GFX_CU_NUM_MAP.keys())}"
            )
        targets.append((gfx, cu_override or GFX_CU_NUM_MAP[gfx]))
    return list(dict.fromkeys(targets))


def _target_mask(tune_df, gfx: str, cu_num: int):
    return (tune_df["gfx"] == gfx) & (tune_df["cu_num"] == cu_num)


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
        mask |= _target_mask(tune_df, gfx, cu_num)
    return tune_df[mask]


def unmatched_targets(tune_df, targets: list) -> list[str]:
    """The targets with no row in tune_df, as 'gfx:cu_num' strings."""
    return [
        f"{gfx}:{cu_num}"
        for gfx, cu_num in targets
        if not _target_mask(tune_df, gfx, cu_num).any()
    ]
