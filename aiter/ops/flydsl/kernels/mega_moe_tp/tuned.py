# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tuned-config CSVs of MegaMoE TP (the small-batch engine's launch configs, the big-M
kernel's chunk plans): rows keyed by (gfx, cu_num, shape ...) plus a token count. A batch
takes the row of the nearest tuned token count at or below it; outside the tuned range the
small engine's configs fall back to the built-in defaults, the big-M plans (shape-level
options) to the nearest end of the range. Paths: AITER_CONFIG_MEGA_MOE_TP /
AITER_CONFIG_MEGA_MOE_TP_BIGM (aiter.jit.core)."""

from __future__ import annotations

import csv
import functools
import os


@functools.cache
def _read(path: str, mtime: float) -> tuple:
    if not os.path.isfile(path):
        return ()
    with open(path) as f:
        return tuple(csv.DictReader(f))


def tuned_rows(path: str, key: dict, tokens_col: str) -> dict:
    """{tokens: row} of the rows of path matching every key column (compared as str)."""
    path = os.path.abspath(path)
    mtime = os.path.getmtime(path) if os.path.isfile(path) else 0.0
    want = {k: str(v) for k, v in key.items()}
    return {
        int(r[tokens_col]): r
        for r in _read(path, mtime)
        if all(r.get(k) == v for k, v in want.items())
    }


def nearest(rows: dict, tokens: int, clamp: bool = False):
    """The row of the nearest tuned count <= tokens inside [min, max] of rows; outside it
    None, or (clamp) the row of the nearer end."""
    if not rows:
        return None
    if not min(rows) <= tokens <= max(rows):
        if not clamp:
            return None
        tokens = min(max(tokens, min(rows)), max(rows))
    return rows[max(t for t in rows if t <= tokens)]


def device_key(device=None) -> dict:
    """gfx / cu_num of a device (the CSVs' leading key columns)."""
    import torch

    from aiter.jit.utils.chip_info import get_gfx

    props = torch.cuda.get_device_properties(device or torch.cuda.current_device())
    return {"gfx": get_gfx(), "cu_num": int(props.multi_processor_count)}
