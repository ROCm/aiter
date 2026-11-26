# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Debug overrides of dispatch decisions, read where the legacy code reads them.

Each is read when a dispatch decision is made (not cached here): decisions are cached per
problem, so this matches the legacy reads inside the cached `get_2stage_cfgs`.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
Please do not modify this code without coordinating with the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""

import os


def bypass_tuned() -> int:
    """`AITER_BYPASS_TUNE_CONFIG`: ignore tuned rows; use the default heuristics."""
    return int(os.environ.get("AITER_BYPASS_TUNE_CONFIG", "0"))


def ksplit_disabled() -> bool:
    """`AITER_KSPLIT=-1`: run tuned rows with ksplit 0."""
    return int(os.environ.get("AITER_KSPLIT", "0")) == -1


def nt_override() -> int:
    """`AITER_USE_NT`: -1 (default) keeps the tuned `nt` column, 0/1 forces it."""
    return int(os.environ.get("AITER_USE_NT", "-1"))


def online_tune() -> bool:
    """`AITER_ONLINE_TUNE=1`: tune missing keys on the fly (legacy code only, for now)."""
    return os.environ.get("AITER_ONLINE_TUNE", "0") == "1"


def explain() -> bool:
    """`AITER_MOE_EXPLAIN=1`: log how each new problem is resolved, and why an active
    path does not claim a call (once per path and reason)."""
    return os.environ.get("AITER_MOE_EXPLAIN", "0") == "1"
