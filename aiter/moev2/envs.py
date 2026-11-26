# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Environment variables read by aiter.moev2, one accessor each.

Values are read once and cached, because they are consulted on every MoE call (including
during graph capture). Tests that change them call `reload()`.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
Please do not modify this code without coordinating with the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""

import functools
import os

AITER_MOEV2 = "AITER_MOEV2"

_OFF = frozenset({"off", "0", "false", "none"})
_ON = frozenset({"on", "1", "true", "all"})


@functools.cache
def moev2_selection():
    """`AITER_MOEV2`: which migration paths run on aiter.moev2.

    Unset or empty: the graduated paths (the default).
    `off`: none; every call runs the legacy code.
    `on`: every registered path, graduated or not.
    `a4w4,fhmoe`: exactly the listed paths.

    Returns "default", "off", "on", or a frozenset of path names.
    """
    raw = os.environ.get(AITER_MOEV2, "").strip().lower()
    if not raw:
        return "default"
    if raw in _OFF:
        return "off"
    if raw in _ON:
        return "on"
    names = frozenset(n.strip() for n in raw.split(",") if n.strip())
    return names or "default"


def reload():
    """Forget cached values so the next accessor call re-reads the environment."""
    moev2_selection.cache_clear()
    from . import paths

    paths.reload()
