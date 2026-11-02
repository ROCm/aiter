# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The limits, the tuning ladders and the dispatch policy of the FlyDSL
all-to-all (``FlyQuickAllToAll``).

The kernels in ``kernels/comm/quick_alltoall_{mesh,ring}.py`` take their limits
from here, as the all-reduce kernels take theirs from ``allreduce_policy``.

Opt-in, like the FlyDSL all-reduce: an engine holds an IPC inbox per process
group, and a quantizing codec changes numerics. Environment:

* ``AITER_FLY_A2A=1`` -- enable. Anything else (or unset) leaves every
  all-to-all on RCCL.
* ``AITER_FLY_A2A_CODEC`` -- wire format: ``none`` (default, lossless), ``int4``
  or ``int6``. A quantizing codec is the explicit opt-in to lossy transport.
* ``AITER_FLY_A2A_ALGORITHM`` -- ``auto`` (default: per payload, see
  ``RING_FROM_BYTES``), ``mesh`` or ``ring``.
* ``AITER_FLY_A2A_MIN_BYTES`` / ``AITER_FLY_A2A_MAX_BYTES`` -- the payload
  window, per-rank input bytes; outside it the call falls through to RCCL.
* ``AITER_FLY_A2A_RING_MIN_BYTES`` -- under ``auto``, where the shifted-pairwise
  ring takes over from the mesh; overrides ``RING_FROM_BYTES``.

Measured on MI350P (8x gfx950, PCIe), bf16, 128 KiB-64 MiB; the ``bench``
scenario of ``test_flydsl_quick_alltoall.py`` repeats the measurement. The
lossless mesh beats RCCL at every size measured at TP4 and TP8 (1.4-8x) and
ties it at TP2 from 16 MiB up; the mesh beats the ring everywhere except TP4
from 4 MiB, where the ring is 9-16% faster. xGMI is unmeasured: it gets the
mesh, the schedule built for a meshed fabric.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from .allreduce_policy import (
    MIN_BATCH_BLOCKS,
    SUPPORTED_ARCHS,
    SUPPORTED_BLOCKS,
    SUPPORTED_WORLDS,
    TwoShotRung,
    detect_link,
)

logger = logging.getLogger("aiter")

__all__ = [
    "A2A_CODECS",
    "A2A_DIRECT_ORDERS",
    "A2A_MESH_SUPER_TILES",
    "A2A_QUANT_CODECS",
    "A2A_RING_SUPER_TILES",
    "DEFAULT_GRID_CAP",
    "MESH_ST_LADDER",
    "MIN_BATCH_BLOCKS",
    "RING_ST_LADDER",
    "SUPPORTED_ARCHS",
    "SUPPORTED_BLOCKS",
    "SUPPORTED_WORLDS",
    "AllToAllPolicy",
    "enabled",
    "mesh_st_ladder",
    "resolve",
    "ring_st_ladder",
]

# Limits
# ======

# Wire formats. "none" is register-direct; the rest go through LDS staging.
# "fp16" is lossless and isolates the transport from the codec in tests.
A2A_CODECS = ("none", "fp16", "int4", "int6")
# Codecs that quantize, i.e. need bf16 in and the fp16 math path.
A2A_QUANT_CODECS = ("int4", "int6")
# Store order of the mesh's register-direct path: one destination's atoms back
# to back ("peer"), or each atom to every destination in turn ("atom").
A2A_DIRECT_ORDERS = ("peer", "atom")
A2A_MESH_SUPER_TILES = (1, 8)
# One publish per step per group, ``N-1`` per group: a bigger super-tile is the
# knob that amortizes them, as in the all-reduce ring.
A2A_RING_SUPER_TILES = (1, 8, 16, 32)
# The grid cap of an engine built without a ladder: four blocks per CU on a
# 304-CU part, clamped to the device's residency at build time.
DEFAULT_GRID_CAP = 304 * 4

# Tuning ladders
# ==============
#
# ``(link, world_size)`` -> rungs, ascending by ``min_bytes``, the whole input
# of one rank. Seeded from the all-reduce ladders; to be refitted with
# ``test_flydsl_quick_alltoall.py --scenario bench``.

MESH_ST_LADDER = {
    (link, ws): (TwoShotRung(0, 1, 128, 256), TwoShotRung(4 << 20, 8, 128, 512))
    for link in ("xgmi", "pcie")
    for ws in SUPPORTED_WORLDS
}

RING_ST_LADDER = {
    (link, ws): (TwoShotRung(0, 8, 128, 256), TwoShotRung(16 << 20, 8, 128, 512))
    for link in ("xgmi", "pcie")
    for ws in SUPPORTED_WORLDS
}


def mesh_st_ladder(world_size: int, link: str = "pcie") -> tuple[TwoShotRung, ...]:
    """Mesh rungs for *(link, world_size)*, or ``()`` when there is no ladder."""
    return MESH_ST_LADDER.get((str(link), int(world_size)), ())


def ring_st_ladder(world_size: int, link: str = "pcie") -> tuple[TwoShotRung, ...]:
    """Ring rungs for *(link, world_size)*, or ``()`` when there is no ladder."""
    return RING_ST_LADDER.get((str(link), int(world_size)), ())


# Dispatch
# ========

ENABLE_VAR = "AITER_FLY_A2A"
CODEC_VAR = "AITER_FLY_A2A_CODEC"
ALGORITHM_VAR = "AITER_FLY_A2A_ALGORITHM"
MIN_BYTES_VAR = "AITER_FLY_A2A_MIN_BYTES"
MAX_BYTES_VAR = "AITER_FLY_A2A_MAX_BYTES"
RING_MIN_BYTES_VAR = "AITER_FLY_A2A_RING_MIN_BYTES"

CODECS = ("none", "int4", "int6")
ALGORITHMS = ("auto", "mesh", "ring")
NO_MAX = 1 << 62

# Default window. FlyDSL wins at the smallest size measured, on every world
# size, so there is no floor; the slot itself is opt-in.
_DEFAULT_MIN_BYTES = 0
_DEFAULT_MAX_BYTES = NO_MAX

# ``(link, world_size)`` -> payload bytes from which ``auto`` takes the ring.
# Absent: the mesh at every size.
RING_FROM_BYTES = {("pcie", 4): 4 << 20}


@dataclass(frozen=True)
class AllToAllPolicy:
    link: str
    algorithm: str
    codec: str
    min_bytes: int
    max_bytes: int
    # Under ``auto``: where the ring takes over from the mesh, or None for never.
    ring_from: int | None = None

    def routes(self, nbytes: int) -> bool:
        """Whether a payload of *nbytes* belongs to FlyDSL at all."""
        return self.min_bytes <= int(nbytes) <= self.max_bytes

    def families(self) -> tuple[str, ...]:
        """Schedules this policy can select, so their engines are built up front."""
        if self.algorithm != "auto":
            return (self.algorithm,)
        return ("mesh",) if self.ring_from is None else ("mesh", "ring")

    def pick(self, nbytes: int) -> str:
        """The schedule an *nbytes* payload runs."""
        if self.algorithm != "auto":
            return self.algorithm
        if self.ring_from is not None and int(nbytes) >= self.ring_from:
            return "ring"
        return "mesh"

    def family_range(self, family: str) -> tuple[int, int]:
        """Inclusive payload range :meth:`pick` routes to *family*, inside the
        window -- what that engine needs compiled ahead of graph capture."""
        lo, hi = self.min_bytes, min(self.max_bytes, 1 << 40)
        if self.algorithm == "auto" and self.ring_from is not None:
            if family == "mesh":
                hi = min(hi, self.ring_from - 1)
            else:
                lo = max(lo, self.ring_from)
        return lo, hi


def _env_int(name: str) -> int | None:
    """A non-negative override from *name*, or None. ``-1`` means "use the table"."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        val = int(raw)
    except ValueError:
        logger.warning(
            "FlyDSL all-to-all: ignoring %s=%r, expected an integer", name, raw
        )
        return None
    return None if val < 0 else val


def enabled() -> bool:
    """Whether ``AITER_FLY_A2A=1``. Only ``"1"`` opts in."""
    return os.environ.get(ENABLE_VAR, "").strip() == "1"


def _env_choice(name: str, choices: tuple[str, ...], default: str) -> str:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw not in choices:
        logger.warning(
            "FlyDSL all-to-all: ignoring %s=%r, expected one of %s", name, raw, choices
        )
        return default
    return raw


def resolve(
    world_size: int, link: str | None = None, codec: str | None = None
) -> AllToAllPolicy:
    """The policy for a *world_size* group on this host, from the environment.

    *codec*, when given, replaces ``AITER_FLY_A2A_CODEC``: a caller whose payload
    must arrive bit-exact pins ``"none"`` so the environment cannot quantize it.
    """
    link = detect_link() if link is None else link
    if codec is not None and codec not in CODECS:
        raise ValueError(f"codec must be one of {CODECS}, got {codec!r}")
    min_bytes = _env_int(MIN_BYTES_VAR)
    max_bytes = _env_int(MAX_BYTES_VAR)
    ring_from = _env_int(RING_MIN_BYTES_VAR)
    return AllToAllPolicy(
        link=link,
        algorithm=_env_choice(ALGORITHM_VAR, ALGORITHMS, "auto"),
        codec=_env_choice(CODEC_VAR, CODECS, "none") if codec is None else codec,
        min_bytes=_DEFAULT_MIN_BYTES if min_bytes is None else min_bytes,
        max_bytes=_DEFAULT_MAX_BYTES if max_bytes is None else max_bytes,
        ring_from=(
            RING_FROM_BYTES.get((link, int(world_size)))
            if ring_from is None
            else ring_from
        ),
    )
