# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""When the dispatcher routes an all-to-all to ``FlyQuickAllToAll``, and how.

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

Measured with ``bench_comm.py --operation a2a`` on MI350P (8x gfx950, PCIe),
bf16, 128 KiB-64 MiB. The lossless mesh beats RCCL at every size measured
at TP4 and TP8 (1.4-8x) and ties it at TP2 from 16 MiB up; the mesh beats the
ring everywhere except TP4 from 4 MiB, where the ring is 9-16% faster. xGMI is
unmeasured: it gets the mesh, the schedule built for a meshed fabric.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from .allreduce_policy import _env_int, detect_link

logger = logging.getLogger("aiter")

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


def enabled() -> bool:
    """Whether ``AITER_FLY_A2A=1``. Only ``"1"`` opts in."""
    return os.environ.get(ENABLE_VAR, "").strip() == "1"


def _env_choice(name: str, choices: tuple[str, ...], default: str) -> str:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw not in choices:
        logger.warning(
            "FlyDSL A2A: ignoring %s=%r, expected one of %s", name, raw, choices
        )
        return default
    return raw


def resolve(world_size: int, link: str | None = None) -> AllToAllPolicy:
    """The policy for a *world_size* group on this host, from the environment."""
    link = detect_link() if link is None else link
    min_bytes = _env_int(MIN_BYTES_VAR)
    max_bytes = _env_int(MAX_BYTES_VAR)
    ring_from = _env_int(RING_MIN_BYTES_VAR)
    return AllToAllPolicy(
        link=link,
        algorithm=_env_choice(ALGORITHM_VAR, ALGORITHMS, "auto"),
        codec=_env_choice(CODEC_VAR, CODECS, "none"),
        min_bytes=_DEFAULT_MIN_BYTES if min_bytes is None else min_bytes,
        max_bytes=_DEFAULT_MAX_BYTES if max_bytes is None else max_bytes,
        ring_from=(
            RING_FROM_BYTES.get((link, int(world_size)))
            if ring_from is None
            else ring_from
        ),
    )
