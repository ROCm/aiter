# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which FlyDSL all-reduce schedule to run, as a function of payload size.

There are three schedules and the fastest one is a function of how many bytes
are being reduced:

* **one-shot** (``OneShotAllReduce``) -- one round, no grid-wide barrier, wire
  volume ``(N-1)*S``. Exact: fp32 accumulate, one bf16 rounding.
* **mesh** (``QuickAllReduceInt4(algorithm="mesh")``) -- two-shot, fanout to all
  ``N-1`` peers twice, wire volume ``2(N-1)/N*S``, INT4 on the wire.
* **ring** (``QuickAllReduceInt4(algorithm="ring")``) -- two-shot, ``2(N-1)``
  hops, same wire volume as the mesh, traded for per-destination locality.

They do not share a dispatcher. Each lives in the aiter slot whose accuracy
contract it already matches, and this module hands each slot its own view of
one shared table row:

* ``resolve_oneshot`` -> ``CustomAllreduce``, which is exact.
* ``resolve_quant``   -> ``QuickAllReduce``, which is allowed to quantize.
"""

from __future__ import annotations

import functools
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger("aiter")

# Fabric between GPUs.
LINKS = ("pcie", "xgmi")

SUPPORTED_WORLDS = (2, 4, 8)

# No ceiling. The ring's inbox is a fixed ring of wire slots sized by ``ST * grid``.
NO_MAX = 1 << 62


@dataclass(frozen=True)
class FamilyPolicy:
    """Family boundaries for one ``(link, world_size)``, in payload bytes.

    ``min_bytes`` is where this whole path starts being worth taking; below it
    the caller should fall through to other alternatives.

    The two one-shot ceilings are measured against *different* alternatives and
    so do not order against each other:

    * ``oneshot_max`` is where the quantized **mesh** overtakes the one-shot.
      It is the quick-reduce slot's exclusive floor.
    * ``oneshot_max_exact`` is where ``cross_device_reduce``/RCCL -- what the
      payload reaches if every FlyDSL family declines -- overtakes it. It is the
      custom-all-reduce slot's ceiling.
    """

    oneshot_max: int
    oneshot_max_exact: int
    mesh_max: int
    min_bytes: int = 0
    max_bytes: int = NO_MAX

    def __post_init__(self):
        if self.oneshot_max <= 0 or self.oneshot_max_exact <= 0:
            raise ValueError(
                f"oneshot_max ({self.oneshot_max}) and oneshot_max_exact "
                f"({self.oneshot_max_exact}) must be positive"
            )
        if self.mesh_max < self.oneshot_max:
            raise ValueError(
                f"mesh_max ({self.mesh_max}) must be >= oneshot_max "
                f"({self.oneshot_max}); the families partition by size"
            )


FAMILY_POLICY: dict[tuple[str, int], FamilyPolicy] = {
    # --- PCIe: Policy from measurements (on gfx950/MI350P) --------------------
    ("pcie", 2): FamilyPolicy(
        oneshot_max=512 << 10, oneshot_max_exact=1536 << 10, mesh_max=3 << 20
    ),
    ("pcie", 4): FamilyPolicy(
        oneshot_max=64 << 10, oneshot_max_exact=(160 << 10) - 1, mesh_max=8 << 20
    ),
    ("pcie", 8): FamilyPolicy(
        oneshot_max=16 << 10, oneshot_max_exact=(80 << 10) - 1, mesh_max=12 << 20
    ),
    # --- xGMI: Policy from measurements (on gfx942) --------------------
    #
    ("xgmi", 2): FamilyPolicy(
        oneshot_max=512 << 10,
        oneshot_max_exact=4 << 20,
        mesh_max=NO_MAX,
    ),
    ("xgmi", 4): FamilyPolicy(
        oneshot_max=512 << 10,
        oneshot_max_exact=(160 << 10) - 1,
        mesh_max=NO_MAX,
    ),
    ("xgmi", 8): FamilyPolicy(
        oneshot_max=256 << 10,
        oneshot_max_exact=256 << 10,
        mesh_max=NO_MAX,
    ),
}

# --- environment variables ---------------------------------------------------------

ENABLE_VAR = "AITER_FLY_AR"
ONESHOT_MAX_VAR = "AITER_FLY_AR_ONESHOT_MAX_BYTES"
MESH_MAX_VAR = "AITER_FLY_AR_MESH_MAX_BYTES"


def _env_int(name: str) -> int | None:
    """A non-negative override from *name*, or None. ``-1`` means "use the table"."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        val = int(raw)
    except ValueError:
        logger.warning("FlyDSL QR: ignoring %s=%r, expected an integer", name, raw)
        return None
    return None if val < 0 else val


def enabled() -> bool:
    """Whether ``AITER_FLY_AR`` opts in to the FlyDSL all-reduce path.

    Opt-in only, and only ``"1"`` opts in -- unset, ``"0"`` and anything else
    means disabled.
    """
    return os.environ.get(ENABLE_VAR, "").strip() == "1"


@functools.lru_cache(maxsize=1)
def detect_link() -> str:
    """``"xgmi"`` or ``"pcie"`` for this host, probed once per process.

    Host-wide, not per group; see ``has_xgmi_peer_links`` for the uniform-node
    assumption that makes that safe.
    """

    from .quick_allreduce_int4 import has_xgmi_peer_links

    return "xgmi" if has_xgmi_peer_links() else "pcie"


def _base(link: str, world_size: int) -> FamilyPolicy:
    if link not in LINKS:
        raise ValueError(f"link must be one of {LINKS}, got {link!r}")
    if world_size not in SUPPORTED_WORLDS:
        raise ValueError(
            f"world_size must be one of {SUPPORTED_WORLDS}, got {world_size}"
        )
    return FAMILY_POLICY[(link, int(world_size))]


def _oneshot_boundary(base: FamilyPolicy) -> int:
    """The measured one-shot/mesh crossover, ``ONESHOT_MAX_VAR`` applied."""

    override = _env_int(ONESHOT_MAX_VAR)
    return base.oneshot_max if override is None else override


@dataclass(frozen=True)
class OneShotPolicy:
    """The exact one-shot's window, as the custom-all-reduce slot sees it."""

    max_bytes: int
    min_bytes: int = 0


@dataclass(frozen=True)
class QuantPolicy:
    """The quantized families' window, as the quick-reduce slot sees it.

    ``floor`` is **exclusive**: dispatch only when ``nbytes > floor``. At or
    below it the exact one-shot is faster, and declining is what lets the
    payload reach the custom-all-reduce slot that hosts it.
    """

    floor: int
    mesh_max: int
    max_bytes: int


def resolve_oneshot(link: str, world_size: int) -> OneShotPolicy:
    """The one-shot's window for a rank, environment overrides applied."""

    base = _base(link, world_size)
    override = _env_int(ONESHOT_MAX_VAR)
    return OneShotPolicy(
        max_bytes=base.oneshot_max_exact if override is None else override,
        min_bytes=base.min_bytes,
    )


def resolve_quant(link: str, world_size: int) -> QuantPolicy:
    """The mesh/ring window for a rank, environment overrides applied."""

    base = _base(link, world_size)
    floor = _oneshot_boundary(base)
    mesh = base.mesh_max
    override_mesh = _env_int(MESH_MAX_VAR)
    if override_mesh is not None:
        mesh = override_mesh
    # The families partition by size; an override must not invert them.
    mesh = max(mesh, floor)
    return QuantPolicy(floor=floor, mesh_max=mesh, max_bytes=base.max_bytes)


def pick_quant_family(nbytes: int, policy: QuantPolicy) -> str:
    """``"mesh"`` | ``"ring"`` for a payload of *nbytes*.

    Assumes ``nbytes > policy.floor``; below that the caller should have
    declined so the exact one-shot gets the payload.
    """
    return "mesh" if nbytes <= policy.mesh_max else "ring"


def quant_families_reachable(policy: QuantPolicy) -> tuple[str, ...]:
    """Quantized families a *policy* can ever select, in size order."""
    out = []
    if policy.mesh_max > policy.floor:
        out.append("mesh")
    if policy.max_bytes > policy.mesh_max:
        out.append("ring")
    return tuple(out)
