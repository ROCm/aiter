# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Every tuned decision of the FlyDSL all-reduce: which kernel runs a payload,
and in which configuration.

Kernels
=======

* **one-shot** (``OneShotAllReduce``) -- one round, no grid-wide barrier, wire
  volume ``(N-1)*S``. Exact: fp32 accumulate, one bf16 rounding.
* **mesh** (``QuickAllReduceInt4(algorithm="mesh")``) -- two-shot, fanout to all
  ``N-1`` peers twice, wire volume ``2(N-1)/N*S``, INT4 on the wire.
* **ring** (``QuickAllReduceInt4(algorithm="ring")``) -- two-shot, ``2(N-1)``
  hops, same wire volume as the mesh, traded for per-destination locality.

The decisions come in two levels.

Level 1: which kernel
=====================

The kernel is a function of the payload size, and the boundaries are a
``FamilyPolicy`` row per ``(link, world_size)`` cell: ``FAMILY_POLICY``, or
``ARCH_FAMILY_POLICY`` on an arch with its own measurements. On the payload
axis, a row reads::

    exact (quick reduce off):
      0 --- one-shot --- oneshot_max --- cross_device_reduce / RCCL ------->

    quantized (AITER_QUICK_REDUCE_QUANTIZATION=INT4):
      0 --- one-shot --- oneshot_max --- cdr --- quant_min --- mesh
                                                     --- mesh_max --- ring --->

The kernels do not share a dispatcher. Each lives in the aiter slot whose
accuracy contract it matches, and this module gives each slot its own view of
the row:

* ``resolve_oneshot`` -> ``CustomAllreduce``, which is exact: the one-shot
  window.
* ``resolve_quant`` -> ``QuickAllReduce``, which can quantize: the mesh and
  ring windows.

``CudaCommunicator.all_reduce`` tries ``QuickAllReduce`` first, so a payload
above ``quant_min`` goes to the mesh or the ring, and the rest go to
``CustomAllreduce``. ``route`` is that composition in one function.
``cross_device_reduce`` can be faster than both FlyDSL families, so
``quant_min`` can be above ``oneshot_max``: the payloads between the two then
go to ``cross_device_reduce`` and stay exact.

Level 2: which configuration
============================

Each kernel has a tuning ladder per ``(link, world_size)``: ``ONESHOT_LADDER``,
``MESH_ST_LADDER`` and ``RING_ST_LADDER``. A ladder is a tuple of rungs in
ascending ``min_bytes`` order, and a payload runs on the last rung whose
``min_bytes`` is at most its size (``pick_rung``). The host builds one engine
per rung. The mesh and ring hosts then refine the super-tile at launch, from
the tile count of the payload; refer to ``QuickAllReduceInt4._pick_cfg``.
"""

from __future__ import annotations

import functools
import logging
import os

from .kernels.comm.allreduce_shared import (
    BLOCK,
    LINKS,
    NO_MAX,
    SUPPORTED_ARCHS,  # noqa: F401  -- re-exported for the dispatchers
    SUPPORTED_WORLDS,
    FamilyPolicy,
    OneShotPolicy,
    OneShotRung,
    QuantPolicy,
    TwoShotRung,
)
from .kernels.comm.one_shot_allreduce import (
    DEFAULT_ATOMS,
    DEFAULT_BLOCK,
    DEFAULT_FANOUT,
    DEFAULT_GRID_CAP,
)

logger = logging.getLogger("aiter")


# Level 1: kernel boundaries
# ==========================

FAMILY_POLICY: dict[tuple[str, int], FamilyPolicy] = {
    # --- PCIe: measured on gfx950 (MI350P) -----------------------------------
    ("pcie", 2): FamilyPolicy(
        oneshot_max=64 << 20,
        quant_min=384 << 10,
        mesh_max=NO_MAX,
    ),
    ("pcie", 4): FamilyPolicy(
        oneshot_max=1 << 20,
        quant_min=128 << 10,
        mesh_max=16 << 20,
    ),
    ("pcie", 8): FamilyPolicy(
        oneshot_max=1 << 20,
        quant_min=96 << 10,
        mesh_max=24 << 20,
    ),
    # --- xGMI: measured on gfx942 (MI300X) -----------------------------------
    ("xgmi", 2): FamilyPolicy(
        oneshot_max=1536 << 10,
        quant_min=384 << 10,
        mesh_max=128 << 20,
    ),
    ("xgmi", 4): FamilyPolicy(
        oneshot_max=384 << 10,
        quant_min=384 << 10,
        mesh_max=128 << 20,
    ),
    ("xgmi", 8): FamilyPolicy(
        oneshot_max=168 << 10,
        quant_min=256 << 10,
        mesh_max=128 << 20,
    ),
}

# Rows measured on one arch, keyed ``(arch, link, world_size)``. On that arch
# they replace the ``FAMILY_POLICY`` row for the same ``(link, world_size)``.
#
# gfx950/xGMI (MI355X), measured with the ``bench`` scenario of the FlyDSL
# all-reduce op tests:
#
# * The ring never wins: it is 1.03x-1.16x slower than the mesh at TP2 up to
#   512 MiB, and 2.2x-5.7x slower at TP4 and TP8. ``mesh_max`` is unbounded.
# * ``oneshot_max`` is where ``cross_device_reduce`` overtakes the one-shot.
#   At TP4 and TP8 that is where it switches to its 2-stage kernel (160 KiB
#   and 80 KiB), which is fast on this fabric.
# * ``quant_min`` is where the mesh overtakes the fastest exact kernel. At TP4
#   and TP8 that kernel is ``cross_device_reduce``'s 2-stage, so the mesh
#   starts well above the one-shot ceiling.
ARCH_FAMILY_POLICY: dict[tuple[str, str, int], FamilyPolicy] = {
    ("gfx950", "xgmi", 2): FamilyPolicy(
        oneshot_max=3 << 20,
        quant_min=448 << 10,
        mesh_max=NO_MAX,
    ),
    ("gfx950", "xgmi", 4): FamilyPolicy(
        # Every payload below the 160 KiB 2-stage switch: payloads are whole
        # multiples of 16 B.
        oneshot_max=(160 << 10) - 16,
        quant_min=960 << 10,
        mesh_max=NO_MAX,
    ),
    ("gfx950", "xgmi", 8): FamilyPolicy(
        oneshot_max=96 << 10,
        quant_min=1600 << 10,
        mesh_max=NO_MAX,
    ),
}


# Level 2: configuration ladders
# ==============================
#
# The ladders were measured on gfx942 for xGMI and on gfx950 for PCIe.

# One-shot, ``OneShotRung``.
#
#   PCIe
#
#     TP2  b256 a1 cap128 -- the 4 KiB tile to 384 KiB, then b512 a4 cap128,
#                            the 32 KiB tile, over the rest of the exact-mode
#                            window, which runs to 64 MiB.
#     TP4  b128 a1 cap128 -- the 2 KiB tile to 96 KiB, then b512 a2 cap64, the
#                            16 KiB tile, to the 1 MiB exact-mode ceiling.
#     TP8  b64 a1 cap64   -- the 1 KiB tile, except b256 a1 cap64, the 4 KiB
#                            tile, from 48 to 192 KiB.
#
#   xGMI
#
#     TP2  atoms=1 cap128 b128/b256 -- the 2 KiB tile to 256 KiB, the most it
#                               covers in one round, then the 4 KiB tile.
#     TP4  atoms=1 cap128 b256 -- one rung over the whole window
#     TP8  atoms=1 cap128 b128 -- one rung over the whole window
ONESHOT_LADDER: dict[tuple[str, int], tuple[OneShotRung, ...]] = {
    ("pcie", 2): (
        OneShotRung(min_bytes=0, atoms=1, grid_cap=128, fanout="peer", block=256),
        OneShotRung(
            min_bytes=384 << 10, atoms=4, grid_cap=128, fanout="peer", block=512
        ),
    ),
    ("pcie", 4): (
        OneShotRung(min_bytes=0, atoms=1, grid_cap=128, fanout="peer", block=128),
        OneShotRung(min_bytes=96 << 10, atoms=2, grid_cap=64, fanout="peer", block=512),
    ),
    ("pcie", 8): (
        OneShotRung(min_bytes=0, atoms=1, grid_cap=64, fanout="peer", block=64),
        OneShotRung(min_bytes=48 << 10, atoms=1, grid_cap=64, fanout="peer", block=256),
        OneShotRung(min_bytes=192 << 10, atoms=1, grid_cap=64, fanout="peer", block=64),
    ),
    ("xgmi", 2): (
        OneShotRung(min_bytes=0, atoms=1, grid_cap=128, fanout="peer", block=128),
        OneShotRung(
            min_bytes=256 << 10, atoms=1, grid_cap=128, fanout="peer", block=256
        ),
    ),
    ("xgmi", 4): (
        OneShotRung(min_bytes=0, atoms=1, grid_cap=128, fanout="peer", block=256),
    ),
    ("xgmi", 8): (
        OneShotRung(min_bytes=0, atoms=1, grid_cap=128, fanout="peer", block=128),
    ),
}

# Mesh, ``TwoShotRung``.
#
#   xGMI  ST=1 everywhere, except TP4 below 4 MiB and TP8 from 6 MiB (ST=8).
#         Block 512 from 4 MiB.
#   PCIe  TP2 ST=1, then ST=8 from 4 MiB. TP4 ST=1, then ST=8 from 768 KiB.
#         TP8 ST=8 everywhere.
MESH_ST_LADDER: dict[tuple[str, int], tuple[TwoShotRung, ...]] = {
    ("xgmi", 2): (
        TwoShotRung(min_bytes=0, super_tile=1, grid_cap=128, block=256),
        TwoShotRung(min_bytes=4 << 20, super_tile=1, grid_cap=128, block=512),
    ),
    ("xgmi", 4): (
        TwoShotRung(min_bytes=0, super_tile=8, grid_cap=128, block=256),
        TwoShotRung(min_bytes=4 << 20, super_tile=1, grid_cap=128, block=512),
    ),
    ("xgmi", 8): (
        TwoShotRung(min_bytes=0, super_tile=1, grid_cap=128, block=256),
        TwoShotRung(min_bytes=4 << 20, super_tile=1, grid_cap=128, block=512),
        TwoShotRung(min_bytes=6 << 20, super_tile=8, grid_cap=128, block=512),
    ),
    ("pcie", 2): (
        TwoShotRung(min_bytes=0, super_tile=1, grid_cap=128, block=256),
        TwoShotRung(min_bytes=4 << 20, super_tile=8, grid_cap=128, block=512),
    ),
    ("pcie", 4): (
        TwoShotRung(min_bytes=0, super_tile=1, grid_cap=128, block=256),
        TwoShotRung(min_bytes=768 << 10, super_tile=8, grid_cap=128, block=512),
    ),
    ("pcie", 8): (TwoShotRung(min_bytes=0, super_tile=8, grid_cap=128, block=512),),
}

# Ring, ``TwoShotRung``. The ring publishes ``2(N-1)`` times per super-tile, so
# a larger payload wants a larger super-tile.
RING_ST_LADDER: dict[tuple[str, int], tuple[TwoShotRung, ...]] = {
    ("xgmi", 2): (
        TwoShotRung(min_bytes=0, super_tile=8, grid_cap=128, block=BLOCK),
        TwoShotRung(min_bytes=24 << 20, super_tile=16, grid_cap=128, block=BLOCK),
    ),
    ("xgmi", 4): (
        TwoShotRung(min_bytes=0, super_tile=8, grid_cap=128, block=BLOCK),
        TwoShotRung(min_bytes=24 << 20, super_tile=16, grid_cap=128, block=BLOCK),
        TwoShotRung(min_bytes=48 << 20, super_tile=32, grid_cap=128, block=BLOCK),
    ),
    ("xgmi", 8): (
        TwoShotRung(min_bytes=0, super_tile=16, grid_cap=128, block=BLOCK),
        TwoShotRung(min_bytes=48 << 20, super_tile=32, grid_cap=128, block=BLOCK),
    ),
    ("pcie", 2): (TwoShotRung(min_bytes=0, super_tile=16, grid_cap=128, block=512),),
    ("pcie", 4): (TwoShotRung(min_bytes=0, super_tile=32, grid_cap=128, block=512),),
    ("pcie", 8): (TwoShotRung(min_bytes=0, super_tile=32, grid_cap=128, block=512),),
}


def oneshot_ladder(world_size: int, link: str = "pcie") -> tuple[OneShotRung, ...]:
    """Rungs for *(link, world_size)*, or a single default rung if unlisted.

    *link* defaults to ``"pcie"`` so a caller that has not resolved the fabric
    gets the conservative table: its fatter tiles cost throughput on xGMI but
    are never wrong in the sense of failing.
    """
    default = OneShotRung(
        min_bytes=0,
        atoms=DEFAULT_ATOMS,
        grid_cap=DEFAULT_GRID_CAP,
        fanout=DEFAULT_FANOUT,
        block=DEFAULT_BLOCK,
    )
    return ONESHOT_LADDER.get((str(link), int(world_size)), (default,))


def mesh_st_ladder(world_size: int, link: str = "pcie") -> tuple[TwoShotRung, ...]:
    """Rungs for *(link, world_size)*, or ``()`` when there is no ladder."""
    return MESH_ST_LADDER.get((str(link), int(world_size)), ())


def ring_st_ladder(world_size: int, link: str = "pcie") -> tuple[TwoShotRung, ...]:
    """Rungs for *(link, world_size)*, or the PCIe TP4 shape for an unlisted one."""
    return RING_ST_LADDER.get((str(link), int(world_size)), RING_ST_LADDER[("pcie", 4)])


# Floor on the block count of the mesh and the ring when they batch publishes
# into super-tiles; refer to ``QuickAllReduceInt4._grid_x``. Shrinking the grid
# trades parallelism for fewer release fences, which is only a good trade once
# there are enough fences to matter.
MIN_BATCH_BLOCKS = 32

# World size from which the ring's reduce-scatter lap defaults to INT6, to
# clear the 18 dB SQNR floor the schedules are held to. The ring requantizes
# the running partial at every hop, and the partial's extremum grows with the
# contributions folded in, so unlike the mesh its error grows with N.
RS_INT6_MIN_WORLD = 8


# Host and environment
# ====================


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
    return os.environ.get("AITER_FLY_AR", "").strip() == "1"


@functools.lru_cache(maxsize=1)
def detect_arch() -> str | None:
    """This host's ``gfx`` arch, or None if it cannot be read."""
    from aiter.jit.utils.chip_info import get_gfx_runtime

    try:
        return get_gfx_runtime()
    except (KeyError, RuntimeError):
        return None


def resolve_link(link: str | None) -> str:
    """*link*, or this host's when it is None. Raises on an unknown link."""
    if link is None:
        link = detect_link()
    if link not in LINKS:
        raise ValueError(f"link must be one of {LINKS}, got {link!r}")
    return link


@functools.lru_cache(maxsize=1)
def detect_link() -> str:
    """``"xgmi"`` or ``"pcie"`` for this host, probed once per process.

    Host-wide, not per group; see ``has_xgmi_peer_links`` for the uniform-node
    assumption that makes that safe.
    """

    from .allreduce_shared import has_xgmi_peer_links

    return "xgmi" if has_xgmi_peer_links() else "pcie"


# Level 1: resolution
# ===================


def family_policy(link: str, world_size: int, arch: str | None = None) -> FamilyPolicy:
    """The table row for *(arch, link, world_size)*, without overrides.

    *arch* defaults to this host's. An arch with no row of its own for the
    cell gets the ``FAMILY_POLICY`` row.
    """
    if link not in LINKS:
        raise ValueError(f"link must be one of {LINKS}, got {link!r}")
    if world_size not in SUPPORTED_WORLDS:
        raise ValueError(
            f"world_size must be one of {SUPPORTED_WORLDS}, got {world_size}"
        )
    if arch is None:
        arch = detect_arch()
    row = ARCH_FAMILY_POLICY.get((arch, link, int(world_size)))
    return FAMILY_POLICY[(link, int(world_size))] if row is None else row


def resolve_oneshot(
    link: str, world_size: int, arch: str | None = None
) -> OneShotPolicy:
    """The one-shot's window for a rank, environment overrides applied.

    *arch* defaults to this host's; refer to ``family_policy``.

    ``AITER_FLY_AR_ONESHOT_MIN_BYTES`` overrides ``oneshot_min`` and
    ``AITER_FLY_AR_ONESHOT_MAX_BYTES`` overrides ``oneshot_max``. ``-1`` on
    either means "use the table".
    """
    row = family_policy(link, world_size, arch)
    max_override = _env_int("AITER_FLY_AR_ONESHOT_MAX_BYTES")
    min_override = _env_int("AITER_FLY_AR_ONESHOT_MIN_BYTES")
    return OneShotPolicy(
        max_bytes=row.oneshot_max if max_override is None else max_override,
        min_bytes=row.oneshot_min if min_override is None else min_override,
    )


def resolve_quant(link: str, world_size: int, arch: str | None = None) -> QuantPolicy:
    """The mesh and ring windows for a rank, environment overrides applied.

    *arch* defaults to this host's; refer to ``family_policy``.

    ``AITER_FLY_AR_ONESHOT_MAX_BYTES`` overrides ``quant_min`` too, so that one
    override moves the one boundary between the one-shot and the mesh.
    ``AITER_FLY_AR_MESH_MAX_BYTES`` overrides ``mesh_max``.
    """
    row = family_policy(link, world_size, arch)
    floor = _env_int("AITER_FLY_AR_ONESHOT_MAX_BYTES")
    if floor is None:
        floor = row.quant_min
    mesh = _env_int("AITER_FLY_AR_MESH_MAX_BYTES")
    if mesh is None:
        mesh = row.mesh_max
    # The families partition by size; an override must not invert them.
    mesh = max(mesh, floor)
    return QuantPolicy(floor=floor, mesh_max=mesh, max_bytes=row.quant_max)


def pick_quant_family(nbytes: int, policy: QuantPolicy) -> str:
    """``"mesh"`` | ``"ring"`` for a payload of *nbytes*.

    Assumes ``nbytes > policy.floor``; below that the caller should have
    declined so the payload reaches an exact kernel.
    """
    return "mesh" if nbytes <= policy.mesh_max else "ring"


def quant_family_range(family: str, policy: QuantPolicy) -> tuple[int, int]:
    """Payload bytes (inclusive) ``pick_quant_family`` sends to *family*."""
    if family == "mesh":
        return policy.floor + 1, min(policy.mesh_max, policy.max_bytes)
    if family == "ring":
        return policy.mesh_max + 1, policy.max_bytes
    raise ValueError(f"family must be 'mesh' or 'ring', got {family!r}")


def quant_families_reachable(policy: QuantPolicy) -> tuple[str, ...]:
    """Quantized families a *policy* can ever select, in size order."""
    out = []
    if policy.mesh_max > policy.floor:
        out.append("mesh")
    if policy.max_bytes > policy.mesh_max:
        out.append("ring")
    return tuple(out)


def route(
    nbytes: int,
    link: str,
    world_size: int,
    *,
    quantized: bool,
    arch: str | None = None,
) -> str:
    """The kernel production dispatch runs for a payload of *nbytes*.

    ``"oneshot"``, ``"mesh"``, ``"ring"``, or ``"fallback"`` for
    ``cross_device_reduce`` and RCCL. *quantized* is whether the quick-reduce
    slot is open (``AITER_QUICK_REDUCE_QUANTIZATION=INT4``).

    The slots are tried in the order of ``CudaCommunicator.all_reduce``:
    quick reduce, then custom all-reduce. Only the payload size is
    considered; the slots also check dtype, alignment and contiguity.
    """
    if quantized:
        quant = resolve_quant(link, world_size, arch)
        if quant.floor < nbytes <= quant.max_bytes:
            return pick_quant_family(nbytes, quant)
    one = resolve_oneshot(link, world_size, arch)
    if one.min_bytes <= nbytes <= one.max_bytes:
        return "oneshot"
    return "fallback"
