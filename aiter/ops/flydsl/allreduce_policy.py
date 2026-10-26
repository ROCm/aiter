# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Every tuned decision of the FlyDSL all-reduce: which kernel runs a payload,
and in which configuration.

Kernels
=======

* **one-shot** (``OneShotAllReduce``) -- one round, no grid-wide barrier, wire
  volume ``(N-1)*S``. Exact: fp32 accumulate, one bf16 rounding.
* **mesh** (``FlyQuickAllReduce(algorithm="mesh")``) -- two-shot, fanout to all
  ``N-1`` peers twice, wire volume ``2(N-1)/N*S``, INT4 on the wire by default
  (INT5, INT6 or FP16 when pinned).
* **ring** (``FlyQuickAllReduce(algorithm="ring")``) -- two-shot, ``2(N-1)``
  hops, same wire volume as the mesh, traded for per-destination locality.

The decisions come in two levels.

Level 1: which algorithm
========================

``FAMILY_POLICIES`` has one ``FamilyPolicy`` per ``(arch, link,
world_size)``. A policy has one ``Window`` of payload sizes per algorithm
(``AllReduceAlgorithm``): ``min`` inclusive, ``max`` exclusive, ``max=None``
for no upper limit. A missing or empty window disables the algorithm, and a
payload in no window goes to other Aiter all-reduce implementations.
On the payload axis, a policy reads::

    quantized (AITER_QUICK_REDUCE_QUANTIZATION=INT4):
      0 --- oneshot ---)    (gap)    [--- mesh ---)[--- ring --->

    exact (quick reduce off):
      0 --- oneshot ---)    cross_device_reduce / RCCL --->

The one-shot window may overlap the mesh and the ring windows:
``QuickAllReduce`` runs before ``CustomAllreduce``, and with quantization on,
the mesh and the ring take their payloads first.

``resolve_family_policy`` gives the policy for a given rank: the table row for the
host's arch and link, with the ``AITER_FLY_AR_<ALG>_{MIN,MAX}_BYTES``
overrides applied.

Level 2: which configuration
============================

Each algorithm has a tuning ladder per ``(arch, link, world_size)``:
``ONESHOT_LADDER``, ``MESH_ST_LADDER`` and ``RING_ST_LADDER``. A ladder is a
tuple of rungs in ascending ``min_bytes`` order, and a payload runs on the last
rung whose ``min_bytes`` is at most its size (``select_rung``). The host builds
one engine per rung.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger("aiter")


# Limits
# ======
#
# What the FlyDSL all-reduce kernels and their hosts support. The other
# modules import these instead of restating them.

SUPPORTED_ARCHS = ("gfx942", "gfx950")
SUPPORTED_WORLDS = (2, 4, 8)
# Fabric between GPUs.
LINKS = ("pcie", "xgmi")
# Threads per workgroup that every kernel can build. The block sets the tile.
SUPPORTED_BLOCKS = (64, 128, 256, 512)
# Payload and buffer address alignment, in bytes: one 16 B store per lane.
PAYLOAD_ALIGN_BYTES = 16
# Largest payload, in bytes: the window of a buffer descriptor.
MAX_PAYLOAD_BYTES = 0xFFFFFFFF


# Ladder rungs
# ============
#
# A ladder is a tuple of rungs in ascending ``min_bytes`` order, and the first
# rung has ``min_bytes == 0``. The host builds one engine per distinct rung
# and, at launch, runs a payload on the last rung whose ``min_bytes`` is at
# most the payload size.


@dataclass(frozen=True)
class OneShotRung:
    """The one-shot kernel configuration for payloads of at least ``min_bytes``.

    ``atoms`` and ``block`` both scale the tile, and their product sets the
    block count. In addition, ``block`` also changes the workgroup size, and ``atoms``
    also changes how many stores one thread has in flight. ``grid_cap`` bounds
    the block count. ``fanout`` selects the order of the stores to the peers.
    """

    min_bytes: int
    atoms: int
    grid_cap: int
    fanout: str
    block: int

    @property
    def cfg(self) -> tuple[int, int, str, int]:
        """The engine key: every field except ``min_bytes``."""
        return (self.atoms, self.grid_cap, self.fanout, self.block)


@dataclass(frozen=True)
class TwoShotRung:
    """The mesh or ring kernel configuration for payloads of at least
    ``min_bytes``.

    ``super_tile`` is the number of tiles a block publishes at once, and
    ``block`` sets the tile. ``grid_cap`` bounds the block count, and with
    ``super_tile`` it sets the size of the IPC inbox.
    """

    min_bytes: int
    super_tile: int
    grid_cap: int
    block: int


def select_rung(ladder, nbytes: int):
    """The rung of *ladder* that runs a payload of *nbytes*."""
    chosen = ladder[0]
    for rung in ladder:
        if nbytes >= rung.min_bytes:
            chosen = rung
    return chosen


# Dispatch policy records
# =======================


class AllReduceAlgorithm(enum.Enum):
    """The FlyDSL all-reduce algorithms."""

    OneShot = "oneshot"
    TwoShotMesh = "mesh"
    TwoShotRing = "ring"


@dataclass(frozen=True)
class Window:
    """A range of payload sizes in bytes: ``min`` inclusive, ``max`` exclusive.

    ``max=None`` means no upper limit. ``min == max`` is an empty window, which
    disables the algorithm it belongs to.
    """

    min: int
    max: int | None = None

    def __post_init__(self):
        if self.min < 0:
            raise ValueError(f"window min ({self.min}) must be non-negative")
        if self.max is not None and self.max < self.min:
            raise ValueError(f"window max ({self.max}) must be >= min ({self.min})")

    @property
    def is_empty(self) -> bool:
        return self.max is not None and self.max <= self.min

    def contains(self, nbytes: int) -> bool:
        return self.min <= nbytes and (self.max is None or nbytes < self.max)

    def inclusive_range(self) -> tuple[int, int]:
        """``(first, last)`` payload sizes, both inclusive, for the callers
        that need finite bounds. An unbounded window ends at
        ``MAX_PAYLOAD_BYTES``."""
        return self.min, MAX_PAYLOAD_BYTES if self.max is None else self.max - 1

    def __str__(self) -> str:
        return f"[{_fmt_bytes(self.min)}, {_fmt_bytes(self.max)})"


def _fmt_bytes(nbytes: int | None) -> str:
    if nbytes is None:
        return "inf"
    for unit, shift in (("MiB", 20), ("KiB", 10)):
        if nbytes >= 1 << shift and nbytes % (1 << shift) == 0:
            return f"{nbytes >> shift} {unit}"
    return f"{nbytes} B"


@dataclass(frozen=True)
class FamilyPolicy:
    """Which algorithm runs which payload sizes, on one platform.

    We have three families of all-reduce kernels: one-shot, mesh and ring.
    Depending on the interconnect type (PCIe or xGMI), the one-shot is best for
    smaller messages, the mesh overtakes for bigger messages, and finally the
    ring is the choice for very large messages. The message sizes at which the
    transitions take place depend on the architecture, the world size and the
    interconnect type, so a policy belongs to one ``(arch, link, world_size)``.

    Each algorithm has a ``Window`` of payload sizes. ``None`` or an empty
    window disables the algorithm. A payload in no window goes to
    ``cross_device_reduce`` or RCCL.

    The mesh and the ring are quantized, so they run only with quantization on
    (``AITER_QUICK_REDUCE_QUANTIZATION=INT4``), and their windows do not
    overlap. The one-shot is exact and runs in both modes. ``QuickAllReduce``
    serves the mesh and the ring before ``CustomAllreduce`` serves the
    one-shot, and when enabled, the mesh and the ring take the payloads
    of their windows first.
    """

    arch: str
    link: str
    world_size: int
    oneshot: Window | None
    mesh: Window | None
    ring: Window | None

    def __post_init__(self):
        if self.arch not in SUPPORTED_ARCHS:
            raise ValueError(
                f"arch must be one of {SUPPORTED_ARCHS}, got {self.arch!r}"
            )
        if self.link not in LINKS:
            raise ValueError(f"link must be one of {LINKS}, got {self.link!r}")
        if self.world_size not in SUPPORTED_WORLDS:
            raise ValueError(
                f"world_size must be one of {SUPPORTED_WORLDS}, got {self.world_size}"
            )
        mesh, ring = self.mesh, self.ring
        if mesh is None or mesh.is_empty or ring is None or ring.is_empty:
            return
        lo, hi = sorted((mesh, ring), key=lambda w: w.min)
        if lo.max is None or lo.max > hi.min:
            raise ValueError(
                f"{self._cell()}: the mesh window {mesh} overlaps the ring "
                f"window {ring}"
            )

    def _cell(self) -> str:
        return f"{self.arch}/{self.link}/TP{self.world_size}"

    def window(
        self, algorithm: AllReduceAlgorithm, *, quantized: bool
    ) -> Window | None:
        """The window of *algorithm*, or None if it is disabled."""
        if algorithm == AllReduceAlgorithm.OneShot:
            w = self.oneshot
        elif not quantized:
            w = None
        elif algorithm == AllReduceAlgorithm.TwoShotMesh:
            w = self.mesh
        else:
            w = self.ring
        return None if w is None or w.is_empty else w

    def algorithms(self, *, quantized: bool) -> tuple[AllReduceAlgorithm, ...]:
        """The enabled algorithms, in payload order."""
        enabled = [
            (w.min, alg)
            for alg in AllReduceAlgorithm
            if (w := self.window(alg, quantized=quantized)) is not None
        ]
        return tuple(alg for _min, alg in sorted(enabled, key=lambda e: e[0]))

    def select(self, nbytes: int, *, quantized: bool) -> AllReduceAlgorithm | None:
        """The algorithm that runs *nbytes*, or None for
        ``cross_device_reduce`` or RCCL. The mesh and the ring come first, as
        in dispatch."""
        for alg in (
            AllReduceAlgorithm.TwoShotMesh,
            AllReduceAlgorithm.TwoShotRing,
            AllReduceAlgorithm.OneShot,
        ):
            w = self.window(alg, quantized=quantized)
            if w is not None and w.contains(nbytes):
                return alg
        return None

    def describe(self, *, quantized: bool) -> str:
        """The enabled windows, for logs."""
        parts = [
            f"{alg.value} {self.window(alg, quantized=quantized)}"
            for alg in self.algorithms(quantized=quantized)
        ]
        return ", ".join(parts) or "no FlyDSL algorithm"


# Level 1: algorithm boundaries
# ==============================

KiB = 1 << 10
MiB = 1 << 20

FAMILY_POLICIES: tuple[FamilyPolicy, ...] = (
    # --- gfx942 (MI300X) / xGMI ----------------------------------------------
    FamilyPolicy(
        arch="gfx942",
        link="xgmi",
        world_size=2,
        oneshot=Window(0, 1536 * KiB),
        mesh=Window(384 * KiB, 128 * MiB),
        ring=Window(128 * MiB, None),
    ),
    FamilyPolicy(
        arch="gfx942",
        link="xgmi",
        world_size=4,
        oneshot=Window(0, 384 * KiB),
        mesh=Window(384 * KiB, 128 * MiB),
        ring=Window(128 * MiB, None),
    ),
    FamilyPolicy(
        arch="gfx942",
        link="xgmi",
        world_size=8,
        oneshot=Window(0, 168 * KiB),
        mesh=Window(256 * KiB, 128 * MiB),
        ring=Window(128 * MiB, None),
    ),
    # --- gfx950 (MI355X) / xGMI ----------------------------------------------
    FamilyPolicy(
        arch="gfx950",
        link="xgmi",
        world_size=2,
        oneshot=Window(0, 3 * MiB),
        mesh=Window(448 * KiB, None),
        ring=None,
    ),
    FamilyPolicy(
        arch="gfx950",
        link="xgmi",
        world_size=4,
        oneshot=Window(0, 160 * KiB),
        mesh=Window(960 * KiB, None),
        ring=None,
    ),
    FamilyPolicy(
        arch="gfx950",
        link="xgmi",
        world_size=8,
        oneshot=Window(0, 96 * KiB),
        mesh=Window(1600 * KiB, None),
        ring=None,
    ),
    # --- gfx950 (MI350P) / PCIe ----------------------------------------------
    FamilyPolicy(
        arch="gfx950",
        link="pcie",
        world_size=2,
        oneshot=Window(0, 64 * MiB),
        mesh=Window(384 * KiB, None),
        ring=None,
    ),
    FamilyPolicy(
        arch="gfx950",
        link="pcie",
        world_size=4,
        oneshot=Window(0, 896 * KiB),
        mesh=Window(128 * KiB, 18 * MiB),
        ring=Window(18 * MiB, None),
    ),
    FamilyPolicy(
        arch="gfx950",
        link="pcie",
        world_size=8,
        oneshot=Window(0, 1216 * KiB),
        mesh=Window(96 * KiB, 24 * MiB),
        ring=Window(24 * MiB, None),
    ),
)


class NoPolicyError(ValueError):
    """There is no FlyDSL all-reduce policy for the platform."""


# Platforms that are not supported, ``(arch, link)``. A lookup raises for them.
NOT_SUPPORTED_PLATFORMS: dict[tuple[str, str], str] = {
    ("gfx942", "pcie"): "gfx942 has no PCIe-only device",
}


def _index(rows):
    index = {}
    for row in rows:
        key = (row.arch, row.link, row.world_size)
        if key in index:
            raise ValueError(f"duplicate table entry for {key}")
        index[key] = row
    return index


_POLICY_BY_KEY = _index(FAMILY_POLICIES)


# Level 2: configuration ladders
# ==============================
#
# The ladders are keyed by ``(arch, link, world_size)``.


# One-shot, ``OneShotRung``.
ONESHOT_LADDER: dict[tuple[str, str, int], tuple[OneShotRung, ...]] = {
    # --- gfx942 (MI300X) / xGMI ----------------------------------------------
    # TP2: 256 KiB is the most the 2 KiB tile covers in one round of 128 blocks.
    ("gfx942", "xgmi", 2): (
        OneShotRung(min_bytes=0, atoms=1, grid_cap=128, fanout="peer", block=128),
        OneShotRung(
            min_bytes=256 * KiB, atoms=1, grid_cap=128, fanout="peer", block=256
        ),
    ),
    ("gfx942", "xgmi", 4): (
        OneShotRung(min_bytes=0, atoms=1, grid_cap=128, fanout="peer", block=256),
    ),
    ("gfx942", "xgmi", 8): (
        OneShotRung(min_bytes=0, atoms=1, grid_cap=128, fanout="peer", block=128),
    ),
    # --- gfx950 (MI355X) / xGMI ----------------------------------------------
    # TODO: measured on gfx942; tune on gfx950.
    ("gfx950", "xgmi", 2): (
        OneShotRung(min_bytes=0, atoms=1, grid_cap=128, fanout="peer", block=128),
        OneShotRung(
            min_bytes=256 * KiB, atoms=1, grid_cap=128, fanout="peer", block=256
        ),
    ),
    ("gfx950", "xgmi", 4): (
        OneShotRung(min_bytes=0, atoms=1, grid_cap=128, fanout="peer", block=256),
    ),
    ("gfx950", "xgmi", 8): (
        OneShotRung(min_bytes=0, atoms=1, grid_cap=128, fanout="peer", block=128),
    ),
    # --- gfx950 (MI350P) / PCIe ----------------------------------------------
    ("gfx950", "pcie", 2): (
        OneShotRung(min_bytes=0, atoms=1, grid_cap=128, fanout="peer", block=256),
        OneShotRung(
            min_bytes=384 * KiB, atoms=4, grid_cap=128, fanout="peer", block=512
        ),
    ),
    ("gfx950", "pcie", 4): (
        OneShotRung(min_bytes=0, atoms=1, grid_cap=128, fanout="peer", block=128),
        OneShotRung(min_bytes=96 * KiB, atoms=2, grid_cap=64, fanout="peer", block=512),
    ),
    ("gfx950", "pcie", 8): (
        OneShotRung(min_bytes=0, atoms=1, grid_cap=64, fanout="peer", block=64),
        OneShotRung(min_bytes=48 * KiB, atoms=1, grid_cap=64, fanout="peer", block=256),
        OneShotRung(min_bytes=192 * KiB, atoms=1, grid_cap=64, fanout="peer", block=64),
    ),
}

# Mesh, ``TwoShotRung``.
MESH_ST_LADDER: dict[tuple[str, str, int], tuple[TwoShotRung, ...]] = {
    # --- gfx942 (MI300X) / xGMI ----------------------------------------------
    ("gfx942", "xgmi", 2): (
        TwoShotRung(min_bytes=0, super_tile=1, grid_cap=128, block=256),
        TwoShotRung(min_bytes=4 * MiB, super_tile=1, grid_cap=128, block=512),
    ),
    ("gfx942", "xgmi", 4): (
        TwoShotRung(min_bytes=0, super_tile=8, grid_cap=128, block=256),
        TwoShotRung(min_bytes=4 * MiB, super_tile=1, grid_cap=128, block=512),
    ),
    ("gfx942", "xgmi", 8): (
        TwoShotRung(min_bytes=0, super_tile=1, grid_cap=128, block=256),
        TwoShotRung(min_bytes=4 * MiB, super_tile=1, grid_cap=128, block=512),
        TwoShotRung(min_bytes=6 * MiB, super_tile=8, grid_cap=128, block=512),
    ),
    # --- gfx950 (MI355X) / xGMI ----------------------------------------------
    # TODO: measured on gfx942; tune on gfx950.
    ("gfx950", "xgmi", 2): (
        TwoShotRung(min_bytes=0, super_tile=1, grid_cap=128, block=256),
        TwoShotRung(min_bytes=4 * MiB, super_tile=1, grid_cap=128, block=512),
    ),
    ("gfx950", "xgmi", 4): (
        TwoShotRung(min_bytes=0, super_tile=8, grid_cap=128, block=256),
        TwoShotRung(min_bytes=4 * MiB, super_tile=1, grid_cap=128, block=512),
    ),
    ("gfx950", "xgmi", 8): (
        TwoShotRung(min_bytes=0, super_tile=1, grid_cap=128, block=256),
        TwoShotRung(min_bytes=4 * MiB, super_tile=1, grid_cap=128, block=512),
        TwoShotRung(min_bytes=6 * MiB, super_tile=8, grid_cap=128, block=512),
    ),
    # --- gfx950 (MI350P) / PCIe ----------------------------------------------
    ("gfx950", "pcie", 2): (
        TwoShotRung(min_bytes=0, super_tile=1, grid_cap=128, block=256),
        TwoShotRung(min_bytes=4 * MiB, super_tile=8, grid_cap=128, block=512),
    ),
    ("gfx950", "pcie", 4): (
        TwoShotRung(min_bytes=0, super_tile=1, grid_cap=128, block=256),
        TwoShotRung(min_bytes=768 * KiB, super_tile=8, grid_cap=128, block=512),
    ),
    ("gfx950", "pcie", 8): (
        TwoShotRung(min_bytes=0, super_tile=8, grid_cap=128, block=512),
    ),
}

# Ring, ``TwoShotRung``. The ring publishes ``2(N-1)`` times per super-tile, so
# a larger payload wants a larger super-tile.
RING_ST_LADDER: dict[tuple[str, str, int], tuple[TwoShotRung, ...]] = {
    # --- gfx942 (MI300X) / xGMI ----------------------------------------------
    ("gfx942", "xgmi", 2): (
        TwoShotRung(min_bytes=0, super_tile=8, grid_cap=128, block=256),
        TwoShotRung(min_bytes=24 * MiB, super_tile=16, grid_cap=128, block=256),
    ),
    ("gfx942", "xgmi", 4): (
        TwoShotRung(min_bytes=0, super_tile=8, grid_cap=128, block=256),
        TwoShotRung(min_bytes=24 * MiB, super_tile=16, grid_cap=128, block=256),
        TwoShotRung(min_bytes=48 * MiB, super_tile=32, grid_cap=128, block=256),
    ),
    ("gfx942", "xgmi", 8): (
        TwoShotRung(min_bytes=0, super_tile=16, grid_cap=128, block=256),
        TwoShotRung(min_bytes=48 * MiB, super_tile=32, grid_cap=128, block=256),
    ),
    # --- gfx950 (MI355X) / xGMI ----------------------------------------------
    # TODO: measured on gfx942; tune on gfx950.
    ("gfx950", "xgmi", 2): (
        TwoShotRung(min_bytes=0, super_tile=8, grid_cap=128, block=256),
        TwoShotRung(min_bytes=24 * MiB, super_tile=16, grid_cap=128, block=256),
    ),
    ("gfx950", "xgmi", 4): (
        TwoShotRung(min_bytes=0, super_tile=8, grid_cap=128, block=256),
        TwoShotRung(min_bytes=24 * MiB, super_tile=16, grid_cap=128, block=256),
        TwoShotRung(min_bytes=48 * MiB, super_tile=32, grid_cap=128, block=256),
    ),
    ("gfx950", "xgmi", 8): (
        TwoShotRung(min_bytes=0, super_tile=16, grid_cap=128, block=256),
        TwoShotRung(min_bytes=48 * MiB, super_tile=32, grid_cap=128, block=256),
    ),
    # --- gfx950 (MI350P) / PCIe ----------------------------------------------
    ("gfx950", "pcie", 2): (
        TwoShotRung(min_bytes=0, super_tile=16, grid_cap=128, block=512),
    ),
    ("gfx950", "pcie", 4): (
        TwoShotRung(min_bytes=0, super_tile=32, grid_cap=128, block=512),
    ),
    ("gfx950", "pcie", 8): (
        TwoShotRung(min_bytes=0, super_tile=32, grid_cap=128, block=512),
    ),
}


def _ladder(table: dict, name: str, arch, link: str, world_size: int):
    arch = detect_arch() if arch is None else arch
    _check_platform(arch, link, int(world_size))
    rungs = table.get((arch, link, int(world_size)))
    if rungs is None:
        raise NoPolicyError(f"no {name} ladder for {arch}/{link}/TP{world_size}")
    return rungs


def oneshot_ladder(
    *, world_size: int, link: str, arch: str | None = None
) -> tuple[OneShotRung, ...]:
    """The one-shot rungs for *(arch, link, world_size)*. *arch* defaults to
    this host's."""
    return _ladder(ONESHOT_LADDER, "one-shot", arch, link, world_size)


def mesh_st_ladder(
    *, world_size: int, link: str, arch: str | None = None
) -> tuple[TwoShotRung, ...]:
    """The mesh rungs for *(arch, link, world_size)*. *arch* defaults to this
    host's."""
    return _ladder(MESH_ST_LADDER, "mesh", arch, link, world_size)


def ring_st_ladder(
    *, world_size: int, link: str, arch: str | None = None
) -> tuple[TwoShotRung, ...]:
    """The ring rungs for *(arch, link, world_size)*. *arch* defaults to this
    host's."""
    return _ladder(RING_ST_LADDER, "ring", arch, link, world_size)


# Floor on the block count of the mesh and the ring when they batch publishes
# into super-tiles. Shrinking the grid trades parallelism for fewer release fences,
# which is only a good trade once there are enough fences to matter.
MIN_BATCH_BLOCKS = 32

# World size from which the ring's reduce-scatter lap defaults to INT6, to
# clear the 18 dB SQNR floor the schedules are held to. The ring requantizes
# the running partial at every hop, and the partial's extremum grows with the
# contributions folded in, so unlike the mesh its error grows with N.
RS_INT6_MIN_WORLD = 8

# Default ``min_bytes`` of a ``FlyQuickAllReduce`` built directly, outside the
# dispatchers: the payload below which its ``allreduce`` refuses to run. The
# dispatchers pass ``min_bytes=0`` and route by the ``FAMILY_POLICIES`` windows,
# so this value never applies to them.
QUANT_MIN_PAYLOAD_BYTES = 128 * KiB


# Host and environment
# ====================


_UNSET = object()


def _env_bound(name: str, *, is_max: bool):
    """The window bound that the variable *name* sets, or ``_UNSET``.

    Unset, empty or ``-1``: ``_UNSET``. ``none`` or ``inf``: None, no upper
    limit, for a maximum only. Anything else that is not a non-negative
    integer: a warning, and ``_UNSET``.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return _UNSET
    text = raw.strip().lower()
    if text in ("none", "inf"):
        if is_max:
            return None
        logger.warning(
            "FlyDSL all-reduce: ignoring %s=%r, a minimum is a number", name, raw
        )
        return _UNSET
    try:
        val = int(text)
    except ValueError:
        val = -2
    if val == -1:
        return _UNSET
    if val < 0:
        logger.warning(
            "FlyDSL all-reduce: ignoring %s=%r, expected bytes, -1, none or inf",
            name,
            raw,
        )
        return _UNSET
    return val


# The environment variables that override each window, as their infix.
_WINDOW_ENV = {
    AllReduceAlgorithm.OneShot: "ONESHOT",
    AllReduceAlgorithm.TwoShotMesh: "MESH",
    AllReduceAlgorithm.TwoShotRing: "RING",
}


def _override(window: Window | None, lo, hi) -> Window | None:
    if lo is _UNSET and hi is _UNSET:
        return window
    if hi == 0:
        # A maximum of 0 disables the algorithm, whatever its minimum.
        return Window(0, 0)
    base = Window(0, None) if window is None else window
    return Window(base.min if lo is _UNSET else lo, base.max if hi is _UNSET else hi)


def apply_env_overrides(policy: FamilyPolicy) -> FamilyPolicy:
    """*policy* with the ``AITER_FLY_AR_<ALG>_{MIN,MAX}_BYTES`` overrides."""

    bounds = {
        alg: (
            _env_bound(f"AITER_FLY_AR_{infix}_MIN_BYTES", is_max=False),
            _env_bound(f"AITER_FLY_AR_{infix}_MAX_BYTES", is_max=True),
        )
        for alg, infix in _WINDOW_ENV.items()
    }
    return dataclasses.replace(
        policy,
        oneshot=_override(policy.oneshot, *bounds[AllReduceAlgorithm.OneShot]),
        mesh=_override(policy.mesh, *bounds[AllReduceAlgorithm.TwoShotMesh]),
        ring=_override(policy.ring, *bounds[AllReduceAlgorithm.TwoShotRing]),
    )


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


# The environment variable that sets the interconnect type.
LINK_ENV = "AITER_FLY_AR_LINK"


def resolve_link(link: str | None) -> str:
    """*link*, or this host's when it is None. Raises on an unknown link."""
    if link is None:
        link = detect_link()
    if link not in LINKS:
        raise ValueError(f"link must be one of {LINKS}, got {link!r}")
    return link


@functools.lru_cache(maxsize=1)
def _probe_link() -> str | None:
    from .allreduce_host_shared import probe_xgmi_peer_links

    peer_link_type = probe_xgmi_peer_links()
    return None if peer_link_type is None else "xgmi" if peer_link_type else "pcie"


def detect_link() -> str:
    """``"xgmi"`` or ``"pcie"`` for this host.

    Try first reading from the KFD topology, once per process
    (``probe_xgmi_peer_links``). Only when that is unreadable, fall back to the
    env variable ``AITER_FLY_AR_LINK``.

    Host-wide, not per group; see ``probe_xgmi_peer_links`` for the
    uniform-node assumption that makes that safe.
    """
    link = _probe_link()

    if link is None:
        raw = os.environ.get(LINK_ENV, "").strip().lower()
        if raw:
            if raw not in LINKS:
                raise ValueError(f"{LINK_ENV} must be one of {LINKS}, got {raw!r}")
            return raw

    if link is None:
        raise ValueError(
            f"cannot tell if this host has xGMI or PCIe links: the KFD topology "
            f"is not readable. Set {LINK_ENV} to one of {LINKS}."
        )
    return link


# Level 1: resolution
# ===================


def _check_platform(arch: str, link: str, world_size: int) -> None:
    if link not in LINKS:
        raise ValueError(f"link must be one of {LINKS}, got {link!r}")
    if world_size not in SUPPORTED_WORLDS:
        raise ValueError(
            f"world_size must be one of {SUPPORTED_WORLDS}, got {world_size}"
        )
    reason = NOT_SUPPORTED_PLATFORMS.get((arch, link))
    if reason is not None:
        raise NoPolicyError(f"no FlyDSL all-reduce for {arch}/{link}: {reason}")


def family_policy(*, arch: str, link: str, world_size: int) -> FamilyPolicy:
    """The table row for *(arch, link, world_size)*, without overrides."""
    _check_platform(arch, link, int(world_size))
    row = _POLICY_BY_KEY.get((arch, link, int(world_size)))
    if row is None:
        raise NoPolicyError(
            f"no FlyDSL all-reduce policy for {arch}/{link}/TP{world_size}"
        )
    return row


def resolve_family_policy(
    *, world_size: int, arch: str | None = None, link: str | None = None
) -> FamilyPolicy:
    """The policy of this rank, with the environment overrides applied."""
    row = family_policy(
        arch=detect_arch() if arch is None else arch,
        link=resolve_link(link),
        world_size=world_size,
    )
    return apply_env_overrides(row)
