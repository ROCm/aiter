# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Every tuned decision of the FlyDSL all-reduce: which kernel runs a payload,
and in which configuration.

Kernels
=======

* **one-shot** (``OneShotAllReduce``) -- one round, no grid-wide barrier, wire
  volume ``(N-1)*S``. Exact: fp32 accumulate, one bf16 rounding.
* **mesh** (``FlyQuickAllReduce(algorithm="mesh")``) -- two-shot, fanout to all
  ``N-1`` peers twice, wire volume ``2(N-1)/N*S``, INT4 on the wire.
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

Fused all-reduce + RMSNorm
==========================

The fused kernels (``OneShotAllReduceRMSNorm``, ``FlyQuickAllReduceRMSNorm``)
have their own tables of both levels: ``FUSED_FAMILY_POLICIES`` and
``FUSED_ONESHOT_LADDER``, ``FUSED_MESH_ST_LADDER``, ``FUSED_RING_ST_LADDER``
plus ``FUSED_QR_ROW_ATOMS``. All three fused families live in one slot
(``QuickAllReduce``). The overrides are
``AITER_FLY_AR_FUSED_<ALG>_{MIN,MAX}_BYTES``.
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
# What the FlyDSL all-reduce kernels and their hosts support.

SUPPORTED_ARCHS = ("gfx942", "gfx950")
SUPPORTED_WORLDS = (2, 4, 8)
# Fabric between GPUs.
LINKS = ("pcie", "xgmi")
# Threads per workgroup that every kernel can build. The block sets the tile.
SUPPORTED_BLOCKS = (64, 128, 256, 512)
# The fused schedules size a workgroup to one token row instead, on any
# BLOCK_ALIGN multiple up to the hardware workgroup limit MAX_BLOCK.
BLOCK_ALIGN = 128
MAX_BLOCK = 1024
# Payload and buffer address alignment, in bytes: one 16 B store per lane.
PAYLOAD_ALIGN_BYTES = 16
# Largest payload, in bytes: the window of a buffer descriptor.
MAX_PAYLOAD_BYTES = 0xFFFFFFFF
# Largest payload a padded fused build may address, in bytes: the 31-bit
# byte-offset range. The host gates reject a padded fused payload above it.
PAD_MASK_MAX_BYTES = 0x7FFFFFFF

# Legal build values
# ------------------
#
# One-shot algorithm.
#
# Atoms per thread per tile. More atoms means a bigger tile, hence fewer blocks
# for a given payload, at the cost of coarser load balance on the last partial
# tile.
SUPPORTED_ATOMS = (1, 2, 4)

# Workgroups one fused token row is split over. 1 is one workgroup per row, the
# unsplit build. The rest are the counts that divide the shipped widths into
# whole waves: 7168 = 14 waves at atoms=1 gives {2, 7, 14}; 4096/8192 give the
# powers of two.
SUPPORTED_SPLITS = (1, 2, 4, 7, 8, 14, 16)

# Which axis of the (peer, atom) fanout runs fastest across consecutive stores.
#
# "peer": a thread pushes all its atoms to one destination before moving to the
# next, so a wave hands each destination a contiguous ``block * 16`` B run.
#
# "atom": consecutive stores walk the peers of one atom. On xGMI the native
# packet is 64 B and there is no per-destination run-length benefit to collect,
# so spreading across links sooner can start more of them in parallel.
FANOUT_ORDERS = ("peer", "atom")

# Mesh algorithm.
#
# Super-tile values the mesh accepts.
SUPER_TILES = (1, 8)

# Wire formats the mesh can build.
MESH_CODECS = ("int4", "fp16")

# Ring algorithm.
#
# Super-tile values the ring accepts.
#
# The ring publishes 2(N-1) times per super-tile.
# On a cacheable inbox every publish is a full L2 writeback. Publishes per rank
# are ``num_tiles / ST * 2(N-1)``.
RING_SUPER_TILES = (1, 8, 16, 32)

# Wire formats accepted per lap.
#
# Both laps take the same set, but they are separate arguments because they are
# separate decisions: the reduce-scatter lap requantizes ``N-1`` times, while
# the all-gather lap forwards the bytes it received without touching them.
RS_CODECS = ("int4", "int6", "fp16")
AG_CODECS = ("int4", "int6", "fp16")

# How the IPC inbox is allocated. ``"auto"`` resolves to one of the others. The
# wire protocol is identical in every mode; only the memory type changes.
INBOX_MEMORY_MODES = ("auto", "uncached", "finegrained")

# Defaults
# ========

# Mesh and ring algorithms.
#
# Floor on the block count of the mesh and the ring when they batch publishes
# into super-tiles. Shrinking the grid trades parallelism for fewer release fences,
# which is only a good trade once there are enough fences to matter.
MIN_BATCH_BLOCKS = 32

# World size from which the ring's reduce-scatter lap defaults to INT6, to
# clear the 18 dB SQNR floor the schedules are held to. The ring requantizes
# the running partial at every hop, and the partial's extremum grows with the
# contributions folded in, so unlike the mesh its error grows with N.
RS_INT6_MIN_WORLD = 8

# The codecs each two-shot algorithm can build, per lap, as ``(rs, ag)``.
_TWO_SHOT_CODECS = {
    "mesh": (MESH_CODECS, MESH_CODECS),
    "ring": (RS_CODECS, AG_CODECS),
}


def default_codecs(algorithm: str, world_size: int) -> tuple[str, str]:
    """The ``(rs_codec, ag_codec)`` of a two-shot *algorithm* at *world_size*."""
    rs_codecs, ag_codecs = _TWO_SHOT_CODECS[algorithm]
    rs = "int6" if int(world_size) >= RS_INT6_MIN_WORLD else "int4"
    ag = "int4"
    return (
        rs if rs in rs_codecs else rs_codecs[0],
        ag if ag in ag_codecs else ag_codecs[0],
    )


def default_inbox_memory(world_size: int, link: str) -> str:
    """The memory type that ``inbox_memory="auto"`` resolves to.

    ``uncached`` on xGMI and ``finegrained`` on PCIe, except at TP2, where it
    is ``uncached`` on PCIe too.
    """
    if int(world_size) == 2 or link == "xgmi":
        return "uncached"
    return "finegrained"


def quantized_floor(
    *, world_size: int, link: str, arch: str | None = None, fused: bool = False
) -> int:
    """The default ``min_bytes`` of a mesh or ring host: the smallest payload
    the policy routes to the mesh or the ring, which is where the quantized
    windows start. 0 when the policy routes no payload to either.

    Below the quantized windows the policy runs an exact all-reduce: the
    quantized wire would cost accuracy for time that a smaller message does not
    save.
    """
    resolve = resolve_fused_policy if fused else resolve_family_policy
    policy = resolve(world_size=world_size, arch=arch, link=link)
    mins = [
        w.min
        for alg in (AllReduceAlgorithm.TwoShotMesh, AllReduceAlgorithm.TwoShotRing)
        if (w := policy.window(alg, quantized=True)) is not None
    ]
    return min(mins, default=0)


# Rung records
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
    the block count. ``fanout``selects the order of the stores to the peers.
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


@dataclass(frozen=True)
class FusedOneShotRung:
    """The fused (all-reduce + RMSNorm) one-shot configuration for payloads of
    at least ``min_bytes``.

    The fused tile is one token row, or one ``1/split`` slice of it, because
    the RMSNorm reduces over the row. So ``atoms`` sets the block width
    (``block = hidden / (split * 8 * atoms)``) instead of the tile, and there
    is no ``block`` field. ``split`` spreads a row over that many workgroups.
    """

    min_bytes: int
    atoms: int
    grid_cap: int
    fanout: str
    split: int


@dataclass(frozen=True)
class FusedTwoShotRung:
    """The fused mesh or ring configuration for payloads of at least
    ``min_bytes``. Without ``block``: a fused build sizes
    its workgroup to the token row (``FUSED_QR_ROW_ATOMS``)."""

    min_bytes: int
    super_tile: int
    grid_cap: int


def select_rung(ladder, nbytes: int):
    """The rung of *ladder* that runs a payload of *nbytes*."""
    chosen = ladder[0]
    for rung in ladder:
        if nbytes >= rung.min_bytes:
            chosen = rung
    return chosen


# Dispatch policy records
# =======================
#
# The tables below are made of these records.


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

    Three families of all-reduce kernels: one-shot, mesh, and ring.
    Depending on the interconnect type (PCIe or xGMI), the one-shot is best for
    smaller messages, the mesh overtakes for bigger messages, and finally the
    ring is the choice for very large messages on PCIe. The message sizes at
    which the transitions take place depend on the architecture, the world size,
    and the interconnect type. Hence, the policy is a function of ``(arch, link, world_size)``.

    Each algorithm has a ``Window`` of payload sizes. ``None`` or an empty
    window disables the algorithm.

    The mesh and the ring are quantized, so they run only with quantization on
    (``AITER_QUICK_REDUCE_QUANTIZATION=INT4``), and their windows do not
    overlap. The one-shot is exact and runs in both modes.
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
#
#   PCIe
#
#     TP2  b256 a1 cap128 -- the 4 KiB tile to 384 KiB, then b512 a4 cap128,
#                            the 32 KiB tile, over the rest of the exact-mode
#                            window, which runs to 64 MiB.
#     TP4  b128 a1 cap128 -- the 2 KiB tile to 96 KiB, then b512 a2 cap64, the
#                            16 KiB tile, to the 896 KiB exact-mode ceiling.
#     TP8  b64 a1 cap64   -- the 1 KiB tile, except b256 a1 cap64, the 4 KiB
#                            tile, from 48 to 192 KiB.
#
#   xGMI
#
#     TP2  atoms=1 cap128 b128/b256 -- the 2 KiB tile to 256 KiB, the most it
#                               covers in one round, then the 4 KiB tile.
#     TP4  atoms=1 cap128 b256 -- one rung over the whole window
#     TP8  atoms=1 cap128 b128 -- one rung over the whole window
ONESHOT_LADDER: dict[tuple[str, str, int], tuple[OneShotRung, ...]] = {
    # --- gfx942 (MI300X) / xGMI ----------------------------------------------
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
#
#   xGMI  ST=1 everywhere, except TP4 below 4 MiB and TP8 from 6 MiB (ST=8).
#         Block 512 from 4 MiB.
#   PCIe  TP2 ST=1, then ST=8 from 4 MiB. TP4 ST=1, then ST=8 from 768 KiB.
#         TP8 ST=8 everywhere.
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


def apply_env_overrides(
    policy: FamilyPolicy, prefix: str = "AITER_FLY_AR"
) -> FamilyPolicy:
    """*policy* with the ``<prefix>_<ALG>_{MIN,MAX}_BYTES`` overrides."""

    bounds = {
        alg: (
            _env_bound(f"{prefix}_{infix}_MIN_BYTES", is_max=False),
            _env_bound(f"{prefix}_{infix}_MAX_BYTES", is_max=True),
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


# Fused all-reduce + RMSNorm
# ==========================
#
# The same two levels as the plain all-reduce, with tables of their own: the
# fused kernels move the same bytes, but the epilogue changes where each
# family pays, and the one-shot tile is pinned to the token row.


# Level 1: algorithm boundaries
#
# The one-shot window is measured against what the caller would run instead,
# the fused ``cross_device_reduce`` or RCCL plus a separate RMSNorm, and above
# it the fused path declines. With quantization on
# (``AITER_QUICK_REDUCE_QUANTIZATION=INT4``) the mesh and the ring take the
# payloads of their windows first.
FUSED_FAMILY_POLICIES: tuple[FamilyPolicy, ...] = (
    # --- gfx942 (MI300X) / xGMI ----------------------------------------------
    # Measured on MI325X. No ring: the mesh wins above the one-shot.
    FamilyPolicy(
        arch="gfx942",
        link="xgmi",
        world_size=2,
        oneshot=Window(0, 32 * MiB),
        mesh=Window(2 * MiB, None),
        ring=None,
    ),
    FamilyPolicy(
        arch="gfx942",
        link="xgmi",
        world_size=4,
        oneshot=Window(0, 512 * KiB),
        mesh=Window(1344 * KiB, None),
        ring=None,
    ),
    FamilyPolicy(
        arch="gfx942",
        link="xgmi",
        world_size=8,
        oneshot=Window(0, 128 * KiB),
        mesh=Window(448 * KiB, None),
        ring=None,
    ),
    # --- gfx950 (MI355X) / xGMI ----------------------------------------------
    # TODO: measured on gfx942; tune on gfx950.
    FamilyPolicy(
        arch="gfx950",
        link="xgmi",
        world_size=2,
        oneshot=Window(0, 32 * MiB),
        mesh=Window(2 * MiB, None),
        ring=None,
    ),
    FamilyPolicy(
        arch="gfx950",
        link="xgmi",
        world_size=4,
        oneshot=Window(0, 512 * KiB),
        mesh=Window(1344 * KiB, None),
        ring=None,
    ),
    FamilyPolicy(
        arch="gfx950",
        link="xgmi",
        world_size=8,
        oneshot=Window(0, 128 * KiB),
        mesh=Window(448 * KiB, None),
        ring=None,
    ),
    # --- gfx950 (MI350P) / PCIe ----------------------------------------------
    FamilyPolicy(
        arch="gfx950",
        link="pcie",
        world_size=2,
        oneshot=Window(0, 32 * MiB),
        mesh=Window(672 * KiB, 768 * KiB),
        ring=Window(768 * KiB, None),
    ),
    FamilyPolicy(
        arch="gfx950",
        link="pcie",
        world_size=4,
        oneshot=Window(0, 32 * MiB),
        mesh=Window(336 * KiB, 21 * MiB),
        ring=Window(21 * MiB, None),
    ),
    FamilyPolicy(
        arch="gfx950",
        link="pcie",
        world_size=8,
        oneshot=Window(0, 32 * MiB),
        mesh=Window(168 * KiB, None),
        ring=None,
    ),
)

_FUSED_POLICY_BY_KEY = _index(FUSED_FAMILY_POLICIES)


# Level 2: configuration ladders


# One-shot, ``FusedOneShotRung``. Rungs select by payload, which at a fixed
# hidden is the token count, so "split the row at M=1, not at M=4" is two rungs.
#
#   PCIe  measured on MI350P.
#   xGMI  measured on MI325X. A split row wins at TP4/TP8 for small M; TP2 never
#         benefits from splitting it.
_FUSED_ONESHOT_XGMI = {
    2: (FusedOneShotRung(min_bytes=0, atoms=1, grid_cap=128, fanout="peer", split=1),),
    4: (
        FusedOneShotRung(min_bytes=0, atoms=1, grid_cap=128, fanout="peer", split=16),
        FusedOneShotRung(
            min_bytes=144 * KiB, atoms=1, grid_cap=128, fanout="peer", split=1
        ),
    ),
    8: (
        FusedOneShotRung(min_bytes=0, atoms=1, grid_cap=64, fanout="peer", split=16),
        FusedOneShotRung(
            min_bytes=56 * KiB, atoms=1, grid_cap=128, fanout="peer", split=16
        ),
    ),
}
FUSED_ONESHOT_LADDER: dict[tuple[str, str, int], tuple[FusedOneShotRung, ...]] = {
    # --- gfx942 (MI300X) / xGMI ----------------------------------------------
    **{("gfx942", "xgmi", ws): rungs for ws, rungs in _FUSED_ONESHOT_XGMI.items()},
    # --- gfx950 (MI355X) / xGMI ----------------------------------------------
    # TODO: measured on gfx942; tune on gfx950.
    **{("gfx950", "xgmi", ws): rungs for ws, rungs in _FUSED_ONESHOT_XGMI.items()},
    # --- gfx950 (MI350P) / PCIe ----------------------------------------------
    ("gfx950", "pcie", 2): (
        FusedOneShotRung(min_bytes=0, atoms=1, grid_cap=128, fanout="peer", split=1),
    ),
    ("gfx950", "pcie", 4): (
        FusedOneShotRung(min_bytes=0, atoms=2, grid_cap=64, fanout="peer", split=1),
        FusedOneShotRung(
            min_bytes=64 * KiB, atoms=1, grid_cap=128, fanout="peer", split=1
        ),
        FusedOneShotRung(
            min_bytes=384 * KiB, atoms=2, grid_cap=32, fanout="peer", split=1
        ),
    ),
    ("gfx950", "pcie", 8): (
        FusedOneShotRung(min_bytes=0, atoms=1, grid_cap=128, fanout="peer", split=8),
        FusedOneShotRung(
            min_bytes=48 * KiB, atoms=1, grid_cap=32, fanout="peer", split=1
        ),
    ),
}

# Mesh, ``FusedTwoShotRung``.
_FUSED_MESH_XGMI = {
    2: (FusedTwoShotRung(min_bytes=0, super_tile=1, grid_cap=128),),
    4: (
        FusedTwoShotRung(min_bytes=0, super_tile=8, grid_cap=128),
        FusedTwoShotRung(min_bytes=4 * MiB, super_tile=1, grid_cap=128),
    ),
    8: (
        FusedTwoShotRung(min_bytes=0, super_tile=1, grid_cap=128),
        FusedTwoShotRung(min_bytes=6 * MiB, super_tile=8, grid_cap=128),
    ),
}
FUSED_MESH_ST_LADDER: dict[tuple[str, str, int], tuple[FusedTwoShotRung, ...]] = {
    # --- gfx942 (MI300X) / xGMI ----------------------------------------------
    **{("gfx942", "xgmi", ws): rungs for ws, rungs in _FUSED_MESH_XGMI.items()},
    # --- gfx950 (MI355X) / xGMI ----------------------------------------------
    # TODO: measured on gfx942; tune on gfx950.
    **{("gfx950", "xgmi", ws): rungs for ws, rungs in _FUSED_MESH_XGMI.items()},
    # --- gfx950 (MI350P) / PCIe ----------------------------------------------
    ("gfx950", "pcie", 2): (FusedTwoShotRung(min_bytes=0, super_tile=1, grid_cap=64),),
    ("gfx950", "pcie", 4): (
        FusedTwoShotRung(min_bytes=0, super_tile=1, grid_cap=32),
        FusedTwoShotRung(min_bytes=896 * KiB, super_tile=8, grid_cap=128),
    ),
    ("gfx950", "pcie", 8): (FusedTwoShotRung(min_bytes=0, super_tile=8, grid_cap=32),),
}

# Ring, ``FusedTwoShotRung``.
_FUSED_RING_XGMI = {
    2: (
        FusedTwoShotRung(min_bytes=0, super_tile=8, grid_cap=128),
        FusedTwoShotRung(min_bytes=24 * MiB, super_tile=16, grid_cap=128),
    ),
    4: (
        FusedTwoShotRung(min_bytes=0, super_tile=8, grid_cap=128),
        FusedTwoShotRung(min_bytes=24 * MiB, super_tile=16, grid_cap=128),
        FusedTwoShotRung(min_bytes=48 * MiB, super_tile=32, grid_cap=128),
    ),
    8: (
        FusedTwoShotRung(min_bytes=0, super_tile=16, grid_cap=128),
        FusedTwoShotRung(min_bytes=48 * MiB, super_tile=32, grid_cap=128),
    ),
}
FUSED_RING_ST_LADDER: dict[tuple[str, str, int], tuple[FusedTwoShotRung, ...]] = {
    # --- gfx942 (MI300X) / xGMI ----------------------------------------------
    **{("gfx942", "xgmi", ws): rungs for ws, rungs in _FUSED_RING_XGMI.items()},
    # --- gfx950 (MI355X) / xGMI ----------------------------------------------
    # TODO: measured on gfx942; tune on gfx950.
    **{("gfx950", "xgmi", ws): rungs for ws, rungs in _FUSED_RING_XGMI.items()},
    # --- gfx950 (MI350P) / PCIe ----------------------------------------------
    ("gfx950", "pcie", 2): (FusedTwoShotRung(min_bytes=0, super_tile=1, grid_cap=128),),
    ("gfx950", "pcie", 4): (
        FusedTwoShotRung(min_bytes=0, super_tile=16, grid_cap=128),
    ),
    ("gfx950", "pcie", 8): (
        FusedTwoShotRung(min_bytes=0, super_tile=32, grid_cap=128),
    ),
}

# ``atoms_per_row`` of a fused mesh or ring build, per ``(arch, link,
# world_size, algorithm)``: the portable form of the block knob, since which
# blocks exist depends on the width and which ``atoms_per_row`` values exist
# does not. A missing entry takes the widest workgroup the row allows.
_FUSED_QR_ROW_ATOMS_XGMI = {(2, "ring"): 4, (4, "mesh"): 2, (8, "mesh"): 1}
FUSED_QR_ROW_ATOMS: dict[tuple[str, str, int, str], int] = {
    **{("gfx942", "xgmi", *k): v for k, v in _FUSED_QR_ROW_ATOMS_XGMI.items()},
    # TODO: measured on gfx942; tune on gfx950.
    **{("gfx950", "xgmi", *k): v for k, v in _FUSED_QR_ROW_ATOMS_XGMI.items()},
    ("gfx950", "pcie", 2, "mesh"): 4,
    ("gfx950", "pcie", 2, "ring"): 2,
    ("gfx950", "pcie", 4, "mesh"): 2,
    ("gfx950", "pcie", 4, "ring"): 2,
    ("gfx950", "pcie", 8, "mesh"): 1,
}


def fused_oneshot_ladder(
    *, world_size: int, link: str, arch: str | None = None
) -> tuple[FusedOneShotRung, ...]:
    """The fused one-shot rungs for *(arch, link, world_size)*. *arch*
    defaults to this host's."""
    return _ladder(FUSED_ONESHOT_LADDER, "fused one-shot", arch, link, world_size)


def fused_mesh_st_ladder(
    *, world_size: int, link: str, arch: str | None = None
) -> tuple[FusedTwoShotRung, ...]:
    """The fused mesh rungs for *(arch, link, world_size)*. *arch* defaults to
    this host's."""
    return _ladder(FUSED_MESH_ST_LADDER, "fused mesh", arch, link, world_size)


def fused_ring_st_ladder(
    *, world_size: int, link: str, arch: str | None = None
) -> tuple[FusedTwoShotRung, ...]:
    """The fused ring rungs for *(arch, link, world_size)*. *arch* defaults to
    this host's."""
    return _ladder(FUSED_RING_ST_LADDER, "fused ring", arch, link, world_size)


def fused_qr_row_atoms(
    *, world_size: int, algorithm: str, link: str, arch: str | None = None
) -> int | None:
    """``atoms_per_row`` of a fused *algorithm* build, or None for the default."""
    arch = detect_arch() if arch is None else arch
    return FUSED_QR_ROW_ATOMS.get((arch, link, int(world_size), str(algorithm)))


# Host and environment

# Comma-separated hidden sizes to build the fused engines for at startup. The
# fused wire layout depends on the width, so a width cannot be built until it
# is known, and building inside a HIP graph capture is not possible.
FUSED_HIDDENS_ENV = "AITER_FLY_AR_FUSED_HIDDENS"

# The prefix of the fused window overrides, ``<prefix>_<ALG>_{MIN,MAX}_BYTES``.
FUSED_ENV_PREFIX = "AITER_FLY_AR_FUSED"


def fused_hiddens(extra: tuple[int, ...] = ()) -> tuple[int, ...]:
    """Widths to build the fused engines for up front, sorted and deduped.

    ``AITER_FLY_AR_FUSED_HIDDENS`` plus *extra*. Sorted so every rank builds in
    the same order: each build is a collective.
    """
    out = {int(h) for h in extra if int(h) > 0}
    raw = os.environ.get(FUSED_HIDDENS_ENV, "")
    for tok in raw.replace(" ", "").split(","):
        if not tok:
            continue
        try:
            val = int(tok)
        except ValueError:
            logger.warning(
                "FlyDSL all-reduce: ignoring %r in %s, expected an integer",
                tok,
                FUSED_HIDDENS_ENV,
            )
            continue
        if val > 0:
            out.add(val)
    return tuple(sorted(out))


# Level 1: resolution


def fused_family_policy(*, arch: str, link: str, world_size: int) -> FamilyPolicy:
    """The fused table row for *(arch, link, world_size)*, without overrides."""
    _check_platform(arch, link, int(world_size))
    row = _FUSED_POLICY_BY_KEY.get((arch, link, int(world_size)))
    if row is None:
        raise NoPolicyError(
            f"no FlyDSL fused all-reduce policy for {arch}/{link}/TP{world_size}"
        )
    return row


def resolve_fused_policy(
    *, world_size: int, arch: str | None = None, link: str | None = None
) -> FamilyPolicy:
    """The fused policy of this rank, with the
    ``AITER_FLY_AR_FUSED_<ALG>_{MIN,MAX}_BYTES`` overrides applied.

    Select with ``quantized=True`` when quantization is on
    (``AITER_QUICK_REDUCE_QUANTIZATION=INT4``).
    """
    row = fused_family_policy(
        arch=detect_arch() if arch is None else arch,
        link=resolve_link(link),
        world_size=world_size,
    )
    return apply_env_overrides(row, prefix=FUSED_ENV_PREFIX)
