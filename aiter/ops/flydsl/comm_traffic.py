# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""What a communication op moves between GPUs, described by the op itself.

A communication op returns a ``Traffic`` from its ``traffic()`` method. A
bandwidth tool can then replay the same data movement with a plain copy
kernel (TransferBench) and grade the op against it, without knowing how the op
works. The description covers which GPU writes to or reads from which, how many
bytes per pair, into what kind of memory, and with what launch geometry. It also
names the RCCL calls that do the same logical work, as a baseline.

Only payload that crosses between GPUs is described. Launches that move no
payload (barriers, small amax exchanges, local dequantization) are left out.

The types are plain data with no torch dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

COLLECTIVES = ("all_reduce", "all_to_all")


class PeerMem(str, Enum):
    """Memory type of the buffers peers write into or read from.

    The values are the ``inbox_memory`` names the FlyDSL ops resolve to.
    """

    COARSE = "coarse"
    FINE = "finegrained"
    UNCACHED = "uncached"


class Pattern(str, Enum):
    """Who moves bytes to whom in one phase. This describes link usage, not
    collective semantics: an all-to-all and an all-gather lap are both
    ``PUSH_ALL``."""

    PUSH_ALL = "push_all"  # rank i writes a chunk into every rank j's buffer
    PULL_ALL = "pull_all"  # rank i reads a chunk from every rank j's buffer
    PUSH_NEXT = "push_next"  # rank i writes a chunk into rank (i + 1) % N
    PULL_PREV = "pull_prev"  # rank i reads a chunk from rank (i - 1) % N


@dataclass(frozen=True)
class Phase:
    """One lap of peer traffic. Phases run back to back.

    ``chunk_bytes`` is the size of the chunk each (source, destination) pair
    moves, in the wire format (payload plus any scales). ``include_self``
    adds the chunk a rank moves into its own buffer, as an all-to-all does.
    ``repeat`` counts identical steps in a row, such as the ``N - 1`` hops of
    a ring lap. ``workgroups`` and ``waves_per_workgroup`` are the launch
    geometry of the kernel running this phase.
    """

    pattern: Pattern
    chunk_bytes: int
    include_self: bool
    repeat: int
    peer_mem: PeerMem
    workgroups: int
    waves_per_workgroup: int

    def __post_init__(self):
        for name in ("chunk_bytes", "repeat", "workgroups", "waves_per_workgroup"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"Phase.{name} must be positive, got {self}")
        if self.include_self and self.pattern not in (
            Pattern.PUSH_ALL,
            Pattern.PULL_ALL,
        ):
            raise ValueError(f"include_self needs an *_ALL pattern, got {self}")


@dataclass(frozen=True)
class RcclCall:
    """One RCCL collective doing the same logical work as the op, per rank."""

    collective: str
    nbytes: int
    dtype: str = "bf16"

    def __post_init__(self):
        if self.collective not in COLLECTIVES:
            raise ValueError(f"collective must be one of {COLLECTIVES}, got {self}")


@dataclass(frozen=True)
class Traffic:
    """Everything a bandwidth tool needs to know about one op call.

    ``logical_bytes`` is the per-rank payload in the caller's dtype: the input
    size of an all-reduce, or the send-buffer size of an all-to-all.
    ``collective`` selects the bus-bandwidth convention.

    ``pipelined`` says the phases run in one launch, with the same grid, and
    the kernel overlaps them (a tile starts its next lap while others are
    still in the previous one). A roof may then stream all their bytes at
    once instead of lap after lap. Phases from separate launches leave it
    ``False``.
    """

    world_size: int
    collective: str
    logical_bytes: int
    phases: tuple[Phase, ...]
    rccl: tuple[RcclCall, ...]
    pipelined: bool = False

    def __post_init__(self):
        if self.collective not in COLLECTIVES:
            raise ValueError(f"collective must be one of {COLLECTIVES}, got {self}")
        if self.world_size < 2 or not self.phases:
            raise ValueError(f"Traffic needs world_size >= 2 and phases, got {self}")
        if self.pipelined and len(
            {(p.workgroups, p.waves_per_workgroup) for p in self.phases}
        ) != 1:
            raise ValueError(f"pipelined phases share one launch grid, got {self}")
