# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Per-rank measurement with an injected timer, and the cross-rank gather.

The tool does not time anything itself. A bench script passes its own
``Timer``. ``measure`` times the op and the RCCL baseline in one call to that
timer, so both get the same method (eager or graph), rounds and rotation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Protocol

from aiter.ops.flydsl.comm_traffic import Traffic

Thunk = Callable[[], None]

RCCL = "rccl"


class Timer(Protocol):
    """A bench script's timing method.

    ``method`` is ``"eager"`` or ``"graph"``. Calling it returns the
    microseconds per call of every thunk on *this* rank. Thunks are
    collectives, so every rank passes the same names in the same order.
    """

    method: str

    def __call__(self, thunks: Mapping[str, Thunk]) -> dict[str, float]: ...


@dataclass
class FnTimer:
    """A ``Timer`` from a plain function, for wrapping a script's timing code."""

    method: str
    fn: Callable[[Mapping[str, Thunk]], dict[str, float]]

    def __call__(self, thunks: Mapping[str, Thunk]) -> dict[str, float]:
        return self.fn(thunks)


@dataclass
class Record:
    """One implementation on one case. ``us`` is the slowest rank's time
    after ``gather``, and this rank's time before it."""

    case: str
    impl: str
    us: float
    timing: str
    traffic: Traffic
    variant: str | None = None
    spread_us: float = 0.0


def measure(
    case: str,
    ops: Mapping[str, tuple[Thunk, Traffic]],
    timer: Timer,
    *,
    rccl: Thunk | None = None,
    variants: Mapping[str, str] | None = None,
) -> list[Record]:
    """Times every op of *ops* (name -> (thunk, traffic)) and the *rccl*
    thunk with one call to *timer*. Returns this rank's records.

    All ops of a case must do the same logical work: the RCCL record carries
    the first op's traffic, for its byte counts.
    """
    if not ops:
        return []
    if RCCL in ops:
        raise ValueError(f"{RCCL!r} is reserved for the baseline")
    thunks = {name: thunk for name, (thunk, _t) in ops.items()}
    if rccl is not None:
        thunks[RCCL] = rccl
    times = timer(thunks)
    variants = variants or {}
    records = [
        Record(case, name, times[name], timer.method, traffic, variants.get(name))
        for name, (_thunk, traffic) in ops.items()
    ]
    if rccl is not None:
        first = next(iter(ops.values()))[1]
        records.append(Record(case, RCCL, times[RCCL], timer.method, first))
    return records


def gather(records: list[Record], cpu_group) -> list[Record] | None:
    """On rank 0, every record with the slowest rank's time and the spread
    to the fastest. None on the other ranks."""
    import torch.distributed as dist

    world = dist.get_world_size(cpu_group)
    per_rank: list = [None] * world
    dist.all_gather_object(per_rank, records, group=cpu_group)
    if dist.get_rank(cpu_group) != 0:
        return None
    merged = []
    for rows in zip(*per_rank, strict=True):
        us = [r.us for r in rows]
        first = rows[0]
        merged.append(
            Record(
                first.case,
                first.impl,
                max(us),
                first.timing,
                first.traffic,
                first.variant,
                max(us) - min(us),
            )
        )
    return merged


def spawn(target, world_size: int, timeout: float = 3600.0, **kwds) -> list:
    """*target* on ranks ``0..world_size-1`` of a fresh spawn. Each call gets
    ``rank``, ``world_size`` and *kwds* as keyword arguments. Returns the
    per-rank return values. *target* must be importable (module level)."""
    ctx = get_context("spawn")
    with ctx.Pool(processes=world_size) as pool:
        futs = [
            pool.apply_async(target, kwds={"rank": r, "world_size": world_size, **kwds})
            for r in range(world_size)
        ]
        return [f.get(timeout=timeout) for f in futs]
