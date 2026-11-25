# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Plans: the segments that cover the op chain, with their parameters.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
Please do not modify this code without coordinating with the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""

from dataclasses import dataclass, field

from ..segments.base import Params, Segment
from .buf import Buf

SOURCES = ("override", "tuned", "profile", "online", "default")


@dataclass(frozen=True)
class SegmentChoice:
    segment: Segment
    params: Params = field(default_factory=Params)

    def __str__(self):
        shown = {k: v for k, v in self.params.items() if v not in ("", None)}
        if not shown:
            return self.segment.name
        return f"{self.segment.name}({', '.join(f'{k}={v}' for k, v in shown.items())})"


# A plan's structure: the segment names, boundaries and boundary (structure) parameters.
# Two plans with the same structure differ only in segment-local parameters.
StructureKey = tuple


@dataclass(frozen=True)
class MoePlan:
    segments: tuple  # SegmentChoice, in chain order
    source: str
    params: Params = field(
        default_factory=Params
    )  # structure parameters (block_m, ...)
    boundaries: tuple = ()  # Buf between consecutive segments

    def __post_init__(self):
        if self.source not in SOURCES:
            raise ValueError(f"unknown plan source {self.source!r}")
        for b in self.boundaries:
            if not isinstance(b, Buf):
                raise TypeError(f"boundary {b!r} is not a Buf")

    def structure(self) -> StructureKey:
        return (
            tuple(c.segment.name for c in self.segments),
            self.boundaries,
            tuple(sorted(self.params.items())),
        )

    def segment(self, name) -> SegmentChoice | None:
        for choice in self.segments:
            if choice.segment.name == name:
                return choice
        return None

    def describe(self) -> str:
        chain = " | ".join(str(c) for c in self.segments)
        params = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{chain}  [{params}] (source={self.source})"
