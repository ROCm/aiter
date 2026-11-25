# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Boundary formats between adjacent segments.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
Please do not modify this code without coordinating with the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""

from dataclasses import dataclass

from ..problem import ScaleSpec

KINDS = ("tokens", "sorted", "inter", "routed_out", "p2p_scatter", "out")


@dataclass(frozen=True)
class Buf:
    kind: str
    fmt: ScaleSpec | None  # None for "sorted", which carries routing, not values
    layout: str = "default"  # "aux_sorted", "v2_sorted", "opus_bucketed", "flat", ...
    block_m: int | None = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown Buf kind {self.kind!r}; known: {KINDS}")

    def __str__(self):
        parts = [self.kind]
        if self.fmt is not None:
            parts.append(str(self.fmt))
        if self.layout != "default":
            parts.append(self.layout)
        if self.block_m is not None:
            parts.append(f"bm={self.block_m}")
        return ":".join(parts)
