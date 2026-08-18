# SPDX-License-Identifier: MIT
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class KernelResources:
    name: str
    vgpr: int
    sgpr: int
    lds_bytes: int
    private_bytes: int
    vgpr_spills: int
    sgpr_spills: int
    workgroup_size: int
    wavefront_size: int

    @property
    def waves_per_workgroup(self) -> int:
        return (self.workgroup_size + self.wavefront_size - 1) // self.wavefront_size

    def lds_workgroup_limit(self, lds_per_cu: int = 160 * 1024) -> int:
        return 2**31 - 1 if self.lds_bytes == 0 else lds_per_cu // self.lds_bytes

    @property
    def spill_free(self) -> bool:
        return self.vgpr_spills == 0 and self.sgpr_spills == 0 and self.private_bytes == 0


def parse_llvm_readobj_notes(text: str, kernel_name: str) -> KernelResources:
    """Parse one-kernel AMDHSA metadata emitted by ``llvm-readobj --notes``."""

    def value(field: str) -> int:
        match = re.search(rf"\.{re.escape(field)}:\s+(\d+)", text)
        if not match:
            raise ValueError(f"missing .{field} in code-object metadata")
        return int(match.group(1))

    return KernelResources(
        name=kernel_name,
        vgpr=value("vgpr_count"),
        sgpr=value("sgpr_count"),
        lds_bytes=value("group_segment_fixed_size"),
        private_bytes=value("private_segment_fixed_size"),
        vgpr_spills=value("vgpr_spill_count"),
        sgpr_spills=value("sgpr_spill_count"),
        workgroup_size=value("max_flat_workgroup_size"),
        wavefront_size=value("wavefront_size"),
    )


def hidden_fraction(comm_time: float, compute_time: float, joint_time: float) -> float:
    """Measured fraction of the shorter phase hidden by overlap."""

    shorter = min(comm_time, compute_time)
    if shorter <= 0:
        return 0.0
    return max(0.0, min(1.0, (comm_time + compute_time - joint_time) / shorter))


def timeline_overlap_ratio(
    comm_intervals: list[tuple[float, float]],
    compute_intervals: list[tuple[float, float]],
) -> float:
    """Pairwise interval overlap divided by the shorter union duration.

    Callers should merge overlapping intervals within each list first; profiler
    traces normally already provide non-overlapping intervals per stream.
    """

    comm_total = sum(max(0.0, end - start) for start, end in comm_intervals)
    compute_total = sum(max(0.0, end - start) for start, end in compute_intervals)
    shorter = min(comm_total, compute_total)
    if shorter <= 0:
        return 0.0
    overlap = 0.0
    for c0, c1 in comm_intervals:
        for g0, g1 in compute_intervals:
            overlap += max(0.0, min(c1, g1) - max(c0, g0))
    return max(0.0, min(1.0, overlap / shorter))
