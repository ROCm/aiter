# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``Traffic`` -> TransferBench tests. Pure Python, no GPU.

Each ``Phase`` becomes one TransferBench test (one config line in advanced
mode, ``-N (SRC EXE DST SUBEXECS BYTES) ...``). Rank r is TransferBench GPU r:
both the ranks and TransferBench enumerate the devices ``HIP_VISIBLE_DEVICES``
leaves visible.

The phase's launch geometry is reproduced with warp-sized sub-executors
(``GFX_SE_TYPE=1``). The kernel's waves are split evenly over the transfers
each GPU executes, and ``GFX_BLOCK_SIZE`` packs them into workgroups of the
op's width.

A memory location on another GPU, and the executing GPU's own destination
buffer in a push, take the op's peer memory type. A push's source and a pull's
destination are ordinary device memory (``G``), as the op's input and output
are.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from aiter.ops.flydsl.comm_traffic import Pattern, PeerMem, Phase, Traffic

# TransferBench memory-type letters.
TB_MEM = {PeerMem.COARSE: "G", PeerMem.FINE: "F", PeerMem.UNCACHED: "U"}

# Each timed iteration loops this many bytes per GPU at least, inside one
# launch (NUM_SUBITERATIONS), so launch cost is amortized away.
SUBITER_TARGET_BYTES = 64 << 20
MAX_SUBITERATIONS = 256

WAVE = 64


@dataclass(frozen=True)
class TbTest:
    """One TransferBench test: a config line, the environment it needs, and
    how many times its time counts toward the phase (``Phase.repeat``)."""

    line: str
    env: tuple[tuple[str, str], ...]
    repeat: int


def round4(nbytes: int) -> int:
    """TransferBench needs a multiple of 4 bytes. Round up, never down, so
    the roof is not made optimistic."""
    return max(4, -(-int(nbytes) // 4) * 4)


def transfers(phase: Phase, world_size: int) -> list[tuple[str, str, str]]:
    """``(src, exe, dst)`` per transfer, rank-major."""
    mem = TB_MEM[phase.peer_mem]
    n = world_size
    out = []
    for i in range(n):
        if phase.pattern in (Pattern.PUSH_ALL, Pattern.PULL_ALL):
            peers = [j for j in range(n) if j != i or phase.include_self]
        elif phase.pattern == Pattern.PUSH_NEXT:
            peers = [(i + 1) % n]
        else:
            peers = [(i - 1) % n]
        for j in peers:
            if phase.pattern in (Pattern.PUSH_ALL, Pattern.PUSH_NEXT):
                out.append((f"G{i}", f"G{i}", f"{mem}{j}"))
            else:
                out.append((f"{mem}{j}", f"G{i}", f"G{i}"))
    return out


def waves_per_transfer(phase: Phase, world_size: int) -> int:
    """The phase's waves split evenly over the transfers each GPU executes."""
    per_gpu = len(transfers(phase, world_size)) // world_size
    return max(1, phase.workgroups * phase.waves_per_workgroup // per_gpu)


def subiterations(phase: Phase, world_size: int) -> int:
    per_gpu = len(transfers(phase, world_size)) // world_size
    moved = per_gpu * round4(phase.chunk_bytes)
    return max(1, min(MAX_SUBITERATIONS, math.ceil(SUBITER_TARGET_BYTES / moved)))


def tb_test(phase: Phase, world_size: int) -> TbTest:
    nbytes = round4(phase.chunk_bytes)
    w = waves_per_transfer(phase, world_size)
    body = " ".join(
        f"({s} {e} {d} {w} {nbytes})" for s, e, d in transfers(phase, world_size)
    )
    count = len(transfers(phase, world_size))
    env = (
        ("GFX_SE_TYPE", "1"),
        ("GFX_BLOCK_SIZE", str(WAVE * phase.waves_per_workgroup)),
        ("NUM_SUBITERATIONS", str(subiterations(phase, world_size))),
    )
    return TbTest(line=f"-{count} {body}", env=env, repeat=phase.repeat)


def tb_tests(traffic: Traffic) -> list[TbTest]:
    """One test per phase, in phase order."""
    return [tb_test(p, traffic.world_size) for p in traffic.phases]


def busbw_factor(collective: str, world_size: int) -> float:
    """The rccl-tests bus-bandwidth factor."""
    n = world_size
    return {"all_reduce": 2 * (n - 1) / n, "all_to_all": (n - 1) / n}[collective]


def fabric_bytes_per_rank(traffic: Traffic) -> int:
    """Bytes each rank sends to *other* GPUs over one call: the bytes that
    cross the fabric."""
    total = 0
    for p in traffic.phases:
        remote = [t for t in transfers(p, traffic.world_size) if t[1] != _dst_gpu(t)]
        total += len(remote) // traffic.world_size * p.chunk_bytes * p.repeat
    return total


def _dst_gpu(t: tuple[str, str, str]) -> str:
    """The GPU at the far end of a transfer, as ``G<i>``."""
    src, exe, dst = t
    other = dst if src == exe else src
    return "G" + other[1:]
