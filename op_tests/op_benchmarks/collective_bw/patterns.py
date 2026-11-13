# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``Traffic`` -> TransferBench tests. Pure Python, no GPU.

A roof is the fastest of a few *candidates*. A candidate is one way to replay
the traffic: a list of TransferBench tests (one config line each, advanced
mode, ``-N (SRC EXE DST SUBEXECS BYTES) ...``) whose times add up. Rank r is
TransferBench GPU r: both the ranks and TransferBench enumerate the devices
``HIP_VISIBLE_DEVICES`` leaves visible.

What a candidate may vary:

* **Waves per transfer.** Sub-executors are single waves (``GFX_SE_TYPE=1``),
  packed into workgroups of the op's width (``GFX_BLOCK_SIZE``). The op's own
  grid split evenly over its transfers is one candidate (``own``). It is not
  the only one: an op wave also packs, quantizes and reduces, so the op's
  grid can be far wider than a copy kernel needs, and a copy kernel that wide
  oversubscribes the links and runs *slower* (a 1365-wave transfer is ~25%
  slower than a 64-wave one on MI350X). ``RoofSettings.wave_sweep`` adds
  fixed widths.
* **Schedule.** ``serial`` runs every phase as its own test and adds them.
  ``fused`` (``Traffic.pipelined`` only) streams every phase's bytes in one
  test, as a kernel that overlaps its laps can.
* **Memory** (RCCL roofs only). The op's roof writes peers' buffers in the
  op's own memory type. RCCL's internals are not modelled, so its roof tries
  every memory type.

Every test is **one pass per launch** (``NUM_SUBITERATIONS=1``), timed inside
the kernel (``tb_text.TestResult.kernel_ms``). Sub-iterations would rewrite
the same buffer within one launch, and in cacheable memory (``G``, ``F``)
the writer's L2 absorbs every pass after the first once the buffer fits in
it: nothing crosses the link, and the time falls as ``1 / NUM_SUBITERATIONS``
(a 7.3 MB TP2 lap at 10 sub-iterations "ran" at ~280 GB/s over one ~64 GB/s
xGMI link). One pass ends with each wave's ``__threadfence_system()``, which
writes its dirty lines back before the wave's stop timestamp, so cacheable
and uncached memory are timed alike and the memory type's real cost shows
(a large INT4 lap is ~20% slower into ``G``/``F`` than into ``U``). The
in-kernel time leaves the host launch out but keeps the wave ramp and the
final fence, which an op also pays once per call. No RCCL candidate pulls
instead of pushing: remote reads can be served from cache across launches.

A push's source and a pull's destination are ordinary device memory
(``G``), as the op's input and output are.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from roof_settings import DEFAULT, RoofSettings

from aiter.ops.flydsl.comm_traffic import Pattern, PeerMem, Phase, Traffic

# TransferBench memory-type letters.
TB_MEM = {PeerMem.COARSE: "G", PeerMem.FINE: "F", PeerMem.UNCACHED: "U"}

WAVE = 64

# Fixed by the method, not settings (see ``roof_settings``): single-wave
# sub-executors, and one pass per launch.
FIXED_ENV = (("GFX_SE_TYPE", "1"), ("NUM_SUBITERATIONS", "1"))

# (src, exe, dst, bytes): one transfer.
Flow = tuple[str, str, str, int]


@dataclass(frozen=True)
class TbTest:
    """One TransferBench test: a config line, the full environment it runs
    with (iterations included), and how many times its time counts toward
    the roof (``Phase.repeat``)."""

    line: str
    env: tuple[tuple[str, str], ...]
    repeat: int


@dataclass(frozen=True)
class Candidate:
    """One way to replay a traffic. Its time is the sum over ``tests`` of
    ``repeat`` x the slowest transfer."""

    label: str
    tests: tuple[TbTest, ...]


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


def _flows(phase: Phase, world_size: int, *, laps: int = 1) -> list[Flow]:
    nbytes = phase.chunk_bytes * laps
    return [(s, e, d, nbytes) for s, e, d in transfers(phase, world_size)]


def _fused_flows(traffic: Traffic) -> list[Flow]:
    """Every phase's bytes (``repeat`` included) as one set of concurrent
    transfers. Phases that use the same link in the same direction share a
    transfer, carrying the sum of their bytes."""
    merged: dict[tuple[str, str, str], int] = {}
    for p in traffic.phases:
        for s, e, d, nbytes in _flows(p, traffic.world_size, laps=p.repeat):
            merged[(s, e, d)] = merged.get((s, e, d), 0) + nbytes
    return [(*k, v) for k, v in merged.items()]


def waves_per_transfer(phase: Phase, world_size: int) -> int:
    """The phase's own waves split evenly over the transfers each GPU executes."""
    per_gpu = len(transfers(phase, world_size)) // world_size
    return max(1, phase.workgroups * phase.waves_per_workgroup // per_gpu)


def env(block_waves: int, settings: RoofSettings = DEFAULT) -> tuple:
    """The full environment of a test with *block_waves*-wave workgroups."""
    return (
        *FIXED_ENV,
        ("GFX_BLOCK_SIZE", str(WAVE * block_waves)),
        ("NUM_ITERATIONS", str(settings.iterations)),
        ("NUM_WARMUPS", str(settings.warmups)),
        *settings.env,
    )


def _test(
    flows: list[Flow],
    waves: int,
    block_waves: int,
    repeat: int,
    settings: RoofSettings,
) -> TbTest:
    body = " ".join(f"({s} {e} {d} {waves} {round4(b)})" for s, e, d, b in flows)
    return TbTest(
        line=f"-{len(flows)} {body}", env=env(block_waves, settings), repeat=repeat
    )


def tb_test(
    phase: Phase,
    world_size: int,
    waves: int | None = None,
    settings: RoofSettings = DEFAULT,
) -> TbTest:
    """One phase as one test, at *waves* per transfer (default: the op's own
    grid split over the transfers)."""
    w = waves_per_transfer(phase, world_size) if waves is None else waves
    return _test(
        _flows(phase, world_size), w, phase.waves_per_workgroup, phase.repeat, settings
    )


def _candidates(
    traffic: Traffic, settings: RoofSettings, *, own: bool, tag: str = ""
) -> list[Candidate]:
    n = traffic.world_size
    phases = traffic.phases
    block = phases[0].waves_per_workgroup
    fuse = traffic.pipelined and len(phases) > 1
    fused = _fused_flows(traffic) if fuse else []
    widths: list[int | None] = [None] if own else []
    widths += list(settings.wave_sweep)
    out = []
    for w in widths:
        name = "own" if w is None else f"w{w}"
        serial = tuple(tb_test(p, n, w, settings) for p in phases)
        out.append(Candidate(f"serial {name}{tag}", serial))
        if fuse:
            # One launch: the grid's waves over all of the fused transfers.
            fw = w
            if fw is None:
                total = phases[0].workgroups * phases[0].waves_per_workgroup
                fw = max(1, total * n // len(fused))
            fused_test = _test(fused, fw, block, 1, settings)
            out.append(Candidate(f"fused {name}{tag}", (fused_test,)))
    return out


def candidates(traffic: Traffic, settings: RoofSettings = DEFAULT) -> list[Candidate]:
    """The op's roof candidates: the op's own pattern and memory, at its own
    geometry and at ``settings.wave_sweep``, serial and (pipelined ops)
    fused. The first is the op's own replay, phase by phase."""
    return _candidates(traffic, settings, own=True)


def rccl_traffic(traffic: Traffic) -> Traffic:
    """The data movement of *traffic*'s RCCL calls, unquantized, as RCCL
    would do it at best: all-pairs, every peer at once.

    An all-reduce is a reduce-scatter and an all-gather of ``nbytes / N`` per
    peer, pipelined in one launch, so each rank sends ``2(N-1)/N`` x
    ``nbytes``: the busbw convention. An all-to-all sends ``nbytes / N`` to
    every rank, its own included. Calls run one after another.

    The memory and grid here are placeholders; ``rccl_candidates`` sets
    them from the settings. Depends only on the RCCL calls and the world
    size, so every op of a case gets the same RCCL roof.
    """
    n = traffic.world_size
    phases = []
    pipelined = False
    for call in traffic.rccl:
        chunk = -(-call.nbytes // n)
        if call.collective == "all_reduce":
            lap = _rccl_phase(chunk, include_self=False)
            phases += [lap, lap]
            pipelined = len(traffic.rccl) == 1
        else:
            phases.append(_rccl_phase(chunk, include_self=True))
    total = sum(c.nbytes for c in traffic.rccl)
    return Traffic(
        world_size=n,
        collective=traffic.collective,
        logical_bytes=total,
        phases=tuple(phases),
        rccl=traffic.rccl,
        pipelined=pipelined,
    )


def _rccl_phase(chunk: int, *, include_self: bool) -> Phase:
    return Phase(
        pattern=Pattern.PUSH_ALL,
        chunk_bytes=chunk,
        include_self=include_self,
        repeat=1,
        peer_mem=PeerMem.UNCACHED,
        workgroups=1,
        waves_per_workgroup=1,
    )


def rccl_candidates(
    rccl: Traffic, settings: RoofSettings = DEFAULT
) -> list[Candidate]:
    """An ``rccl_traffic``'s roof candidates: ``settings.wave_sweep`` x
    serial/fused x ``settings.rccl_mems``, in workgroups of
    ``settings.rccl_waves_per_workgroup`` waves."""
    out = []
    block = settings.rccl_waves_per_workgroup
    for mem in settings.rccl_mems:
        phases = tuple(
            replace(p, peer_mem=mem, waves_per_workgroup=block) for p in rccl.phases
        )
        tag = f" {TB_MEM[mem]}"
        out += _candidates(
            replace(rccl, phases=phases), settings, own=False, tag=tag
        )
    return out


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
