# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Runtime correctness and timing for FlyDSL quick all-reduce (``FlyQuickAllReduce``).

A default run covers what production dispatch can run on this host, and each
sweep ends in a markdown table:

* ``test_quick_allreduce`` -- the shipping configuration (codecs and
  super-tile left to the per-world defaults and ladders, as production dispatch
  constructs the engine), timed with ``run_perftest``. The payloads come from
  ``allreduce_policy``: for every schedule it routes to at each world size, the
  smallest payload that reaches each kernel it can select there, plus the top
  of its window. One row per schedule and world size captures the all-reduce
  into a CUDA graph and replays it. A schedule the policy never selects on this
  host (the ring on xGMI) gets one row, for a user who moves the boundary with
  ``AITER_FLY_AR_MESH_MAX_BYTES``.
* ``test_quick_allreduce_coverage`` -- the kernels those rows ran include
  every kernel the engine's own ``cfgs_for`` says the window selects.
* ``test_quick_allreduce`` again, as a second table -- the shipping INT4
  ladder with ``block`` overridden on every rung, at the narrow blocks where a
  VMEM store-data hazard once showed up.
* ``test_quick_allreduce_edge_inputs`` -- payloads that land on the E4M3
  scale's edge cases, and degenerate groups that must stay finite.
* ``test_quick_allreduce_transport`` -- the ``fp16`` wire format, a lossless
  passthrough, on an exactly representable input: the result must be
  bit-identical to the fp32 reference, which separates a chunk-addressing or
  flag-protocol bug from a codec one. Run through each production schedule's
  ladder on the shipping payloads, so it covers the geometry that ships.

``--extended`` adds what production never selects: the legacy fixed shipping
shapes, the full ``block`` sweep, the fp16 transport over a matrix of pinned
``super_tile``/``block``, and
``test_quick_allreduce_pinned_codec`` -- the ring with its wire formats
pinned per lap: all-INT4 at TP8, and one lap lossless to isolate the other.

The fused epilogue (``FlyQuickAllReduceRMSNorm``: all-reduce + residual add +
RMSNorm) has its own ``test_quick_allreduce_rmsnorm_*`` sweeps, one table each:
SQNR against an fp32 oracle, bit-exactness of ``residual_out`` on the lossless
wire (native, padded-width and pinned-super-tile geometries), mesh rank
agreement, and
``test_quick_allreduce_host_checks``.

Every mesh row also checks that all ranks wrote bit-identical output: each
rank decodes every chunk from the same packets, its own included -- its own
from the packet it sent, since it never round-trips through its own inbox.
The ring's owner stores its chunk before the all-gather quantization, so
its ranks legitimately differ and only report the count.

Every rank is a ``multiprocessing`` spawn worker that builds its own engine.
All rows that share an engine configuration ride one spawn. The oracle is an
untimed fp32 NCCL all-reduce of the same per-rank inputs. INT4/INT6 are lossy,
so those rows gate on SQNR, a calibrated mismatch ratio and a per-tile SQNR
floor.

The kernels see a flat payload, so the derived shapes use one width, 4096,
whose rows land exactly on every policy and ladder boundary.
FlyQuickAllReduce runs on gfx942/gfx950 at TP in {2, 4, 8}; other archs skip,
and ``main()`` skips a world size when fewer GPUs are visible than TP.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
from multiprocessing import freeze_support

import torch
from flydsl_comm_test_utils import (
    ARCH,
    SUPPORTED_ARCHS,
    FailureLog,
    SpawnRegistry,
    fmt_bytes,
    lanes_differing,
    min_tile_sqnr_db,
    rel_mae,
    run_on_ranks,
    sqnr_db,
    summarize,
    worst,
)

import aiter
from aiter import dtypes
from aiter.ops.flydsl import allreduce_policy as fly_policy
from aiter.ops.flydsl.kernels.collectives_shared import (
    ATOMS,
    DEFAULT_GRID_CAP,
    SUPPORTED_WORLDS,
    clamp_grid_cap,
)
from aiter.ops.flydsl.kernels.quick_allreduce_codec import SUPPORTED_BLOCKS
from aiter.ops.flydsl.kernels.quick_allreduce_mesh import mesh_st_ladder
from aiter.ops.flydsl.kernels.quick_allreduce_ring import ring_st_ladder
from aiter.ops.flydsl.quick_allreduce import (
    _resolve_inbox_flags,
    batches_publishes,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

ALGORITHMS = ("mesh", "ring")

# One SQNR floor for both schedules, in their shipping configuration.
#
# The ring's reduce-scatter lap requantizes N-1 times where the mesh requantizes
# once, and the partial sum it requantizes grows with the contributions folded
# in, so on an all-INT4 wire the ring's SQNR degrades with N: 22.2 dB at TP2,
# 18.7 at TP4, ~15 at TP8. Defaulting the ring's reduce-scatter lap to INT6 at
# TP8 lifts it to ~21 dB, so anything below 18.0 is a regression, not a known
# cost of the schedule.
SQNR_MIN_DB = 18.0

# INT4 at TP8 on the ring is still a supported configuration -- pinning both
# laps reaches it -- and is held to what it actually delivers, not to the
# shipping floor.
SQNR_MIN_DB_TP8_INT4_RING = 15.0

# A tile the kernel never wrote scores ~0 dB; codec noise stays above 8.
# Per-tile rather than whole-payload, so one unwritten tile cannot be averaged
# away by the rest of a large message. The tile is the engine's own, which
# ``block`` sets.
TILE_SQNR_MIN_DB = 8.0

# Calibrated to the INT4 group-16 codec vs fp32 all-reduce, not bit identity.
CLOSE_RTOL = 1e-1
CLOSE_ATOL = 1e-1
CLOSE_ERR_RATIO = 0.5

# Each replay advances the per-block colour and alternates the inbox parity
# slot, so a graph that froze either would pass the first replay and fail a
# later one. Four covers both parities twice.
GRAPH_REPLAYS = 4

# Shipping payloads are derived from the dispatch policy (``_ship_payloads``)
# and laid out at this width. A row is 8 KiB and every policy and ladder
# boundary is a multiple of that, so rounding a payload up to whole rows never
# carries it across one.
HIDDEN = 4096
_ROW_BYTES = HIDDEN * 2

# Where an unbounded dispatch window is cut for its production-scale row.
TOP_PAYLOAD_BYTES = 64 << 20

# Under one tile at every block, so a single block owns the whole payload.
SUB_TILE_SHAPE = (8, 1024)

# The fixed shapes the shipping sweep used before it was derived from the
# policy. ``--extended`` only: they time the schedules outside the windows
# production routes to them.
LEGACY_SHAPES_PER_WORLD_SIZE = {
    8: [(8, 1024), (512, 5120), (9216, 4096), (32768, 5120)],
    4: [(512, 5120), (9216, 4096)],
    2: [(512, 5120), (9216, 4096)],
}

# (tp, algorithm, tokens, hidden, fill).
EDGE_CASES = (
    (2, "mesh", 16, 1024, "pos_underflow"),
    (2, "mesh", 16, 1024, "neg_underflow"),
    (2, "mesh", 16, 1024, "overflow_512"),
    (2, "mesh", 16, 1024, "zeros"),
    (2, "mesh", 512, 5120, "degenerate"),
    (8, "ring", 512, 5120, "degenerate"),
)

# (tp, tokens, hidden, rs_codec, ag_codec), ring algo only. ``--extended``
# only: production leaves both laps at the per-world default.
PINNED_CODEC_CASES = (
    (8, 512, 5120, "int4", "int4"),
    (8, 512, 5120, "fp16", "int4"),
    (8, 512, 5120, "int4", "fp16"),
)

# (tp, algorithm, tokens, hidden, super_tile, block) on the lossless
# fp16 wire, with the geometry pinned. ``--extended`` only: a default run
# drives the fp16 wire through each schedule's own ladder instead, on the
# payloads that reach every kernel production selects.
TRANSPORT_CASES = (
    (8, "ring", 8, 1024, 1, 256),
    (8, "ring", 512, 5120, 1, 256),
    (8, "ring", 4096, 4096, 8, 256),
    (4, "ring", 512, 5120, 1, 256),
    (4, "ring", 4096, 4096, 8, 256),
    (2, "ring", 512, 5120, 1, 256),
    (8, "mesh", 512, 5120, 1, 256),
    (8, "mesh", 4096, 4096, 8, 256),
    (8, "mesh", 512, 5120, 1, 128),
    (8, "mesh", 4096, 4096, 8, 64),
    (8, "ring", 512, 5120, 8, 128),
    (4, "mesh", 512, 5120, 1, 64),
    (4, "mesh", 4096, 4096, 8, 64),
    (4, "mesh", 512, 5120, 1, 128),
    (4, "mesh", 4096, 4096, 8, 128),
    (4, "mesh", 512, 5120, 1, 256),
    (4, "mesh", 4096, 4096, 8, 256),
    (4, "mesh", 512, 5120, 1, 512),
    (4, "mesh", 4096, 4096, 8, 512),
    (4, "ring", 512, 5120, 1, 64),
    (4, "ring", 4096, 4096, 8, 128),
    (4, "ring", 4096, 4096, 8, 512),
    (2, "mesh", 512, 5120, 1, 64),
    (2, "mesh", 4096, 4096, 8, 128),
    (2, "mesh", 512, 5120, 1, 512),
    (2, "mesh", 4096, 4096, 8, 256),
    (2, "ring", 4096, 4096, 8, 64),
    (2, "ring", 512, 5120, 1, 512),
    # Sub-tile and single-tile payloads, where one block owns the lot.
    (4, "mesh", 8, 1024, 1, 64),
    (2, "mesh", 8, 1024, 8, 512),
)
TRANSPORT_GRID_CAP = 64

# (tp, algorithm, tokens, hidden, block): the shipping INT4 ladder with
# ``block`` overridden on every rung.
#
# The TP4 mesh rows at blocks 64 and 128 are where the VMEM store-data hazard in
# ``_store_v4i32_peer`` was caught, back when the mesh still had a variant that
# round-tripped through its own inbox: INT4's peer-major fanout at those widths
# is where the register allocator recycles the store's data VGPRs. The fp16
# transport rows at the same geometry never did. Those two run by default; no
# production rung uses either width, but the hazard lives in code every mesh
# rung shares.
KNOB_CASES = (
    (4, "mesh", 512, 5120, 64),
    (4, "mesh", 512, 5120, 128),
)

# The rest of the knob sweep. ``--extended`` only.
EXTENDED_KNOB_CASES = (
    (8, "mesh", 512, 5120, 128),
    (8, "ring", 512, 5120, 128),
    (4, "mesh", 9216, 4096, 128),
    (4, "mesh", 9216, 4096, 512),
    (4, "ring", 9216, 4096, 64),
    (2, "mesh", 512, 5120, 128),
    (2, "mesh", 9216, 4096, 512),
    (2, "ring", 9216, 4096, 128),
)


_FILLS = (
    "normal",
    "degenerate",
    "exact",
    "pos_underflow",
    "neg_underflow",
    "overflow_512",
    "zeros",
)


def _make_inp(
    tokens: int, hidden: int, fill: str, *, rank: int, device: torch.device
) -> torch.Tensor:
    """One rank's contribution, for whichever edge case *fill* names."""
    shape = (tokens, hidden)
    gen = torch.Generator().manual_seed(1234 + rank)
    if fill == "normal":
        src = torch.randn(shape, generator=gen, dtype=torch.float32) * 0.1
    elif fill == "degenerate":
        # Half the rows exactly zero, half far below the E4M3 magnitude floor
        # of 2^-7. Both drive the group extremum to (or under) zero, which is
        # where the encode reciprocal blows up.
        src = torch.randn(shape, generator=gen, dtype=torch.float32) * 0.1
        src[0::2] = 0.0
        src[1::2] *= 1e-8
    elif fill == "exact":
        # Grid of 1/16, magnitude < 0.5: exact in both bf16 and fp16, and every
        # partial sum over up to 8 ranks stays exact in both too (integer
        # multiple of 1/16, magnitude <= 4 -- 7 significant bits). Paired with
        # the fp16 wire this makes the whole reduce lossless, so the result is
        # bit-identical to the fp32 reference regardless of accumulation order.
        src = torch.randint(-8, 8, shape, generator=gen).float() * (2.0**-4)
    elif fill in ("pos_underflow", "neg_underflow", "overflow_512", "zeros"):
        val = {
            "pos_underflow": 2.0**-8,
            "neg_underflow": -(2.0**-8),
            # Drives the E4M3 scale above its largest exponent, which the
            # encoder has to saturate. Only rank 0 carries the value: if every
            # rank sent 512 the reduced sum would also saturate the INT4 group
            # codec, and the case would fail on codec range rather than on
            # scale encoding.
            "overflow_512": 512.0 if rank == 0 else 0.0,
            "zeros": 0.0,
        }[fill]
        return torch.full(shape, val, device=device, dtype=torch.bfloat16)
    else:
        raise ValueError(f"unknown fill {fill!r}; expected one of {_FILLS}")
    return src.to(device=device, dtype=torch.bfloat16)


def _tile_bytes(block: int) -> int:
    return block * ATOMS * 16


def _num_tiles(nbytes: int, block: int) -> int:
    tile = _tile_bytes(block)
    return max(1, (nbytes + tile - 1) // tile)


def _ladder(algorithm: str, world_size: int, link: str) -> tuple:
    if algorithm == "mesh":
        return mesh_st_ladder(world_size, link)
    return ring_st_ladder(world_size, link)


def _expected_cfg(
    nbytes: int,
    *,
    ladder: tuple,
    batched: bool,
    grid_by_cfg: dict[tuple, int],
    block: int | None = None,
) -> tuple[int, int]:
    """Mirror of ``FlyQuickAllReduce._pick_cfg``: the ``(super_tile, block)``
    kernel an engine with no super-tile pinned runs *nbytes* on. *block* is the
    override the engine was built with, ``None`` for the rung's own.

    Two rules compose. The payload one: the schedule's ladder assigns a
    super-tile by size -- publishes per rank are ``num_tiles / ST * 2(N-1)``,
    so a bigger payload wants a bigger one. The interconnect one: when the
    engine *batched* publishes (a release fence, or the ring on PCIe) it takes
    the super-tile as soon as there is a whole one, while otherwise ST=1 is
    preferred until there are more tiles than blocks.

    *grid_by_cfg* must hold the engines' *clamped* grids, not the requested
    caps: the host reduces them to the measured resident workgroups per CU,
    and the clamped value is what the selection compares against.
    """
    want, b = 1, None
    for floor, rung_st, _cap, rung_b in ladder:
        if nbytes >= floor:
            want, b = rung_st, rung_b
    b = b if block is None else block
    if want == 1:
        return 1, b
    tiles = _num_tiles(nbytes, b)
    if batched:
        return (want if tiles >= want else 1), b
    return (want if tiles > grid_by_cfg[(want, b)] else 1), b


def _expected_st(
    nbytes: int,
    *,
    algorithm: str,
    world_size: int,
    link: str,
    inbox_memory: str,
    grid_by_cfg: dict[tuple, int],
    block: int | None = None,
) -> int:
    """The super-tile ``_expected_cfg`` picks, on the host a rank reported.

    Whether publishes are batched is a property of the host, so it comes from
    the rank's reported ``inbox_memory`` and ``link`` rather than being assumed.
    """
    return _expected_cfg(
        nbytes,
        ladder=_ladder(algorithm, world_size, link),
        batched=batches_publishes(inbox_memory, algorithm, link),
        grid_by_cfg=grid_by_cfg,
        block=block,
    )[0]


def _rung_grids(world_size: int, ladder: tuple) -> dict[tuple, int]:
    """Each rung's clamped grid, computed as the engine computes it.

    The parent has no engine to ask: payloads are chosen before any spawn.
    """
    cu_count = int(torch.cuda.get_device_properties(0).multi_processor_count)
    grids = {}
    for _floor, st, cap, b in ladder:
        grids.setdefault(
            (st, b),
            clamp_grid_cap(
                min(cap, DEFAULT_GRID_CAP),
                arch=ARCH,
                world_size=world_size,
                super_tile=st,
                cu_count=cu_count,
                block=b,
            ),
        )
    return grids


def _kernel_payloads(
    world_size: int, algorithm: str, link: str, lo: int, hi: int
) -> dict[tuple, int]:
    """Smallest whole-row payload in ``lo..hi`` bytes (inclusive) that selects
    each ``(super_tile, block)`` kernel the shipping engine can run there.

    Within one rung the choice is a fixed kernel, or the rung's super-tile
    once the tile count crosses a threshold and its ST=1 fallback below it, so
    the start of each rung's slice and that threshold between them reach
    everything.
    """
    ladder = _ladder(algorithm, world_size, link)
    inbox_memory = _resolve_inbox_flags("auto", world_size)[1]
    batched = batches_publishes(inbox_memory, algorithm, link)
    grids = _rung_grids(world_size, ladder)
    ends = [floor - 1 for floor, *_ in ladder[1:]] + [hi]
    out: dict[tuple, int] = {}
    for (floor, st, _cap, b), end in zip(ladder, ends):
        start, end = max(lo, floor), min(hi, end)
        probes = [start]
        if st > 1:
            # The first payload with a whole super-tile when batched, and with
            # more tiles than blocks when not.
            tiles = st - 1 if batched else grids[(st, b)]
            probes.append(tiles * _tile_bytes(b) + 1)
        for probe in probes:
            nbytes = -(-max(probe, start) // _ROW_BYTES) * _ROW_BYTES
            if nbytes <= end:
                cfg = _expected_cfg(
                    nbytes, ladder=ladder, batched=batched, grid_by_cfg=grids
                )
                out[cfg] = min(out.get(cfg, nbytes), nbytes)
    return out


def _ship_payloads(
    world_size: int, algorithm: str, link: str
) -> tuple[list[int], tuple[int, int] | None]:
    """Shipping-sweep payloads for one schedule, and the dispatch window they
    cover.

    A schedule the policy selects on this host gets the smallest payload
    reaching each kernel it can run in its window, plus the window's top (cut
    at ``TOP_PAYLOAD_BYTES``) for a production-scale timing row. The window
    comes back so the run can check that kernel list against the engine's own.
    """
    policy = fly_policy.resolve_quant(link, world_size)
    if algorithm in fly_policy.quant_families_reachable(policy):
        lo, hi = fly_policy.quant_family_range(algorithm, policy)
        payloads = set(_kernel_payloads(world_size, algorithm, link, lo, hi).values())
        top = min(hi, TOP_PAYLOAD_BYTES) // _ROW_BYTES * _ROW_BYTES
        if top >= lo:
            payloads.add(top)
        return sorted(payloads), (lo, hi)
    by_cfg = _kernel_payloads(
        world_size, algorithm, link, policy.floor + 1, policy.max_bytes
    )
    _floor, st, _cap, b = _ladder(algorithm, world_size, link)[0]
    return [by_cfg.get((st, b), min(by_cfg.values()))], None


def _metrics(
    out: torch.Tensor, ref: torch.Tensor, rank: int, tile_bytes: int, group
) -> dict:
    got = out.to(torch.float32)
    mismatch = got != ref
    n_mismatch = int(mismatch.sum().item())
    first_bad = -1
    if n_mismatch:
        first_bad = int(torch.nonzero(mismatch.reshape(-1), as_tuple=False)[0].item())
    diff = (got - ref).abs()
    err = checkAllclose(
        ref,
        got,
        rtol=CLOSE_RTOL,
        atol=CLOSE_ATOL,
        tol_err_ratio=CLOSE_ERR_RATIO,
        printLog=False,
        msg=f"quick_allreduce rank {rank}",
    )
    return {
        "sqnr_db": sqnr_db(got, ref),
        "min_tile_sqnr_db": min_tile_sqnr_db(got, ref, tile_bytes),
        "lanes_differing": lanes_differing(out, group),
        "rel_mae": rel_mae(got, ref),
        "err": float(err),
        "n_mismatch": n_mismatch,
        "max_abs_err": float(diff.max().item()) if diff.numel() else 0.0,
        "first_bad": first_bad,
    }


def _run_rank(
    rank: int,
    tp: int,
    init_method: str,
    engine_kw: dict,
    cases: list[tuple],
    window: tuple[int, int] | None = None,
) -> list[dict]:
    """One rank of one spawn: build the engine, then run every case on it.

    A case is ``(tokens, hidden, fill, graph, time_it)``. The engine is built
    once, so every case shares its compiled binaries and IPC inbox. *window*
    is the payload range production dispatch routes to this engine, if any;
    every row then carries the kernels the engine itself says that range
    selects.
    """
    import torch.distributed as dist

    from aiter.ops.flydsl import FlyQuickAllReduce

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        world_size=tp,
        rank=rank,
        device_id=device,
    )
    # FlyQuickAllReduce exchanges IPC metadata over a non-NCCL group; NCCL
    # stays for the fp32 reference all-reduce.
    gloo = dist.new_group(backend="gloo")
    group = dist.group.WORLD

    fly = FlyQuickAllReduce(
        group=gloo,
        device=device,
        rank=rank,
        world_size=tp,
        # The cases deliberately include sub-threshold shapes (8x1024 is
        # 16 KiB, well under the default floor) to cover the partial-tile path.
        min_bytes=0,
        **engine_kw,
    )
    fly.preload()
    production_cfgs = None
    if window is not None:
        production_cfgs = [(int(st), int(b)) for st, b in fly.cfgs_for(*window)]

    rows = []
    try:
        for ntok, hidden, fill, graph, time_it in cases:
            inp = _make_inp(ntok, hidden, fill, rank=rank, device=device)
            ref = inp.to(torch.float32)
            dist.all_reduce(ref, group=group)
            dist.barrier()

            nbytes = int(inp.numel()) * int(inp.element_size())
            cfg_used, _ = fly._pick_cfg(nbytes)
            tile_bytes = _tile_bytes(cfg_used[1])

            out = torch.zeros_like(inp)
            fly.allreduce(inp, out)
            torch.cuda.synchronize()
            dist.barrier()
            m = _metrics(out, ref, rank, tile_bytes, group)

            g = None
            if graph:
                # Captured on the current stream, which allreduce() launches on
                # when it is given none; the IPC inbox is allocated at init, so
                # the captured launch is safe to replay.
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    fly.allreduce(inp, out)
                for _ in range(GRAPH_REPLAYS):
                    out.zero_()
                    g.replay()
                    torch.cuda.synchronize()
                    dist.barrier()
                    m = worst(m, _metrics(out, ref, rank, tile_bytes, group))

            st_used, block_used = cfg_used
            row = {
                **m,
                "link": fly.link,
                "inbox_memory": fly.inbox_memory,
                # Resolved, not requested: these come from the per-world-size
                # default unless the caller pinned a lap, and a regression
                # should name the codec that produced it.
                "rs_codec": fly.rs_codec,
                "ag_codec": fly.ag_codec,
                "st_used": int(st_used),
                "block_used": int(block_used),
                "grid_by_cfg": {cfg: int(e.grid) for cfg, e in fly._by_cfg.items()},
                "production_cfgs": production_cfgs,
                "us": None,
            }
            if time_it:
                dist.barrier(group=group)
                torch.cuda.synchronize()
                if g is not None:
                    fn = g.replay
                else:

                    def fn(eng=fly, src=inp, dst=out):
                        eng.allreduce(src, dst)
                        return dst

                # cuda.Event timing, not run_perftest's default profiler timer:
                # `import aiter` creates a GPU context in the parent, and on
                # some ROCm/torch builds a child spawned after that records no
                # GPU events in torch.profiler, so the default timer fails
                # reducing an empty trace.
                _, us = run_perftest(fn, use_cuda_event=True)
                row["us"] = float(us)
            rows.append(row)
            del inp, out, ref, g
            torch.cuda.empty_cache()
    finally:
        fly.close()
        dist.destroy_process_group()
    return rows


# Rows are registered up front, grouped by the engine they need, so that each
# engine is built by exactly one spawn however many tables read from it. A case
# is ``(tokens, hidden, fill, graph, time_it)``; ``registry.windows`` holds the
# production dispatch window of a shipping engine whose kernel coverage is
# checked.
registry = SpawnRegistry(_run_rank)
failures = FailureLog()


def _identity_fails(rows: list[dict], algorithm: str) -> list[str]:
    """Cross-rank bit-identity, asserted for the mesh onlr."""
    if algorithm != "mesh":
        return []
    return [
        f"rank {rank}: {row['lanes_differing']} bf16 lanes differ between ranks"
        for rank, row in enumerate(rows)
        if row["lanes_differing"]
    ]


def _check_sqnr(
    label: str,
    rows: list[dict],
    *,
    floor: float,
    expected_st: int | None,
    algorithm: str,
) -> bool:
    fails = _identity_fails(rows, algorithm)
    for rank, row in enumerate(rows):
        if expected_st is not None and row["st_used"] != expected_st:
            fails.append(f"rank {rank}: ST={row['st_used']}, expected {expected_st}")
        if row["sqnr_db"] < floor:
            fails.append(
                f"rank {rank}: SQNR {row['sqnr_db']:.2f} dB < {floor} "
                f"(rel MAE {row['rel_mae']:.3e})"
            )
        if row["min_tile_sqnr_db"] < TILE_SQNR_MIN_DB:
            fails.append(
                f"rank {rank}: min-tile SQNR {row['min_tile_sqnr_db']:.2f} dB "
                f"< {TILE_SQNR_MIN_DB}"
            )
        if row["err"] >= CLOSE_ERR_RATIO:
            fails.append(
                f"rank {rank}: checkAllclose err {row['err']:.3f} >= {CLOSE_ERR_RATIO}"
            )
    return failures.check(label, fails)


def _shipping_st(
    rows: list[dict],
    nbytes: int,
    algorithm: str,
    tp: int,
    block: int | None = None,
) -> int:
    return _expected_st(
        nbytes,
        algorithm=algorithm,
        world_size=tp,
        link=rows[0]["link"],
        inbox_memory=rows[0]["inbox_memory"],
        grid_by_cfg=rows[0]["grid_by_cfg"],
        block=block,
    )


def _summary(rows: list[dict]) -> dict:
    return {
        "gfx": ARCH,
        "inbox_memory": rows[0]["inbox_memory"],
        "rs_codec": rows[0]["rs_codec"],
        "ag_codec": rows[0]["ag_codec"],
        "st_used": rows[0]["st_used"],
        "block_used": rows[0]["block_used"],
        "lanes_differing": max(r["lanes_differing"] for r in rows),
        "err": max(r["err"] for r in rows),
        "sqnr_db": min(r["sqnr_db"] for r in rows),
        "min_tile_sqnr_db": min(r["min_tile_sqnr_db"] for r in rows),
    }


def _ship_key(
    tp: int,
    algorithm: str,
    grid_cap: int | None,
    block: int | None = None,
) -> tuple:
    kw = {"algorithm": algorithm}
    for name, val in (("grid_cap", grid_cap), ("block", block)):
        if val is not None:
            kw[name] = val
    return registry.key(tp, **kw)


def _transport_key(
    tp: int,
    algorithm: str,
    super_tile: int | None = None,
    block: int | None = None,
) -> tuple:
    """The fp16-wire engine: the schedule's own ladder when *super_tile* is
    None, otherwise that geometry pinned at ``TRANSPORT_GRID_CAP``."""
    kw = {"algorithm": algorithm, "rs_codec": "fp16", "ag_codec": "fp16"}
    if super_tile is not None:
        kw.update(
            super_tile=super_tile,
            grid_cap=TRANSPORT_GRID_CAP,
            block=block,
        )
    return registry.key(tp, **kw)


# ---------------------------------------------------------------------------
# Fused epilogue: FlyQuickAllReduceRMSNorm (all-reduce + residual add + RMSNorm)
# ---------------------------------------------------------------------------

RMS_EPS = 1e-6

# Floor for the fused ``out`` and ``residual_out`` against the fp32 oracle.
#
# Lower than SQNR_MIN_DB, and not because the epilogue is noisy: the reference
# for a fused run adds the residual, and the residual is signal the all-reduce
# noise is measured against. The numbers land 21-27 dB across TP2..TP8, so 18.0
# leaves the same headroom the plain floor does.
FUSED_SQNR_MIN_DB = 18.0

# ``out`` recomputed from the kernel's own ``residual_out`` must match the
# kernel's ``out``. Both are bf16 and the two rstds differ (the kernel norms the
# unrounded fp32 x, this reference norms the rounded bf16 one), so it is a high
# floor rather than equality -- but far above anything a row-grouping or
# weight-indexing bug could survive, since those move whole rows.
FUSED_SELF_SQNR_MIN_DB = 40.0

# Sentinel rows after each fused output. A tile holds at most ATOMS (8) rows, so
# a partial last tile has at most 7 dead rows; 8 covers every one of them. The
# bits are a bf16 NaN, which no live store produces.
_GUARD_ROWS = 8
_GUARD_BITS = 0x7FC1


def _fused_inputs(tokens, hidden, rank, device):
    """Per-rank input, and the residual and gain every rank shares.

    Values stay inside fp16's exponent range so the ``fp16`` passthrough wire
    is genuinely lossless -- the exactness case below depends on it.

    ``inp`` and ``residual`` are followed by _GUARD_ROWS rows of ones. A load
    past row M would otherwise read whatever the allocator left there -- which
    can be a previous case's NaN sentinel, and a dead row computed from it
    writes the sentinel's own bits back, hiding the store the guard is for.
    """

    def _head(values):
        buf = torch.ones(tokens + _GUARD_ROWS, hidden, dtype=dtypes.bf16, device=device)
        buf[:tokens] = values.to(device, dtypes.bf16)
        return buf[:tokens]

    g = torch.Generator().manual_seed(1234 + rank)
    inp = _head(torch.randn(tokens, hidden, generator=g) * 0.25)
    gs = torch.Generator().manual_seed(99)
    residual = _head(torch.randn(tokens, hidden, generator=gs) * 0.5)
    weight = (torch.randn(hidden, generator=gs) * 0.1 + 1.0).to(device, dtypes.bf16)
    return inp, residual, weight


def _run_rank_fused(
    rank: int,
    tp: int,
    init_method: str,
    cases: list[tuple[int, int]],
    algorithm: str,
    codecs: tuple[str | None, str | None],
    solo: bool,
    engine_kw: dict | None = None,
) -> list[dict]:
    """One rank of a fused run, over every case in one spawn.

    Every case rides a single engine object and a single process, as the plain
    worker above does. That is not only startup cost: each fused engine holds an
    IPC inbox per (hidden, super-tile) rung, hundreds of MiB at the ring's high
    rungs, and a pool per test leaves those in flight while the next pool tries
    to allocate. ``FlyQuickAllReduceRMSNorm`` builds per-hidden engines on
    demand, so one object covers every width here.

    ``solo`` zeroes every rank but 0, which -- paired with the ``fp16``
    passthrough wire -- makes the all-reduce exact. That is the only
    configuration in which ``residual_out`` can be checked for bit-equality,
    and it is the cheapest detector for a dropped bf16 round-trip: SQNR on a
    quantized wire is far too loose to notice one, and the error it hides
    compounds per layer.
    """
    import torch.distributed as dist

    from aiter.ops.flydsl.quick_allreduce import FlyQuickAllReduceRMSNorm

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        world_size=tp,
        rank=rank,
        device_id=device,
    )
    gloo = dist.new_group(backend="gloo")
    rs_codec, ag_codec = codecs

    eng = FlyQuickAllReduceRMSNorm(
        group=gloo,
        device=device,
        rank=rank,
        world_size=tp,
        algorithm=algorithm,
        rs_codec=rs_codec,
        ag_codec=ag_codec,
        **(engine_kw or {}),
    )
    # The cases here are deliberately small; the floor is a speed policy, not a
    # correctness limit, so it must not decide what this test covers.
    eng.min_bytes = 0
    rows = []
    try:
        for tokens, hidden in cases:
            inp, residual, weight = _fused_inputs(tokens, hidden, rank, device)
            if solo and rank != 0:
                inp.zero_()

            # Build and JIT this width up front, so the checked call below is
            # never the one that compiles.
            eng.preload(hidden)

            # Both outputs are the head of a buffer with a sentinel tail of
            # _GUARD_ROWS rows, so a store past row M -- a dead row of a partial
            # last tile -- shows up here rather than in whatever allocation
            # happens to follow the tensor.
            guard = torch.full(
                (2, tokens + _GUARD_ROWS, hidden), _GUARD_BITS, dtype=torch.int16
            ).to(device)
            out = guard[0, :tokens].view(dtypes.bf16)
            res_out = guard[1, :tokens].view(dtypes.bf16)
            eng.allreduce_rmsnorm(
                inp, residual, weight, RMS_EPS, out=out, residual_out=res_out
            )
            torch.cuda.synchronize()
            dist.barrier()
            tail_intact = bool((guard[:, tokens:] == _GUARD_BITS).all())

            # fp32 oracle of the contract, through an untimed NCCL all-reduce.
            ar = inp.to(torch.float32)
            dist.all_reduce(ar, group=dist.group.WORLD)
            x = ar.to(dtypes.bf16).to(torch.float32) + residual.to(torch.float32)
            res_ref = x.to(dtypes.bf16)
            rstd = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + RMS_EPS)
            out_ref = (x * rstd * weight.to(torch.float32)).to(dtypes.bf16)

            # The epilogue judged on its own, with the codec noise factored
            # out: re-norm the kernel's residual_out and compare to its out.
            xr = res_out.to(torch.float32)
            out_self = (
                xr
                * torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + RMS_EPS)
                * weight.to(torch.float32)
            ).to(dtypes.bf16)

            rows.append(
                {
                    "rank": rank,
                    "tokens": tokens,
                    "hidden": hidden,
                    "variant": eng.variant(hidden, int(inp.numel()) * 2),
                    "out_sqnr_db": sqnr_db(
                        out.to(torch.float32), out_ref.to(torch.float32)
                    ),
                    "res_sqnr_db": sqnr_db(
                        res_out.to(torch.float32), res_ref.to(torch.float32)
                    ),
                    "self_sqnr_db": sqnr_db(
                        out.to(torch.float32), out_self.to(torch.float32)
                    ),
                    "res_exact": bool(torch.equal(res_out, res_ref)),
                    "tail_intact": tail_intact,
                    "finite": bool(torch.isfinite(out.to(torch.float32)).all()),
                    # Cheap cross-rank fingerprints; see the mesh identity test.
                    "out_bits": int(
                        out.view(torch.int16).to(torch.int64).abs().sum().item()
                    ),
                    "res_bits": int(
                        res_out.view(torch.int16).to(torch.int64).abs().sum().item()
                    ),
                }
            )
            del inp, residual, weight, out, res_out, ar, x, res_ref, out_ref, xr
            del out_self, guard
            torch.cuda.empty_cache()
    finally:
        eng.close()
        dist.destroy_process_group()
    return rows


def _spawn_fused(
    world_size: int,
    cases: list[tuple[int, int]],
    *,
    algorithm: str = "ring",
    codecs: tuple[str | None, str | None] = (None, None),
    solo: bool = False,
    engine_kw: dict | None = None,
) -> list[list[dict]]:
    # A fused build is a persistent kernel whose blocks wait on the same block
    # id at every peer, so a co-residency bug hangs rather than returning a
    # wrong number. The timeout is what turns that into a failed test instead
    # of a wedged run.
    return run_on_ranks(
        world_size,
        _run_rank_fused,
        timeout=float(os.environ.get("FLYDSL_QR_TIMEOUT", "3600")),
        cases=cases,
        algorithm=algorithm,
        codecs=codecs,
        solo=solo,
        engine_kw=engine_kw,
    )


_FUSED_BATCH_CACHE: dict[tuple, dict[tuple[int, int], list[dict]]] = {}


def _fused_batch(key: tuple, cases: list[tuple[int, int]], **spawn_kwargs) -> dict:
    """One ``_spawn_fused`` per *key*, memoized, indexed by shape.

    Same bargain as ``_result`` above and for a sharper reason: a fused
    engine's IPC inboxes are large, and a pool per test leaves the previous
    one's still resident when the next allocates.
    """
    if key not in _FUSED_BATCH_CACHE:
        ranks = _spawn_fused(key[0], cases, **spawn_kwargs)
        _FUSED_BATCH_CACHE[key] = {
            case: [rank_rows[i] for rank_rows in ranks] for i, case in enumerate(cases)
        }
    return _FUSED_BATCH_CACHE[key]


_FUSED_CASES = (
    # (tp, tokens, hidden, label). Widths chosen for what they exercise:
    # 4096 is an 8-wave block, 7168 a 14-wave one (the only non-power-of-two
    # width, and DeepSeek's), 5120 exercises rows_per_chunk > 1 at TP4, and
    # 2048 is the narrowest block the mesh fanout accepts. The last case of
    # each world size is the co-residency stress: the payload is far larger
    # than the grid, so every block loops over many tiles.
    (2, 512, 4096, "tp2-4096"),
    (2, 256, 2048, "tp2-2048-narrow-block"),
    (2, 4096, 8192, "tp2-8192-wide-block-many-tiles"),
    (4, 1024, 5120, "tp4-5120"),
    (8, 1024, 7168, "tp8-7168"),
    (8, 512, 8192, "tp8-8192-wide-block"),
)


#: The fused mesh is tested at both super-tile paths it has: ST=1 carries the
#: own share and chunk in registers across the waits, ST>1 reloads the share
#: from the input and parks the chunk in ``out`` between its three loops.
_FUSED_MESH_STS = (1, 8)

#: Widths with no native row geometry, which run on a *padded* workgroup.
#:
#: Graded with a **lossless** wire on purpose. Padding must be invisible to the
#: answer, so under fp16 a padded build has to produce the same bits a native
#: one would.
_FUSED_PAD_CASES = (
    (2, 512, 1536, "tp2-1536pad2048"),
    (2, 11, 1536, "tp2-1536pad2048-partial-tile"),
)

# (tp, tokens, hidden) of the lossless one-live-rank residual equality.
_FUSED_EXACT_CASES = ((2, 512, 4096),)

# (tp, tokens, hidden) of the mesh cross-rank bit-identity check.
_FUSED_MESH_AGREE_CASES = ((4, 1024, 5120),)

#: ``tokens`` is large enough that a pinned ST=8 really runs ST=8 at every
#: world size and inbox type here; the test checks it.
_FUSED_MESH_ST_EXACT_CASES = ((2, 4096, 4096), (4, 4096, 4096), (2, 4096, 1536))

# Off the row-sized-block grid: 2560 and 640 are not multiples of 128 threads
# per block, 12288 is wider than 1024.
_FUSED_REJECTED_HIDDENS = (2560, 640, 12288)

# With a lossless wire and one live rank the all-reduce is exact, so ``out``
# differs from the oracle only by rsqrt's hardware approximation.
FUSED_EXACT_OUT_SQNR_MIN_DB = 60.0


def _pinned_st_kw(super_tile: int) -> dict:
    return {"super_tile": super_tile}


def _fused_sqnr_fails(rows: list[dict]) -> list[str]:
    """The SQNR gates of ``test_quick_allreduce_rmsnorm_sqnr``, per rank."""
    fails = []
    for row in rows:
        if not row["finite"]:
            fails.append(f"rank {row['rank']}: non-finite output")
        if not row["tail_intact"]:
            fails.append(f"rank {row['rank']}: stored past the last row")
        if row["out_sqnr_db"] < FUSED_SQNR_MIN_DB:
            fails.append(
                f"rank {row['rank']}: out SQNR {row['out_sqnr_db']:.2f} dB < "
                f"{FUSED_SQNR_MIN_DB}"
            )
        if row["res_sqnr_db"] < FUSED_SQNR_MIN_DB:
            fails.append(
                f"rank {row['rank']}: residual_out SQNR {row['res_sqnr_db']:.2f} dB "
                f"< {FUSED_SQNR_MIN_DB}"
            )
        if row["self_sqnr_db"] < FUSED_SELF_SQNR_MIN_DB:
            fails.append(
                f"rank {row['rank']}: out vs norm(residual_out) "
                f"{row['self_sqnr_db']:.2f} dB < {FUSED_SELF_SQNR_MIN_DB} -- the "
                "epilogue disagrees with its own residual"
            )
    return fails


def _fused_tail_fails(rows: list[dict], why: str) -> list[str]:
    bad = [r["rank"] for r in rows if not r["tail_intact"]]
    return [f"stored past the last row on ranks {bad} -- {why}"] if bad else []


def _fused_exact_fails(rows: list[dict], why: str) -> list[str]:
    """``residual_out`` bit-exact and ``out`` near-exact, on a lossless wire."""
    bad = [r["rank"] for r in rows if not r["res_exact"]]
    fails = (
        [f"residual_out is not bit-exact on ranks {bad} with a lossless wire -- {why}"]
        if bad
        else []
    )
    fails += [
        f"rank {r['rank']}: out SQNR {r['out_sqnr_db']:.2f} dB, not above "
        f"{FUSED_EXACT_OUT_SQNR_MIN_DB}"
        for r in rows
        if not r["out_sqnr_db"] > FUSED_EXACT_OUT_SQNR_MIN_DB
    ]
    return fails


def _fused_disagree_fails(rows: list[dict]) -> list[str]:
    first = rows[0]
    return [
        f"rank {r['rank']} disagrees with rank 0"
        for r in rows
        if (r["out_bits"], r["res_bits"]) != (first["out_bits"], first["res_bits"])
    ]


def _fused_summary(rows: list[dict]) -> dict:
    return {
        "gfx": ARCH,
        "variant": rows[0]["variant"],
        "out_sqnr_db": min(r["out_sqnr_db"] for r in rows),
        "res_sqnr_db": min(r["res_sqnr_db"] for r in rows),
        "self_sqnr_db": min(r["self_sqnr_db"] for r in rows),
        "res_exact": all(r["res_exact"] for r in rows),
        "tail_intact": all(r["tail_intact"] for r in rows),
    }


@benchmark()
def test_quick_allreduce_rmsnorm_sqnr(tp, tokens, hidden, label, algorithm):
    """The fused epilogue against an fp32 oracle, plus its own self-consistency.

    Three gates per rank: ``out`` and ``residual_out`` against the oracle, and
    ``out`` against a norm recomputed from the kernel's own ``residual_out``.
    The third is the one that isolates the epilogue -- it is blind to codec
    noise, so a row-grouping or weight-indexing bug cannot hide behind it.

    Every case sharing (tp, algorithm) rides one spawn; see ``_fused_batch``.
    """
    cases = [(t, h) for w, t, h, _ in _FUSED_CASES if w == tp]
    batch = _fused_batch((tp, "sqnr", algorithm), cases, algorithm=algorithm)
    rows = batch[(tokens, hidden)]
    failures.check(f"{label}/{algorithm}", _fused_sqnr_fails(rows))
    return _fused_summary(rows)


@benchmark()
def test_quick_allreduce_rmsnorm_padded_is_bit_exact(
    tp, tokens, hidden, label, algorithm
):
    """A padded width on a lossless wire is bit-exact, same as a native one.

    This is the test that isolates the padded addressing. The kernel runs a
    block covering ``h_pad`` and masks the lanes past the real row; a mask one
    atom too wide folds the next row's first 16 B into this row's norm, and a
    store that is not dropped overwrites the next row's first 16 B. Both move
    bits that an equality sees and that an SQNR floor would not.
    """
    batch = _fused_batch(
        (tp, "pad", algorithm),
        [(t, h) for w, t, h, _ in _FUSED_PAD_CASES if w == tp],
        algorithm=algorithm,
        codecs=("fp16", "fp16"),
        solo=True,
    )
    rows = batch[(tokens, hidden)]
    fails = _fused_tail_fails(rows, "a dead row of the partial last tile is not masked")
    fails += _fused_exact_fails(rows, "a pad lane is reaching the real row")
    failures.check(f"{label}/{algorithm}", fails)
    return _fused_summary(rows)


@benchmark()
def test_quick_allreduce_rmsnorm_residual_is_bit_exact(tp, tokens, hidden, algorithm):
    """With a lossless wire and one live rank, ``residual_out`` is exact.

    ``out`` still goes through ``rsqrt``, whose hardware approximation
    legitimately differs from torch's, so only the residual is an equality.
    This is what pins the bf16 round-trip before the residual add: keeping the
    extra fp32 mantissa bits would be *more* accurate and would silently
    diverge from the unfused path it has to match.
    """
    batch = _fused_batch(
        (tp, "exact", algorithm),
        [(tokens, hidden)],
        algorithm=algorithm,
        codecs=("fp16", "fp16"),
        solo=True,
    )
    rows = batch[(tokens, hidden)]
    failures.check(
        f"tp{tp} {tokens}x{hidden} {algorithm}",
        _fused_exact_fails(rows, "the bf16 round-trip or the residual add has drifted"),
    )
    return _fused_summary(rows)


@benchmark()
def test_quick_allreduce_rmsnorm_mesh_ranks_agree(tp, tokens, hidden):
    """Every mesh rank must compute the same bits.

    The mesh dequantizes the same wire bytes for every atom, so its output is a
    pure function of what went over the wire and the ranks cannot diverge. The
    **ring** deliberately can: at the seam op it stores its own chunk from the
    unquantized accumulator, which is a better value than the one its peers
    receive -- so this gate is mesh-only, and a ring that passed it would mean
    that optimization had been lost.
    """
    rows = _fused_batch((tp, "sqnr", "mesh"), [(tokens, hidden)], algorithm="mesh")[
        (tokens, hidden)
    ]
    failures.check(f"tp{tp} {tokens}x{hidden} mesh", _fused_disagree_fails(rows))
    return _fused_summary(rows)


@benchmark()
def test_quick_allreduce_rmsnorm_mesh_pinned_st(tp, tokens, hidden, label, super_tile):
    """The fused mesh at a pinned super-tile: the SQNR gates, and every rank
    agreeing.

    A rank's own share never goes through its codec on the reduce-scatter lap,
    and its own reduced chunk never goes through its inbox on the all-gather
    lap. That chunk is the dequant of the very packet its peers receive, so the
    ranks must still agree bit for bit; a rank that used its unquantized
    accumulator instead would not.

    A pinned ST=8 still falls back to ST=1 on payloads with too few tiles, so
    only the variant says which path a case took.
    """
    cases = [(t, h) for w, t, h, _ in _FUSED_CASES if w == tp]
    batch = _fused_batch(
        (tp, "st", "mesh", super_tile),
        cases,
        algorithm="mesh",
        engine_kw=_pinned_st_kw(super_tile),
    )
    rows = batch[(tokens, hidden)]
    fails = _fused_sqnr_fails(rows) + _fused_disagree_fails(rows)
    failures.check(f"{label}/mesh/st{super_tile}", fails)
    return _fused_summary(rows)


@benchmark()
def test_quick_allreduce_rmsnorm_mesh_st8_is_exercised(tp):
    """At least one case per world size really takes the ST>1 path, so the
    pinned-ST=8 rows above are not all silently ST=1."""
    cases = [(t, h) for w, t, h, _ in _FUSED_CASES if w == tp]
    batch = _fused_batch(
        (tp, "st", "mesh", 8), cases, algorithm="mesh", engine_kw=_pinned_st_kw(8)
    )
    variants = sorted({r["variant"] for rows in batch.values() for r in rows})
    fails = []
    if not any("_st8_" in v for v in variants):
        fails.append(f"no ST=8 variant: {variants}")
    failures.check(f"tp{tp} mesh ST=8 coverage", fails)
    return {"gfx": ARCH, "variants": variants}


@benchmark()
def test_quick_allreduce_rmsnorm_mesh_pinned_st_is_bit_exact(
    tp, tokens, hidden, super_tile
):
    """A pinned super-tile on a lossless wire with one live rank:
    ``residual_out`` exact.

    The own slot sits at ``self_rank * rank_atoms`` in the tile; an off-by-one
    there moves a whole atom, which only an equality sees. TP4 puts some rank's
    slot in the middle of the tile, and 1536 runs on a padded workgroup -- the
    case where ST>1's reload of the own share must take the per-row route.
    """
    batch = _fused_batch(
        (tp, "st-exact", "mesh", hidden, super_tile),
        [(tokens, hidden)],
        algorithm="mesh",
        codecs=("fp16", "fp16"),
        solo=True,
        engine_kw=_pinned_st_kw(super_tile),
    )
    rows = batch[(tokens, hidden)]
    fails = [
        f"rank {r['rank']} ran {r['variant']}, not st{super_tile}"
        for r in rows
        if f"_st{super_tile}_" not in r["variant"]
    ]
    fails += _fused_exact_fails(rows, "the own slot is misplaced")
    failures.check(f"tp{tp} {tokens}x{hidden} st{super_tile}", fails)
    return _fused_summary(rows)


def _expect_raises(what: str, pattern: str, fn) -> list[str]:
    try:
        fn()
    except ValueError as exc:
        if re.search(pattern, str(exc)):
            return []
        return [f"{what}: ValueError {str(exc)!r} does not match {pattern!r}"]
    return [f"{what}: did not raise ValueError"]


def _host_rejects_unsupported_hidden() -> list[str]:
    """Widths off the row-sized-block grid raise, naming the constraint.

    A fused block is ``hidden/8`` threads and has to be a multiple of 128 (the
    64 B fabric sector grid) and at most 1024, so 2560 and 640 are off-grid and
    12288 is too wide.
    """
    from aiter.ops.flydsl.kernels.quick_allreduce_fusions import (
        quick_reduce_hidden_supported,
        quick_reduce_row_block,
    )

    fails = []
    for hidden in _FUSED_REJECTED_HIDDENS:
        if quick_reduce_hidden_supported(hidden, 8):
            fails.append(f"hidden={hidden}: reported supported")
        fails += _expect_raises(
            f"hidden={hidden}",
            "hidden",
            lambda h=hidden: quick_reduce_row_block(h, 8),
        )
    return fails


def _host_supported_hiddens() -> list[str]:
    """Every multiple of 1024 up to 8192 builds at every world size."""
    from aiter.ops.flydsl.kernels.quick_allreduce_fusions import quick_reduce_row_block

    fails = []
    for world_size in SUPPORTED_WORLDS:
        for hidden in range(1024, 8192 + 1, 1024):
            block, atoms_per_row = quick_reduce_row_block(hidden, world_size)
            if not (atoms_per_row == 1 and block == hidden // 8):
                fails.append(
                    f"hidden={hidden} tp={world_size}: block={block} "
                    f"atoms_per_row={atoms_per_row}"
                )
    return fails


def _host_padded_row_block_is_least_wire() -> list[str]:
    """The padded pick is the narrowest legal width at or above hidden dim.

    Guards the selection rule rather than the kernel: padding costs
    ``(h_pad-hidden)/hidden`` extra wire on a bandwidth-bound schedule, so
    taking anything but the least is a silent throughput loss.

    Also pins the half that matters more -- a width with a native geometry must
    resolve to ``h_pad == hidden`` and the same ``(block, atoms_per_row)`` it
    always did, or every shipped shape starts paying for this.
    """
    from aiter.ops.flydsl.kernels.quick_allreduce_fusions import (
        quick_reduce_padded_row_block_options,
        quick_reduce_row_block_options,
    )

    fails = []
    for world_size in SUPPORTED_WORLDS:
        for hidden in range(8, 32768 + 1, 8):
            opts = quick_reduce_padded_row_block_options(hidden, world_size)
            if not opts:
                continue
            where = f"hidden={hidden} tp={world_size}"
            block, atoms_per_row, h_pad = opts[0]
            if not (h_pad >= hidden and block * atoms_per_row * 8 == h_pad):
                fails.append(f"{where}: malformed leading option {opts[0]}")
            if h_pad != min(o[2] for o in opts):
                fails.append(f"{where}: leads with h_pad={h_pad}, not the least")
            native = quick_reduce_row_block_options(hidden, world_size)
            if native and (block, atoms_per_row, h_pad) != (*native[0], hidden):
                fails.append(
                    f"{where}: native geometry {native[0]} but the padded list "
                    f"leads with {(block, atoms_per_row, h_pad)}"
                )
    return fails


def _host_mesh_fanout_quad_budget_gates_narrow_blocks() -> list[str]:
    """The mesh needs a quad per (peer, sector) of a stripe, and the gate knows.

    The row geometry is necessary but not sufficient: ``hidden=1024`` at TP8
    gives a 128-thread block, which passes every row constraint and then has
    too few quads for the fanout to issue in one pass. Before this predicate
    existed the host advertised that width and the factory raised on it.
    """
    from aiter.ops.flydsl.kernels.quick_allreduce_mesh import (
        make_quick_allreduce_mesh_kernel,
        mesh_fanout_fits,
    )

    fails = []
    if mesh_fanout_fits(128, 8, "int4"):
        fails.append("mesh_fanout_fits(128, 8, 'int4') is True")
    if not mesh_fanout_fits(256, 8, "int4"):
        fails.append("mesh_fanout_fits(256, 8, 'int4') is False")
    # The predicate and the factory must agree, or the gate lies again.
    fails += _expect_raises(
        "hidden=1024 tp=8 factory",
        "quads",
        lambda: make_quick_allreduce_mesh_kernel(
            world_size=8, grid=64, rank=0, fusion="rmsnorm", hidden=1024
        ),
    )
    for world_size in SUPPORTED_WORLDS:
        for block in range(128, 1024 + 1, 128):
            fits = mesh_fanout_fits(block, world_size, "int4")
            try:
                make_quick_allreduce_mesh_kernel(
                    world_size=world_size,
                    grid=64,
                    rank=0,
                    fusion="rmsnorm",
                    hidden=block * 8,
                )
                built = True
            except ValueError as exc:
                if "quads" not in str(exc):
                    continue  # LDS or another limit; not what this check is about
                built = False
            if built != fits:
                fails.append(
                    f"tp={world_size} block={block}: predicate says {fits}, "
                    f"factory built={built}"
                )
    return fails


def _host_row_block_is_first_option() -> list[str]:
    """Enumerating the row geometries did not move the pick.

    ``quick_reduce_row_block`` is the widest-block entry of
    ``quick_reduce_row_block_options``, and every existing caller takes it, so
    this is the guard that the refactor is invisible to the shipped kernels.
    """
    from aiter.ops.flydsl.kernels.quick_allreduce_fusions import (
        quick_reduce_row_block,
        quick_reduce_row_block_options,
    )

    fails = []
    for world_size in SUPPORTED_WORLDS:
        for hidden in list(range(1024, 16384 + 1, 1024)) + [5120, 7168]:
            where = f"hidden={hidden} tp={world_size}"
            opts = quick_reduce_row_block_options(hidden, world_size)
            # Widest first, and each entry really is block * 8 * atoms.
            if list(opts) != sorted(opts, reverse=True):
                fails.append(f"{where}: options not widest first: {opts}")
            fails += [
                f"{where}: block={block} atoms={atoms} does not cover hidden"
                for block, atoms in opts
                if block * 8 * atoms != hidden
            ]
            if opts:
                if quick_reduce_row_block(hidden, world_size) != opts[0]:
                    fails.append(f"{where}: pick is not the first option {opts[0]}")
            else:
                fails += _expect_raises(
                    where,
                    "no fused build",
                    lambda h=hidden, w=world_size: quick_reduce_row_block(h, w),
                )
    return fails


def _host_fused_one_shot_block_options() -> list[str]:
    """The fused one-shot's whole block axis, and its inverse.

    ``block * atoms * 8 == hidden`` with atoms in ``SUPPORTED_ATOMS`` and the
    block a whole number of waves at most 1024 -- so the axis is short and
    width-dependent, which is the thing a tuner has to be told rather than
    allowed to assume.
    """
    from aiter.ops.flydsl.kernels.one_shot_allreduce import (
        fused_atoms_for_block,
        fused_block,
        fused_block_options,
    )

    expected = {
        8192: ((1024, 1), (512, 2), (256, 4)),
        # atoms=4 would want 224 threads at 7168, which is not a whole wave.
        7168: ((896, 1), (448, 2)),
        5120: ((640, 1), (320, 2)),
        # atoms=1 would want 2048 threads, over the 1024 limit.
        16384: ((1024, 2), (512, 4)),
        6000: (),
    }
    fails = [
        f"hidden={hidden}: block options {fused_block_options(hidden)}, "
        f"expected {want}"
        for hidden, want in expected.items()
        if fused_block_options(hidden) != want
    ]
    for hidden in (2048, 4096, 5120, 7168, 8192, 16384):
        for block, atoms in fused_block_options(hidden):
            if fused_atoms_for_block(hidden, block) != atoms:
                fails.append(f"hidden={hidden} block={block}: atoms inverse differs")
            if fused_block(hidden, atoms) != block:
                fails.append(f"hidden={hidden} atoms={atoms}: block inverse differs")
    fails += _expect_raises(
        "hidden=7168 block=256", "896, 448", lambda: fused_atoms_for_block(7168, 256)
    )
    return fails


_HOST_CHECKS = {
    "rejects_unsupported_hidden": _host_rejects_unsupported_hidden,
    "supported_hiddens": _host_supported_hiddens,
    "padded_row_block_is_least_wire": _host_padded_row_block_is_least_wire,
    "mesh_fanout_quad_budget": _host_mesh_fanout_quad_budget_gates_narrow_blocks,
    "row_block_is_first_option": _host_row_block_is_first_option,
    "fused_one_shot_block_options": _host_fused_one_shot_block_options,
}


@benchmark()
def test_quick_allreduce_host_checks(check):
    """Host-side, GPU-free checks of the fused row-geometry helpers."""
    fails = _HOST_CHECKS[check]()
    if len(fails) > 10:
        fails = fails[:10] + [f"... and {len(fails) - 10} more"]
    return {"gfx": ARCH, "passed": failures.check(f"host-side {check}", fails)}


@benchmark()
def test_quick_allreduce(
    tokens,
    hidden,
    dtype,
    tp,
    algorithm,
    grid_cap=None,
    graph=False,
    block=None,
):
    """Shipping configuration: no codec or super-tile pinned."""
    rows = registry.result(
        _ship_key(tp, algorithm, grid_cap, block),
        (tokens, hidden, "normal", graph, True),
    )
    nbytes = tokens * hidden * 2
    _check_sqnr(
        f"tp={tp} {algorithm} {tokens}x{hidden} graph={graph} block={block}",
        rows,
        floor=SQNR_MIN_DB,
        expected_st=_shipping_st(rows, nbytes, algorithm, tp, block),
        algorithm=algorithm,
    )
    # (tp - 1) adds per element; codec ALU work is not counted.
    flops = tokens * hidden * (tp - 1)
    us = max(r["us"] for r in rows)
    ret = _summary(rows)
    ret.update(
        {
            "flydsl us": us,
            "flydsl TFLOPS": flops / us / 1e6,
            "flydsl TB/s": nbytes / us / 1e6,
            "flydsl err": ret.pop("err"),
        }
    )
    return ret


@benchmark()
def test_quick_allreduce_edge_inputs(tokens, hidden, tp, algorithm, fill):
    """Edge-case payloads on the shipping engine; correctness only."""
    rows = registry.result(
        _ship_key(tp, algorithm, None), (tokens, hidden, fill, False, False)
    )
    label = f"tp={tp} {algorithm} {tokens}x{hidden} fill={fill}"
    if fill == "degenerate":
        # A group whose extremum is zero decodes to a zero scale, so the encode
        # reciprocal saturates; before it was clamped, that reached the codec
        # as Inf and 0 * Inf poisoned the tile. Only finiteness is asserted.
        failures.check(
            label,
            [
                f"rank {rank}: rel MAE {row['rel_mae']}"
                for rank, row in enumerate(rows)
                if not math.isfinite(row["rel_mae"])
            ],
        )
    else:
        _check_sqnr(
            label,
            rows,
            floor=SQNR_MIN_DB,
            expected_st=_shipping_st(rows, tokens * hidden * 2, algorithm, tp),
            algorithm=algorithm,
        )
    ret = _summary(rows)
    ret["rel_mae"] = max(r["rel_mae"] for r in rows)
    return ret


@benchmark()
def test_quick_allreduce_pinned_codec(tokens, hidden, tp, rs_codec, ag_codec):
    """The ring with both laps' wire formats pinned; correctness only."""
    key = registry.key(tp, algorithm="ring", rs_codec=rs_codec, ag_codec=ag_codec)
    rows = registry.result(key, (tokens, hidden, "normal", False, False))
    label = f"tp={tp} ring {tokens}x{hidden} rs={rs_codec} ag={ag_codec}"
    _check_sqnr(
        label,
        rows,
        floor=SQNR_MIN_DB_TP8_INT4_RING,
        expected_st=_shipping_st(rows, tokens * hidden * 2, "ring", tp),
        algorithm="ring",
    )
    failures.check(
        label,
        [
            f"rank {rank}: resolved codecs {row['rs_codec']}/{row['ag_codec']}"
            for rank, row in enumerate(rows)
            if (row["rs_codec"], row["ag_codec"]) != (rs_codec, ag_codec)
        ],
    )
    return _summary(rows)


@benchmark()
def test_quick_allreduce_transport(tokens, hidden, tp, algorithm, super_tile, block):
    """fp16 wire, exact-grid input: bit-identical to the fp32 reference.

    Exercises chunk/slot addressing, the flag protocol, the super-tile loop and
    the accumulate order with the codec taken out of the picture. With
    super_tile None the engine walks its ladder, and the ``*_used`` columns
    name the geometry that ran; pinned, there is no selection to check.
    """
    rows = registry.result(
        _transport_key(tp, algorithm, super_tile, block),
        (tokens, hidden, "exact", False, False),
    )
    failures.check(
        f"tp={tp} {algorithm} {tokens}x{hidden} st={super_tile} block={block} "
        "fp16-exact",
        [
            f"rank {rank}: {row['n_mismatch']} mismatched elements, "
            f"max |err| {row['max_abs_err']:.3e}, "
            f"first bad flat index {row['first_bad']}"
            for rank, row in enumerate(rows)
            if row["n_mismatch"]
        ],
    )
    return {
        "gfx": ARCH,
        "st_used": rows[0]["st_used"],
        "block_used": rows[0]["block_used"],
        "n_mismatch": max(r["n_mismatch"] for r in rows),
        "max_abs_err": max(r["max_abs_err"] for r in rows),
    }


@benchmark()
def test_quick_allreduce_coverage(tp, algorithm, window):
    """Every kernel production dispatch can select on this host ran in the
    shipping sweep.

    The engine's own ``cfgs_for`` is the reference, so a retuned ladder or
    policy that the payload derivation does not follow fails here rather than
    silently leaving a kernel untested.
    """
    key = _ship_key(tp, algorithm, None)
    by_case = {c: registry.result(key, c) for c in registry.cases[key]}
    production = set(next(iter(by_case.values()))[0]["production_cfgs"])
    ran = {
        (rows[0]["st_used"], rows[0]["block_used"])
        for case, rows in by_case.items()
        if case[2] == "normal"
    }
    missing = sorted(production - ran)
    failures.check(
        f"tp={tp} {algorithm} coverage of {window}",
        [f"production kernels never run: {missing}"] if missing else [],
    )
    return {
        "gfx": ARCH,
        "production_kernels": sorted(production),
        "missing": missing,
    }


def main():
    if ARCH not in SUPPORTED_ARCHS:
        aiter.logger.warning("FlyQuickAllReduce unsupported on %s; skipping", ARCH)
        return
    n_gpu = torch.cuda.device_count()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.d_dtypes["bf16"]],
        help="Payload dtype (bf16 only).\n    e.g.: -d bf16",
    )
    parser.add_argument(
        "--tp",
        type=int,
        nargs="*",
        default=list(SUPPORTED_WORLDS),
        help="World sizes to sweep (2, 4, 8). Default all; sizes with fewer\n"
        "visible GPUs are skipped.\n    e.g.: --tp 8",
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        nargs="*",
        default=list(ALGORITHMS),
        choices=ALGORITHMS,
        help="Schedules to sweep. Default both.\n    e.g.: -a ring",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=None,
        help="(tokens, hidden) pairs for the shipping sweep, at every TP.\n"
        "Default: derived from the dispatch policy, one payload per kernel\n"
        "production selects on this host.\n"
        "    e.g.: -s 512,5120 9216,5120",
    )
    parser.add_argument(
        "--grid-cap",
        type=int,
        nargs="*",
        default=[None],
        help="Persistent-launch block caps for the shipping sweep. Default: the\n"
        "engine's own. The engine clamps a cap to the measured resident\n"
        "workgroups per CU.",
    )
    parser.add_argument(
        "--block",
        type=int,
        nargs="*",
        default=[None],
        choices=(*SUPPORTED_BLOCKS, None),
        help="Threads per block for the shipping sweep, on every rung. Default:\n"
        "each rung's own.",
    )
    parser.add_argument(
        "-o",
        "--out",
        default=None,
        help="Optional JSON output path for the shipping-sweep rows.",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Also run what production never selects: the legacy fixed\n"
        "shipping shapes, the full block sweep, the pinned-codec ring and the\n"
        "pinned-geometry fp16 transport matrix.",
    )
    args = parser.parse_args()

    tps = []
    for tp in args.tp:
        if tp not in SUPPORTED_WORLDS:
            aiter.logger.warning("unsupported world_size=%s; skipping", tp)
        elif n_gpu < tp:
            aiter.logger.warning(
                "tp=%s needs %s GPUs, have %s; skipping", tp, tp, n_gpu
            )
        else:
            tps.append(tp)
    algos = args.algorithm
    dts = [d for d in args.dtype if d == dtypes.bf16]
    if len(dts) != len(args.dtype):
        aiter.logger.warning("FlyQuickAllReduce payload is bf16; skipping others")

    if args.mnk is not None:
        for mnk in args.mnk:
            if not isinstance(mnk, tuple) or len(mnk) != 2:
                raise ValueError(f"-s expects tokens,hidden; got {mnk!r}")
    link = fly_policy.detect_link()
    # Payloads, and the production window when there is one, per schedule.
    plans = {
        (tp, algorithm): _ship_payloads(tp, algorithm, link)
        for tp, algorithm in itertools.product(tps, algos)
    }
    # Engines exactly as production builds them: only these are checked for
    # kernel coverage.
    shipping_engine = (
        args.mnk is None and args.grid_cap == [None] and args.block == [None]
    )

    # Register every row before running any, so rows sharing an engine share
    # a spawn.
    ship = []
    coverage = []
    for (tp, algorithm), (payloads, window) in plans.items():
        if args.mnk is not None:
            shapes = [(int(t), int(h)) for t, h in args.mnk]
        else:
            shapes = [(nbytes // _ROW_BYTES, HIDDEN) for nbytes in payloads]
            if args.extended:
                shapes += [
                    s for s in LEGACY_SHAPES_PER_WORLD_SIZE[tp] if s not in shapes
                ]
        for grid_cap, block in itertools.product(args.grid_cap, args.block):
            for tokens, hidden in shapes:
                row = (tokens, hidden, tp, algorithm, grid_cap, False, block)
                if row not in ship:
                    ship.append(row)
        if window is not None:
            # Captured into a CUDA graph at every world size, on the smallest
            # shape, as a serving framework replays it.
            tokens, hidden = shapes[0]
            ship.append((tokens, hidden, tp, algorithm, args.grid_cap[0], True, None))
            if shipping_engine:
                registry.windows[_ship_key(tp, algorithm, None)] = window
                lo, hi = window
                label = f"({fmt_bytes(lo - 1, inf_at=fly_policy.NO_MAX)}, {fmt_bytes(hi, inf_at=fly_policy.NO_MAX)}]"
                coverage.append((tp, algorithm, label))
    # Knob rows only in a default run: pinning --block already sweeps it over
    # the shipping shapes.
    knobs = []
    if args.block == [None]:
        knobs = [
            (tokens, hidden, tp, algorithm, None, False, block)
            for tp, algorithm, tokens, hidden, block in (
                KNOB_CASES + (EXTENDED_KNOB_CASES if args.extended else ())
            )
            if tp in tps and algorithm in algos
        ]
    if dts:
        for tokens, hidden, tp, algorithm, grid_cap, graph, block in ship + knobs:
            registry.register(
                _ship_key(tp, algorithm, grid_cap, block),
                (tokens, hidden, "normal", graph, True),
            )
    edge = [c for c in EDGE_CASES if c[0] in tps and c[1] in algos]
    for tp, algorithm, tokens, hidden, fill in edge:
        registry.register(
            _ship_key(tp, algorithm, None), (tokens, hidden, fill, False, False)
        )
    pinned = []
    if args.extended:
        pinned = [c for c in PINNED_CODEC_CASES if c[0] in tps and "ring" in algos]
    for tp, tokens, hidden, rs, ag in pinned:
        registry.register(
            registry.key(tp, algorithm="ring", rs_codec=rs, ag_codec=ag),
            (tokens, hidden, "normal", False, False),
        )
    # The fp16 wire through each production schedule's own ladder, on the
    # payloads that reach its kernels, plus one a single block owns.
    transport = [
        (tp, algorithm, *shape, None, None)
        for (tp, algorithm), (payloads, window) in plans.items()
        if window is not None
        for shape in [SUB_TILE_SHAPE] + [(n // _ROW_BYTES, HIDDEN) for n in payloads]
    ]
    if args.extended:
        transport += [c for c in TRANSPORT_CASES if c[0] in tps and c[1] in algos]
    for tp, algorithm, tokens, hidden, st, block in transport:
        registry.register(
            _transport_key(tp, algorithm, st, block),
            (tokens, hidden, "exact", False, False),
        )

    def _int4_rows(cases):
        return [
            test_quick_allreduce(
                tokens,
                hidden,
                dtype,
                tp,
                algorithm,
                grid_cap=grid_cap,
                graph=graph,
                block=block,
            )
            for tokens, hidden, tp, algorithm, grid_cap, graph, block in cases
        ]

    for dtype in dts:
        rows = _int4_rows(ship)
        summarize("flydsl quick allreduce INT4", rows)
        summarize(
            "flydsl quick allreduce INT4 production kernel coverage",
            [
                test_quick_allreduce_coverage(tp, algorithm, window)
                for tp, algorithm, window in coverage
            ],
        )
        summarize("flydsl quick allreduce INT4 block", _int4_rows(knobs))
        if args.out and rows:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as fh:
                json.dump(
                    {
                        "meta": {"gfx": ARCH, "timer": "run_perftest cuda_event"},
                        "rows": rows,
                    },
                    fh,
                    indent=2,
                    default=str,
                )
            aiter.logger.info("wrote %s", args.out)
    summarize(
        "flydsl quick allreduce INT4 edge inputs",
        [
            test_quick_allreduce_edge_inputs(tokens, hidden, tp, algorithm, fill)
            for tp, algorithm, tokens, hidden, fill in edge
        ],
    )
    summarize(
        "flydsl quick allreduce INT4 pinned codec",
        [
            test_quick_allreduce_pinned_codec(tokens, hidden, tp, rs, ag)
            for tp, tokens, hidden, rs, ag in pinned
        ],
    )
    summarize(
        "flydsl quick allreduce transport (fp16 wire, bit-exact)",
        [
            test_quick_allreduce_transport(tokens, hidden, tp, algorithm, st, block)
            for tp, algorithm, tokens, hidden, st, block in transport
        ],
    )

    # Fused all-reduce + residual add + RMSNorm epilogue. The mesh-only rows
    # need the mesh among the requested schedules.
    fused = [c for c in _FUSED_CASES if c[0] in tps]
    pad = [c for c in _FUSED_PAD_CASES if c[0] in tps]
    mesh = "mesh" in algos
    summarize(
        "flydsl quick allreduce rmsnorm sqnr",
        [
            test_quick_allreduce_rmsnorm_sqnr(tp, tokens, hidden, label, algorithm)
            for (tp, tokens, hidden, label), algorithm in itertools.product(
                fused, algos
            )
        ],
    )
    summarize(
        "flydsl quick allreduce rmsnorm padded width (fp16 wire, bit-exact)",
        [
            test_quick_allreduce_rmsnorm_padded_is_bit_exact(
                tp, tokens, hidden, label, algorithm
            )
            for (tp, tokens, hidden, label), algorithm in itertools.product(pad, algos)
        ],
    )
    summarize(
        "flydsl quick allreduce rmsnorm residual (fp16 wire, bit-exact)",
        [
            test_quick_allreduce_rmsnorm_residual_is_bit_exact(
                tp, tokens, hidden, algorithm
            )
            for (tp, tokens, hidden), algorithm in itertools.product(
                [c for c in _FUSED_EXACT_CASES if c[0] in tps], algos
            )
        ],
    )
    summarize(
        "flydsl quick allreduce rmsnorm mesh rank agreement",
        [
            test_quick_allreduce_rmsnorm_mesh_ranks_agree(tp, tokens, hidden)
            for tp, tokens, hidden in _FUSED_MESH_AGREE_CASES
            if mesh and tp in tps
        ],
    )
    summarize(
        "flydsl quick allreduce rmsnorm mesh pinned super-tile",
        [
            test_quick_allreduce_rmsnorm_mesh_pinned_st(
                tp, tokens, hidden, label, super_tile
            )
            for (tp, tokens, hidden, label), super_tile in itertools.product(
                fused if mesh else [], _FUSED_MESH_STS
            )
        ],
    )
    summarize(
        "flydsl quick allreduce rmsnorm mesh ST=8 coverage",
        [
            test_quick_allreduce_rmsnorm_mesh_st8_is_exercised(tp)
            for tp in sorted({c[0] for c in fused})
            if mesh
        ],
    )
    summarize(
        "flydsl quick allreduce rmsnorm mesh pinned super-tile (fp16 wire, bit-exact)",
        [
            test_quick_allreduce_rmsnorm_mesh_pinned_st_is_bit_exact(
                tp, tokens, hidden, super_tile
            )
            for (tp, tokens, hidden), super_tile in itertools.product(
                [c for c in _FUSED_MESH_ST_EXACT_CASES if c[0] in tps and mesh],
                _FUSED_MESH_STS,
            )
        ],
    )
    summarize(
        "flydsl quick allreduce rmsnorm host-side geometry checks",
        [test_quick_allreduce_host_checks(check) for check in _HOST_CHECKS],
    )

    failures.raise_if_failed("FlyQuickAllReduce")


if __name__ == "__main__":
    freeze_support()
    from time import perf_counter

    start = perf_counter()
    main()
    end = perf_counter()
    aiter.logger.info(f"Test execution took {end-start:.2f}s")
