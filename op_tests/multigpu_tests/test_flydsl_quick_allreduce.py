# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Runtime correctness and timing for FlyDSL quick all-reduce (``FlyQuickAllReduce``).

``--scenario`` selects what a run does: ``verify`` (the default) checks the
kernels, and ``bench`` compares them with aiter's HIP quick all-reduce.

**verify** covers what production dispatch can run on this host, and each
sweep ends in a markdown table:

* ``test_quick_allreduce`` -- the shipping configuration (codecs and
  super-tile left to the per-world defaults and ladders, as production dispatch
  constructs the engine), timed on a CUDA graph of ``GRAPH_INNER``
  back-to-back calls and checked again after every replay. The payloads come
  from ``allreduce_policy``: for every schedule it routes to at each world
  size, the smallest payload that reaches each kernel it can select there,
  plus the top of its window. A schedule the policy never selects on this
  host (the ring on xGMI) gets one row, for a user who moves the boundary with
  ``AITER_FLY_AR_MESH_MAX_BYTES``.
* ``test_quick_allreduce_coverage`` -- the kernels those rows ran include
  every kernel the engine's own ``cfgs_for`` says the window selects.
* ``test_quick_allreduce_edge_inputs`` -- payloads that land on the group
  scale's edge cases, sums above E4M3's old ceiling of 480 on every rank, and
  degenerate groups that must stay finite.
* ``test_quick_allreduce_divisor`` -- the mesh's shared divisor at TP8: an
  outlier fill must reach the SQNR floor at divisor 1 (the E5M2 scale does not
  clip it) and lose at most ``DIVISOR_MAX_LOSS_DB`` at divisor 8, and a
  small-magnitude fill must keep full SQNR at the default.
* ``test_quick_allreduce_transport`` -- the ``fp16`` wire format, a lossless
  passthrough, on an exactly representable input: the result must be
  bit-identical to the fp32 reference, which separates a chunk-addressing or
  flag-protocol bug from a codec one. Run through each production schedule's
  ladder on the shipping payloads, so it covers the geometry that ships.
* ``test_quick_allreduce_relay`` -- the TP2 mesh with relay devices set, checked
  against a direct engine on the same inputs. A default run adds these rows
  only on xGMI, when four GPUs are visible and this arch has relay defaults.
  The direct TP2 rows always run. Relay rows run eager; 2.5 and 80 MiB shapes
  also run captured into a CUDA graph, replayed with the inputs refilled
  between replays. A block-64 row with every block relaying, eager and graph,
  makes each block re-read its bounce slots, which catches a relay acquire that
  no longer empties L1.
* ``test_quick_allreduce_setup_failure`` -- at TP2, one rank fails the inbox
  allocation or the peer-handle open while the engines are built. Every rank
  must raise, none may hang in an exchange, and a clean engine must build
  afterwards. Every engine shares this setup code.

``--extended`` adds what production never selects: the legacy fixed shipping
shapes; ``test_quick_allreduce`` again, as a second table, with ``block``
overridden on every rung of the shipping ladder (``KNOB_CASES``), including
the narrow blocks where a VMEM store-data hazard once showed up; the fp16
transport over a matrix of pinned ``super_tile``/``block``; and
``test_quick_allreduce_pinned_codec`` -- the ring with its wire formats
pinned per lap: all-INT4 at TP8, and one lap lossless to isolate the other;
and the mesh on its opt-in INT5 and INT6 wire formats at TP4 and TP8.

Every row also checks that all ranks wrote bit-identical output: each rank
decodes every chunk from the same packets, its own included -- its own from the
packet it sent, since it never round-trips through its own inbox. The ring's
owner decodes its chunk from the packet that it quantizes for the all-gather,
as the mesh does, so the ring holds to the same check.

``test_quick_allreduce_run_ahead`` makes ``RUN_AHEAD_ITERS`` back-to-back calls
on the shipping engine, each on a new input and into its own output, with
no barrier between them and rank 0 deliberately late. The sweeps above wait for
all ranks after every call, so they never reuse a colour or an inbox slot while a
peer is still reading it. A stale or overwritten slot is the wrong answer for a
new input, and the SQNR and cross-rank identity checks see it.

**bench** times each FlyDSL schedule (``mesh``, ``ring``) against the one
stock kernel of the INT4 regime: ``hip``, aiter's HIP quick all-reduce
(``QuickAllReduce`` with ``AITER_QUICK_REDUCE_QUANTIZATION=INT4``), on the
same payloads. The comparison is ``hip us / <schedule> us``, so above 1.0 the
FlyDSL schedule is faster; ``flydsl vs hip`` is the schedule the policy routes
the payload to (``route``), where that is a quantized one. The HIP kernel runs
below its production floor (``QuickAllReduce._QR_MIN_SIZE``) too: that floor
is a speed policy, and the comparison is between kernels. ``hip prod`` says
whether production would run it on the payload.

``flydsl_allreduce_bench.py`` describes how the comparison is kept unbiased.
The default payloads are the verify sweep's shipping payloads of both
schedules; ``-s`` or ``--shape-csv`` replace them, ``-a`` picks the schedules,
and ``--grid-cap`` and ``--block`` pin their geometry. ``--report``,
``--csv`` and ``-o`` save the tables.

Every rank is a ``multiprocessing`` spawn worker that builds its own engine.
All rows that share an engine configuration ride one spawn. The oracle is an
untimed fp32 NCCL all-reduce of the same per-rank inputs. INT4/INT6 are lossy,
so those rows gate on SQNR, a calibrated mismatch ratio and a per-tile SQNR
floor.

The kernels see a flat payload, so the derived shapes use one width, 4096,
whose rows land exactly on every policy and ladder boundary.
FlyQuickAllReduce runs on gfx942/gfx950 at TP in {2, 4, 8}; ``main()`` fails on
an unsupported arch, and on a world size that is named with ``--tp`` but has
fewer GPUs than ranks. The default sweep skips the world sizes that do not fit,
and fails if none does.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import os
import sys
from multiprocessing import Pool, freeze_support, set_start_method
from time import perf_counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
# The repo root for ``aiter``, this directory for ``flydsl_allreduce_bench``.
# The ranks are spawned, so they re-import this file and need the same paths.
for _path in (_REPO_ROOT, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import flydsl_allreduce_bench as fab
import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest

set_start_method("spawn", force=True)

from aiter.ops.flydsl import allreduce_policy as fly_policy
from aiter.ops.flydsl.allreduce_policy import (
    MAX_PAYLOAD_BYTES,
    SUPPORTED_ARCHS,
    SUPPORTED_BLOCKS,
    SUPPORTED_WORLDS,
    TwoShotRung,
    mesh_st_ladder,
    ring_st_ladder,
)
from aiter.ops.flydsl.kernels.comm.allreduce_shared import (
    ATOMS,
    BLOCK,
    DEFAULT_GRID_CAP,
)
from aiter.ops.flydsl.kernels.comm.quick_allreduce_mesh import (
    MESH_RELAY_DEFAULTS,
    clamp_grid_cap,
)
from aiter.ops.flydsl.quick_allreduce import (
    DEFAULT_DIVISOR,
    _resolve_inbox_flags,
    batches_publishes,
)

ARCH = get_gfx_runtime()
ALGORITHMS = ("mesh", "ring")

# One SQNR floor for both schedules, in their shipping configuration.
#
# The ring's reduce-scatter lap requantizes N-1 times where the mesh requantizes
# once, and the partial sum it requantizes grows with the contributions folded
# in, so on an all-INT4 wire the ring's SQNR degrades with N: 17.9 dB at TP4
# and 15.3 at TP8 with the E5M2 group scale (gfx950). Defaulting the
# ring's reduce-scatter lap to INT6 from TP4 (RS_INT6_MIN_WORLD) lifts it to
# ~21 dB, so anything below 18.0 is a regression, not a known cost of the
# schedule.
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

# Calls captured in each timed CUDA graph.
GRAPH_INNER = fab.GRAPH_INNER

# bench: the regime of the HIP quick all-reduce, and its accuracy floor. It
# measures ~18.3 dB; the floor leaves ~4 dB of headroom.
HIP_QR_REGIME = "INT4"
HIP_SQNR_FLOOR_DB = 14.0

# Shipping payloads are derived from the dispatch policy (``_ship_payloads``)
# and laid out at this width. A row is 8 KiB and every policy and ladder
# boundary is a multiple of that, so rounding a payload up to whole rows never
# carries it across one.
HIDDEN = 4096
_ROW_BYTES = HIDDEN * 2

# Back-to-back calls of ``test_quick_allreduce_run_ahead``, and the shape of each
# (a 4 MiB payload). ``_run_ahead_shape`` moves it into the schedule's dispatch
# window when it is outside it.
RUN_AHEAD_ITERS = 48
RUN_AHEAD_SHAPE = (512, HIDDEN)

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
# The divisor cases: ``outlier`` at this divisor against ``outlier`` at 1. With
# the E5M2 group scale divisor 1 no longer clips, so the opt-in divisor has
# nothing to recover and may cost at most this much.
OUTLIER_DIVISOR = 8.0
DIVISOR_MAX_LOSS_DB = 1.0

EDGE_CASES = (
    (2, "mesh", 16, 1024, "pos_underflow"),
    (2, "mesh", 16, 1024, "neg_underflow"),
    (2, "mesh", 16, 1024, "overflow_512"),
    (2, "mesh", 16, 1024, "large_every_rank"),
    (8, "ring", 512, 5120, "large_every_rank"),
    (2, "mesh", 16, 1024, "zeros"),
    (2, "mesh", 512, 5120, "degenerate"),
    (8, "ring", 512, 5120, "degenerate"),
)

# (tp, algorithm, tokens, hidden, rs_codec, ag_codec). ``--extended`` only:
# production leaves both laps at the per-world default.
PINNED_CODEC_CASES = (
    (8, "ring", 512, 5120, "int4", "int4"),
    # The ring's shipping default at TP8 (RS_INT6_MIN_WORLD), pinned as well.
    (8, "ring", 512, 5120, "int6", "int4"),
    (8, "ring", 512, 5120, "fp16", "int4"),
    (8, "ring", 512, 5120, "int4", "fp16"),
    # The mesh's opt-in wire formats, on a decode payload and on the top of
    # the mesh window, which TP8 serves from its ST=8 rung.
    *(
        (tp, "mesh", tokens, hidden, codec, codec)
        for tp in (4, 8)
        for codec in ("int5", "int6")
        for tokens, hidden in ((512, 5120), (TOP_PAYLOAD_BYTES // _ROW_BYTES, HIDDEN))
    ),
)

# The mesh's pinned INT5/INT6 floors, so a wider codec that decodes at INT4
# quality fails.
MESH_PINNED_SQNR_MIN_DB = {"int5": 23.0, "int6": 29.0}

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
# ``block`` overridden on every rung. ``--extended`` only: no production rung
# uses most of these widths.
#
# The TP4 mesh rows at blocks 64 and 128 are where the VMEM store-data hazard in
# ``_store_v4i32_peer`` was caught, back when the mesh still had a variant that
# round-tripped through its own inbox: INT4's peer-major fanout at those widths
# is where the register allocator recycles the store's data VGPRs. The fp16
# transport rows at the same geometry never did.
KNOB_CASES = (
    (4, "mesh", 512, 5120, 64),
    (4, "mesh", 512, 5120, 128),
    (8, "mesh", 512, 5120, 128),
    (8, "ring", 512, 5120, 128),
    (4, "mesh", 9216, 4096, 128),
    (4, "mesh", 9216, 4096, 512),
    (4, "ring", 9216, 4096, 64),
    (2, "mesh", 512, 5120, 128),
    (2, "mesh", 9216, 4096, 512),
    (2, "ring", 9216, 4096, 128),
)

# TP2 relay rows, on the mesh. Back-to-back calls per case, each on a different
# input, so a payload that reaches the peer late shows up as a differing lane.
RELAY_CALLS = 4
# Relays for the TP pair on devices 0 and 1.
DEFAULT_RELAY = (2, 3)
DEFAULT_RELAY_FRACTION = (3, 8)
# (tokens, hidden, codec). Sizes: a single tile, 10 MiB (partial persistent
# grid on gfx950's g512 block-256 rung; above 4 MiB on two-rung ladders), a
# payload whose last tile is partial, a full-grid multiple at 80 MiB, and 120
# MiB. The lossless wire runs the two that exercise the offsets hardest.
RELAY_CASES = (
    (8, 1024, "int4"),
    (1024, 5120, "int4"),
    (8191, 5120, "int4"),
    (8192, 5120, "int4"),
    (12288, 5120, "int4"),
    (1024, 5120, "fp16"),
    (8191, 5120, "fp16"),
    (1024, 5120, "int5"),
    (1024, 5120, "int6"),
)
# The default relay rows that also run under CUDA-graph replay: 80 MiB (full
# g512 grid on gfx950; block-512 rung on gfx942) and 2.5 MiB (80 tiles at
# block-256).
RELAY_GRAPH_CASES = ((8192, 5120, "int4"), (256, 5120, "int4"))
# (tokens, hidden, codec, block, grid_cap): 16 tiles of 8 KiB over 8 blocks, all
# relaying, so every block reads its bounce slots twice with too little traffic
# in between to evict them from L1. A relay block's plain loads read the
# previous tile's packet here unless its acquire invalidates L1, so this fails
# if that is dropped or narrowed to workgroup scope. Eager and under graph
# replay.
RELAY_REUSE_CASES = ((64, 1024, "int4", 64, 8),)
RELAY_REUSE_FRACTION = (1, 1)
# (tokens, hidden, codec, relay_fraction, super_tile), ``--extended`` only.
EXTENDED_RELAY_CASES = (
    (8192, 5120, "int4", (1, 4), None),
    (8192, 5120, "int4", (1, 2), None),
    (8192, 5120, "int4", (3, 4), None),
    (8192, 5120, "int4", (1, 1), None),
    (8192, 5120, "int4", DEFAULT_RELAY_FRACTION, 8),
    (8192, 5120, "int6", DEFAULT_RELAY_FRACTION, None),
)

# Seconds to wait for each rank of a spawn. The kernels spin on flags written
# by peers, so a protocol bug or a dead rank hangs the rest.
SPAWN_TIMEOUT_S = 600

# Setup steps of ``_StEngine`` that a rank can fail alone, by the
# ``UncachedIpcHeap`` method made to raise: the inbox allocation, before the
# handle exchange, and opening the peer's handle, after it.
SETUP_FAULTS = {"alloc": "alloc", "open": "open_mem_handle"}
SETUP_FAULT_RANK = 1
# A rank left behind in an exchange would wait out the gloo timeout, far longer.
SETUP_FAULT_TIMEOUT_S = 300

_FILLS = (
    "normal",
    "outlier",
    "small",
    "degenerate",
    "exact",
    "pos_underflow",
    "neg_underflow",
    "overflow_512",
    "large_every_rank",
    "zeros",
)


def _make_inp(
    tokens: int,
    hidden: int,
    fill: str,
    *,
    rank: int,
    device: torch.device,
    call: int = 0,
) -> torch.Tensor:
    """One rank's contribution, for whichever edge case *fill* names.

    *call* picks a different draw for each back-to-back call of one case;
    ``call=0`` is the single-call input.
    """
    shape = (tokens, hidden)
    gen = torch.Generator().manual_seed(1234 + rank + 7919 * call)
    if fill in ("normal", "outlier", "small"):
        std = {"normal": 0.1, "outlier": 100.0, "small": 0.005}[fill]
        src = torch.randn(shape, generator=gen, dtype=torch.float32) * std
    elif fill == "degenerate":
        # Half the rows exactly zero, half far below the scale's magnitude
        # floor (2^-12 for INT4, 2^-10 for INT6). Both drive the group extremum to (or under) zero, which is
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
        # For the same reason, it cannot tell one accumulation order from another.
        src = torch.randint(-8, 8, shape, generator=gen).float() * (2.0**-4)
    elif fill in (
        "pos_underflow",
        "neg_underflow",
        "overflow_512",
        "large_every_rank",
        "zeros",
    ):
        val = {
            "pos_underflow": 2.0**-8,
            "neg_underflow": -(2.0**-8),
            # Above E4M3's old largest scale (480) on rank 0 only, so only
            # the first pack sees it.
            "overflow_512": 512.0 if rank == 0 else 0.0,
            # Every partial and the reduced sum above 480: the E4M3 scale
            # clipped both packs here (SQNR about 1 dB), and on the ring every
            # requantized hop. MiniMax-M3 all-reduce outputs reach 6400-16800.
            "large_every_rank": 2048.0,
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
        return mesh_st_ladder(world_size=world_size, link=link)
    return ring_st_ladder(world_size=world_size, link=link)


def _ships(world_size: int, algorithm: str, link: str) -> bool:
    """Whether production has a ladder for algorithm at world_size on this
    host. A schedule without one is never built, so nothing that runs the
    shipped ladder sweeps it. The pinned-ladder transport cases still run its
    kernel."""
    table = fly_policy.LADDERS[fly_policy.AllReduceAlgorithm(algorithm)]
    return (ARCH, link, int(world_size)) in table


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
    for rung in ladder:
        if nbytes >= rung.min_bytes:
            want, b = rung.super_tile, rung.block
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
    for rung in ladder:
        grids.setdefault(
            (rung.super_tile, rung.block),
            clamp_grid_cap(
                min(rung.grid_cap, DEFAULT_GRID_CAP),
                arch=ARCH,
                world_size=world_size,
                super_tile=rung.super_tile,
                cu_count=cu_count,
                block=rung.block,
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
    ends = [rung.min_bytes - 1 for rung in ladder[1:]] + [hi]
    out: dict[tuple, int] = {}
    for rung, end in zip(ladder, ends):
        st, b = rung.super_tile, rung.block
        start, end = max(lo, rung.min_bytes), min(hi, end)
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
    policy = fly_policy.resolve_family_policy(world_size=world_size, link=link)
    window = policy.window(fly_policy.AllReduceAlgorithm(algorithm), quantized=True)
    if window is not None:
        lo, hi = window.inclusive_range()
        payloads = set(_kernel_payloads(world_size, algorithm, link, lo, hi).values())
        top = min(hi, TOP_PAYLOAD_BYTES) // _ROW_BYTES * _ROW_BYTES
        if top >= lo:
            payloads.add(top)
        return sorted(payloads), (lo, hi)
    # Every payload a quantized algorithm runs, from the smallest.
    quant = [
        w
        for alg in (
            fly_policy.AllReduceAlgorithm.TwoShotMesh,
            fly_policy.AllReduceAlgorithm.TwoShotRing,
        )
        if (w := policy.window(alg, quantized=True)) is not None
    ]
    lo = min((w.min for w in quant), default=0)
    by_cfg = _kernel_payloads(world_size, algorithm, link, lo, MAX_PAYLOAD_BYTES)
    first = _ladder(algorithm, world_size, link)[0]
    return [by_cfg.get((first.super_tile, first.block), min(by_cfg.values()))], None


def _run_ahead_shape(window: tuple[int, int] | None) -> tuple[int, int]:
    """``RUN_AHEAD_SHAPE``, moved to the nearest whole-row payload of *window*,
    the inclusive dispatch range of the schedule, so the run-ahead runs a kernel
    that production dispatches. The top of an unbounded window is cut at
    ``TOP_PAYLOAD_BYTES``; a window starting above that runs at its start.
    """
    tokens, hidden = RUN_AHEAD_SHAPE
    if window is None:
        return RUN_AHEAD_SHAPE
    lo, hi = window
    rlo = -(-lo // _ROW_BYTES) * _ROW_BYTES
    rhi = max(min(hi, TOP_PAYLOAD_BYTES) // _ROW_BYTES * _ROW_BYTES, rlo)
    nbytes = min(max(tokens * hidden * 2, rlo), rhi)
    return nbytes // _ROW_BYTES, HIDDEN


def _fmt_bytes(nbytes: int) -> str:
    if nbytes >= MAX_PAYLOAD_BYTES:
        return "inf"
    for unit, shift in (("MiB", 20), ("KiB", 10)):
        if nbytes >= 1 << shift:
            return f"{nbytes / (1 << shift):g} {unit}"
    return f"{nbytes} B"


def _sqnr(ref_pow: torch.Tensor, mse: torch.Tensor) -> torch.Tensor:
    score = torch.where(
        (ref_pow <= 0) & (mse <= 0),
        torch.full_like(mse, float("inf")),
        10.0 * torch.log10(ref_pow / mse),
    )
    return torch.nan_to_num(score, nan=float("-inf"), neginf=float("-inf"))


def _sqnr_db(got: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        _sqnr((reference * reference).mean(), ((got - reference) ** 2).mean()).item()
    )


def _min_tile_sqnr_db(
    got: torch.Tensor, reference: torch.Tensor, tile_bytes: int
) -> float:
    """Worst per-tile SQNR, so one unwritten tile cannot be averaged away."""
    tile_elems = tile_bytes // 2
    g = got.reshape(-1)
    r = reference.reshape(-1)
    n = int(g.numel())
    n_full = (n // tile_elems) * tile_elems
    vals = []
    if n_full:
        gt = g[:n_full].view(-1, tile_elems)
        rt = r[:n_full].view(-1, tile_elems)
        mse = ((gt - rt) ** 2).mean(dim=1)
        pow_ = (rt * rt).mean(dim=1)
        vals.append(float(_sqnr(pow_, mse).min().item()))
    if n > n_full:
        vals.append(_sqnr_db(g[n_full:], r[n_full:]))
    return min(vals) if vals else _sqnr_db(got, reference)


def _rel_mae(got: torch.Tensor, reference: torch.Tensor) -> float:
    scale = float(reference.abs().mean().item())
    err = float((got - reference).abs().mean().item())
    return err / scale if scale else 0.0


def _lanes_differing(out: torch.Tensor, group) -> int:
    """bf16 lanes where any rank's output differs, bit for bit, from rank 0's."""
    import torch.distributed as dist

    gathered = [torch.empty_like(out) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, out.contiguous(), group=group)
    bits = [g.view(torch.int16) for g in gathered]
    return max(int((b != bits[0]).sum().item()) for b in bits)


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
        "sqnr_db": _sqnr_db(got, ref),
        "min_tile_sqnr_db": _min_tile_sqnr_db(got, ref, tile_bytes),
        "lanes_differing": _lanes_differing(out, group),
        "rel_mae": _rel_mae(got, ref),
        "err": float(err),
        "n_mismatch": n_mismatch,
        "max_abs_err": float(diff.max().item()) if diff.numel() else 0.0,
        "first_bad": first_bad,
    }


def _worst(a: dict, b: dict) -> dict:
    """Per-field worst of two metric dicts, for a row checked several times."""
    out = dict(a)
    for key in ("sqnr_db", "min_tile_sqnr_db"):
        out[key] = min(a[key], b[key])
    for key in ("err", "n_mismatch", "max_abs_err", "lanes_differing"):
        out[key] = max(a[key], b[key])
    # NaN must win, so compare with isfinite rather than max().
    if not math.isfinite(b["rel_mae"]) or b["rel_mae"] > a["rel_mae"]:
        out["rel_mae"] = b["rel_mae"]
    if a["first_bad"] < 0:
        out["first_bad"] = b["first_bad"]
    return out


def _engine_row(fly, m: dict, cfg_used: tuple, production_cfgs) -> dict:
    """The row fields every case reports about the engine it ran on."""
    st_used, block_used = cfg_used
    return {
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


def _time_eager(eng, inp: torch.Tensor, out: torch.Tensor, group) -> float:
    """Eager per-call latency of *eng* on *inp*, in microseconds.

    cuda.Event timing, not run_perftest's default profiler timer:
    `import aiter` creates a GPU context in the parent, and on some ROCm/torch
    builds a child spawned after that records no GPU events in torch.profiler,
    so the default timer fails reducing an empty trace.
    """
    import torch.distributed as dist

    dist.barrier(group=group)
    torch.cuda.synchronize()

    def fn():
        eng.allreduce(inp, out)
        return out

    _, us = run_perftest(fn, use_cuda_event=True)
    return float(us)


def _run_relay_case(fly, direct, case: tuple, *, rank: int, device, group, window):
    """One case on a relay engine and its direct twin, on the same inputs.

    Under *graph*, the relay engine's *calls* all-reduces are captured into one
    graph. Each replay runs on inputs refilled in place; the direct engine's
    eager output on the same inputs is what it is compared with.
    """
    import torch.distributed as dist

    ntok, hidden, fill, graph, time_it, calls = case

    def draw(replay):
        return [
            _make_inp(
                ntok, hidden, fill, rank=rank, device=device, call=replay * calls + c
            )
            for c in range(calls)
        ]

    inps = draw(0)
    relay_outs = [torch.zeros_like(inp) for inp in inps]
    direct_outs = [torch.zeros_like(inp) for inp in inps]
    nbytes = int(inps[0].numel()) * int(inps[0].element_size())
    cfg_used, _ = fly._pick_cfg(nbytes)
    tile_bytes = _tile_bytes(cfg_used[1])

    def run_all(eng, outs):
        for inp, out in zip(inps, outs):
            eng.allreduce(inp, out)

    relay_graph = direct_graph = None
    if graph:
        # Captured on the current stream, which allreduce() launches on when it
        # is given none, after one eager pass per engine.
        for eng, outs in ((fly, relay_outs), (direct, direct_outs)):
            run_all(eng, outs)
            torch.cuda.synchronize()
            dist.barrier()
        relay_graph, direct_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        for g, eng, outs in (
            (relay_graph, fly, relay_outs),
            (direct_graph, direct, direct_outs),
        ):
            with torch.cuda.graph(g):
                run_all(eng, outs)

    m = d = None
    differing = 0
    for replay in range(GRAPH_REPLAYS if graph else 1):
        if replay:
            for inp, new in zip(inps, draw(replay)):
                inp.copy_(new)
        refs = []
        for inp in inps:
            ref = inp.to(torch.float32)
            dist.all_reduce(ref, group=group)
            refs.append(ref)
        for out in relay_outs + direct_outs:
            out.zero_()
        dist.barrier()

        # Back to back with no sync between calls, so each call's flags and
        # colour follow the previous one on the wire; then the direct engine
        # on the same inputs, eager.
        if graph:
            relay_graph.replay()
        else:
            run_all(fly, relay_outs)
        torch.cuda.synchronize()
        dist.barrier()
        run_all(direct, direct_outs)
        torch.cuda.synchronize()
        dist.barrier()

        for ref, r_out, d_out in zip(refs, relay_outs, direct_outs):
            mi = _metrics(r_out, ref, rank, tile_bytes, group)
            di = _metrics(d_out, ref, rank, tile_bytes, group)
            m = mi if m is None else _worst(m, mi)
            d = di if d is None else _worst(d, di)
            differing = max(
                differing,
                int((r_out.view(torch.int16) != d_out.view(torch.int16)).sum()),
            )

    row = _engine_row(
        fly,
        m,
        cfg_used,
        None if window is None else [tuple(c) for c in fly.cfgs_for(*window)],
    )
    row.update(
        {
            "direct_sqnr_db": d["sqnr_db"],
            "direct_err": d["err"],
            "direct_n_mismatch": d["n_mismatch"],
            "relay_vs_direct_lanes": differing,
            "relay_used": fly.uses_relay(nbytes),
            "relay_tile_fraction": fly.relay_tile_fraction(nbytes),
            "relay_bytes": fly.relay_bytes,
            "variant": fly.variant(nbytes),
            "direct_us": None,
        }
    )
    if time_it:
        if graph:
            # Per call, from a replay of *calls* of them.
            for key, g in (("us", relay_graph), ("direct_us", direct_graph)):
                dist.barrier(group=group)
                torch.cuda.synchronize()
                _, us = run_perftest(g.replay, use_cuda_event=True)
                row[key] = float(us) / calls
        else:
            row["us"] = _time_eager(fly, inps[0], relay_outs[0], group)
            row["direct_us"] = _time_eager(direct, inps[0], direct_outs[0], group)
    return row


def _init_dist(rank: int, tp: int, init_method: str, hip: bool):
    """Join a TP group as production does. Returns the ``GroupCoordinator``.

    ``QuickAllReduce`` is built only if *hip* is set: its IPC buffer is large,
    and only the bench scenario runs its kernel.
    """
    # This test builds its own FlyDSL engines. The dispatcher must not build
    # more into ``QuickAllReduce``, which would then send the HIP kernel's
    # payloads to FlyDSL.
    os.environ["AITER_FLY_AR"] = "0"
    os.environ["AITER_QUICK_REDUCE_QUANTIZATION"] = HIP_QR_REGIME if hip else "NONE"
    # No candidate here uses CustomAllreduce.
    set_custom_all_reduce(False)
    init_distributed_environment(
        world_size=tp, rank=rank, distributed_init_method=init_method
    )
    ensure_model_parallel_initialized(tp, 1)
    return get_tp_group()


def _test_ladder(
    algorithm: str,
    world_size: int,
    *,
    super_tile: int | None = None,
    grid_cap: int | None = None,
    block: int | None = None,
) -> tuple:
    """The ladder of a test engine, from the knobs this test pins.

    *super_tile* pins that super-tile for every payload size, as one rung at
    *grid_cap* (default ``DEFAULT_GRID_CAP``) and *block* (default ``BLOCK``).
    Otherwise the production ladder runs, with *block* overriding and
    *grid_cap* bounding every rung.
    """
    if super_tile is not None:
        return (
            TwoShotRung(
                min_bytes=0,
                super_tile=int(super_tile),
                grid_cap=DEFAULT_GRID_CAP if grid_cap is None else int(grid_cap),
                block=BLOCK if block is None else int(block),
            ),
        )
    link = fly_policy.detect_link()
    ladder = _ladder(algorithm, world_size, link)
    return tuple(
        dataclasses.replace(
            r,
            grid_cap=r.grid_cap if grid_cap is None else min(r.grid_cap, grid_cap),
            block=r.block if block is None else int(block),
        )
        for r in ladder
    )


def _engine(rank: int, tp: int, gloo, device, engine_kw: dict):
    """A ``FlyQuickAllReduce`` with every binary compiled.

    The ``super_tile``, ``grid_cap`` and ``block`` keys of *engine_kw* become
    the engine's ladder; refer to ``_test_ladder``.
    """
    from aiter.ops.flydsl import FlyQuickAllReduce

    engine_kw = dict(engine_kw)
    knobs = {
        k: engine_kw.pop(k)
        for k in ("super_tile", "grid_cap", "block")
        if k in engine_kw
    }
    if knobs:
        engine_kw["ladder"] = _test_ladder(
            engine_kw.get("algorithm", "mesh"), tp, **knobs
        )
    fly = FlyQuickAllReduce(
        group=gloo,
        device=device,
        rank=rank,
        world_size=tp,
        # The cases deliberately include sub-threshold shapes (8x1024 is
        # 16 KiB, well under the default floor) to cover the partial-tile
        # path, and the bench times every payload it is given.
        min_bytes=0,
        **engine_kw,
    )
    fly.preload()
    return fly


def _teardown(engines, gloo) -> None:
    import torch.distributed as dist

    torch.cuda.synchronize()
    dist.barrier(group=gloo)
    for fly in engines:
        fly.close()
    destroy_model_parallel()
    destroy_distributed_environment()


def _run_ahead_part(it: int, r: int, tokens: int, hidden: int, device) -> torch.Tensor:
    """Rank *r*'s input of call *it*. Every rank can build every other rank's,
    so the reference needs no collective."""
    gen = torch.Generator(device=device).manual_seed(7000 + it * 16 + r)
    x = torch.randn(tokens, hidden, generator=gen, device=device, dtype=torch.float32)
    return (x * 0.1).to(torch.bfloat16)


def _run_ahead(fly, rank: int, tp: int, device, tokens: int, hidden: int, group):
    """``RUN_AHEAD_ITERS`` back-to-back calls with a new input each, into
    separate outputs and with no host sync or barrier between them. Rank 0 is
    dragged now and then so that the others run ahead into the next call.

    Afterwards, every output is checked against its own fp32 reference, and
    against the other ranks' outputs.
    """
    drag = torch.randn(2048, 2048, device=device, dtype=torch.float32)
    outs = []
    for it in range(RUN_AHEAD_ITERS):
        if rank == 0 and it % 4 == 0:
            drag = drag @ drag.T * 1e-6
        inp = _run_ahead_part(it, rank, tokens, hidden, device)
        out = torch.empty_like(inp)
        fly.allreduce(inp, out)
        outs.append(out)
    torch.cuda.synchronize()
    sqnr = float("inf")
    lanes = 0
    for it, out in enumerate(outs):
        ref = sum(
            _run_ahead_part(it, r, tokens, hidden, device).float() for r in range(tp)
        )
        sqnr = min(sqnr, _sqnr_db(out.float(), ref))
        lanes = max(lanes, _lanes_differing(out, group))
    return {"calls": len(outs), "sqnr_db": sqnr, "lanes_differing": lanes}


def _run_rank(
    rank: int,
    tp: int,
    init_method: str,
    engine_kw: dict,
    cases: list[tuple],
    timing: tuple[int, int],
    window: tuple[int, int] | None = None,
) -> list[dict]:
    """One rank of one verify spawn: build the engine, then run every case.

    A case is ``(tokens, hidden, fill, time_it)``, with an optional fifth
    element: the ``divisor`` its all-reduces pass. The engine is built once, so
    every case shares its compiled binaries and IPC inbox. *timing* is
    ``(reps, rounds)`` for ``fab.time_graphs``. *window* is the payload range
    production dispatch routes to this engine, if any; every row then carries
    the kernels the engine itself says that range selects.
    """
    import torch.distributed as dist

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    tp_group = _init_dist(rank, tp, init_method, hip=False)
    # FlyQuickAllReduce exchanges IPC metadata over the gloo group; NCCL stays
    # for the fp32 reference all-reduce.
    gloo = tp_group.cpu_group
    group = tp_group.device_group
    fly = _engine(rank, tp, gloo, device, engine_kw)
    direct = None
    if "relay_devices" in engine_kw:
        direct = _engine(
            rank,
            tp,
            gloo,
            device,
            {k: v for k, v in engine_kw.items() if not k.startswith("relay_")},
        )
    production_cfgs = None
    if window is not None:
        production_cfgs = [(int(st), int(b)) for st, b in fly.cfgs_for(*window)]
    reps, rounds = timing

    rows = []
    try:
        for case in cases:
            if direct is not None:
                rows.append(
                    _run_relay_case(
                        fly,
                        direct,
                        case,
                        rank=rank,
                        device=device,
                        group=group,
                        window=window,
                    )
                )
                torch.cuda.empty_cache()
                continue
            ntok, hidden, fill, time_it, *extra = case
            divisor = extra[0] if extra else DEFAULT_DIVISOR
            if fill == "run_ahead":
                rows.append(
                    {
                        **_run_ahead(fly, rank, tp, device, ntok, hidden, group),
                        "link": fly.link,
                        "inbox_memory": fly.inbox_memory,
                    }
                )
                continue
            inp = _make_inp(ntok, hidden, fill, rank=rank, device=device)
            ref = inp.to(torch.float32)
            dist.all_reduce(ref, group=group)
            dist.barrier(group=group)

            nbytes = int(inp.numel()) * int(inp.element_size())
            cfg_used, _ = fly._pick_cfg(nbytes)
            tile_bytes = _tile_bytes(cfg_used[1])

            out = torch.zeros_like(inp)
            fly.allreduce(inp, out, divisor=divisor)
            torch.cuda.synchronize()
            dist.barrier(group=group)
            m = _metrics(out, ref, rank, tile_bytes, group)

            us = None
            if time_it:
                # The IPC inbox is allocated at init, so the captured launch is
                # safe to replay.
                g = fab.capture(
                    lambda src=inp, dst=out, div=divisor: fly.allreduce(
                        src, dst, divisor=div
                    )
                )
                for _ in range(GRAPH_REPLAYS):
                    out.zero_()
                    g.replay()
                    torch.cuda.synchronize()
                    dist.barrier(group=group)
                    m = _worst(m, _metrics(out, ref, rank, tile_bytes, group))
                us = fab.time_graphs({"flydsl": g}, gloo, reps=reps, rounds=rounds)[
                    "flydsl"
                ]
                del g

            st_used, block_used = cfg_used
            rows.append(
                {
                    **m,
                    "link": fly.link,
                    "inbox_memory": fly.inbox_memory,
                    # Resolved, not requested: these come from the
                    # per-world-size default unless the caller pinned a lap,
                    # and a regression should name the codec that produced it.
                    "rs_codec": fly.rs_codec,
                    "ag_codec": fly.ag_codec,
                    "st_used": int(st_used),
                    "block_used": int(block_used),
                    "grid_by_cfg": {cfg: int(e.grid) for cfg, e in fly._by_cfg.items()},
                    "production_cfgs": production_cfgs,
                    "us": us,
                }
            )
            del inp, out, ref
            torch.cuda.empty_cache()
    finally:
        _teardown([fly] + ([] if direct is None else [direct]), gloo)
    return rows


def _run_setup_fault_rank(rank: int, tp: int, init_method: str, fault: str) -> dict:
    """One rank of a setup-fault spawn: build an engine with *fault* injected
    on ``SETUP_FAULT_RANK``, then build a clean one on every rank."""
    import torch.distributed as dist

    from aiter.ops.flydsl import FlyQuickAllReduce
    from aiter.ops.flydsl.ipc_heap import UncachedIpcHeap

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="gloo", init_method=init_method, world_size=tp, rank=rank
    )
    name = SETUP_FAULTS[fault]
    real = vars(UncachedIpcHeap)[name]

    def _fail(*_args, **_kwargs):
        raise RuntimeError(f"injected {name} failure")

    def _build():
        FlyQuickAllReduce(
            group=dist.group.WORLD, device=device, rank=rank, world_size=tp
        ).close()

    row = {"err": None, "s": None, "rebuild_err": None}
    try:
        if rank == SETUP_FAULT_RANK:
            setattr(UncachedIpcHeap, name, staticmethod(_fail))
        start = perf_counter()
        try:
            _build()
        except Exception as exc:  # noqa: BLE001
            row["err"] = f"{type(exc).__name__}: {exc}"
        finally:
            row["s"] = perf_counter() - start
            setattr(UncachedIpcHeap, name, real)
        try:
            _build()
        except Exception as exc:  # noqa: BLE001
            row["rebuild_err"] = f"{type(exc).__name__}: {exc}"
    finally:
        dist.destroy_process_group()
    return row


def _hip_runs(qr, nbytes: int) -> bool:
    """Whether the HIP quick all-reduce can run a bf16 payload of *nbytes*.

    ``QuickAllReduce._should_hip`` without the production size floor.
    """
    return qr is not None and nbytes % 16 == 0 and nbytes <= qr.qr_max_size


def _bench_rank(
    rank: int,
    tp: int,
    init_method: str,
    algorithms: tuple[str, ...],
    engine_kw: dict,
    cases: list[tuple],
    timing: tuple[int, int],
) -> list[dict]:
    """One rank of one bench spawn: every candidate on every case.

    One engine per schedule in *algorithms*, each with *engine_kw*, and the
    HIP quick all-reduce. A case is ``(tokens, hidden)``. Returns, per case,
    whether production would run the HIP kernel on it and each candidate's
    per-rank result.
    """
    import torch.distributed as dist

    from aiter.dist.device_communicators.quick_all_reduce import QuickReduceRegime

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    tp_group = _init_dist(rank, tp, init_method, hip=True)
    gloo = tp_group.cpu_group
    group = tp_group.device_group
    qr = tp_group.device_communicator.qr_comm
    if qr is not None and qr.disabled:
        qr = None
    level = QuickReduceRegime[HIP_QR_REGIME]
    # Built in a fixed order: each engine's IPC exchange is a collective.
    engines = {
        a: _engine(rank, tp, gloo, device, {"algorithm": a, **engine_kw})
        for a in algorithms
    }
    reps, rounds = timing

    # The input and every candidate's output, for the whole spawn. Refer to
    # flydsl_allreduce_bench for why the addresses must not change.
    names = (*algorithms, "hip")
    max_bytes = max(m * hidden * 2 for m, hidden in cases)
    bufs = {
        name: torch.empty(max_bytes, dtype=torch.uint8, device=device)
        for name in ("inp", *names)
    }

    def _view(name, m, hidden):
        return bufs[name][: m * hidden * 2].view(torch.bfloat16).view(m, hidden)

    def _fly(eng, x, out):
        return lambda: eng.allreduce(x, out)

    def _hip(x, out):
        def thunk():
            # Set in the thunk, as production sets it once at init: nothing
            # else here changes it, but nothing then depends on that.
            qr.qr_quant_level = level
            qr.quick_all_reduce(x, out=out)

        return thunk

    rows = []
    try:
        for m, hidden in cases:
            x = _view("inp", m, hidden)
            x.copy_(_make_inp(m, hidden, "normal", rank=rank, device=device))
            ref = x.to(torch.float32)
            dist.all_reduce(ref, group=group)
            nbytes = m * hidden * 2
            outs = {name: _view(name, m, hidden) for name in names}

            thunks = {a: _fly(eng, x, outs[a]) for a, eng in engines.items()}
            hip_prod = False
            if _hip_runs(qr, nbytes):
                thunks["hip"] = _hip(x, outs["hip"])
                qr.qr_quant_level = level
                hip_prod = qr._should_hip(x)

            graphs, res = {}, {}
            for name, thunk in thunks.items():
                graphs[name] = fab.capture(thunk)
                got = fab.graded_replay(graphs[name], outs[name], gloo).float()
                res[name] = {"sqnr_db": _sqnr_db(got, ref)}
            times = fab.time_graphs(graphs, gloo, reps=reps, rounds=rounds)
            for name, us in times.items():
                res[name]["us"] = us
            for algorithm, eng in engines.items():
                res[algorithm]["variant"] = eng.variant(nbytes)
            if "hip" in res:
                wire = "fp16" if qr.use_fp16_kernels else "bf16"
                res["hip"]["variant"] = f"{HIP_QR_REGIME.lower()}/{wire}"
            rows.append({"hip prod": hip_prod, "candidates": res})
            del graphs
    finally:
        _teardown(engines.values(), gloo)
    return rows


def _dispatch_rank(rank: int, tp: int, init_method: str, skewed: bool = False) -> dict:
    """One rank of the production-dispatch spawn.

    Unlike every other spawn here, this one does not build its own engines: it
    stands up the real production ``QuickAllReduce`` with ``AITER_FLY_AR=1`` and
    ``AITER_QUICK_REDUCE_QUANTIZATION=INT4``, then drives payloads across each
    schedule's dispatch window through the top-level ``CudaCommunicator.all_reduce``,
    which tries quick reduce before custom all-reduce. ``_init_fly_backend`` builds
    the engines; ``_should_fly`` is the gate ``quick_all_reduce`` applies to route a
    payload to one.

    Per probe it records which schedule the policy selects, which engine actually
    ran (a wrapper on each engine's ``allreduce`` records it), and the accuracy of
    the all-reduce the top-level returned. Unbounded windows are cut at
    ``TOP_PAYLOAD_BYTES``, as everywhere else in this file.

    *skewed* leaves FlyDSL off on the last rank alone. The init must then
    return on every rank, with nothing built, instead of leaving the other
    ranks waiting in the engines' collectives.
    """
    import torch.distributed as dist

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    os.environ["AITER_FLY_AR"] = "0" if skewed and rank == tp - 1 else "1"
    os.environ["AITER_QUICK_REDUCE_QUANTIZATION"] = "INT4"
    # No candidate here uses custom all-reduce; only the quick-reduce slot.
    set_custom_all_reduce(False)
    init_distributed_environment(
        world_size=tp, rank=rank, distributed_init_method=init_method
    )
    ensure_model_parallel_initialized(tp, 1)
    tp_group = get_tp_group()
    gloo = tp_group.cpu_group
    group = tp_group.device_group
    comm = tp_group.device_communicator
    qr = comm.qr_comm
    built = qr is not None and qr._fly_policy is not None

    probes = []
    try:
        if built and not skewed:
            ran = []
            for alg, eng in qr._fly_engines.items():

                def allreduce(*args, _run=eng.allreduce, _alg=alg, **kw):
                    ran.append(_alg)
                    return _run(*args, **kw)

                eng.allreduce = allreduce
            # Whole-row payloads at both ends of every served window, plus one row
            # below the smallest (expected to route to HIP or the fallback).
            sizes = set()
            lows = []
            for alg in qr._fly_engines:
                lo, hi = qr._fly_policy.window(alg, quantized=True).inclusive_range()
                rlo = -(-lo // _ROW_BYTES) * _ROW_BYTES
                rhi = min(hi, TOP_PAYLOAD_BYTES) // _ROW_BYTES * _ROW_BYTES
                if rlo <= rhi:
                    sizes.update((rlo, rhi))
                    lows.append(rlo)
            if lows:
                below = min(lows) - _ROW_BYTES
                if below > 0:
                    sizes.add(below)
            for nbytes in sorted(sizes):
                m = nbytes // _ROW_BYTES
                inp = _make_inp(m, HIDDEN, "normal", rank=rank, device=device)
                ref = inp.to(torch.float32)
                dist.all_reduce(ref, group=group)
                ran.clear()
                out = comm.all_reduce(inp)
                torch.cuda.synchronize()
                alg = qr._fly_policy.select(nbytes, quantized=True)
                probes.append(
                    {
                        "nbytes": nbytes,
                        "algorithm": None if alg is None else alg.value,
                        "expect_served": alg in qr._fly_engines,
                        "served": bool(ran),
                        "ran": [a.value for a in ran],
                        "sqnr_db": _sqnr_db(out.float(), ref),
                    }
                )
    finally:
        torch.cuda.synchronize()
        dist.barrier(group=gloo)
        destroy_model_parallel()
        destroy_distributed_environment()
    return {"built": built, "probes": probes}


def _capture_first_rank(rank: int, tp: int, init_method: str) -> dict:
    """One rank of the first-call-in-capture spawn.

    Stands up the production ``QuickAllReduce`` with ``AITER_FLY_AR=1`` and
    ``AITER_QUICK_REDUCE_QUANTIZATION=INT4``, as ``_dispatch_rank`` does, but
    makes no eager call. The first launch of every kernel the mesh and the ring
    serve is inside one HIP graph capture.

    It records the served kernels that were not loaded at init. That check
    does not depend on whether HIP tolerates a module load during capture. It
    then replays the graph and checks every output against an fp32 reference.
    """
    import torch.distributed as dist

    from aiter.dist.parallel_state import graph_capture
    from aiter.ops.flydsl.allreduce_host_shared import launcher_loaded

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    os.environ["AITER_FLY_AR"] = "1"
    os.environ["AITER_QUICK_REDUCE_QUANTIZATION"] = "INT4"
    set_custom_all_reduce(False)
    init_distributed_environment(
        world_size=tp, rank=rank, distributed_init_method=init_method
    )
    ensure_model_parallel_initialized(tp, 1)
    tp_group = get_tp_group()
    gloo = tp_group.cpu_group
    group = tp_group.device_group
    comm = tp_group.device_communicator
    qr = comm.qr_comm
    built = qr is not None and qr._fly_policy is not None
    res = {"built": built, "unloaded": [], "rows": []}

    try:
        if built:
            link = fly_policy.detect_link()
            cases = []
            for alg, eng in qr._fly_engines.items():
                lo, hi = qr._fly_policy.window(alg, quantized=True).inclusive_range()
                res["unloaded"] += [
                    f"{alg.value} {key}"
                    for key in eng.cfgs_for(lo, hi)
                    if not launcher_loaded(eng._by_cfg[key].launch)
                ]
                payloads = _kernel_payloads(
                    tp, alg.value, link, lo, min(hi, TOP_PAYLOAD_BYTES)
                )
                for cfg, nbytes in sorted(payloads.items(), key=lambda kv: kv[1]):
                    inp = _make_inp(
                        nbytes // _ROW_BYTES, HIDDEN, "normal", rank=rank, device=device
                    )
                    ref = inp.to(torch.float32)
                    dist.all_reduce(ref, group=group)
                    cases.append((alg, cfg, nbytes, inp, ref))

            ran = []
            for alg, eng in qr._fly_engines.items():

                def allreduce(*args, _run=eng.allreduce, _alg=alg, **kw):
                    ran.append(_alg)
                    return _run(*args, **kw)

                eng.allreduce = allreduce
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            error = None
            try:
                with graph_capture() as gc, torch.cuda.graph(g, stream=gc.stream):
                    outs = [comm.all_reduce(case[3]) for case in cases]
            except Exception as e:  # noqa: BLE001
                error = f"{type(e).__name__}: {e}"
            res["capture_error"] = error
            res["ran"] = [a.value for a in ran]
            res["expected"] = [case[0].value for case in cases]
            # Replay only if every rank captured: the graph launches peers'
            # kernels that wait for each other.
            vote = torch.tensor([error is None], dtype=torch.int32)
            dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=gloo)
            if vote.item():
                sqnr = [float("inf")] * len(cases)
                for _ in range(GRAPH_REPLAYS):
                    torch.cuda.synchronize()
                    dist.barrier(group=gloo)
                    for out in outs:
                        out.fill_(float("nan"))
                    torch.cuda.synchronize()
                    dist.barrier(group=gloo)
                    g.replay()
                    torch.cuda.synchronize()
                    for i, (case, out) in enumerate(zip(cases, outs)):
                        sqnr[i] = min(sqnr[i], _sqnr_db(out.float(), case[4]))
                res["rows"] = [
                    {
                        "algorithm": alg.value,
                        "cfg": str(cfg),
                        "nbytes": nbytes,
                        "sqnr_db": s,
                    }
                    for (alg, cfg, nbytes, _inp, _ref), s in zip(cases, sqnr)
                ]
    finally:
        torch.cuda.synchronize()
        dist.barrier(group=gloo)
        destroy_model_parallel()
        destroy_distributed_environment()
    return res


def _spawn(target, world_size: int, timeout_s=SPAWN_TIMEOUT_S, **kwds) -> list:
    """*target* on every rank of a new spawn, with *kwds* and the rendezvous."""
    if world_size not in SUPPORTED_WORLDS:
        raise ValueError(f"unsupported world_size={world_size}")
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    pool = Pool(processes=world_size)
    try:
        results = [
            pool.apply_async(
                target,
                kwds={
                    "rank": rank,
                    "tp": world_size,
                    "init_method": init_method,
                    **kwds,
                },
            )
            for rank in range(world_size)
        ]
        ranks = [fut.get(timeout=timeout_s) for fut in results]
    except Exception:
        pool.terminate()
        raise
    else:
        pool.close()
    finally:
        pool.join()
    return ranks


# Rows are registered up front, grouped by the engine they need, so that each
# engine is built by exactly one spawn however many tables read from it.
# A spawn key is ``(tp, sorted engine kwargs)``; a case is
# ``(tokens, hidden, fill, time_it)``, or ``(tokens, hidden)`` in a bench
# spawn, whose key also names its schedules.
_CASES: dict[tuple, list[tuple]] = {}
# Production dispatch window of a shipping engine whose kernel coverage is
# checked, per spawn key.
_WINDOWS: dict[tuple, tuple[int, int]] = {}
_RESULTS: dict[tuple, dict[tuple, list[dict]]] = {}
# Per-rank production-dispatch results, per ``(tp, skewed)``. Not keyed by
# knobs: the dispatch spawn builds the production communicator, not a
# knob-pinned engine.
_DISPATCH: dict[tuple[int, bool], list] = {}
_FAILURES: list[str] = []
# ``(reps, rounds)`` for every timed graph; set by ``main()``.
_TIMING = (fab.DEFAULT_REPS, fab.DEFAULT_ROUNDS)


def _key(tp: int, **engine_kw) -> tuple:
    return (tp, tuple(sorted(engine_kw.items())))


def _register(key: tuple, case: tuple) -> None:
    cases = _CASES.setdefault(key, [])
    if case not in cases:
        cases.append(case)


def _result(key: tuple, case: tuple) -> list[dict]:
    """Per-rank rows for *case*, spawning *key*'s engine on first use."""
    if key not in _RESULTS:
        cases = _CASES[key]
        kw = dict(key[1])
        if "algorithms" in kw:
            ranks = _spawn(
                _bench_rank,
                key[0],
                algorithms=kw.pop("algorithms"),
                engine_kw=kw,
                cases=cases,
                timing=_TIMING,
            )
        else:
            ranks = _spawn(
                _run_rank,
                key[0],
                engine_kw=kw,
                cases=cases,
                timing=_TIMING,
                window=_WINDOWS.get(key),
            )
        _RESULTS[key] = {
            c: [rank_rows[i] for rank_rows in ranks] for i, c in enumerate(cases)
        }
    return _RESULTS[key][case]


def _dispatch_ranks(tp: int, skewed: bool = False) -> list[dict]:
    if (tp, skewed) not in _DISPATCH:
        _DISPATCH[tp, skewed] = _spawn(_dispatch_rank, tp, skewed=skewed)
    return _DISPATCH[tp, skewed]


def _check(label: str, fails: list[str]) -> bool:
    if fails:
        msg = f"{label}: " + "; ".join(fails)
        aiter.logger.error(msg)
        _FAILURES.append(msg)
    return not fails


def _identity_fails(rows: list[dict], algorithm: str) -> list[str]:
    """Cross-rank bit-identity, asserted for both schedules."""
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
    check_close: bool = True,
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
        if check_close and row["err"] >= CLOSE_ERR_RATIO:
            fails.append(
                f"rank {rank}: checkAllclose err {row['err']:.3f} >= {CLOSE_ERR_RATIO}"
            )
    return _check(label, fails)


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
    return _key(tp, **kw)


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
    return _key(tp, **kw)


@benchmark()
def test_quick_allreduce(
    tokens,
    hidden,
    dtype,
    tp,
    algorithm,
    grid_cap=None,
    block=None,
):
    """Shipping configuration: no codec or super-tile pinned."""
    rows = _result(
        _ship_key(tp, algorithm, grid_cap, block),
        (tokens, hidden, "normal", True),
    )
    nbytes = tokens * hidden * 2
    _check_sqnr(
        f"tp={tp} {algorithm} {tokens}x{hidden} block={block}",
        rows,
        floor=SQNR_MIN_DB,
        expected_st=_shipping_st(rows, nbytes, algorithm, tp, block),
        algorithm=algorithm,
    )
    # (tp - 1) adds per element; codec ALU work is not counted.
    flops = tokens * hidden * (tp - 1)
    # The slowest rank is when the collective finishes.
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


def _bench_key(tp: int, algorithms, grid_cap, block) -> tuple:
    kw = {"algorithms": tuple(algorithms)}
    for name, val in (("grid_cap", grid_cap), ("block", block)):
        if val is not None:
            kw[name] = val
    return _key(tp, **kw)


@benchmark()
def bench_quick_allreduce(
    tokens,
    hidden,
    tp,
    algorithms,
    route,
    grid_cap=None,
    block=None,
):
    """Each FlyDSL schedule against the HIP quick all-reduce.

    *route* is the algorithm production dispatch picks for the payload:
    ``mesh``, ``ring``, ``oneshot``, or ``fallback`` for
    ``cross_device_reduce`` and RCCL.
    """
    rows = _result(_bench_key(tp, algorithms, grid_cap, block), (tokens, hidden))
    names = [n for n in (*algorithms, "hip") if n in rows[0]["candidates"]]
    got = {n: fab.collapse([r["candidates"][n] for r in rows]) for n in names}

    fails = []
    for name, c in got.items():
        floor = HIP_SQNR_FLOOR_DB if name == "hip" else SQNR_MIN_DB
        if c["SQNR dB"] < floor:
            fails.append(f"{name}: SQNR {c['SQNR dB']:.2f} dB below {floor} dB")
        if c["variant"] is not None and c["variant"].startswith("MIXED"):
            fails.append(f"{name}: ranks ran different kernels: {c['variant']}")
    _check(f"tp={tp} {tokens}x{hidden} grid_cap={grid_cap} block={block}", fails)

    hip_us = got["hip"]["us"] if "hip" in got else None
    ret = {
        "gfx": ARCH,
        "KiB": tokens * hidden * 2 / 1024,
        "hip prod": rows[0]["hip prod"],
        "flydsl vs hip": (
            fab.speedup(hip_us, got[route]["us"]) if route in got else None
        ),
    }
    for algorithm in algorithms:
        ret[f"{algorithm} vs hip"] = fab.speedup(hip_us, got[algorithm]["us"])
    for name, c in got.items():
        ret.update({f"{name} {field}": v for field, v in c.items()})
    return ret


@benchmark()
def test_quick_allreduce_edge_inputs(tokens, hidden, tp, algorithm, fill):
    """Edge-case payloads on the shipping engine; correctness only."""
    rows = _result(_ship_key(tp, algorithm, None), (tokens, hidden, fill, False))
    label = f"tp={tp} {algorithm} {tokens}x{hidden} fill={fill}"
    if fill == "degenerate":
        # A group whose extremum is zero decodes to a zero scale, so the encode
        # reciprocal saturates; before it was clamped, that reached the codec
        # as Inf and 0 * Inf poisoned the tile. Only finiteness is asserted.
        _check(
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
def test_quick_allreduce_divisor(tokens, hidden, tp, fill, divisor):
    """The shared divisor on the shipping mesh engine; correctness only.

    ``small`` at the default divisor must keep the plain INT4 accuracy. An
    ``outlier`` fill used to clip the E4M3 scale at divisor 1, which the
    divisor existed to recover. The E5M2 scale does not clip it, so divisor 1
    must reach the SQNR floor and the opt-in divisor may cost at most
    ``DIVISOR_MAX_LOSS_DB``.
    """
    key = _ship_key(tp, "mesh", None)
    nbytes = tokens * hidden * 2
    label = f"tp={tp} mesh {tokens}x{hidden} fill={fill} divisor={divisor:g}"
    rows = _result(key, (tokens, hidden, fill, False, divisor))
    ret = _summary(rows)
    ret["divisor"] = divisor
    ret["sqnr_vs_divisor1_db"] = None
    if fill == "outlier":
        base = _result(key, (tokens, hidden, fill, False, DEFAULT_DIVISOR))
        base_db = min(r["sqnr_db"] for r in base)
        ret["sqnr_vs_divisor1_db"] = ret["sqnr_db"] - base_db
        fails = []
        if base_db < SQNR_MIN_DB:
            fails.append(
                f"SQNR {base_db:.2f} dB at divisor 1 < {SQNR_MIN_DB}; "
                "the group scale clips"
            )
        if ret["sqnr_vs_divisor1_db"] < -DIVISOR_MAX_LOSS_DB:
            fails.append(
                f"SQNR {ret['sqnr_db']:.2f} dB is "
                f"{-ret['sqnr_vs_divisor1_db']:.2f} dB below divisor 1; the "
                f"opt-in divisor should cost at most {DIVISOR_MAX_LOSS_DB} dB"
            )
        _check(label, fails)
    # CLOSE_ATOL is calibrated to the 0.1-std fill, not to the outlier's 100.
    _check_sqnr(
        label,
        rows,
        floor=SQNR_MIN_DB,
        expected_st=_shipping_st(rows, nbytes, "mesh", tp),
        algorithm="mesh",
        check_close=fill != "outlier",
    )
    return ret


@benchmark()
def test_quick_allreduce_pinned_codec(
    tokens, hidden, tp, algorithm, rs_codec, ag_codec
):
    """One schedule with both laps' wire formats pinned; correctness only."""
    key = _key(tp, algorithm=algorithm, rs_codec=rs_codec, ag_codec=ag_codec)
    rows = _result(key, (tokens, hidden, "normal", False))
    label = f"tp={tp} {algorithm} {tokens}x{hidden} rs={rs_codec} ag={ag_codec}"
    if algorithm == "mesh":
        floor = MESH_PINNED_SQNR_MIN_DB[rs_codec]
    else:
        floor = SQNR_MIN_DB_TP8_INT4_RING
    _check_sqnr(
        label,
        rows,
        floor=floor,
        expected_st=_shipping_st(rows, tokens * hidden * 2, algorithm, tp),
        algorithm=algorithm,
    )
    _check(
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

    Exercises chunk/slot addressing, the flag protocol and the super-tile loop
    with the codec taken out of the picture. It does not check the accumulate
    order: on the 1/16 grid every partial sum is exact, so any order gives the
    same bits. The one-shot test checks its order against ``_rank_order_sum``,
    and the codec tests cover the arithmetic. With super_tile None the engine
    walks its ladder, and the ``*_used`` columns name the geometry that ran;
    pinned, there is no selection to check.
    """
    rows = _result(
        _transport_key(tp, algorithm, super_tile, block),
        (tokens, hidden, "exact", False),
    )
    _check(
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


def _relay_key(
    tp,
    relay_devices,
    relay_fraction,
    codec,
    super_tile,
    grid_cap=None,
    min_bytes=0,
    block=None,
) -> tuple:
    """The engine for a relay row, on *relay_devices* at *relay_fraction*.

    ``min_bytes`` 0 relays from the first byte. ``None`` keeps the host's
    per-arch cutoff. Pinned geometry is translated into a test ladder by
    ``_engine``.
    ``block`` pins the block size on every rung; ``None`` keeps each rung's.
    """
    kw = {
        "algorithm": "mesh",
        "relay_devices": tuple(relay_devices),
        "relay_fraction": tuple(relay_fraction),
    }
    if min_bytes is not None:
        kw["relay_min_bytes"] = min_bytes
    if codec != "int4":
        kw.update(rs_codec=codec, ag_codec=codec)
    if super_tile is not None:
        kw["super_tile"] = super_tile
    if grid_cap is not None:
        kw["grid_cap"] = grid_cap
    if block is not None:
        kw["block"] = block
    return _key(tp, **kw)


def _relay_case(
    tokens: int,
    hidden: int,
    codec: str,
    calls: int = RELAY_CALLS,
    graph: bool = False,
) -> tuple:
    exact = codec == "fp16"
    return (
        tokens,
        hidden,
        "exact" if exact else "normal",
        graph,
        not exact,
        calls,
    )


@benchmark()
def test_quick_allreduce_relay(
    tokens,
    hidden,
    tp,
    relay_devices,
    relay_fraction,
    codec,
    super_tile=None,
    grid_cap=None,
    expect_relay=True,
    min_bytes=0,
    calls=RELAY_CALLS,
    graph=False,
    block=None,
):
    """TP2 relay against its direct twin on the same inputs.

    Gates: SQNR and bit-identity across ranks as for any mesh row, zero lanes
    where the relay's output differs from the direct engine's on any of the
    back-to-back calls, and the relay engine running exactly when
    *expect_relay* says so. The lossless wire must also match the fp32
    reference exactly. Latencies are reported, never asserted; under *graph*
    they are per call.
    """
    exact = codec == "fp16"
    rows = _result(
        _relay_key(
            tp,
            relay_devices,
            relay_fraction,
            codec,
            super_tile,
            grid_cap,
            min_bytes,
            block,
        ),
        _relay_case(tokens, hidden, codec, calls, graph),
    )
    nbytes = tokens * hidden * 2
    label = (
        f"tp={tp} relay={tuple(relay_devices)} f={tuple(relay_fraction)} {codec} "
        f"{tokens}x{hidden} st={super_tile} grid_cap={grid_cap} block={block} "
        f"graph={graph}"
    )
    expected_st = super_tile or _shipping_st(rows, nbytes, "mesh", tp)
    if exact:
        _check(
            label,
            _identity_fails(rows, "mesh")
            + [
                f"rank {rank}: {row['n_mismatch']} mismatched elements, "
                f"max |err| {row['max_abs_err']:.3e}"
                for rank, row in enumerate(rows)
                if row["n_mismatch"]
            ],
        )
    else:
        _check_sqnr(
            label, rows, floor=SQNR_MIN_DB, expected_st=expected_st, algorithm="mesh"
        )
    _check(
        label,
        [
            f"rank {rank}: relay output differs from direct in "
            f"{row['relay_vs_direct_lanes']} bf16 lanes"
            for rank, row in enumerate(rows)
            if row["relay_vs_direct_lanes"]
        ]
        + [
            f"rank {rank}: the relay engine "
            f"{'did not run' if expect_relay else 'ran below its cutoff'} "
            f"({row['variant']})"
            for rank, row in enumerate(rows)
            if bool(row["relay_used"]) != expect_relay
        ],
    )
    ret = {
        "gfx": ARCH,
        "sqnr dB": min(r["sqnr_db"] for r in rows),
        "min tile sqnr dB": min(r["min_tile_sqnr_db"] for r in rows),
        "relay err": max(r["err"] for r in rows),
        "direct err": max(r["direct_err"] for r in rows),
        "lanes_differing": max(r["lanes_differing"] for r in rows),
        "relay_vs_direct_lanes": max(r["relay_vs_direct_lanes"] for r in rows),
        "relay_used": all(r["relay_used"] for r in rows),
        "relay_tile_fraction": rows[0]["relay_tile_fraction"],
        "relay_MiB": rows[0]["relay_bytes"] / (1 << 20),
        "variant": rows[0]["variant"],
    }
    if not exact:
        flops = tokens * hidden * (tp - 1)
        for name, key in (("direct", "direct_us"), ("relay", "us")):
            us = max(r[key] for r in rows)
            ret[f"{name} us"] = us
            ret[f"{name} TFLOPS"] = flops / us / 1e6
            ret[f"{name} TB/s"] = nbytes / us / 1e6
    else:
        ret["n_mismatch"] = max(r["n_mismatch"] for r in rows)
    return ret


@benchmark()
def test_quick_allreduce_run_ahead(tokens, hidden, tp, algorithm):
    """Back-to-back calls with new inputs, under rank skew; correctness only."""
    rows = _result(_ship_key(tp, algorithm, None), (tokens, hidden, "run_ahead", False))
    fails = _identity_fails(rows, algorithm)
    fails += [
        f"rank {rank}: SQNR {row['sqnr_db']:.2f} dB < {SQNR_MIN_DB} over "
        f"{row['calls']} calls"
        for rank, row in enumerate(rows)
        if row["sqnr_db"] < SQNR_MIN_DB
    ]
    _check(f"tp={tp} {algorithm} run-ahead", fails)
    return {
        "gfx": ARCH,
        "calls": rows[0]["calls"],
        "lanes_differing": max(r["lanes_differing"] for r in rows),
        "sqnr_db": min(r["sqnr_db"] for r in rows),
    }


@benchmark()
def test_quick_allreduce_setup_failure(tp, fault):
    """A ``_StEngine`` setup step failing on one rank fails engine construction
    on every rank.

    ``SETUP_FAULT_RANK`` fails *fault* while the host builds its engines. Gates:
    every rank raises a RuntimeError naming that rank and the injected error,
    none is left waiting in an exchange, and a clean engine then builds on
    every rank. The time to the raise is reported, not asserted.
    """
    label = f"tp={tp} setup fault={fault} on rank {SETUP_FAULT_RANK}"
    try:
        rows = _spawn(
            _run_setup_fault_rank,
            tp,
            timeout_s=SETUP_FAULT_TIMEOUT_S,
            fault=fault,
        )
    except Exception as exc:  # noqa: BLE001
        why = str(exc) or f"no result within {SETUP_FAULT_TIMEOUT_S} s"
        _check(label, [f"spawn failed: {type(exc).__name__}: {why}"])
        return {"gfx": ARCH, "fault": fault, "raised_on_all": False, "rebuilt": False}
    expected = (
        f"rank {SETUP_FAULT_RANK}: RuntimeError: "
        f"injected {SETUP_FAULTS[fault]} failure"
    )
    fails = []
    for rank, row in enumerate(rows):
        err = row["err"]
        if err is None:
            fails.append(f"rank {rank}: the engine built")
        elif not err.startswith("RuntimeError: ") or expected not in err:
            fails.append(f"rank {rank}: raised {err!r}, expected {expected!r}")
        if row["rebuild_err"] is not None:
            fails.append(f"rank {rank}: rebuild failed: {row['rebuild_err']}")
    _check(label, fails)
    return {
        "gfx": ARCH,
        "fault": fault,
        "raised_on_all": all(r["err"] for r in rows),
        "max raise s": max(r["s"] for r in rows),
        "rebuilt": not any(r["rebuild_err"] for r in rows),
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
    by_case = {c: _result(key, c) for c in _CASES[key]}
    production = set(next(iter(by_case.values()))[0]["production_cfgs"])
    ran = {
        (rows[0]["st_used"], rows[0]["block_used"])
        for case, rows in by_case.items()
        if case[2] == "normal"
    }
    missing = sorted(production - ran)
    _check(
        f"tp={tp} {algorithm} coverage of {window}",
        [f"production kernels never run: {missing}"] if missing else [],
    )
    return {
        "gfx": ARCH,
        "production_kernels": sorted(production),
        "missing": missing,
    }


@benchmark()
def test_quick_allreduce_dispatch(tp):
    """Production dispatch with ``AITER_FLY_AR=1``: the quick-reduce slot builds the
    FlyDSL mesh/ring engines and ``CudaCommunicator.all_reduce`` routes each payload
    in a schedule's window to it, reducing it within the INT4 floor.
    """
    ranks = _dispatch_ranks(tp)
    fails = []
    for rank, r in enumerate(ranks):
        if not r["built"]:
            fails.append(
                f"rank {rank}: the FlyDSL backend was not built under AITER_FLY_AR=1"
            )
            continue
        if not any(p["expect_served"] for p in r["probes"]):
            fails.append(f"rank {rank}: no payload landed in a dispatch window")
        for p in r["probes"]:
            if p["served"] != p["expect_served"]:
                fails.append(
                    f"rank {rank}: {p['nbytes']} B routed to FlyDSL={p['served']}, "
                    f"expected {p['expect_served']} ({p['algorithm']})"
                )
            elif p["served"] and p["ran"] != [p["algorithm"]]:
                fails.append(
                    f"rank {rank}: {p['nbytes']} B ran {p['ran']}, expected "
                    f"[{p['algorithm']!r}]"
                )
            if p["expect_served"] and p["sqnr_db"] < SQNR_MIN_DB:
                fails.append(
                    f"rank {rank}: {p['nbytes']} B SQNR {p['sqnr_db']:.2f} dB below "
                    f"the {SQNR_MIN_DB} dB floor"
                )
    _check(f"tp={tp} dispatch", fails)
    probes0 = ranks[0]["probes"]
    served = [p for p in probes0 if p["expect_served"]]
    return {
        "gfx": ARCH,
        "built": ranks[0]["built"],
        "served_probes": len(served),
        "total_probes": len(probes0),
        "min_sqnr_db": min((p["sqnr_db"] for p in served), default=float("nan")),
    }


@benchmark()
def test_quick_allreduce_dispatch_skewed_rank(tp):
    """``AITER_FLY_AR=1`` on every rank but the last: the init returns on every
    rank (the spawn timeout catches a hang) and none builds the FlyDSL backend."""
    ranks = _dispatch_ranks(tp, skewed=True)
    _check(
        f"tp={tp} dispatch with one rank skewed",
        [
            f"rank {rank}: built the FlyDSL backend while the last rank has "
            "AITER_FLY_AR=0"
            for rank, r in enumerate(ranks)
            if r["built"]
        ],
    )
    return {"gfx": ARCH, "built": [r["built"] for r in ranks]}


@benchmark()
def test_quick_allreduce_first_call_in_capture(tp):
    """Production dispatch with ``AITER_FLY_AR=1`` and no eager call: every
    served kernel is loaded at init, and the graph whose capture holds the
    first launch of each one replays within the INT4 floor."""
    ranks = _spawn(_capture_first_rank, tp)
    fails = []
    for rank, r in enumerate(ranks):
        if not r["built"]:
            fails.append(
                f"rank {rank}: the FlyDSL backend was not built under AITER_FLY_AR=1"
            )
            continue
        if r["unloaded"]:
            fails.append(
                f"rank {rank}: served kernels not loaded at init: {r['unloaded']}"
            )
        if r["capture_error"]:
            fails.append(f"rank {rank}: capture failed: {r['capture_error']}")
            continue
        if r["ran"] != r["expected"]:
            fails.append(
                f"rank {rank}: captured calls ran {r['ran']}, expected {r['expected']}"
            )
        for row in r["rows"]:
            if not row["sqnr_db"] >= SQNR_MIN_DB:
                fails.append(
                    f"rank {rank}: {row['algorithm']} {row['cfg']} at "
                    f"{row['nbytes']} B SQNR {row['sqnr_db']:.2f} dB below the "
                    f"{SQNR_MIN_DB} dB floor"
                )
    _check(f"tp={tp} first call in capture", fails)
    rows0 = ranks[0]["rows"]
    return {
        "gfx": ARCH,
        "kernels": [f"{row['algorithm']} {row['cfg']}" for row in rows0],
        "payloads": [row["nbytes"] for row in rows0],
        "unloaded": sum(len(r["unloaded"]) for r in ranks),
        "min_sqnr_db": min((row["sqnr_db"] for row in rows0), default=float("nan")),
    }


def _summarize(name: str, rows: list[dict]) -> None:
    if rows:
        aiter.logger.info(
            "%s summary (markdown):\n%s",
            name,
            pd.DataFrame(rows).to_markdown(index=False),
        )


def _verify(args, tps, algos, dts, link) -> None:
    # Payloads, and the production window when there is one, per schedule.
    plans = {
        (tp, algorithm): _ship_payloads(tp, algorithm, link)
        for tp, algorithm in itertools.product(tps, algos)
        if _ships(tp, algorithm, link)
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
                row = (tokens, hidden, tp, algorithm, grid_cap, block)
                if row not in ship:
                    ship.append(row)
        if window is not None and shipping_engine:
            _WINDOWS[_ship_key(tp, algorithm, None)] = window
            lo, hi = window
            label = f"[{_fmt_bytes(lo)}, {_fmt_bytes(hi)}]"
            coverage.append((tp, algorithm, label))
    # Knob rows with --extended, unless --block already sweeps the block over
    # the shipping shapes.
    knobs = []
    if args.extended and args.block == [None]:
        knobs = [
            (tokens, hidden, tp, algorithm, None, block)
            for tp, algorithm, tokens, hidden, block in KNOB_CASES
            if (tp, algorithm) in plans
        ]
    if dts:
        for tokens, hidden, tp, algorithm, grid_cap, block in ship + knobs:
            _register(
                _ship_key(tp, algorithm, grid_cap, block),
                (tokens, hidden, "normal", True),
            )
    run_ahead = [
        (tp, algorithm, *_run_ahead_shape(window))
        for (tp, algorithm), (_payloads, window) in plans.items()
    ]
    for tp, algorithm, tokens, hidden in run_ahead:
        _register(_ship_key(tp, algorithm, None), (tokens, hidden, "run_ahead", False))
    edge = [c for c in EDGE_CASES if (c[0], c[1]) in plans]
    for tp, algorithm, tokens, hidden, fill in edge:
        _register(_ship_key(tp, algorithm, None), (tokens, hidden, fill, False))
    pinned = []
    if args.extended:
        pinned = [c for c in PINNED_CODEC_CASES if (c[0], c[1]) in plans]
    for tp, algorithm, tokens, hidden, rs, ag in pinned:
        _register(
            _key(tp, algorithm=algorithm, rs_codec=rs, ag_codec=ag),
            (tokens, hidden, "normal", False),
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
        _register(
            _transport_key(tp, algorithm, st, block),
            (tokens, hidden, "exact", False),
        )

    # The divisor on the mesh at TP8, on the payloads that reach every mesh
    # kernel: the ST=1 and ST>1 epilogues handle our own chunk differently.
    divisor = []
    if (8, "mesh") in plans:
        divisor = [
            (n // _ROW_BYTES, HIDDEN, 8, fill, div)
            for n in plans[(8, "mesh")][0]
            for fill, div in (
                ("small", DEFAULT_DIVISOR),
                ("outlier", OUTLIER_DIVISOR),
            )
        ]
    for tokens, hidden, tp, fill, div in divisor:
        key = _ship_key(tp, "mesh", None)
        _register(key, (tokens, hidden, fill, False, div))
        if fill == "outlier":
            _register(key, (tokens, hidden, fill, False, DEFAULT_DIVISOR))

    def _rows(cases):
        return [
            test_quick_allreduce(
                tokens,
                hidden,
                dtype,
                tp,
                algorithm,
                grid_cap=grid_cap,
                block=block,
            )
            for tokens, hidden, tp, algorithm, grid_cap, block in cases
        ]

    for dtype in dts:
        rows = _rows(ship)
        _summarize("flydsl quick allreduce", rows)
        _summarize(
            "flydsl quick allreduce production kernel coverage",
            [
                test_quick_allreduce_coverage(tp, algorithm, window)
                for tp, algorithm, window in coverage
            ],
        )
        _summarize("flydsl quick allreduce block", _rows(knobs))
        _write_json(args.out, rows)
    _summarize(
        "flydsl quick allreduce run-ahead",
        [
            test_quick_allreduce_run_ahead(tokens, hidden, tp, algorithm)
            for tp, algorithm, tokens, hidden in run_ahead
        ],
    )
    _summarize(
        "flydsl quick allreduce edge inputs",
        [
            test_quick_allreduce_edge_inputs(tokens, hidden, tp, algorithm, fill)
            for tp, algorithm, tokens, hidden, fill in edge
        ],
    )
    _summarize(
        "flydsl quick allreduce divisor",
        [
            test_quick_allreduce_divisor(tokens, hidden, tp, fill, div)
            for tokens, hidden, tp, fill, div in divisor
        ],
    )
    _summarize(
        "flydsl quick allreduce pinned codec",
        [
            test_quick_allreduce_pinned_codec(tokens, hidden, tp, algorithm, rs, ag)
            for tp, algorithm, tokens, hidden, rs, ag in pinned
        ],
    )
    _summarize(
        "flydsl quick allreduce transport (fp16 wire, bit-exact)",
        [
            test_quick_allreduce_transport(tokens, hidden, tp, algorithm, st, block)
            for tp, algorithm, tokens, hidden, st, block in transport
        ],
    )
    os.environ["AITER_QUICK_REDUCE_QUANTIZATION"] = HIP_QR_REGIME
    aiter.qr_max_size()
    _summarize(
        "flydsl quick allreduce production dispatch",
        [test_quick_allreduce_dispatch(tp) for tp in tps],
    )
    _summarize(
        "flydsl quick allreduce dispatch with one rank skewed",
        [test_quick_allreduce_dispatch_skewed_rank(tp) for tp in tps],
    )
    _summarize(
        "flydsl quick allreduce first call in capture",
        [test_quick_allreduce_first_call_in_capture(tp) for tp in tps],
    )


def _route(policy, nbytes: int) -> str:
    """The algorithm production dispatch runs *nbytes* on with quantization
    on, or ``fallback`` for ``cross_device_reduce`` and RCCL."""
    alg = policy.select(nbytes, quantized=True)
    return "fallback" if alg is None else alg.value


def _bench(args, tps, algos, link) -> None:
    algos = tuple(a for a in ALGORITHMS if a in algos)
    rows = []
    for tp in tps:
        shipped = tuple(a for a in algos if _ships(tp, a, link))
        if not shipped:
            continue
        if args.mnk is not None:
            shapes = [(int(t), int(h)) for t, h in args.mnk]
        else:
            payloads = {n for a in shipped for n in _ship_payloads(tp, a, link)[0]}
            shapes = [(n // _ROW_BYTES, HIDDEN) for n in sorted(payloads)]
        policy = fly_policy.resolve_family_policy(world_size=tp, link=link)
        for grid_cap, block in itertools.product(args.grid_cap, args.block):
            # Every case of a spawn is registered before its first result.
            for shape in shapes:
                _register(_bench_key(tp, shipped, grid_cap, block), shape)
            rows += [
                bench_quick_allreduce(
                    tokens,
                    hidden,
                    tp,
                    shipped,
                    route=_route(policy, tokens * hidden * 2),
                    grid_cap=grid_cap,
                    block=block,
                )
                for tokens, hidden in shapes
            ]

    drop = ["algorithms"]
    drop += [k for k in ("grid_cap", "block") if getattr(args, k) == [None]]
    tables = fab.bench_tables(
        rows,
        candidates=(*algos, "hip"),
        drop=drop,
        title="flydsl quick allreduce vs hip",
    )
    for title, df in tables:
        aiter.logger.info("%s (markdown):\n%s", title, fab.to_markdown(df))
    cast = os.environ.get("AITER_QUICK_REDUCE_CAST_BF16_TO_FP16", "1")
    header = fab.provenance(
        args,
        link,
        [
            (
                f"- hip: QuickAllReduce at {HIP_QR_REGIME}, "
                f"AITER_QUICK_REDUCE_CAST_BF16_TO_FP16={cast}"
            ),
            "- `X vs hip` is `hip us / X us`: above 1.0 the schedule is faster",
        ],
    )
    fab.write_outputs(args, "FlyDSL quick all-reduce INT4 bench", header, tables, rows)
    _write_json(args.out, rows)


def _write_json(path: str | None, rows: list[dict]) -> None:
    """``-o``: the main table's rows as JSON."""
    if not path or not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(
            {
                "meta": {
                    "gfx": ARCH,
                    "timer": f"CUDA graph of {GRAPH_INNER} calls, "
                    f"{_TIMING[0]} replays per sample, median of {_TIMING[1]} "
                    "rounds",
                },
                "rows": rows,
            },
            fh,
            indent=2,
            default=str,
        )
    aiter.logger.info("wrote %s", path)


def main():
    n_gpu = torch.cuda.device_count()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__,
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
        default=None,
        help="World sizes to sweep (2, 4, 8). Default all, skipping sizes with\n"
        "fewer visible GPUs. A size that is named but cannot run is an error.\n"
        "    e.g.: --tp 8",
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
        help="Persistent-launch block caps for the shipping sweep and the relay\n"
        "rows. Default: the engine's own. The engine clamps a cap to the\n"
        "measured resident workgroups per CU.",
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
        "--relay",
        type=dtypes.str2tuple,
        nargs="*",
        default=None,
        help="TP2 mesh relay rows: pairs of devices that relay rank0->rank1 and\n"
        "rank1->rank0, outside the TP pair. Default: 2,3 on xGMI when 4 GPUs\n"
        "are visible and this arch has relay defaults. Pass an empty value to\n"
        "run no relay rows.\n"
        "    e.g.: --relay 2,3",
    )
    parser.add_argument(
        "--relay-fraction",
        type=dtypes.str2tuple,
        nargs="*",
        default=[DEFAULT_RELAY_FRACTION],
        help="Share num,den of blocks that relay, for the relay rows.\n"
        "    e.g.: --relay-fraction 1,2 3,4",
    )
    parser.add_argument(
        "--relay-codec",
        nargs="*",
        default=["int4", "fp16"],
        choices=("int4", "int5", "int6", "fp16"),
        help="Wire formats of the relay rows; fp16 is the lossless one.",
    )
    parser.add_argument(
        "-o",
        "--out",
        default=None,
        help="Optional JSON output path for the shipping-sweep rows, or for\n"
        "the bench rows.",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Also run what production never selects: the legacy fixed\n"
        "shipping shapes, the block sweep, the pinned-codec ring and the\n"
        "pinned-geometry fp16 transport matrix.",
    )
    fab.add_bench_args(parser)
    args = parser.parse_args()
    fab.check_bench_args(parser, args)
    if ARCH not in SUPPORTED_ARCHS:
        raise SystemExit(f"FlyQuickAllReduce is unsupported on {ARCH}")
    global _TIMING
    _TIMING = (args.reps, args.rounds)

    tps = fab.select_world_sizes(
        args.tp or list(SUPPORTED_WORLDS),
        explicit=bool(args.tp),
        n_gpu=n_gpu,
        supported=SUPPORTED_WORLDS,
    )
    algos = args.algorithm
    dts = [d for d in args.dtype if d == dtypes.bf16]
    if len(dts) != len(args.dtype):
        aiter.logger.warning("FlyQuickAllReduce payload is bf16; skipping others")

    if args.mnk is not None:
        for mnk in args.mnk:
            if not isinstance(mnk, tuple) or len(mnk) != 2:
                raise ValueError(f"-s expects tokens,hidden; got {mnk!r}")
    link = fly_policy.detect_link()

    if args.scenario == "bench":
        if dts:
            # quick_all_reduce.py reads this variable at import time, and
            # disables QuickAllReduce for the whole process if it does not
            # name a regime. The spawned ranks inherit it. Build the HIP
            # module here, not in every rank at once.
            os.environ["AITER_QUICK_REDUCE_QUANTIZATION"] = HIP_QR_REGIME
            aiter.qr_max_size()
            _bench(args, tps, algos, link)
    else:
        _verify(args, tps, algos, dts, link)

    # TP2 mesh relay rows. Each distinct (fraction, codec, super-tile) is an
    # engine and so a spawn. The direct TP2 rows above always run; these are
    # the second case, and only when a relay is usable on this host.
    relay_pairs = args.relay
    if link != "xgmi":
        if relay_pairs:
            aiter.logger.warning("relay rows need xGMI; link is %s, skipping", link)
        relay_pairs = [] if relay_pairs is not None else None
    if relay_pairs is None:
        has_default = (link, 2, ARCH) in MESH_RELAY_DEFAULTS
        relay_pairs = [DEFAULT_RELAY] if n_gpu >= 4 and has_default else []
    malformed = [p for p in relay_pairs if not isinstance(p, tuple)]
    if malformed:
        aiter.logger.warning(
            "--relay expects device,device pairs; skipping %s", malformed
        )
    relay_pairs = [p for p in relay_pairs if isinstance(p, tuple)]
    relay = []
    threshold_added = False
    if args.scenario == "verify" and 2 in tps and "mesh" in algos and dts:
        for pair in relay_pairs:
            if (
                len(pair) != 2
                or min(pair) < 0
                or max(pair) >= n_gpu
                or {0, 1} & set(pair)
            ):
                aiter.logger.warning(
                    "relay %s needs two devices outside 0,1 and below %s; skipping",
                    pair,
                    n_gpu,
                )
                continue
            cases = [
                (int(t), int(h), codec)
                for codec in args.relay_codec
                for t, h in (
                    args.mnk
                    if args.mnk is not None
                    else [(t, h) for t, h, c in RELAY_CASES if c == codec]
                )
            ]
            # The graph rows are the graph shapes, or the first shape of each
            # codec when ``-s`` picks them.
            if args.mnk is None:
                graph_cases = [
                    (t, h, c) for t, h, c in RELAY_GRAPH_CASES if c in args.relay_codec
                ]
            else:
                graph_cases = [
                    next(c for c in cases if c[2] == codec)
                    for codec in dict.fromkeys(args.relay_codec)
                ]
            relay_cases = [(*c, False) for c in cases] + [
                (*c, True) for c in graph_cases
            ]
            relay += [
                (
                    tokens,
                    hidden,
                    2,
                    pair,
                    tuple(fraction),
                    codec,
                    None,
                    grid_cap,
                    True,
                    0,
                    RELAY_CALLS,
                    graph,
                    None,
                )
                for tokens, hidden, codec, graph in relay_cases
                for fraction in args.relay_fraction
                for grid_cap in args.grid_cap
            ]
            if args.mnk is None:
                relay += [
                    (
                        tokens,
                        hidden,
                        2,
                        pair,
                        RELAY_REUSE_FRACTION,
                        codec,
                        1,
                        grid_cap,
                        True,
                        0,
                        RELAY_CALLS,
                        graph,
                        block,
                    )
                    for tokens, hidden, codec, block, grid_cap in RELAY_REUSE_CASES
                    if codec in args.relay_codec
                    for graph in (False, True)
                ]
            if args.extended:
                relay += [
                    (
                        tokens,
                        hidden,
                        2,
                        pair,
                        fraction,
                        codec,
                        st,
                        None,
                        True,
                        0,
                        RELAY_CALLS,
                        False,
                        None,
                    )
                    for tokens, hidden, codec, fraction, st in EXTENDED_RELAY_CASES
                ]
            # One engine at the host's cutoff, walked below, onto, above, and
            # back below it. The four rows share that engine, so the return to
            # the direct inbox happens after the relay inbox has been used.
            # The return visit uses one call so it is a different case from the
            # first visit and actually runs again.
            default = MESH_RELAY_DEFAULTS.get((link, 2, ARCH))
            if (
                not threshold_added
                and default is not None
                and args.mnk is None
                and args.grid_cap == [None]
            ):
                hidden = 5120
                at = default[0] // (hidden * 2)
                if at > 1 and at * hidden * 2 == default[0]:
                    fraction = tuple(args.relay_fraction[0])
                    relay += [
                        (
                            tokens,
                            hidden,
                            2,
                            pair,
                            fraction,
                            "int4",
                            None,
                            None,
                            used,
                            None,
                            calls,
                            False,
                            None,
                        )
                        for tokens, used, calls in (
                            (at - 1, False, RELAY_CALLS),
                            (at, True, RELAY_CALLS),
                            (at + 16, True, RELAY_CALLS),
                            (at - 1, False, 1),
                        )
                    ]
                    threshold_added = True
    for (
        tokens,
        hidden,
        tp,
        pair,
        fraction,
        codec,
        st,
        grid_cap,
        expect,
        floor,
        calls,
        graph,
        block,
    ) in relay:
        _register(
            _relay_key(tp, pair, fraction, codec, st, grid_cap, floor, block),
            _relay_case(tokens, hidden, codec, calls, graph),
        )

    _summarize(
        "flydsl quick allreduce TP2 relay",
        [test_quick_allreduce_relay(*row) for row in relay],
    )
    _summarize(
        "flydsl quick allreduce engine setup failure",
        [
            test_quick_allreduce_setup_failure(2, fault)
            for fault in (SETUP_FAULTS if 2 in tps else ())
        ],
    )

    if _FAILURES:
        raise SystemExit(
            f"{len(_FAILURES)} FlyQuickAllReduce check(s) failed:\n  "
            + "\n  ".join(_FAILURES)
        )


if __name__ == "__main__":
    freeze_support()
    start = perf_counter()
    main()
    end = perf_counter()
    aiter.logger.info(f"Test execution took {end - start:.2f}s")
