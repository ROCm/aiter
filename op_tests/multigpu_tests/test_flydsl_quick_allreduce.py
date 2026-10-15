# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Runtime correctness and timing for FlyDSL quick all-reduce (``FlyQuickAllReduce``).

``--scenario`` selects what a run does: ``verify`` (the default) checks the
kernels, and ``bench`` compares them with aiter's HIP quick all-reduce.

**verify** covers what production dispatch can run on this host, and each
sweep ends in a markdown table:

* ``test_quick_allreduce_int4`` -- the shipping configuration (codecs and
  super-tile left to the per-world defaults and ladders, as production dispatch
  constructs the engine), timed on a CUDA graph of ``GRAPH_INNER``
  back-to-back calls and checked again after every replay. The payloads come
  from ``allreduce_policy``: for every schedule it routes to at each world
  size, the smallest payload that reaches each kernel it can select there,
  plus the top of its window. A schedule the policy never selects on this
  host (the ring on xGMI) gets one row, for a user who moves the boundary with
  ``AITER_FLY_AR_MESH_MAX_BYTES``.
* ``test_quick_allreduce_int4_coverage`` -- the kernels those rows ran include
  every kernel the engine's own ``cfgs_for`` says the window selects.
* ``test_quick_allreduce_int4_edge_inputs`` -- payloads that land on the E4M3
  scale's edge cases, and degenerate groups that must stay finite.
* ``test_quick_allreduce_transport`` -- the ``fp16`` wire format, a lossless
  passthrough, on an exactly representable input: the result must be
  bit-identical to the fp32 reference, which separates a chunk-addressing or
  flag-protocol bug from a codec one. Run through each production schedule's
  ladder on the shipping payloads, so it covers the geometry that ships.

``--extended`` adds what production never selects: the legacy fixed shipping
shapes; ``test_quick_allreduce_int4`` again, as a second table, with ``block``
overridden on every rung of the shipping ladder (``KNOB_CASES``), including
the narrow blocks where a VMEM store-data hazard once showed up; the fp16
transport over a matrix of pinned ``super_tile``/``block``; and
``test_quick_allreduce_int4_pinned_codec`` -- the ring with its wire formats
pinned per lap: all-INT4 at TP8, and one lap lossless to isolate the other.

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

``--operation ar_rmsnorm`` runs the fused epilogue instead
(``FlyQuickAllReduceRMSNorm``: all-reduce + residual add + RMSNorm), and verify
runs both unless one is named. Its ``test_quick_allreduce_rmsnorm_*`` sweeps
end in one table each: SQNR against an fp32 oracle, bit-exactness of
``residual_out`` on the lossless wire (native, padded-width and
pinned-super-tile geometries), padded widths on the INT4 wire (SQNR), mesh
rank agreement, and ``test_quick_allreduce_host_checks``.

With ``--operation ar_rmsnorm``, bench times each fused schedule
(``FlyQuickAllReduceRMSNorm`` as production builds it) against
``FUSED_STOCK``: ``hip_fused``, the HIP quick all-reduce's own fused kernel,
where its row constraint allows, and ``hip+norm``, the HIP quick all-reduce
followed by ``rmsnorm2d_fwd_with_add``. ``route`` is the family the fused
policy picks with ``AITER_QUICK_REDUCE_QUANTIZATION=INT4``. Its default shapes are
``FUSED_BENCH_SHAPES``.

``flydsl_allreduce_bench.py`` describes how the comparison is kept unbiased.
The default payloads are the verify sweep's shipping payloads of both
schedules; ``-s`` or ``--shape-csv`` replace them, ``-a`` picks the schedules,
and ``--grid-cap`` and ``--block`` pin their geometry. ``--candidates`` adds
the pinned configurations of a candidate file
(``flydsl_allreduce_bench.load_candidates``) as more columns of the same
spawn, and ``--candidate-set`` selects its rows. ``--report``, ``--csv`` and
``-o`` save the tables.

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
import re
import sys
from multiprocessing import Pool, freeze_support, get_context, set_start_method

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
from aiter.test_common import benchmark, checkAllclose

set_start_method("spawn", force=True)

from aiter.ops.flydsl import allreduce_policy as fly_policy
from aiter.ops.flydsl.allreduce_policy import mesh_st_ladder, ring_st_ladder
from aiter.ops.flydsl.kernels.comm.allreduce_shared import (
    ATOMS,
    BLOCK,
    DEFAULT_GRID_CAP,
    MAX_PAYLOAD_BYTES,
    SUPPORTED_ARCHS,
    SUPPORTED_BLOCKS,
    SUPPORTED_WORLDS,
    TwoShotRung,
)
from aiter.ops.flydsl.kernels.comm.quick_allreduce_mesh import clamp_grid_cap
from aiter.ops.flydsl.quick_allreduce import (
    _resolve_inbox_flags,
    batches_publishes,
)

try:
    ARCH = get_gfx_runtime()
except (KeyError, RuntimeError):
    ARCH = None
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

# Calls captured in each timed CUDA graph.
GRAPH_INNER = fab.GRAPH_INNER

# bench: the regime of the HIP quick all-reduce, and its accuracy floor. It
# measures ~18.3 dB; the floor leaves ~4 dB of headroom.
HIP_QR_REGIME = "INT4"
HIP_SQNR_FLOOR_DB = 14.0
# bench, ar_rmsnorm: the stock quantized fused paths. ``hip_fused`` is the HIP
# quick all-reduce's own fused kernel (``qr_all_reduce_rmsnorm``), which takes
# rows that divide its 32 KiB tile; ``hip+norm`` is the HIP quick all-reduce,
# then ``rmsnorm2d_fwd_with_add``.
FUSED_STOCK = ("hip_fused", "hip+norm")
# bench, ar_rmsnorm: default shapes, decode to prefill at three model widths.
FUSED_BENCH_SHAPES = [
    (m, h) for h in (4096, 7168, 8192) for m in (16, 64, 256, 1024, 4096)
]

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

# Seconds to wait for each rank of a spawn. The kernels spin on flags written
# by peers, so a protocol bug or a dead rank hangs the rest.
SPAWN_TIMEOUT_S = 600

# ``--operation``: the plain all-reduce, or all-reduce + residual add + RMSNorm.
OPERATIONS = ("ar", "ar_rmsnorm")

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
        # For the same reason, it cannot tell one accumulation order from another.
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
        return mesh_st_ladder(world_size=world_size, link=link)
    return ring_st_ladder(world_size=world_size, link=link)


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
        msg=f"quick_allreduce_int4 rank {rank}",
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

    A case is ``(tokens, hidden, fill, time_it)``. The engine is built once, so
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
    production_cfgs = None
    if window is not None:
        production_cfgs = [(int(st), int(b)) for st, b in fly.cfgs_for(*window)]
    reps, rounds = timing

    rows = []
    try:
        for ntok, hidden, fill, time_it in cases:
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
            fly.allreduce(inp, out)
            torch.cuda.synchronize()
            dist.barrier(group=group)
            m = _metrics(out, ref, rank, tile_bytes, group)

            us = None
            if time_it:
                # The IPC inbox is allocated at init, so the captured launch is
                # safe to replay.
                g = fab.capture(lambda src=inp, dst=out: fly.allreduce(src, dst))
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
        _teardown([fly], gloo)
    return rows


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
    extra: tuple = (),
) -> list[dict]:
    """One rank of one bench spawn: every candidate on every case.

    One engine per schedule in *algorithms*, each with *engine_kw*, and the
    HIP quick all-reduce. *extra* is ``(name, algorithm, knobs)`` per pinned
    engine from ``--candidates``. A case is ``(tokens, hidden)``. Returns, per
    case, whether production would run the HIP kernel on it and each
    candidate's per-rank result.
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
    for name, algorithm, knobs in extra:
        engines[name] = _engine(
            rank, tp, gloo, device, {"algorithm": algorithm, **dict(knobs)}
        )
    reps, rounds = timing

    # The input and every candidate's output, for the whole spawn. Refer to
    # flydsl_allreduce_bench for why the addresses must not change.
    names = (*engines, "hip")
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
            for name, eng in engines.items():
                res[name]["variant"] = eng.variant(nbytes)
            if "hip" in res:
                wire = "fp16" if qr.use_fp16_kernels else "bf16"
                res["hip"]["variant"] = f"{HIP_QR_REGIME.lower()}/{wire}"
            rows.append({"hip prod": hip_prod, "candidates": res})
            del graphs
    finally:
        _teardown(engines.values(), gloo)
    return rows


def _dispatch_rank(rank: int, tp: int, init_method: str) -> dict:
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
    """
    import torch.distributed as dist

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    os.environ["AITER_FLY_AR"] = "1"
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
        if built:
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


def _spawn(target, world_size: int, **kwds) -> list:
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
        ranks = [fut.get(timeout=SPAWN_TIMEOUT_S) for fut in results]
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
# Per-rank production-dispatch results, per world size. Keyed by tp alone: the
# dispatch spawn builds the production communicator, not a knob-pinned engine.
_DISPATCH: dict[int, list] = {}
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
            op = kw.pop("op")
            ranks = _spawn(
                _bench_rank if op == "ar" else _bench_fused_rank,
                key[0],
                algorithms=kw.pop("algorithms"),
                extra=kw.pop("extra"),
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


def _dispatch_ranks(tp: int) -> list[dict]:
    if tp not in _DISPATCH:
        _DISPATCH[tp] = _spawn(_dispatch_rank, tp)
    return _DISPATCH[tp]


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
def test_quick_allreduce_int4(
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


def _bench_key(tp: int, algorithms, grid_cap, block, extra=(), op="ar") -> tuple:
    kw = {"algorithms": tuple(algorithms), "extra": tuple(extra), "op": op}
    for name, val in (("grid_cap", grid_cap), ("block", block)):
        if val is not None:
            kw[name] = val
    return _key(tp, **kw)


@benchmark()
def bench_quick_allreduce_int4(
    tokens,
    hidden,
    tp,
    algorithms,
    route,
    grid_cap=None,
    block=None,
    extra=(),
):
    """Each FlyDSL schedule against the HIP quick all-reduce.

    *route* is the algorithm production dispatch picks for the payload:
    ``mesh``, ``ring``, ``oneshot``, or ``fallback`` for
    ``cross_device_reduce`` and RCCL. *extra* are the ``--candidates``
    engines, timed in the same spawn.
    """
    key = _bench_key(tp, algorithms, grid_cap, block, extra)
    rows = _result(key, (tokens, hidden))
    pinned = tuple(name for name, _alg, _knobs in extra)
    names = [n for n in (*algorithms, *pinned, "hip") if n in rows[0]["candidates"]]
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
        "operation": "ar",
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
def test_quick_allreduce_int4_edge_inputs(tokens, hidden, tp, algorithm, fill):
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
def test_quick_allreduce_int4_pinned_codec(tokens, hidden, tp, rs_codec, ag_codec):
    """The ring with both laps' wire formats pinned; correctness only."""
    key = _key(tp, algorithm="ring", rs_codec=rs_codec, ag_codec=ag_codec)
    rows = _result(key, (tokens, hidden, "normal", False))
    label = f"tp={tp} ring {tokens}x{hidden} rs={rs_codec} ag={ag_codec}"
    _check_sqnr(
        label,
        rows,
        floor=SQNR_MIN_DB_TP8_INT4_RING,
        expected_st=_shipping_st(rows, tokens * hidden * 2, "ring", tp),
        algorithm="ring",
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
def test_quick_allreduce_int4_coverage(tp, algorithm, window):
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
                    "out_sqnr_db": _sqnr_db(
                        out.to(torch.float32), out_ref.to(torch.float32)
                    ),
                    "res_sqnr_db": _sqnr_db(
                        res_out.to(torch.float32), res_ref.to(torch.float32)
                    ),
                    "self_sqnr_db": _sqnr_db(
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
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    timeout = float(os.environ.get("FLYDSL_QR_TIMEOUT", "3600"))
    pool = get_context("spawn").Pool(processes=world_size)
    try:
        futs = [
            pool.apply_async(
                _run_rank_fused,
                kwds={
                    "rank": rank,
                    "tp": world_size,
                    "init_method": init_method,
                    "cases": cases,
                    "algorithm": algorithm,
                    "codecs": codecs,
                    "solo": solo,
                    "engine_kw": engine_kw,
                },
            )
            for rank in range(world_size)
        ]
        # A fused build is a persistent kernel whose blocks wait on the same
        # block id at every peer, so a co-residency bug hangs rather than
        # returning a wrong number. The timeout is what turns that into a
        # failed test instead of a wedged run.
        ranks = [f.get(timeout=timeout) for f in futs]
    except Exception:
        pool.terminate()
        raise
    else:
        pool.close()
    finally:
        pool.join()
    return ranks


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
#:
#: 2880 -> 3072 is the gpt-oss width (BLOCK 384, three waves) and is run at every
#: world size, where a rank owns 4, 2 and 1 atoms of a tile. 7680 -> 8192 at TP8
#: is the widest block there is (1024 threads, one atom per row), so a pad lane
#: reaching past it has nowhere left to hide. The ``11``-token rows leave a
#: partial last tile.
_FUSED_PAD_CASES = (
    (2, 512, 1536, "tp2-1536pad2048"),
    (2, 11, 1536, "tp2-1536pad2048-partial-tile"),
    (2, 512, 2880, "tp2-2880pad3072"),
    (4, 512, 2880, "tp4-2880pad3072"),
    (4, 11, 2880, "tp4-2880pad3072-partial-tile"),
    (8, 512, 2880, "tp8-2880pad3072"),
    (8, 11, 2880, "tp8-2880pad3072-partial-tile"),
    (8, 64, 7680, "tp8-7680pad8192-widest-block"),
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
    _check(f"{label}/{algorithm}", _fused_sqnr_fails(rows))
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
    _check(f"{label}/{algorithm}", fails)
    return _fused_summary(rows)


@benchmark()
def test_quick_allreduce_rmsnorm_padded_sqnr(tp, tokens, hidden, label, algorithm):
    """Padded widths on the shipping INT4 wire, graded like the native ones.

    The bit-exact test above isolates the addressing on a lossless wire; this one
    runs the codec that ships. A padded row puts zero columns into the quantized
    tile, so a pad lane that picks up a non-zero value (or a group scale that
    straddles the row boundary) shows up as lost SQNR on ``out`` and
    ``residual_out``, or as a broken ``out`` vs ``norm(residual_out)``
    self-consistency, not as a bit difference.
    """
    batch = _fused_batch(
        (tp, "pad-sqnr", algorithm),
        [(t, h) for w, t, h, _ in _FUSED_PAD_CASES if w == tp],
        algorithm=algorithm,
    )
    rows = batch[(tokens, hidden)]
    _check(f"{label}/{algorithm}", _fused_sqnr_fails(rows))
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
    _check(
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
    _check(f"tp{tp} {tokens}x{hidden} mesh", _fused_disagree_fails(rows))
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
    _check(f"{label}/mesh/st{super_tile}", fails)
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
    _check(f"tp{tp} mesh ST=8 coverage", fails)
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
    _check(f"tp{tp} {tokens}x{hidden} st{super_tile}", fails)
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
    from aiter.ops.flydsl.kernels.comm.quick_allreduce_fusions import (
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
    from aiter.ops.flydsl.kernels.comm.quick_allreduce_fusions import (
        quick_reduce_row_block,
    )

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
    from aiter.ops.flydsl.kernels.comm.quick_allreduce_fusions import (
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


def _host_padded_hidden_limits() -> list[str]:
    """What padding can and cannot address, per world size.

    The only requirement is a whole number of 16 B atoms (a multiple of 8 bf16).
    The ceiling is the 1024-thread block times the atoms a rank owns of a tile
    (``ATOMS // world_size``: 4, 2, 1), so 32768 / 16384 / 8192 at TP2 / TP4 /
    TP8. Past it, and off the atom grid, the host raises naming the constraint.
    """
    from aiter.ops.flydsl.kernels.comm.quick_allreduce_fusions import (
        quick_reduce_padded_row_block_options,
    )

    ceilings = {2: 32768, 4: 16384, 8: 8192}
    fails = []
    for world_size in SUPPORTED_WORLDS:
        ceiling = ceilings[world_size]
        widest = max(
            h
            for h in range(8, 2 * 32768 + 1, 8)
            if quick_reduce_padded_row_block_options(h, world_size)
        )
        if widest != ceiling:
            fails.append(
                f"tp={world_size}: widest padded hidden {widest}, not {ceiling}"
            )
        for hidden in (ceiling + 8, 2 * ceiling):
            fails += _expect_raises(
                f"hidden={hidden} tp={world_size}",
                "no padded fused build",
                lambda h=hidden, w=world_size: _padded_pick(h, w),
            )
        for hidden in (2884, 1001, 2879):
            if quick_reduce_padded_row_block_options(hidden, world_size):
                fails.append(f"hidden={hidden} tp={world_size}: not a multiple of 8")
            fails += _expect_raises(
                f"hidden={hidden} tp={world_size}",
                "multiple of 8",
                lambda h=hidden, w=world_size: _padded_pick(h, w),
            )
    return fails


def _padded_pick(hidden: int, world_size: int):
    from aiter.ops.flydsl.kernels.comm.allreduce_shared import BLOCK_ALIGN
    from aiter.ops.flydsl.kernels.comm.quick_allreduce_fusions import (
        _quick_reduce_atoms_choices,
        padded_row_block,
    )

    return padded_row_block(
        hidden,
        atoms_choices=_quick_reduce_atoms_choices(world_size),
        align=BLOCK_ALIGN,
    )


def _host_mesh_fanout_quad_budget_gates_narrow_blocks() -> list[str]:
    """The mesh needs a quad per (peer, sector) of a stripe, and the gate knows.

    The row geometry is necessary but not sufficient: ``hidden=1024`` at TP8
    gives a 128-thread block, which passes every row constraint and then has
    too few quads for the fanout to issue in one pass. Before this predicate
    existed the host advertised that width and the factory raised on it.
    """
    from aiter.ops.flydsl.kernels.comm.quick_allreduce_mesh import (
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
    from aiter.ops.flydsl.kernels.comm.quick_allreduce_fusions import (
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
    from aiter.ops.flydsl.kernels.comm.one_shot_allreduce import (
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
    "padded_hidden_limits": _host_padded_hidden_limits,
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
    return {"gfx": ARCH, "passed": _check(f"host-side {check}", fails)}


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
            if tp in tps and algorithm in algos
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
    edge = [c for c in EDGE_CASES if c[0] in tps and c[1] in algos]
    for tp, algorithm, tokens, hidden, fill in edge:
        _register(_ship_key(tp, algorithm, None), (tokens, hidden, fill, False))
    pinned = []
    if args.extended:
        pinned = [c for c in PINNED_CODEC_CASES if c[0] in tps and "ring" in algos]
    for tp, tokens, hidden, rs, ag in pinned:
        _register(
            _key(tp, algorithm="ring", rs_codec=rs, ag_codec=ag),
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

    def _int4_rows(cases):
        return [
            test_quick_allreduce_int4(
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
        rows = _int4_rows(ship)
        _summarize("flydsl quick allreduce INT4", rows)
        _summarize(
            "flydsl quick allreduce INT4 production kernel coverage",
            [
                test_quick_allreduce_int4_coverage(tp, algorithm, window)
                for tp, algorithm, window in coverage
            ],
        )
        _summarize("flydsl quick allreduce INT4 block", _int4_rows(knobs))
        _write_json(args.out, rows)
    _summarize(
        "flydsl quick allreduce run-ahead",
        [
            test_quick_allreduce_run_ahead(tokens, hidden, tp, algorithm)
            for tp, algorithm, tokens, hidden in run_ahead
        ],
    )
    _summarize(
        "flydsl quick allreduce INT4 edge inputs",
        [
            test_quick_allreduce_int4_edge_inputs(tokens, hidden, tp, algorithm, fill)
            for tp, algorithm, tokens, hidden, fill in edge
        ],
    )
    _summarize(
        "flydsl quick allreduce INT4 pinned codec",
        [
            test_quick_allreduce_int4_pinned_codec(tokens, hidden, tp, rs, ag)
            for tp, tokens, hidden, rs, ag in pinned
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


def _verify_fused(tps, algos) -> None:
    """The fused (all-reduce + residual add + RMSNorm) verify sweeps. The
    mesh-only rows need the mesh among the requested schedules."""
    fused = [c for c in _FUSED_CASES if c[0] in tps]
    pad = [c for c in _FUSED_PAD_CASES if c[0] in tps]
    mesh = "mesh" in algos
    _summarize(
        "flydsl quick allreduce rmsnorm sqnr",
        [
            test_quick_allreduce_rmsnorm_sqnr(tp, tokens, hidden, label, algorithm)
            for (tp, tokens, hidden, label), algorithm in itertools.product(
                fused, algos
            )
        ],
    )
    _summarize(
        "flydsl quick allreduce rmsnorm padded width (fp16 wire, bit-exact)",
        [
            test_quick_allreduce_rmsnorm_padded_is_bit_exact(
                tp, tokens, hidden, label, algorithm
            )
            for (tp, tokens, hidden, label), algorithm in itertools.product(pad, algos)
        ],
    )
    _summarize(
        "flydsl quick allreduce rmsnorm padded width (INT4 wire, sqnr)",
        [
            test_quick_allreduce_rmsnorm_padded_sqnr(
                tp, tokens, hidden, label, algorithm
            )
            for (tp, tokens, hidden, label), algorithm in itertools.product(pad, algos)
        ],
    )
    _summarize(
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
    _summarize(
        "flydsl quick allreduce rmsnorm mesh rank agreement",
        [
            test_quick_allreduce_rmsnorm_mesh_ranks_agree(tp, tokens, hidden)
            for tp, tokens, hidden in _FUSED_MESH_AGREE_CASES
            if mesh and tp in tps
        ],
    )
    _summarize(
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
    _summarize(
        "flydsl quick allreduce rmsnorm mesh ST=8 coverage",
        [
            test_quick_allreduce_rmsnorm_mesh_st8_is_exercised(tp)
            for tp in sorted({c[0] for c in fused})
            if mesh
        ],
    )
    _summarize(
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
    _summarize(
        "flydsl quick allreduce rmsnorm host-side geometry checks",
        [test_quick_allreduce_host_checks(check) for check in _HOST_CHECKS],
    )


def _fused_applies(algorithm: str, knobs: dict, tp: int, hidden: int) -> bool:
    """Whether a pinned fused mesh/ring builds exactly *knobs* at *hidden*.

    ``atoms_per_row`` resolves to the nearest one a width has, and the column
    would then time another build under this candidate's name; a ``grid_cap``
    above the CU count is clamped away by the one-workgroup-per-CU rule. Both
    leave the cell empty instead.
    """
    from aiter.ops.flydsl.quick_allreduce import (
        ALGORITHMS as HOST_ALGORITHMS,
    )
    from aiter.ops.flydsl.quick_allreduce import (
        FlyQuickAllReduceRMSNorm,
        _resolve_codecs,
    )

    cap = knobs.get("grid_cap")
    if (
        cap is not None
        and cap > torch.cuda.get_device_properties(0).multi_processor_count
    ):
        return False
    apr = knobs.get("atoms_per_row")
    if apr is not None and (8 // tp) % apr:
        return False
    algo = HOST_ALGORITHMS[algorithm]
    eng = FlyQuickAllReduceRMSNorm.__new__(FlyQuickAllReduceRMSNorm)
    eng.world_size = tp
    eng._algo = algo
    eng.rs_codec, _ag = _resolve_codecs(algo, tp, None, None)
    eng.block = None
    eng.atoms_per_row = apr
    if not eng.supports_hidden(hidden):
        return False
    return apr is None or eng._geom_for(hidden)[1] == apr


def _hip_fused_runs(qr, nbytes: int, hidden: int) -> bool:
    """Whether the HIP fused kernel can run: ``_hip_runs``, and a row that
    divides its 32 KiB tile (``should_quick_allreduce_rmsnorm``)."""
    row = hidden * 2
    return _hip_runs(qr, nbytes) and 0 < row <= 32768 and 32768 % row == 0


def _fused_bench_thunks(engines, qr, level, outs, x, residual, weight, ar_buf, eps):
    """The ``ar_rmsnorm`` bench thunks, by candidate name, for one case."""
    from aiter import qr_all_reduce_rmsnorm, rmsnorm2d_fwd_with_add

    thunks = {}
    for name, eng in engines.items():

        def _fly(e=eng, o=outs[name]):
            e.allreduce_rmsnorm(x, residual, weight, eps, out=o[0], residual_out=o[1])

        thunks[name] = _fly
    nbytes = x.numel() * 2
    hidden = x.shape[-1]
    if _hip_fused_runs(qr, nbytes, hidden):

        def _hip_fused(o=outs["hip_fused"]):
            # As QuickAllReduce.quick_all_reduce_rmsnorm, into fixed outputs.
            qr_all_reduce_rmsnorm(
                qr._ptr,
                x,
                residual,
                o[1],
                o[0],
                weight,
                eps,
                hidden,
                level.value,
                qr.use_fp16_kernels,
            )

        thunks["hip_fused"] = _hip_fused
    if _hip_runs(qr, nbytes):

        def _hip_norm(o=outs["hip+norm"]):
            qr.qr_quant_level = level
            qr.quick_all_reduce(x, out=ar_buf)
            rmsnorm2d_fwd_with_add(o[0], ar_buf, residual, o[1], weight, eps)

        thunks["hip+norm"] = _hip_norm
    return thunks


def _bench_fused_rank(
    rank: int,
    tp: int,
    init_method: str,
    algorithms: tuple[str, ...],
    engine_kw: dict,
    cases: list[tuple],
    timing: tuple[int, int],
    extra: tuple = (),
) -> list[dict]:
    """One rank of one ``ar_rmsnorm`` bench spawn: every candidate on every case.

    One ``FlyQuickAllReduceRMSNorm`` per schedule in *algorithms*, configured
    as production builds it, the *extra* engines from ``--candidates``, and
    ``FUSED_STOCK``. A case is ``(tokens, hidden)``. Returns, per case,
    whether production would run the HIP fused kernel on it and each
    candidate's per-rank result.
    """
    import torch.distributed as dist

    from aiter.dist.device_communicators.quick_all_reduce import QuickReduceRegime
    from aiter.ops.flydsl.quick_allreduce import FlyQuickAllReduceRMSNorm

    del engine_kw  # the knobs come per candidate
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    tp_group = _init_dist(rank, tp, init_method, hip=True)
    gloo = tp_group.cpu_group
    group = tp_group.device_group
    qr = tp_group.device_communicator.qr_comm
    if qr is not None and qr.disabled:
        qr = None
    level = QuickReduceRegime[HIP_QR_REGIME]
    link = fly_policy.detect_link()
    # Built in a fixed order: each engine's IPC exchange is a collective, and
    # so is each width's build (``preload`` below).
    specs = [
        (
            a,
            a,
            {
                "atoms_per_row": fly_policy.fused_qr_row_atoms(
                    world_size=tp, algorithm=a, link=link
                )
            },
        )
        for a in algorithms
    ]
    specs += [(name, a, dict(knobs)) for name, a, knobs in extra]
    engines = {}
    for name, algorithm, knobs in specs:
        engines[name] = FlyQuickAllReduceRMSNorm(
            group=gloo,
            device=device,
            rank=rank,
            world_size=tp,
            algorithm=algorithm,
            link=link,
            min_bytes=0,
            **knobs,
        )
    spec_of = {name: (algorithm, knobs) for name, algorithm, knobs in specs}
    reps, rounds = timing
    eps = fab.RMS_EPS

    # The inputs and every candidate's outputs, for the whole spawn. Refer to
    # flydsl_allreduce_bench for why the addresses must not change.
    names = (*engines, *FUSED_STOCK)
    max_bytes = max(m * hidden * 2 for m, hidden in cases)
    bufs = {
        name: torch.empty(max_bytes, dtype=torch.uint8, device=device)
        for name in (
            "inp",
            "residual",
            "ar",
            *(f"{n}:{o}" for n in names for o in ("out", "res")),
        )
    }

    def _view(name, m, hidden):
        return bufs[name][: m * hidden * 2].view(torch.bfloat16).view(m, hidden)

    rows = []
    try:
        for m, hidden in cases:
            inp, residual_src, weight = _fused_inputs(m, hidden, rank, device)
            x = _view("inp", m, hidden)
            x.copy_(inp)
            residual = _view("residual", m, hidden)
            residual.copy_(residual_src)
            total = x.to(torch.float32)
            dist.all_reduce(total, group=group)
            ref_out, ref_res = fab.fused_reference(total, residual, weight, eps)
            nbytes = m * hidden * 2
            outs = {
                n: (_view(f"{n}:out", m, hidden), _view(f"{n}:res", m, hidden))
                for n in names
            }
            # A --candidates engine whose pinned geometry this width does not
            # have is left out: n/a.
            live = {
                name: eng
                for name, eng in engines.items()
                if name in algorithms or _fused_applies(*spec_of[name], tp, hidden)
            }
            for eng in live.values():
                eng.preload(hidden)
            thunks = _fused_bench_thunks(
                live,
                qr,
                level,
                outs,
                x,
                residual,
                weight,
                _view("ar", m, hidden),
                eps,
            )
            hip_prod = False
            if "hip_fused" in thunks:
                qr.qr_quant_level = level
                hip_prod = qr._should_hip(x)

            graphs, res = {}, {}
            for name, thunk in thunks.items():
                graphs[name] = fab.capture(thunk)
                out, res_out = fab.graded_replay(graphs[name], outs[name], gloo)
                res[name] = {
                    "sqnr_db": min(
                        _sqnr_db(out.float(), ref_out.float()),
                        _sqnr_db(res_out.float(), ref_res.float()),
                    )
                }
            times = fab.time_graphs(graphs, gloo, reps=reps, rounds=rounds)
            for name, us in times.items():
                res[name]["us"] = us
            for name, eng in live.items():
                res[name]["variant"] = eng.variant(hidden, nbytes)
            wire = "fp16" if qr is not None and qr.use_fp16_kernels else "bf16"
            for name in FUSED_STOCK:
                if name in res:
                    res[name]["variant"] = f"{HIP_QR_REGIME.lower()}/{wire}"
            rows.append({"hip prod": hip_prod, "candidates": res})
            del graphs
    finally:
        _teardown(engines.values(), gloo)
    return rows


def _fused_route(policy, nbytes: int) -> str:
    """The family the fused policy routes *nbytes* to with
    ``AITER_QUICK_REDUCE_QUANTIZATION=INT4``, or ``fallback``."""
    alg = policy.select(nbytes, quantized=True)
    return "fallback" if alg is None else alg.value


@benchmark()
def bench_quick_allreduce_rmsnorm(tokens, hidden, tp, algorithms, route, extra=()):
    """Each fused FlyDSL schedule against the HIP quick all-reduce's fused
    kernel and its two-launch form.

    *route* is the family the fused policy picks for the payload. *extra* are
    the ``--candidates`` engines, timed in the same spawn.
    """
    key = _bench_key(tp, algorithms, None, None, extra, "ar_rmsnorm")
    rows = _result(key, (tokens, hidden))
    pinned = tuple(name for name, _alg, _knobs in extra)
    names = [
        n for n in (*algorithms, *pinned, *FUSED_STOCK) if n in rows[0]["candidates"]
    ]
    got = {n: fab.collapse([r["candidates"][n] for r in rows]) for n in names}

    fails = []
    for name, c in got.items():
        floor = HIP_SQNR_FLOOR_DB if name in FUSED_STOCK else FUSED_SQNR_MIN_DB
        if c["SQNR dB"] < floor:
            fails.append(f"{name}: SQNR {c['SQNR dB']:.2f} dB below {floor} dB")
        if c["variant"] is not None and c["variant"].startswith("MIXED"):
            fails.append(f"{name}: ranks ran different kernels: {c['variant']}")
    _check(f"ar_rmsnorm tp={tp} {tokens}x{hidden}", fails)

    base = "hip_fused" if "hip_fused" in got else "hip+norm"
    base_us = got[base]["us"] if base in got else None
    ret = {
        "gfx": ARCH,
        "operation": "ar_rmsnorm",
        "KiB": tokens * hidden * 2 / 1024,
        "route": route,
        "hip prod": rows[0]["hip prod"],
        "hip base": base if base in got else None,
        "flydsl vs hip": (
            fab.speedup(base_us, got[route]["us"]) if route in got else None
        ),
    }
    for algorithm in algorithms:
        ret[f"{algorithm} vs hip"] = fab.speedup(base_us, got[algorithm]["us"])
    for name, c in got.items():
        ret.update({f"{name} {field}": v for field, v in c.items()})
    return ret


def _bench_fused(args, tps, algos, link) -> None:
    extra = _candidates(args, "ar_rmsnorm", algos)
    shapes = (
        FUSED_BENCH_SHAPES
        if args.mnk is None
        else [(int(t), int(h)) for t, h in args.mnk]
    )
    rows = []
    for tp in tps:
        policy = fly_policy.resolve_fused_policy(world_size=tp, link=link)
        key = _bench_key(tp, algos, None, None, extra, "ar_rmsnorm")
        for shape in shapes:
            _register(key, shape)
        rows += [
            bench_quick_allreduce_rmsnorm(
                tokens,
                hidden,
                tp,
                algos,
                route=_fused_route(policy, tokens * hidden * 2),
                extra=extra,
            )
            for tokens, hidden in shapes
        ]
    rows = [{k: v for k, v in r.items() if k != "extra"} for r in rows]

    tables = fab.bench_tables(
        rows,
        candidates=(*algos, *(name for name, _a, _k in extra), *FUSED_STOCK),
        drop=["algorithms"],
        title="flydsl quick allreduce + rmsnorm vs hip",
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
            (
                "- `hip base` is `hip_fused` where its row constraint allows, "
                "else `hip+norm`"
            ),
            "- `X vs hip` is `hip base us / X us`: above 1.0 the schedule is faster",
        ],
    )
    fab.write_outputs(
        args, "FlyDSL quick all-reduce + RMSNorm bench", header, tables, rows
    )
    _write_json(args.out, rows)


def _route(policy, nbytes: int) -> str:
    """The algorithm production dispatch runs *nbytes* on with quantization
    on, or ``fallback`` for ``cross_device_reduce`` and RCCL."""
    alg = policy.select(nbytes, quantized=True)
    return "fallback" if alg is None else alg.value


def _candidates(args, operation: str, algos) -> tuple:
    """``(name, algorithm, knobs)`` of every ``--candidates`` row for *algos*."""
    if args.candidates is None:
        return ()
    return tuple(
        (c.name, c.algorithm, tuple(sorted(c.knobs.items())))
        for c in fab.load_candidates(
            args.candidates,
            operation=operation,
            algorithms=tuple(algos),
            candidate_set=args.candidate_set,
        )
    )


def _bench(args, tps, algos, link) -> None:
    algos = tuple(a for a in ALGORITHMS if a in algos)
    if args.operation == "ar_rmsnorm":
        _bench_fused(args, tps, algos, link)
        return
    extra = _candidates(args, "ar", algos)
    rows = []
    for tp in tps:
        if args.mnk is not None:
            shapes = [(int(t), int(h)) for t, h in args.mnk]
        else:
            payloads = {n for a in algos for n in _ship_payloads(tp, a, link)[0]}
            shapes = [(n // _ROW_BYTES, HIDDEN) for n in sorted(payloads)]
        policy = fly_policy.resolve_family_policy(world_size=tp, link=link)
        for grid_cap, block in itertools.product(args.grid_cap, args.block):
            # Every case of a spawn is registered before its first result.
            for shape in shapes:
                _register(_bench_key(tp, algos, grid_cap, block, extra), shape)
            rows += [
                bench_quick_allreduce_int4(
                    tokens,
                    hidden,
                    tp,
                    algos,
                    route=_route(policy, tokens * hidden * 2),
                    grid_cap=grid_cap,
                    block=block,
                    extra=extra,
                )
                for tokens, hidden in shapes
            ]
    rows = [{k: v for k, v in r.items() if k != "extra"} for r in rows]

    drop = ["algorithms"]
    drop += [k for k in ("grid_cap", "block") if getattr(args, k) == [None]]
    tables = fab.bench_tables(
        rows,
        candidates=(*algos, *(name for name, _a, _k in extra), "hip"),
        drop=drop,
        title="flydsl quick allreduce INT4 vs hip",
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
    parser.add_argument(
        "--operation",
        choices=OPERATIONS,
        default=None,
        help="ar: the plain all-reduce. ar_rmsnorm: all-reduce + residual add +\n"
        "RMSNorm. verify runs both unless one is named; bench runs ar unless\n"
        "ar_rmsnorm is named.",
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
        if args.operation in (None, "ar"):
            _verify(args, tps, algos, dts, link)
        if args.operation in (None, "ar_rmsnorm"):
            _verify_fused(tps, algos)

    if _FAILURES:
        raise SystemExit(
            f"{len(_FAILURES)} FlyQuickAllReduce check(s) failed:\n  "
            + "\n  ".join(_FAILURES)
        )


if __name__ == "__main__":
    freeze_support()
    from time import perf_counter

    start = perf_counter()
    main()
    end = perf_counter()
    aiter.logger.info(f"Test execution took {end - start:.2f}s")
