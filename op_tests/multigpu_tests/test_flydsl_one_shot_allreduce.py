# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and timing for the exact one-shot (1-stage) all-reduce
(``OneShotAllReduce``) and its fused form (``OneShotAllReduceRMSNorm``:
all-reduce + residual add + RMSNorm).

``--scenario`` selects what a run does: ``verify`` (the default) checks the
kernel, and ``bench`` compares it with the stock kernels production runs today.

**verify** runs four sweeps, each ending in a markdown table. A default run
drives the engine as production does -- no tuning knob pinned, so it walks
``ONESHOT_LADDER`` and picks a rung by payload size -- on payloads derived from
``allreduce_policy``: at every world size, both ends of each rung's slice of
the window the policy routes to the one-shot. ``--atoms``, ``--grid-cap``,
``--fanout`` or ``--block`` pin a configuration instead.

``test_one_shot_allreduce`` checks three things per shape, and the second
matters more than the first:

1. The sum is right, against an fp32 reference, at the bf16 rounding floor, and
   bit for bit against the fp32 rank-order sum that the kernel computes.
2. The result is **bit-identical on every rank**. The kernel accumulates in a
   fixed rank order for exactly this reason, and an SQNR check cannot see an
   ordering bug -- both answers would be equally "accurate".
3. It is timed on a CUDA graph of ``GRAPH_INNER`` back-to-back calls, and 1
   and 2 are checked again after every replay.

``test_one_shot_allreduce_coverage`` checks that those rows ran every rung the
engine's own ``cfgs_for`` says the production window selects, so a retuned
ladder or policy the payload derivation does not follow fails rather than
silently leaving a rung untested.

``test_one_shot_allreduce_run_ahead`` makes repeated back-to-back calls under
deliberate rank skew. The inbox rotates through three buffers by colour, and
the safety argument depends on a straggler's read of round k finishing before
anyone's push for round k+2 lands in the buffer it re-armed; a quiescent test
never exercises that.

``test_one_shot_allreduce_fresh_inputs`` makes back-to-back calls with a new
input on every call, at varying sizes and under rank skew, and checks each
result bit for bit against an fp32 rank-order reference. A static input cannot
see a read of a stale inbox slot: the stale data is the right answer. The
slots are re-armed rather than flagged, so the run also feeds the kernel the
NaN pattern it uses as a sentinel.

``--extended`` adds what production does not run: spot checks of pinned
configurations no shipped rung uses, at every world size, and the spot-check
shapes on the production engine too.

``--operation`` picks the kernel: ``ar`` the plain all-reduce, ``ar_rmsnorm``
the fused one. verify runs both unless one is named. The fused suites
(``test_one_shot_allreduce_rmsnorm*``) check what the plain ones do, plus one
more thing: ``residual_out`` must be **bit-exact** against the reference, not
merely close. Both sides sum in rank order in fp32 and round exactly once, so
equality is the correct expectation, and it is the only cheap way to catch a
missing bf16 round-trip before the residual add, an error that SQNR on ``out``
is far too loose to see and that compounds per layer in a real model. Their
ranks are subprocesses of this script (``--rank``) rather than spawn workers.

**bench** times the one-shot (``flydsl``) against the stock kernels on the
same payloads, all exact:

* ``cdr`` -- ``CustomAllreduce`` with ``use_new`` on, what production runs.
  Its C++ dispatch picks the kernel by world size and payload, so the
  ``variant`` column names the one that ran: ``ll``, ``ll128_<block>``,
  ``1stage`` or ``2stage``.
* ``cdr_naive`` -- ``CustomAllreduce`` with ``use_new`` off: ``1stage``,
  ``1stage_naive`` or ``2stage_naive``.
* ``rccl`` -- PyNccl, production's fallback.

The ``cdr`` kernels use their input in place under capture, registered when
the capture ends, as production does. Where production disables that
(``enable_register_for_capturing``: gfx1250, or PyTorch expandable segments),
they copy the input into the IPC pool as production does, and the variant
ends in ``+staged``.

Each row has two comparisons, both as ``other us / flydsl us``, so above 1.0
means the one-shot is faster:

* ``flydsl vs prod`` -- against the kernel ``CudaCommunicator.all_reduce``
  dispatches with the FlyDSL path off and quick reduce at its default
  (``NONE``): ``cdr`` where ``should_custom_ar`` accepts the payload, ``rccl``
  otherwise.
* ``flydsl vs best`` -- against the fastest stock kernel that clears its
  accuracy floor, which need not be the one production runs.

``--operation ar_rmsnorm`` benches the fused one-shot instead
(``OneShotAllReduceRMSNorm`` on its shipped ladder), against
``FUSED_STOCK``: ``cdr_fused_1stage`` and ``cdr_fused_2stage``,
``CustomAllreduce``'s fused kernel, and ``cdr+norm`` and ``rccl+norm``, each
all-reduce followed by ``rmsnorm2d_fwd_with_add``. ``prod`` is what
``CudaCommunicator.fused_allreduce_rmsnorm`` runs with the FlyDSL path off,
and ``route``/``route exact`` the family the fused policy picks with
quantization on and off. Its default shapes are ``FUSED_BENCH_SHAPES``.

``flydsl_allreduce_bench.py`` describes how the comparison is kept unbiased.
The default payloads are the verify sweep's plus the DeepSeek-V4 decode
shapes; ``-s`` or ``--shape-csv`` replace them, and the knob options pin the
one-shot's configuration. ``--candidates`` adds the pinned configurations of
a candidate file (``flydsl_allreduce_bench.load_candidates``) as more columns
of the same spawn, and ``--candidate-set`` selects its rows; the tuning suite
in ``flydsl_allreduce_tuning/`` drives the bench that way. ``--report`` and
``--csv`` save the tables.

Every rank is a ``multiprocessing`` spawn worker; one spawn per engine
configuration and world size runs every sweep. OneShotAllReduce runs on
gfx942/gfx950 at TP in {2, 4, 8}; other archs skip, and ``main()`` skips a
world size when fewer GPUs are visible than TP.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool, freeze_support, set_start_method

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
# The repo root for ``aiter``, this directory for ``flydsl_allreduce_bench``.
# The ranks are spawned, so they re-import this file and need the same paths.
for _path in (_REPO_ROOT, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import flydsl_allreduce_bench as fab
import pandas as pd
import pytest
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
from aiter.ops.flydsl.allreduce_policy import (
    SUPPORTED_ARCHS,
    SUPPORTED_BLOCKS,
    SUPPORTED_WORLDS,
    FusedOneShotRung,
    detect_link,
    oneshot_ladder,
)

try:
    ARCH = get_gfx_runtime()
except (KeyError, RuntimeError):
    ARCH = None

HIDDEN = 7168

# Production payloads are derived from the dispatch policy
# (``_production_payloads``) and laid out in rows of this many bf16. Every
# ladder floor is a whole number of 2 KiB rows, and a row is under every rung's
# tile, so the lowest payload is the sub-tile, single-block corner.
ROW = 1024
_ROW_BYTES = ROW * 2

# Shapes for the spot checks. m x HIDDEN spans 14 KiB to 224 KiB. m = 1, 3, 5
# end in a partial last tile and m = 8, 16 on an exact multiple at every rung's
# tile width. The narrow shapes reach the sub-tile, single-block corner HIDDEN
# cannot.
SPOT_SHAPES = [(m, HIDDEN) for m in (1, 3, 5, 8, 16)] + [
    (1, 1024),
    (1, 2048),
    (1, 3072),
]

# Each replay advances the device-side colour and rotates the inbox buffer, so
# only repeated replays show the captured launch advancing that state
# rather than freezing it.
GRAPH_REPLAYS = 4

# Calls captured in each timed CUDA graph.
GRAPH_INNER = fab.GRAPH_INNER

# bench: the ``CustomAllreduce`` kernels, as column prefix -> ``use_new``.
CDR_KERNELS = {"cdr": True, "cdr_naive": False}
# bench: every stock candidate, in the order each spawn captures and times them.
STOCK = (*CDR_KERNELS, "rccl")
# bench: the accuracy floor of the stock kernels. All are exact, at the bf16
# rounding floor (~55 dB; RCCL a little lower, since it reduces in bf16).
STOCK_SQNR_FLOOR_DB = 40.0
# bench: DeepSeek-V4 decode shapes, timed next to the policy-derived payloads.
BENCH_SHAPES = [(m, HIDDEN) for m in (1, 2, 4, 8)]
# bench, ar_rmsnorm: the stock fused paths, in the order each spawn captures
# them. ``cdr_fused_*`` is ``CustomAllreduce``'s fused kernel at both stage
# counts; ``<ar>+norm`` is that all-reduce and then ``rmsnorm2d_fwd_with_add``.
FUSED_STOCK = ("cdr_fused_1stage", "cdr_fused_2stage", "cdr+norm", "rccl+norm")
# bench, ar_rmsnorm: the accuracy floor of every candidate, on ``out`` and on
# ``residual_out``. ``out`` goes through an approximate rsqrt, so it sits well
# below the plain all-reduce's floor.
FUSED_BENCH_SQNR_FLOOR_DB = 30.0
# bench, ar_rmsnorm: default shapes, decode to prefill at three model widths.
FUSED_BENCH_SHAPES = [(m, h) for h in (4096, 7168, 8192) for m in (1, 4, 16, 64, 256)]
# bench: the largest payload a ``--candidates`` one-shot engine is timed at.
# Not a correctness limit -- the kernel is exact at every size -- but the
# one-shot handshakes once per tile, so far past any policy window it costs
# minutes per candidate to measure a size nothing routes to it.
# AITER_BENCH_FLY1S_MAX_KB moves it for one run.
CANDIDATE_MAX_BYTES = 64 << 20


def _candidate_max_bytes() -> int:
    kb = os.environ.get("AITER_BENCH_FLY1S_MAX_KB")
    return int(kb) << 10 if kb else CANDIDATE_MAX_BYTES


def _fused_applies(knobs: dict, hidden: int) -> bool:
    """Whether a pinned fused one-shot builds exactly *knobs* at *hidden*.

    A pinned ``block`` or ``split`` exists at only some widths. The engine
    would fall back to the nearest geometry, and the column would then time
    another build under this candidate's name, so the cell is left empty.
    """
    from aiter.ops.flydsl.kernels.comm.one_shot_allreduce import (
        fused_block_options,
        fused_hidden_supported,
        fused_padded_block_options,
        fused_split_options,
    )

    split, block = knobs.get("split"), knobs.get("block")
    if split is not None and split > 1:
        return any(
            k == split and (block is None or b == block)
            for k, b, _a in fused_split_options(hidden)
        )
    if block is not None:
        return any(b == block for b, _a in fused_block_options(hidden)) or any(
            b == block for b, _a, _h in fused_padded_block_options(hidden)
        )
    atoms = knobs.get("atoms", 1)
    return fused_hidden_supported(hidden, atoms) or any(
        a == atoms for _b, a, _h in fused_padded_block_options(hidden)
    )


# (tp, knobs): pinned configurations that ``--extended`` spot-checks next to
# the shipped ladder, for widths no shipped rung uses yet. It runs each at every
# world size; the tp is the one each was written for. Block 512 is the widest
# workgroup; at atoms=4 it is a 32 KiB tile, so every spot shape is a single
# partial tile.
SPOT_CONFIGS = (
    (
        4,
        {
            "atoms": 1,
            "grid_cap": 64,
            "fanout": "peer",
            "block": 512,
        },
    ),
    (
        2,
        {
            "atoms": 4,
            "grid_cap": 64,
            "fanout": "peer",
            "block": 512,
        },
    ),
    (
        8,
        {
            "atoms": 2,
            "grid_cap": 64,
            "fanout": "peer",
            "block": 256,
        },
    ),
)

RUN_AHEAD_M = 5
RUN_AHEAD_ITERS = 200
# Fresh-input stress: calls, and the token counts they cycle through at HIDDEN.
# 160 x 7168 is 2.2 MiB, several tiles per block at every rung's grid cap.
FRESH_ITERS = 300
FRESH_TOKENS = (1, 2, 3, 8, 1, 160, 4, 1)
SQNR_FLOOR_DB = 45.0

# Seconds to wait for each rank of a spawn. The kernels spin on inbox slots
# written by peers, so a protocol bug or a dead rank hangs the rest; this fails the
# spawn instead of leaving it to CI's per-file timeout. A full default run of
# either FlyDSL all-reduce test takes a few minutes, JIT included.
SPAWN_TIMEOUT_S = 600

# ``--operation``: the plain all-reduce, or all-reduce + residual add + RMSNorm.
OPERATIONS = ("ar", "ar_rmsnorm")

# Fused (all-reduce + residual add + RMSNorm) coverage. The tile is one token
# row there, so `hidden` is baked into the kernel and every distinct width is a
# distinct build -- unlike the plain kernel, where hidden is just payload bytes.
#
# 4096 at m in {1, 32} is the Qwen3-235B decode shape that motivated the kernel;
# 7168 keeps parity with the plain list above; 2048 and 8192 are the ends of the
# supported range (BLOCK 256 and 1024 at atoms=1).
FUSED_HIDDENS = (2048, 4096, 7168, 8192)
FUSED_M = (1, 2, 5, 8, 32)
RMS_EPS = 1e-6

_FUSED_SHAPE_CASES = tuple(
    (m, hidden, f"{m}x{hidden}") for hidden in FUSED_HIDDENS for m in FUSED_M
)

# Widths with no native row geometry, which run on a *padded* workgroup: BLOCK
# covers h_pad and the lanes past the real row are masked off.
#
# 896 -> 1024 and 2304 -> 2560 pad within one atom of the row; 2880 -> 3072 is
# the gpt-oss width and the one that motivated this.
#
# ``m`` has to reach past one row. With a single token every pad lane addresses
# past the end of the tensor, where the descriptor bound masks it for free --
# so m=1 passes even with the per-lane mask removed entirely. m=32 is what puts
# a *real* row under the pad, which is the only arrangement where a leaked load
# reads live data and a leaked store corrupts a neighbour.
#
# 9000 -> 9216 is the first kind of width above 8192: atoms=1 tops out at 1024
# threads x 8 elements, so a row that wide pads onto a two-atom block (BLOCK 576).
FUSED_PAD_HIDDENS = (896, 2304, 2880, 9000)
_FUSED_PAD_SHAPE_CASES = tuple(
    (m, hidden, f"{m}x{hidden}pad") for hidden in FUSED_PAD_HIDDENS for m in (1, 5, 32)
)

# `atoms` sets the *block width* in a fused build (BLOCK = hidden/(8*atoms)),
# not the tile width, so these cases exercise the reduction ladder rather than
# the wire -- the plain kernel's atoms cases already cover the wire.
#
# atoms=4 at hidden=4096 is BLOCK=128, two waves: the low end, where the LDS
# partial count drops to 2. An off-by-one in the `BLOCK//64` unrolled combine
# survives BLOCK=512 and dies here.
#
# It also catches a stride bug the default cannot see at all: the per-atom term
# in the fanout is `atom * BLOCK * ATOM_I32`, which is identically zero at
# atoms=1 no matter how wrong BLOCK is.
FUSED_ATOMS_CASES = ((2, 4096), (2, 8192), (4, 4096), (4, 8192))

# The same geometries reached from the other side: pin `block` and let the
# engine solve for atoms at each width.
FUSED_BLOCK_CASES = ((512, 4096), (256, 8192), (448, 7168))

# Fresh-input stress on the fused kernel, at the Qwen3-235B width. ``None``
# walks the shipped ladder; 4 forces a split build, whose exchange words rotate
# with the inbox colour.
FUSED_FRESH_HIDDEN = 4096
FUSED_FRESH_SPLITS = (None, 4)

# Split rows: ``split`` workgroups per token row, joined through the HBM
# exchange word. Every split the shipped widths have below 16, with 7168's odd
# 7 in particular (its grid cap rounds to 63). M=3 and M=5 leave a group count
# that does not divide the cap, so the last pass of the grid-stride loop is
# partial.
FUSED_SPLIT_CASES = ((2, 4096), (4, 4096), (2, 7168), (7, 7168), (4, 8192), (16, 8192))
FUSED_SPLIT_M = (1, 2, 3, 5)
# Calls per shape in a split spawn. The exchange word is never reset and each
# workgroup's ``prev`` persists across launches, so a bug in that bookkeeping
# only shows from the second call on; every repeat must reproduce the first
# call's bits.
FUSED_SPLIT_REPEAT = 50
# Input magnitudes at the two ends of the fixed-point exchange: its resolution
# (2**-16 per writer) against a small row, and its range against a large one.
FUSED_SPLIT_SCALES = (1e-2, 300.0)


def _production_payloads(tp: int, link: str) -> tuple[list[int], tuple[int, int]]:
    """Whole-row payloads covering the window the policy routes to the
    one-shot, and that window, in bytes (inclusive).

    The ladder picks a rung by floor alone, so each rung serves one contiguous
    slice of the window; both ends of every slice are taken, which reaches
    every rung and the largest payload each is asked to move.
    """
    window = fly_policy.resolve_family_policy(world_size=tp, link=link).window(
        fly_policy.AllReduceAlgorithm.OneShot, quantized=False
    )
    lo, hi = window.inclusive_range()
    ladder = oneshot_ladder(world_size=tp, link=link)
    ends = [rung.min_bytes - 1 for rung in ladder[1:]] + [hi]
    payloads = set()
    for rung, end in zip(ladder, ends):
        first = -(-max(lo, rung.min_bytes, 1) // _ROW_BYTES) * _ROW_BYTES
        last = min(hi, end) // _ROW_BYTES * _ROW_BYTES
        if first <= last:
            payloads.update((first, last))
    return sorted(payloads), (lo, hi)


def _sqnr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    ref = ref.double()
    err = ref - got.double()
    p = (ref * ref).mean().item()
    e = (err * err).mean().item()
    if e == 0:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(p / e)).item()


def _parts(m: int, hidden: int, tp: int, seed: int, device) -> list[torch.Tensor]:
    """Every rank's contribution, generated identically on every rank.

    Same seed everywhere, then a per-rank scale, so each rank can build the
    reference locally without another collective.
    """
    torch.manual_seed(seed)
    return [
        torch.randn(m, hidden, dtype=torch.bfloat16, device=device) * (r + 1)
        for r in range(tp)
    ]


def _metrics(out, ref, tp: int, group, exact=None) -> dict:
    """Accuracy against fp32, bit-identity across ranks, and, when *exact* is
    given, bit-identity with it (the rank-order reference).

    *group* is a gloo group: the bits are gathered on the host.
    """
    import torch.distributed as dist

    # Widened to int32 because gloo rejects int16 ("Invalid scalar type"); the
    # widening is exact, so the comparison is still on bits.
    bits = out.view(torch.int16).to(torch.int32).cpu()
    gathered = [torch.empty_like(bits) for _ in range(tp)]
    dist.all_gather(gathered, bits, group=group)
    lanes_differing = max(int((g != gathered[0]).sum()) for g in gathered)
    err = checkAllclose(
        ref, out.float(), printLog=False, msg="one_shot_allreduce vs fp32"
    )
    return {
        "sqnr_db": _sqnr_db(ref, out),
        "lanes_differing": lanes_differing,
        "err": float(err),
        "exact_mismatches": (
            0
            if exact is None
            else int((out.view(torch.int16) != exact.view(torch.int16)).sum())
        ),
    }


def _rank_order_sum(parts: list[torch.Tensor]) -> torch.Tensor:
    """fp32 sum in rank order, rounded to bf16 once: what the kernel computes,
    bit for bit."""
    acc = parts[0].float()
    for p in parts[1:]:
        acc = acc + p.float()
    return acc.to(torch.bfloat16)


def _fresh_inputs(eng, rank: int, tp: int, device) -> dict:
    """``FRESH_ITERS`` calls, each on a new input and into its own output, with
    no host sync in between, then every result checked bit for bit.

    Rank 0 is dragged now and then so the others run ahead. Every 7th call
    carries the bf16 NaN ``0xFFFF`` in some lanes of every rank, which is the
    inbox sentinel; the result there need only be NaN.
    """
    drag = torch.randn(2048, 2048, device=device, dtype=torch.float32)
    calls = []
    for it in range(FRESH_ITERS):
        m = FRESH_TOKENS[it % len(FRESH_TOKENS)]
        parts = _parts(m, HIDDEN, tp, 7 + it, device)
        if it % 7 == 3:
            for p in parts:
                p.view(torch.int16).view(-1)[it % 5 :: 97] = -1
        if rank == 0 and it % 5 == 0:
            drag = drag @ drag.T * 1e-6
        inp = parts[rank].contiguous()
        out = torch.empty_like(inp)
        eng.allreduce(inp, out)
        calls.append((out, _rank_order_sum(parts)))
    torch.cuda.synchronize()
    mismatches = 0
    for out, ref in calls:
        nan = ref.isnan()
        mismatches += int((out.view(torch.int16) != ref.view(torch.int16))[~nan].sum())
        mismatches += int((~out.isnan() & nan).sum())
    return {"calls": len(calls), "mismatches": mismatches}


def _worst(a: dict, b: dict) -> dict:
    return {
        "sqnr_db": min(a["sqnr_db"], b["sqnr_db"]),
        "lanes_differing": max(a["lanes_differing"], b["lanes_differing"]),
        "err": max(a["err"], b["err"]),
        "exact_mismatches": max(a["exact_mismatches"], b["exact_mismatches"]),
    }


def _init_dist(rank: int, tp: int, init_method: str, stock: bool):
    """Join a TP group as production does. Returns the ``GroupCoordinator``.

    ``CustomAllreduce`` is built only if *stock* is set: its IPC pool is
    large, and only the bench scenario runs its kernels.
    """
    # This test builds its own FlyDSL engine. The dispatcher must not build a
    # second one into ``CustomAllreduce``, which would then send the ``cdr``
    # payloads to FlyDSL.
    os.environ["AITER_FLY_AR"] = "0"
    # No candidate here uses quick reduce, so do not allocate its IPC buffer.
    # That is also production's default, so ``prod`` is the default path.
    os.environ["AITER_QUICK_REDUCE_QUANTIZATION"] = "NONE"
    set_custom_all_reduce(stock)
    init_distributed_environment(
        world_size=tp, rank=rank, distributed_init_method=init_method
    )
    ensure_model_parallel_initialized(tp, 1)
    return get_tp_group()


def _test_ladder(tp: int, atoms=None, grid_cap=None, fanout=None, block=None):
    """The production ladder with the knobs this test pins overriding that
    field on every rung. An unpinned knob keeps the ladder's value."""
    pins = {
        name: value
        for name, value in (
            ("atoms", atoms),
            ("grid_cap", grid_cap),
            ("fanout", fanout),
            ("block", block),
        )
        if value is not None
    }
    ladder = oneshot_ladder(world_size=tp, link=detect_link())
    return tuple(dataclasses.replace(r, **pins) for r in ladder)


def _engine(rank: int, tp: int, group, device, engine_kw: dict):
    """The one-shot engine, every rung compiled.

    With no knob pinned in *engine_kw*, it walks the production ladder;
    otherwise it runs ``_test_ladder``.
    """
    from aiter.ops.flydsl.one_shot_allreduce import OneShotAllReduce

    eng = OneShotAllReduce(
        group=group,
        device=device,
        rank=rank,
        world_size=tp,
        # MAX_PAYLOAD_BYTES is a speed policy, not a correctness limit -- the
        # kernel is exact at every size -- so it must not decide what this test
        # covers. Lifted so the shape list stays free to include sizes
        # production would route elsewhere.
        max_bytes=1 << 30,
        ladder=_test_ladder(tp, **engine_kw) if engine_kw else None,
    )
    eng.preload()
    return eng


def _teardown(eng, group) -> None:
    import torch.distributed as dist

    torch.cuda.synchronize()
    dist.barrier(group=group)
    eng.close()
    destroy_model_parallel()
    destroy_distributed_environment()


def _run_rank(
    rank: int,
    tp: int,
    init_method: str,
    engine_kw: dict,
    cases: list[tuple],
    timing: tuple[int, int],
    window: tuple[int, int] | None = None,
) -> dict:
    """One rank of one verify spawn: every case, then the run-ahead loop.

    A case is ``(tokens, hidden)``. The engine is built once, so every case
    shares its compiled rungs and IPC inboxes. *timing* is ``(reps, rounds)``
    for ``fab.time_graphs``. *window* is the payload range production dispatch
    routes to this engine, if any; the result then carries the rungs the
    engine itself says that range selects.
    """
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    cpu_group = _init_dist(rank, tp, init_method, stock=False).cpu_group
    eng = _engine(rank, tp, cpu_group, device, engine_kw)
    production_cfgs = None if window is None else eng.cfgs_for(*window)
    reps, rounds = timing

    rows = []
    try:
        for m, hidden in cases:
            parts = _parts(m, hidden, tp, 1234 + m * 8191 + hidden, device)
            inp = parts[rank].contiguous()
            ref = torch.stack(parts).float().sum(0)
            # The kernel's own order, bit for bit. Every case is a production
            # payload, so this covers both ends of every rung.
            exact = _rank_order_sum(parts)
            out = torch.zeros_like(inp)
            eng.allreduce(inp, out)
            torch.cuda.synchronize()
            res = _metrics(out, ref, tp, cpu_group, exact)

            g = fab.capture(lambda e=eng, src=inp, dst=out: e.allreduce(src, dst))
            for _ in range(GRAPH_REPLAYS):
                out.zero_()
                g.replay()
                torch.cuda.synchronize()
                res = _worst(res, _metrics(out, ref, tp, cpu_group, exact))
            nbytes = inp.numel() * inp.element_size()
            res["us"] = fab.time_graphs(
                {"flydsl": g}, cpu_group, reps=reps, rounds=rounds
            )["flydsl"]
            res["variant"] = eng.variant(nbytes)
            res["cfg"] = eng._pick_cfg(nbytes)
            del g
            rows.append(res)

        # Run-ahead: many back-to-back calls with rank 0 deliberately late, so
        # the others get a chance to run ahead into the next inbox buffer.
        parts = _parts(RUN_AHEAD_M, HIDDEN, tp, 99, device)
        inp = parts[rank].contiguous()
        out = torch.empty_like(inp)
        ref = torch.stack(parts).float().sum(0)
        drag = torch.randn(4096, 4096, device=device, dtype=torch.float32)
        bad = 0
        checks = 0
        for it in range(RUN_AHEAD_ITERS):
            if rank == 0 and it % 3 == 0:
                for _ in range(3):
                    drag = drag @ drag.T * 1e-6
            eng.allreduce(inp, out)
            if it % 25 == 0:
                torch.cuda.synchronize()
                checks += 1
                bad += _sqnr_db(ref, out) < SQNR_FLOOR_DB
        torch.cuda.synchronize()
        checks += 1
        bad += _sqnr_db(ref, out) < SQNR_FLOOR_DB
        run_ahead = {"checks": checks, "bad_checks": bad}
        fresh = _fresh_inputs(eng, rank, tp, device)
    finally:
        _teardown(eng, cpu_group)
    return {
        "rows": rows,
        "run_ahead": run_ahead,
        "fresh": fresh,
        "production_cfgs": production_cfgs,
    }


def _bench_rank(
    rank: int,
    tp: int,
    init_method: str,
    engine_kw: dict,
    cases: list[tuple],
    timing: tuple[int, int],
    extra: tuple = (),
) -> list[dict]:
    """One rank of one bench spawn: every candidate on every case.

    A case is ``(tokens, hidden)``. *extra* is ``(name, knobs)`` per pinned
    engine from ``--candidates``, timed next to ``flydsl``. Returns, per case,
    the production candidate's name and each candidate's per-rank result.
    """
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    tp_group = _init_dist(rank, tp, init_method, stock=True)
    cpu_group = tp_group.cpu_group
    comm = tp_group.device_communicator
    ca = comm.ca_comm
    if ca is not None and ca.disabled:
        ca = None
    nccl = comm.pynccl_comm
    if nccl is not None and nccl.disabled:
        nccl = None
    eng = _engine(rank, tp, cpu_group, device, engine_kw)
    # Built in a fixed order: each engine's IPC exchange is a collective.
    fly = {"flydsl": eng}
    fly.update(
        (name, _engine(rank, tp, cpu_group, device, dict(knobs)))
        for name, knobs in extra
    )
    reps, rounds = timing

    # The input and every candidate's output, for the whole spawn. Refer to
    # flydsl_allreduce_bench for why the addresses must not change.
    max_bytes = max(m * hidden * 2 for m, hidden in cases)
    bufs = {
        name: torch.empty(max_bytes, dtype=torch.uint8, device=device)
        for name in ("inp", *fly, *STOCK)
    }

    def _view(name, m, hidden):
        return bufs[name][: m * hidden * 2].view(torch.bfloat16).view(m, hidden)

    def _cdr(x, out, use_new):
        def thunk():
            # As CustomAllreduce.custom_all_reduce: in place under capture,
            # through the pre-registered pool otherwise.
            reg = (
                ca.enable_register_for_capturing
                and torch.cuda.is_current_stream_capturing()
            )
            ca.all_reduce(x, out=out, use_new=use_new, registered_input=reg)

        return thunk

    rows = []
    try:
        for m, hidden in cases:
            parts = _parts(m, hidden, tp, 1234 + m * 8191 + hidden, device)
            x = _view("inp", m, hidden)
            x.copy_(parts[rank])
            ref = torch.stack(parts).float().sum(0)
            nbytes = m * hidden * 2
            outs = {name: _view(name, m, hidden) for name in (*fly, *STOCK)}

            # A --candidates engine past its size cap is left out: n/a.
            thunks = {
                name: lambda x=x, o=outs[name], e=e: e.allreduce(x, o)
                for name, e in fly.items()
                if name == "flydsl" or nbytes <= _candidate_max_bytes()
            }
            # The same gate production applies before it runs a cdr kernel;
            # where it rejects the payload, production runs RCCL.
            custom = ca is not None and ca.should_custom_ar(x)
            if custom:
                for name, use_new in CDR_KERNELS.items():
                    thunks[name] = _cdr(x, outs[name], use_new)
            if nccl is not None:
                # What CudaCommunicator.all_reduce falls back to: an
                # out-of-place ncclAllReduce on the current stream.
                thunks["rccl"] = lambda x=x, o=outs["rccl"]: nccl.all_reduce(x, o)

            graphs, res = {}, {}
            for name, thunk in thunks.items():
                graphs[name] = fab.capture(thunk)
                got = fab.graded_replay(graphs[name], outs[name], cpu_group)
                res[name] = {
                    "sqnr_db": _sqnr_db(ref, got),
                    "err": float(
                        checkAllclose(
                            ref, got.float(), printLog=False, msg=f"{name} vs fp32"
                        )
                    ),
                }
            times = fab.time_graphs(graphs, cpu_group, reps=reps, rounds=rounds)
            for name, us in times.items():
                res[name]["us"] = us
            for name, e in fly.items():
                if name in res:
                    res[name]["variant"] = e.variant(nbytes)
            for name, use_new in CDR_KERNELS.items():
                if name in res:
                    res[name]["variant"] = fab.cdr_variant(
                        tp, nbytes, use_new=use_new, arch=ARCH
                    )
                    if not ca.enable_register_for_capturing:
                        res[name]["variant"] += "+staged"
            if "rccl" in res:
                res["rccl"]["variant"] = "pynccl"
            rows.append({"prod": "cdr" if custom else "rccl", "candidates": res})
            del graphs
    finally:
        for name, e in fly.items():
            if name != "flydsl":
                e.close()
        _teardown(eng, cpu_group)
    return rows


def _dispatch_rank(rank: int, tp: int, init_method: str) -> dict:
    """One rank of the production-dispatch spawn.

    Unlike every other spawn here, this one does not build its own engine: it
    stands up the real production ``CustomAllreduce`` with ``AITER_FLY_AR=1``
    and drives payloads across the one-shot's dispatch range through the
    top-level ``CudaCommunicator.all_reduce``, so the QR-before-CAR ordering and
    ``should_custom_ar`` gate run on the way in. ``_init_fly_oneshot`` is what
    builds the slot; ``_fly_oneshot_range`` is the byte range dispatch actually
    routes to it (the policy window intersected with ``AITER_CUSTOM_AR_*``).
    """
    import torch.distributed as dist

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    # Enable the FlyDSL dispatch, and leave quick reduce off: NONE disables
    # qr_comm, so the custom-AR slot is reached, which is also production's
    # default for a plain bf16 all-reduce.
    os.environ["AITER_FLY_AR"] = "1"
    os.environ["AITER_QUICK_REDUCE_QUANTIZATION"] = "NONE"
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp, rank=rank, distributed_init_method=init_method
    )
    ensure_model_parallel_initialized(tp, 1)
    tp_group = get_tp_group()
    cpu_group = tp_group.cpu_group
    comm = tp_group.device_communicator
    ca = comm.ca_comm
    if ca is not None and ca.disabled:
        ca = None
    built = ca is not None and ca._fly_oneshot is not None

    probes = []
    try:
        if built:
            eng = ca._fly_oneshot
            calls = [0]

            def allreduce(*args, _run=eng.allreduce, **kw):
                calls[0] += 1
                return _run(*args, **kw)

            eng.allreduce = allreduce
            lo, hi = ca._fly_oneshot_range(ca._fly_window)
            # Whole-row payloads at both ends of the served range, one in its
            # middle, and one row outside each end (expected to route elsewhere).
            served_lo = -(-lo // _ROW_BYTES) * _ROW_BYTES
            served_hi = hi // _ROW_BYTES * _ROW_BYTES
            sizes = set()
            if served_lo <= served_hi:
                mid = (served_lo + served_hi) // 2 // _ROW_BYTES * _ROW_BYTES
                sizes.update((served_lo, mid, served_hi))
                below = served_lo - _ROW_BYTES
                if below > 0:
                    sizes.add(below)
                sizes.add(served_hi + _ROW_BYTES)
            for nbytes in sorted(sizes):
                m = nbytes // _ROW_BYTES
                parts = _parts(m, ROW, tp, 4242 + nbytes, device)
                x = parts[rank].contiguous()
                ref = torch.stack(parts).float().sum(0)
                before = calls[0]
                out = comm.all_reduce(x)
                torch.cuda.synchronize()
                exact = _rank_order_sum(parts)
                probes.append(
                    {
                        "nbytes": nbytes,
                        "expect_served": served_lo <= nbytes <= served_hi,
                        "served": calls[0] > before,
                        "sqnr_db": _sqnr_db(ref, out.float()),
                        "bit_mismatches": int(
                            (out.view(torch.int16) != exact.view(torch.int16)).sum()
                        ),
                    }
                )
    finally:
        torch.cuda.synchronize()
        dist.barrier(group=cpu_group)
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


# Rows are registered up front, grouped by engine, so each engine is built by
# exactly one spawn. A spawn key is ``(tp, sorted engine kwargs)``; a case is
# ``(tokens, hidden)``.
_CASES: dict[tuple, list[tuple]] = {}
# Production dispatch window of an unpinned engine whose rung coverage is
# checked, per spawn key.
_WINDOWS: dict[tuple, tuple[int, int]] = {}
_RESULTS: dict[tuple, list] = {}
# Per-rank production-dispatch results, per world size. Keyed by tp alone: the
# dispatch spawn builds the production communicator, not a knob-pinned engine.
_DISPATCH: dict[int, list] = {}
_FAILURES: list[str] = []
# ``(reps, rounds)`` for every timed graph; set by ``main()``.
_TIMING = (fab.DEFAULT_REPS, fab.DEFAULT_ROUNDS)

# Tuning knobs the command line can pin. They are test-function arguments, so
# they select the engine, but not table columns: the ``variant`` column names
# the binary that actually ran, which is what a pinned knob changes.
KNOBS = ("atoms", "grid_cap", "fanout", "block")


def _engine_kw(atoms, grid_cap, fanout, block) -> dict:
    """OneShotAllReduce kwargs for the knobs that are pinned (not None)."""
    kw = {
        "atoms": atoms,
        "grid_cap": grid_cap,
        "fanout": fanout,
        "block": block,
    }
    return {k: v for k, v in kw.items() if v is not None}


def _key(tp: int, engine_kw: dict) -> tuple:
    return (tp, tuple(sorted(engine_kw.items())))


def _ranks(key: tuple) -> list[dict]:
    if key not in _RESULTS:
        _RESULTS[key] = _spawn(
            _run_rank,
            key[0],
            engine_kw=dict(key[1]),
            cases=_CASES[key],
            timing=_TIMING,
            window=_WINDOWS.get(key),
        )
    return _RESULTS[key]


def _dispatch_ranks(tp: int) -> list[dict]:
    if tp not in _DISPATCH:
        _DISPATCH[tp] = _spawn(_dispatch_rank, tp)
    return _DISPATCH[tp]


def _bench_key(tp: int, engine_kw: dict, extra: tuple = (), op: str = "ar") -> tuple:
    return (tp, tuple(sorted(engine_kw.items())), extra, op)


def _bench_ranks(key: tuple) -> list[list[dict]]:
    if key not in _RESULTS:
        tp, engine_kw, extra, op = key
        _RESULTS[key] = _spawn(
            _bench_rank if op == "ar" else _bench_fused_rank,
            tp,
            **({"engine_kw": dict(engine_kw)} if op == "ar" else {}),
            cases=_CASES[key],
            timing=_TIMING,
            extra=extra,
        )
    return _RESULTS[key]


def _check(label: str, fails: list[str]) -> None:
    if fails:
        msg = f"{label}: " + "; ".join(fails)
        aiter.logger.error(msg)
        _FAILURES.append(msg)


@benchmark()
def test_one_shot_allreduce(
    tokens,
    hidden,
    dtype,
    tp,
    atoms=None,
    grid_cap=None,
    fanout=None,
    block=None,
):
    engine_kw = _engine_kw(atoms, grid_cap, fanout, block)
    key = _key(tp, engine_kw)
    i = _CASES[key].index((tokens, hidden))
    rows = [r["rows"][i] for r in _ranks(key)]
    fails = []
    for rank, row in enumerate(rows):
        if row["sqnr_db"] < SQNR_FLOOR_DB:
            fails.append(
                f"rank {rank}: SQNR {row['sqnr_db']:.2f} dB below the "
                f"{SQNR_FLOOR_DB} dB bf16 floor"
            )
        if row["lanes_differing"]:
            fails.append(
                f"rank {rank}: {row['lanes_differing']} bf16 lanes differ between "
                "ranks (accumulation order is not rank-stable)"
            )
        if row["exact_mismatches"]:
            fails.append(
                f"rank {rank}: {row['exact_mismatches']} bf16 lanes differ from "
                "the rank-order reference"
            )
    _check(f"tp={tp} {tokens}x{hidden} {engine_kw}", fails)
    nbytes = tokens * hidden * 2
    # (tp - 1) adds per element.
    flops = tokens * hidden * (tp - 1)
    # The slowest rank is when the collective finishes.
    us = max(r["us"] for r in rows)
    return {
        "gfx": ARCH,
        "variant": rows[0]["variant"],
        "sqnr_db": min(r["sqnr_db"] for r in rows),
        "bit_identical": not any(r["lanes_differing"] for r in rows),
        "flydsl us": us,
        "flydsl TFLOPS": flops / us / 1e6,
        "flydsl TB/s": nbytes / us / 1e6,
        "flydsl err": max(r["err"] for r in rows),
    }


@benchmark()
def bench_one_shot_allreduce(
    tokens,
    hidden,
    tp,
    atoms=None,
    grid_cap=None,
    fanout=None,
    block=None,
    extra=(),
):
    """The one-shot against production's kernel and the best stock kernel.

    *extra* are the ``--candidates`` engines, timed in the same spawn.
    """
    engine_kw = _engine_kw(atoms, grid_cap, fanout, block)
    key = _bench_key(tp, engine_kw, extra)
    i = _CASES[key].index((tokens, hidden))
    per_rank = [r[i] for r in _bench_ranks(key)]
    pinned = tuple(name for name, _knobs in extra)
    names = [n for n in ("flydsl", *pinned, *STOCK) if n in per_rank[0]["candidates"]]
    got = {n: fab.collapse([r["candidates"][n] for r in per_rank]) for n in names}

    label = f"tp={tp} {tokens}x{hidden} {engine_kw}"
    fails = []
    for name, c in got.items():
        floor = STOCK_SQNR_FLOOR_DB if name in STOCK else SQNR_FLOOR_DB
        if c["SQNR dB"] < floor:
            fails.append(f"{name}: SQNR {c['SQNR dB']:.2f} dB below {floor} dB")
        if c["variant"] is not None and c["variant"].startswith("MIXED"):
            fails.append(f"{name}: ranks ran different kernels: {c['variant']}")
    _check(label, fails)

    def _name(n):
        return n if n == "rccl" else f"{n}:{got[n]['variant']}"

    fly_us = got["flydsl"]["us"]
    prod = per_rank[0]["prod"]
    prod_us = got[prod]["us"] if prod in got else None
    # A stock kernel below its floor failed above, and cannot be the best.
    passing = [
        n for n in STOCK if n in got and got[n]["SQNR dB"] >= STOCK_SQNR_FLOOR_DB
    ]
    best = min(passing, key=lambda n: got[n]["us"], default=None)
    best_us = got[best]["us"] if best else None
    ret = {
        "gfx": ARCH,
        "operation": "ar",
        "KiB": tokens * hidden * 2 / 1024,
        "prod": _name(prod) if prod in got else prod,
        "prod us": prod_us,
        "flydsl vs prod": fab.speedup(prod_us, fly_us),
        "best stock": _name(best) if best else None,
        "best stock us": best_us,
        "flydsl vs best": fab.speedup(best_us, fly_us),
    }
    for name, c in got.items():
        ret.update({f"{name} {field}": v for field, v in c.items()})
    return ret


@benchmark()
def test_one_shot_allreduce_run_ahead(
    tokens,
    hidden,
    tp,
    iters,
    atoms=None,
    grid_cap=None,
    fanout=None,
    block=None,
):
    engine_kw = _engine_kw(atoms, grid_cap, fanout, block)
    runs = [r["run_ahead"] for r in _ranks(_key(tp, engine_kw))]
    _check(
        f"tp={tp} run-ahead {engine_kw}",
        [
            f"rank {rank}: {r['bad_checks']} bad checks of {r['checks']}"
            for rank, r in enumerate(runs)
            if r["bad_checks"]
        ],
    )
    return {
        "gfx": ARCH,
        "checks": runs[0]["checks"],
        "bad_checks": sum(r["bad_checks"] for r in runs),
    }


@benchmark()
def test_one_shot_allreduce_fresh_inputs(
    tp,
    iters,
    atoms=None,
    grid_cap=None,
    fanout=None,
    block=None,
):
    engine_kw = _engine_kw(atoms, grid_cap, fanout, block)
    runs = [r["fresh"] for r in _ranks(_key(tp, engine_kw))]
    _check(
        f"tp={tp} fresh inputs {engine_kw}",
        [
            f"rank {rank}: {r['mismatches']} bf16 lanes differ from the "
            f"rank-order reference over {r['calls']} calls"
            for rank, r in enumerate(runs)
            if r["mismatches"]
        ],
    )
    return {
        "gfx": ARCH,
        "calls": runs[0]["calls"],
        "mismatches": sum(r["mismatches"] for r in runs),
    }


@benchmark()
def test_one_shot_allreduce_coverage(tp, window):
    """Every rung production dispatch can select on this host ran on the
    unpinned engine.

    The engine's own ``cfgs_for`` is the reference, so a retuned ladder or
    policy that ``_production_payloads`` does not follow fails here.
    """
    key = _key(tp, {})
    ranks = _ranks(key)
    production = set(ranks[0]["production_cfgs"])
    ran = {row["cfg"] for row in ranks[0]["rows"]}
    missing = sorted(production - ran)
    _check(
        f"tp={tp} coverage of {window}",
        [f"production rungs never run: {missing}"] if missing else [],
    )
    return {
        "gfx": ARCH,
        "production_rungs": sorted(production),
        "missing": missing,
    }


@benchmark()
def test_one_shot_allreduce_dispatch(tp):
    """Production dispatch with ``AITER_FLY_AR=1``: the custom all-reduce slot
    builds the FlyDSL one-shot and ``CudaCommunicator.all_reduce`` routes the
    payloads in its dispatch range to it.
    """
    ranks = _dispatch_ranks(tp)
    fails = []
    for rank, r in enumerate(ranks):
        if not r["built"]:
            fails.append(
                f"rank {rank}: the FlyDSL one-shot was not built under AITER_FLY_AR=1"
            )
            continue
        if not any(p["expect_served"] for p in r["probes"]):
            fails.append(f"rank {rank}: no payload landed in the dispatch range")
        for p in r["probes"]:
            if p["served"] != p["expect_served"]:
                fails.append(
                    f"rank {rank}: {p['nbytes']} B routed to FlyDSL={p['served']}, "
                    f"expected {p['expect_served']}"
                )
            if p["expect_served"] and p["sqnr_db"] < SQNR_FLOOR_DB:
                fails.append(
                    f"rank {rank}: {p['nbytes']} B SQNR {p['sqnr_db']:.2f} dB below "
                    f"the {SQNR_FLOOR_DB} dB bf16 floor"
                )
            if p["served"] and p["bit_mismatches"]:
                fails.append(
                    f"rank {rank}: {p['nbytes']} B differs from the rank-order "
                    f"sum in {p['bit_mismatches']} lanes"
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


# Payloads above this are not probed: the quantized windows are unbounded.
FUSED_DISPATCH_TOP_BYTES = 64 << 20
# ``out`` and ``residual_out`` floor of a quantized (INT4) fused call, as in
# test_flydsl_quick_allreduce.
FUSED_QUANTIZED_SQNR_FLOOR_DB = 18.0


def _fused_dispatch_rank(rank: int, tp: int, init_method: str, regime: str) -> dict:
    """One rank of the fused production-dispatch spawn.

    Stands up the real communicators with ``AITER_FLY_AR=1`` and
    ``AITER_QUICK_REDUCE_QUANTIZATION=<regime>``, and drives payloads across
    the fused policy's windows through ``CudaCommunicator.fused_allreduce_rmsnorm``.
    ``CustomAllreduce`` owns the exact backend (the one-shot), ``QuickAllReduce``
    the quantized one (the mesh and the ring, INT4 only). A wrapper on each
    backend records which one ran.
    """
    import torch.distributed as dist

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    os.environ["AITER_FLY_AR"] = "1"
    os.environ["AITER_QUICK_REDUCE_QUANTIZATION"] = regime
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp, rank=rank, distributed_init_method=init_method
    )
    ensure_model_parallel_initialized(tp, 1)
    tp_group = get_tp_group()
    cpu_group = tp_group.cpu_group
    comm = tp_group.device_communicator
    backends = {
        "exact": getattr(comm.ca_comm, "_fly_rms", None),
        "quantized": getattr(comm.qr_comm, "_fly_rms", None),
    }
    built = {name: b is not None for name, b in backends.items()}

    hidden = FUSED_FRESH_HIDDEN
    row_bytes = hidden * 2
    quantized = regime == "INT4"
    policy = fly_policy.resolve_fused_policy(world_size=tp)
    probes = []
    try:
        ran = []
        for name, b in backends.items():
            if b is None:
                continue

            def fused(*args, _run=b.fly_fused_ar_rms, _name=name, **kw):
                ran.append(_name)
                return _run(*args, **kw)

            b.fly_fused_ar_rms = fused
        # Whole-row payloads at both ends of every window, the quantized ones
        # too in either regime, and one row past each end of the one-shot.
        sizes = set()
        for alg in policy.algorithms(quantized=True):
            lo, hi = policy.window(alg, quantized=True).inclusive_range()
            rlo = max(-(-lo // row_bytes), 1) * row_bytes
            rhi = min(hi, FUSED_DISPATCH_TOP_BYTES) // row_bytes * row_bytes
            if rlo <= rhi:
                sizes.update((rlo, rhi))
                if alg == fly_policy.AllReduceAlgorithm.OneShot:
                    sizes.add(rhi + row_bytes)
                    if rlo > row_bytes:
                        sizes.add(rlo - row_bytes)
        for nbytes in sorted(sizes):
            m = nbytes // row_bytes
            parts, residual, weight = _fused_inputs(m, hidden, tp, device)
            ran.clear()
            out, res_out = comm.fused_allreduce_rmsnorm(
                parts[rank].contiguous(), residual, weight, RMS_EPS
            )
            torch.cuda.synchronize()
            out_ref, res_ref = _fused_reference(parts, residual, weight, RMS_EPS)
            alg = policy.select(nbytes, quantized=quantized)
            expect = (
                None
                if alg is None
                else (
                    "exact"
                    if alg == fly_policy.AllReduceAlgorithm.OneShot
                    else "quantized"
                )
            )
            probes.append(
                {
                    "nbytes": nbytes,
                    "expect": expect,
                    "ran": list(ran),
                    "out_sqnr_db": _sqnr_db(out_ref.float(), out.float()),
                    "res_sqnr_db": _sqnr_db(res_ref.float(), res_out.float()),
                    "res_mismatches": int(
                        (res_out.view(torch.int16) != res_ref.view(torch.int16)).sum()
                    ),
                }
            )
    finally:
        torch.cuda.synchronize()
        dist.barrier(group=cpu_group)
        destroy_model_parallel()
        destroy_distributed_environment()
    return {"built": built, "probes": probes}


@benchmark()
def test_fused_allreduce_rmsnorm_dispatch(tp, regime):
    """Production fused dispatch with ``AITER_FLY_AR=1``: custom all-reduce
    serves the exact one-shot in every regime; quick reduce serves the mesh and
    the ring at INT4, ahead of the one-shot, and is not built at NONE.
    """
    ranks = _spawn(_fused_dispatch_rank, tp, regime=regime)
    fails = []
    for rank, r in enumerate(ranks):
        want_built = {"exact": True, "quantized": regime == "INT4"}
        if r["built"] != want_built:
            fails.append(f"rank {rank}: built {r['built']}, expected {want_built}")
            continue
        for p in r["probes"]:
            want = [] if p["expect"] is None else [p["expect"]]
            if p["ran"] != want:
                fails.append(
                    f"rank {rank}: {p['nbytes']} B ran {p['ran']}, expected {want}"
                )
                continue
            if p["expect"] == "exact":
                if p["out_sqnr_db"] < SQNR_FLOOR_DB:
                    fails.append(
                        f"rank {rank}: {p['nbytes']} B out SQNR "
                        f"{p['out_sqnr_db']:.2f} dB below {SQNR_FLOOR_DB} dB"
                    )
                if p["res_mismatches"]:
                    fails.append(
                        f"rank {rank}: {p['nbytes']} B residual_out not bit-exact "
                        f"in {p['res_mismatches']} lanes"
                    )
            elif p["expect"] == "quantized":
                db = min(p["out_sqnr_db"], p["res_sqnr_db"])
                if db < FUSED_QUANTIZED_SQNR_FLOOR_DB:
                    fails.append(
                        f"rank {rank}: {p['nbytes']} B SQNR {db:.2f} dB below "
                        f"{FUSED_QUANTIZED_SQNR_FLOOR_DB} dB"
                    )
    _check(f"fused dispatch tp={tp} {regime}", fails)
    probes0 = ranks[0]["probes"]
    return {
        "gfx": ARCH,
        "probes": len(probes0),
        "exact": sum(p["ran"] == ["exact"] for p in probes0),
        "quantized": sum(p["ran"] == ["quantized"] for p in probes0),
        "fallback": sum(not p["ran"] for p in probes0),
        "failures": len(fails),
    }


# Fused all-reduce + RMSNorm
# ==========================


def _fused_reference(parts, residual, weight, eps):
    """The C++ contract of ``allreduce_fusion_kernel_1stage<T, T, N, false>``.

    Returns ``(out_ref, residual_out_ref)``. The bf16 round-trip on the third
    line is not an accident and not an optimization: the unfused path
    all-reduces into a bf16 tensor before adding the residual, and the fused
    kernel is required to be a drop-in, so it throws the extra f32 mantissa
    bits away too. Dropping it here would make this reference disagree with
    both kernels.
    """
    ar = torch.zeros_like(parts[0], dtype=torch.float32)
    for p in parts:  # rank order, matching the kernel's accumulation order
        ar += p.float()
    ar = ar.to(torch.bfloat16).float()
    s = ar + residual.float()
    residual_out = s.to(torch.bfloat16)
    rstd = torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps)
    return (s * rstd * weight.float()).to(torch.bfloat16), residual_out


def _fused_inputs(m, hidden, tp, device, scale=1.0):
    """Per-rank contributions, residual and gain. Same seed on every rank.

    *scale* multiplies the contributions and the residual (not the gain), which
    moves the row's sum of squares by ``scale**2``.
    """
    torch.manual_seed(1234 + m * 8191 + hidden)
    parts = [
        torch.randn(m, hidden, dtype=torch.bfloat16, device=device) * (r + 1)
        for r in range(tp)
    ]
    residual = torch.randn(m, hidden, dtype=torch.bfloat16, device=device)
    weight = torch.randn(hidden, dtype=torch.bfloat16, device=device)
    if scale != 1.0:
        parts = [(p.float() * scale).to(torch.bfloat16) for p in parts]
        residual = (residual.float() * scale).to(torch.bfloat16)
    return parts, residual, weight


def _bits_agree(tensor, tp, dist) -> int | None:
    """Rank of the first peer whose raw bits differ from rank 0's, or None.

    Widened to int32 because gloo rejects int16 ("Invalid scalar type"); the
    widening is exact, so the comparison is still on bits.
    """
    bits = tensor.view(torch.int16).to(torch.int32).cpu()
    gathered = [torch.empty_like(bits) for _ in range(tp)]
    dist.all_gather(gathered, bits)
    for r, g in enumerate(gathered):
        if not torch.equal(g, gathered[0]):
            return r
    return None


def _fused_fresh_inputs(eng, rank: int, tp: int, hidden: int, device) -> list[str]:
    """``FRESH_ITERS`` fused calls, each on new inputs and into its own outputs,
    with no host sync in between, then every result checked.

    Same skew and sentinel pattern as ``_fresh_inputs``. ``residual_out`` must
    match the reference bit for bit wherever the reference is not NaN, and be
    NaN where it is. ``out`` goes through ``rsqrt``, so it is graded on SQNR,
    over the rows the sentinel left finite: a split build clamps a NaN partial
    in its exchange, so a NaN row's norm need not be NaN.
    """
    eng.preload(hidden)
    torch.manual_seed(5)
    weight = torch.randn(hidden, dtype=torch.bfloat16, device=device)
    drag = torch.randn(2048, 2048, device=device, dtype=torch.float32)
    calls = []
    for it in range(FRESH_ITERS):
        m = FRESH_TOKENS[it % len(FRESH_TOKENS)]
        parts = _parts(m, hidden, tp, 7 + it, device)
        residual = torch.randn(m, hidden, dtype=torch.bfloat16, device=device)
        if it % 7 == 3:
            for p in parts:
                p.view(torch.int16).view(-1)[it % 5 :: 97] = -1
        if rank == 0 and it % 5 == 0:
            drag = drag @ drag.T * 1e-6
        inp = parts[rank].contiguous()
        out, res_out = eng.allreduce_rmsnorm(inp, residual, weight, RMS_EPS)
        calls.append(
            (it, out, res_out, *_fused_reference(parts, residual, weight, RMS_EPS))
        )
    torch.cuda.synchronize()
    bad = []
    for it, out, res_out, out_ref, res_ref in calls:
        nan = res_ref.isnan()
        n = int((res_out.view(torch.int16) != res_ref.view(torch.int16))[~nan].sum())
        n += int((~res_out.isnan() & nan).sum())
        if n:
            bad.append(f"call {it}: residual_out differs in {n} bf16 lanes")
        finite = ~out_ref.isnan().any(-1)
        if finite.any():
            db = _sqnr_db(out_ref[finite].float(), out[finite].float())
            if db < SQNR_FLOOR_DB:
                bad.append(f"call {it}: out SQNR {db:.2f} dB")
    if len(bad) > 5:
        bad = bad[:5] + [f"... {len(bad) - 5} more of {len(calls)} calls"]
    return bad


def _log_path(world_size: int, rank: int, mode: str, tag: str) -> str:
    """One log per (world size, rank, mode, knobs) so concurrent spawn
    configurations cannot overwrite each other's failure tails.

    *tag* has to name every pinned knob, not just ``atoms``: a block-pinned
    fused spawn leaves atoms unset, so two of them would otherwise share a path.
    """
    return f"/tmp/flydsl_one_shot_allreduce_tp{world_size}_{mode}_{tag}_rank{rank}.log"


def _fused_spawn(
    world_size: int,
    pairs: list[tuple[int, int]],
    *,
    atoms: int | None = None,
    grid_cap: int | None = None,
    fanout: str | None = None,
    block: int | None = None,
    mode: str = "fused",
    iters: int = RUN_AHEAD_ITERS,
    split: int | None = None,
    input_scale: float = 1.0,
    repeat: int = 1,
) -> list:
    """Subprocess-based spawn for fused (``OneShotAllReduceRMSNorm``) tests.

    Each rank is relaunched as a subprocess with ``--rank``/``--mode`` flags so
    it enters ``_run_rank_fused`` directly. Results are collected via a JSON
    file written by rank 0. Returns a list of bad-lists, one per rank.
    """
    if world_size not in SUPPORTED_WORLDS:
        raise ValueError(f"unsupported world_size={world_size}")
    n_gpu = torch.cuda.device_count()
    if n_gpu < world_size:
        import pytest

        pytest.skip(f"OneShotAllReduceRMSNorm needs {world_size} GPUs, have {n_gpu}")
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    out_path = os.path.join(
        tempfile.mkdtemp(prefix="flydsl_one_shot_fused_"), "rank0.json"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        f"{_REPO_ROOT}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else _REPO_ROOT
    )
    env["PYTHONUNBUFFERED"] = "1"
    if ARCH is not None:
        env.setdefault("FLYDSL_GPU_ARCH", ARCH)
    tokens = ",".join(str(t) for t, _ in pairs)
    hiddens = ",".join(str(h) for _, h in pairs)
    tag = f"a{atoms}"
    if block is not None:
        tag += f"_b{block}"
    if split is not None:
        tag += f"_k{split}"
    if grid_cap is not None:
        tag += f"_g{grid_cap}"
    if input_scale != 1.0:
        tag += f"_x{input_scale:g}"
    procs = []
    logs = []
    for rank in range(world_size):
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--rank",
            str(rank),
            "--init-method",
            init_method,
            "--tp",
            str(world_size),
            "--mode",
            mode,
            "--tokens",
            tokens,
            "--hiddens",
            hiddens,
            "--iters",
            str(iters),
        ]
        if atoms is not None:
            cmd += ["--atoms", str(atoms)]
        if grid_cap is not None:
            cmd += ["--grid-cap", str(grid_cap)]
        if fanout is not None:
            cmd += ["--fanout", fanout]
        if block is not None:
            cmd += ["--block", str(block)]
        if split is not None:
            cmd += ["--split", str(split)]
        if input_scale != 1.0:
            cmd += ["--input-scale", repr(float(input_scale))]
        if repeat != 1:
            cmd += ["--repeat", str(repeat)]
        if rank == 0:
            cmd += ["--out", out_path]
        log = open(  # noqa: SIM115
            _log_path(world_size, rank, mode, tag),
            "w",
        )
        procs.append(
            subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        )
        logs.append(log)
    rc = 0
    deadline = time.time() + float(
        os.environ.get("FLYDSL_QR_TIMEOUT", str(SPAWN_TIMEOUT_S))
    )
    for proc in procs:
        try:
            rc |= proc.wait(timeout=max(1.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            proc.kill()
            rc |= 1
    for log in logs:
        log.close()
    if rc != 0:
        tails = []
        for rank in range(world_size):
            path = _log_path(world_size, rank, mode, tag)
            try:
                with open(path) as fh:
                    tails.append(f"===== rank {rank} =====\n{fh.read()[-4000:]}")
            except OSError:
                pass
        raise RuntimeError("OneShotAllReduceRMSNorm ranks failed\n" + "\n".join(tails))
    with open(out_path) as fh:
        payload = json.load(fh)
    ranks = payload["ranks"]
    if len(ranks) != world_size:
        raise RuntimeError(
            f"OneShotAllReduceRMSNorm gathered {len(ranks)} ranks, expected {world_size}"
        )
    return ranks


def _run_rank_fused(args, rank, device, dist) -> None:
    """``OneShotAllReduceRMSNorm``: the same three questions as the plain kernel,
    plus one the plain kernel cannot ask.

    ``residual_out`` is checked for **bit-exactness**, not SQNR. Both the kernel
    and ``_fused_reference`` sum in rank order in fp32 and round exactly once,
    so equality is the correct expectation and anything else is a bug. It is
    also the only cheap detector for a missing bf16 round-trip: SQNR on ``out``
    is far too loose to notice one, and the error it hides compounds per layer.

    ``out`` goes through ``rsqrt``, whose hardware approximation legitimately
    differs from torch's, so it is graded on SQNR.

    ``fused_slice_lag`` builds with the factory's test-only
    ``debug_slice_delay``, injected by wrapping the factory rather than through
    the engine's API: slice 0 of every row group stalls between contributing to
    the exchange and reading it, so its siblings run ahead into the next
    colour's word.

    ``fused_fresh`` is the fused counterpart of ``_fresh_inputs``.
    """
    from aiter.ops.flydsl import one_shot_allreduce as _host
    from aiter.ops.flydsl.one_shot_allreduce import OneShotAllReduceRMSNorm

    if args.mode == "fused_slice_lag":
        _factory = _host.make_one_shot_allreduce_kernel

        def _lagging_factory(**kw):
            if kw.get("split", 1) > 1:
                kw["debug_slice_delay"] = 4
            return _factory(**kw)

        _host.make_one_shot_allreduce_kernel = _lagging_factory

    eng = OneShotAllReduceRMSNorm(
        group=dist.group.WORLD,
        device=device,
        rank=rank,
        world_size=args.tp,
        # atoms and block are the same knob here -- the row has to fit one
        # workgroup -- so the caller pins one and leaves the other None.
        atoms=args.atoms,
        grid_cap=args.grid_cap,
        fanout=args.fanout,
        block=args.block,
        split=args.split,
        # As in the plain test: the payload ceiling is a speed policy, not a
        # correctness limit, so it must not decide what this test covers.
        max_bytes=1 << 30,
    )

    def _check(m, hidden, tag):
        bad = []
        parts, residual, weight = _fused_inputs(
            m, hidden, args.tp, device, scale=args.input_scale
        )
        inp = parts[rank].contiguous()
        # Build and JIT this width up front, so the checked call below is
        # never the one that compiles.
        eng.preload(hidden)

        out, res_out = eng.allreduce_rmsnorm(inp, residual, weight, RMS_EPS)
        torch.cuda.synchronize()

        # Later calls must reproduce the first one's bits exactly: a split
        # build carries its exchange state across launches, and only a
        # second call reads what the first one left behind.
        for i in range(1, args.repeat):
            o2, r2 = eng.allreduce_rmsnorm(inp, residual, weight, RMS_EPS)
            torch.cuda.synchronize()
            if not (
                torch.equal(o2.view(torch.int16), out.view(torch.int16))
                and torch.equal(r2.view(torch.int16), res_out.view(torch.int16))
            ):
                bad.append(f"{tag}: call {i} differs from call 0")
                break

        out_ref, res_ref = _fused_reference(parts, residual, weight, RMS_EPS)

        db = _sqnr_db(out_ref.float(), out.float())
        if db < SQNR_FLOOR_DB:
            bad.append(
                f"{tag}: out SQNR {db:.2f} dB below the {SQNR_FLOOR_DB} dB floor"
            )

        if not torch.equal(res_out.view(torch.int16), res_ref.view(torch.int16)):
            n = int((res_out.view(torch.int16) != res_ref.view(torch.int16)).sum())
            bad.append(
                f"{tag}: residual_out differs from the reference in {n} of "
                f"{res_out.numel()} bf16 lanes (the all-reduce is not rank-ordered, "
                "or the bf16 round-trip before the residual add is missing)"
            )

        for name, t in (("out", out), ("residual_out", res_out)):
            r = _bits_agree(t, args.tp, dist)
            if r is not None:
                bad.append(f"{tag}: {name} on rank {r} differs from rank 0")
        return bad

    failures = []
    if args.mode in ("fused_run_ahead", "fused_slice_lag"):
        m, hidden = args.tokens[0], args.hiddens[0]
        parts, residual, weight = _fused_inputs(m, hidden, args.tp, device)
        inp = parts[rank].contiguous()
        out = torch.empty_like(inp)
        res_out = torch.empty_like(inp)
        eng.preload(hidden)
        out_ref, res_ref = _fused_reference(parts, residual, weight, RMS_EPS)
        drag = torch.randn(4096, 4096, device=device, dtype=torch.float32)
        bad = 0
        checks = 0
        for it in range(args.iters):
            # Rank 0 arrives late every third call, so the others run ahead into
            # the next inbox buffer -- the case the buffer rotation exists for,
            # and one a quiescent loop never reaches. (Slice lag skews the
            # workgroups of one row instead, inside the kernel.)
            if args.mode == "fused_run_ahead" and rank == 0 and it % 3 == 0:
                for _ in range(3):
                    drag = drag @ drag.T * 1e-6
            eng.allreduce_rmsnorm(
                inp, residual, weight, RMS_EPS, out=out, residual_out=res_out
            )
            if it % 25 == 0:
                torch.cuda.synchronize()
                checks += 1
                if _sqnr_db(
                    out_ref.float(), out.float()
                ) < SQNR_FLOOR_DB or not torch.equal(
                    res_out.view(torch.int16), res_ref.view(torch.int16)
                ):
                    bad += 1
        torch.cuda.synchronize()
        if bad or _sqnr_db(out_ref.float(), out.float()) < SQNR_FLOOR_DB:
            failures.append(f"{args.mode} loop: {bad} bad checks of {checks}")
    elif args.mode == "fused_fresh":
        failures.extend(
            _fused_fresh_inputs(eng, rank, args.tp, args.hiddens[0], device)
        )
    else:
        for m, hidden in zip(args.tokens, args.hiddens, strict=True):
            failures.append(_check(m, hidden, f"{m}x{hidden}"))

    gathered = [None] * args.tp
    dist.all_gather_object(gathered, failures)
    if rank == 0 and args.out:
        with open(args.out, "w") as fh:
            json.dump({"ranks": gathered}, fh)
    dist.barrier()
    eng.close()
    dist.destroy_process_group()


# One `_fused_spawn` call per group of shapes that share every axis baked
# into the compiled kernel as a Python-level constant.
_BATCH_CACHE: dict[tuple, dict[tuple[int, int], list]] = {}


def _index_by_shape(
    ranks: list, pairs: list[tuple[int, int]]
) -> dict[tuple[int, int], list]:
    """Reshape `_fused_spawn`'s per-rank, per-shape bad-lists into a per-shape view.

    ``ranks[r]`` holds one bad-list per pair, in ``pairs`` order. Slicing out
    one shape's bad-list from every rank reproduces exactly what a single-shape
    ``_fused_spawn`` call would have returned for that rank.
    """
    return {
        pair: [rank_shapes[i] for rank_shapes in ranks] for i, pair in enumerate(pairs)
    }


def _batch_cache_lookup(
    key: tuple, pairs: list[tuple[int, int]], **spawn_kwargs
) -> dict:
    """One ``_fused_spawn`` call per `key`, all shapes computed at once.

    ``key`` is ``(world_size, mode)`` or ``(world_size, mode, atoms)``; every
    axis in it is one that forces a separate spawn. Anything passed in
    ``spawn_kwargs`` is such an axis too, so it has to be reflected in *key* --
    two lookups that differ only in a kwarg would otherwise share one cached
    spawn.
    """
    if key not in _BATCH_CACHE:
        world_size, mode = key[0], key[1]
        spawn_kwargs.setdefault("atoms", key[2] if len(key) > 2 else None)
        ranks = _fused_spawn(world_size, pairs, mode=mode, **spawn_kwargs)
        _BATCH_CACHE[key] = _index_by_shape(ranks, pairs)
    return _BATCH_CACHE[key]


# ---------------------------------------------------------------------------
# Plain (pool-based) scaffolding -- used by OneShotAllReduce test suite
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("world_size", SUPPORTED_WORLDS)
@pytest.mark.parametrize("m,hidden,label", _FUSED_SHAPE_CASES)
def test_one_shot_allreduce_rmsnorm(m, hidden, label, world_size):
    """``OneShotAllReduceRMSNorm`` against an fp32 reference of the C++ contract.

    Per shape: ``out`` at the bf16 SQNR floor, ``residual_out`` **bit-exact**,
    and both outputs bit-identical across ranks. See ``_run_rank_fused`` for why
    the middle one is an equality rather than a tolerance.
    """
    group_pairs = [(mm, hh) for mm, hh, _ in _FUSED_SHAPE_CASES]
    batch = _batch_cache_lookup((world_size, "fused"), group_pairs)
    for rank, bad in enumerate(batch[(m, hidden)]):
        assert not bad, f"{label}, tp={world_size}, rank {rank}: " + "; ".join(bad)


@pytest.mark.parametrize("world_size", SUPPORTED_WORLDS)
@pytest.mark.parametrize("m,hidden,label", _FUSED_PAD_SHAPE_CASES)
def test_one_shot_allreduce_rmsnorm_padded(m, hidden, label, world_size):
    """Widths that only exist because the workgroup is padded past the row.

    Graded exactly like the native widths -- and that is the point. Padding is
    supposed to be invisible to the answer: the pad lanes load 0, push 0 on the
    wire and add 0 to the sum of squares, and their stores are dropped, so
    ``residual_out`` must still be **bit-exact** against the fp32 reference.

    Anything that leaks shows up here rather than as a tolerance wobble. A
    mask that is one atom too wide reads the next row's first 16 B into the
    norm; a store that is not dropped overwrites the next row's first 16 B.
    Both change bits that the equality check sees.
    """
    group_pairs = [(mm, hh) for mm, hh, _ in _FUSED_PAD_SHAPE_CASES]
    # key[2] is read as ``atoms`` by ``_batch_cache_lookup``; the trailing
    # "pad" is only there to keep this group's spawn out of the plain fused
    # cache entry, which shares both the mode and the atoms value.
    batch = _batch_cache_lookup((world_size, "fused", None, "pad"), group_pairs)
    for rank, bad in enumerate(batch[(m, hidden)]):
        assert not bad, f"{label}, tp={world_size}, rank {rank}: " + "; ".join(bad)


def test_one_shot_allreduce_rmsnorm_padding_is_least_wire():
    """The padded pick is the narrowest legal width at or above hidden dim.

    Host-side and GPU-free. Guards the selection rule rather than the kernel:
    padding costs ``(h_pad-hidden)/hidden`` extra wire on a bandwidth-bound
    schedule, so taking anything but the least is a silent throughput loss.

    Also pins the half of the contract that matters more -- a width with a
    native geometry must not pad at all, or every shipped shape pays for this.
    """
    from aiter.ops.flydsl.kernels.comm.one_shot_allreduce import (
        fused_block_options,
        fused_padded_block,
        fused_padded_block_options,
    )

    # Every width the padded GPU cases use must have *no* native geometry, or
    # those cases would be passing on the unpadded path and proving nothing.
    for hidden in FUSED_PAD_HIDDENS:
        assert not fused_block_options(hidden), (
            f"hidden={hidden} is in FUSED_PAD_HIDDENS but builds natively, so "
            "test_one_shot_allreduce_rmsnorm_padded is not testing padding"
        )
        assert fused_padded_block(hidden)[2] > hidden

    for hidden in range(8, 40961, 8):
        opts = fused_padded_block_options(hidden)
        if not opts:
            continue
        block, atoms, h_pad = fused_padded_block(hidden)
        assert h_pad >= hidden
        assert h_pad == min(o[2] for o in opts), (
            f"hidden={hidden} padded to {h_pad}, but {min(o[2] for o in opts)} "
            "is legal and moves fewer bytes"
        )
        assert block * atoms * 8 == h_pad
        native = fused_block_options(hidden)
        if native:
            assert (h_pad, block, atoms) == (hidden, *native[0]), (
                f"hidden={hidden} has a native geometry {native[0]} but "
                f"resolved to a padded {(block, atoms, h_pad)}"
            )


def test_one_shot_allreduce_rmsnorm_padded_hidden_limits():
    """What padding can and cannot address on the fused one-shot.

    Host-side and GPU-free. The only requirement is a whole number of 16 B atoms
    (a multiple of 8 bf16); the ceiling is the 1024-thread block times the widest
    ``SUPPORTED_ATOMS`` (4), i.e. 32768. Past it, and off the atom grid, the
    host raises naming the constraint rather than building something else.
    """
    from aiter.ops.flydsl.allreduce_policy import SUPPORTED_ATOMS
    from aiter.ops.flydsl.kernels.comm.one_shot_allreduce import (
        fused_padded_block,
        fused_padded_block_options,
    )

    ceiling = 1024 * 8 * max(SUPPORTED_ATOMS)
    assert ceiling == 32768
    widest = max(
        h for h in range(8, 2 * ceiling + 1, 8) if fused_padded_block_options(h)
    )
    assert widest == ceiling, f"widest padded hidden {widest}, not {ceiling}"

    for hidden in (ceiling + 8, 2 * ceiling):
        with pytest.raises(ValueError, match="no padded fused build"):
            fused_padded_block(hidden)
    for hidden in (2884, 1001, 2879):
        assert not fused_padded_block_options(hidden)
        with pytest.raises(ValueError, match="multiple of 8"):
            fused_padded_block(hidden)


def test_one_shot_allreduce_rmsnorm_geom_for_pinned_block_pads():
    """A pinned block a width lacks natively resolves through the padded set.

    ``OneShotAllReduceRMSNorm._geom_for`` is the resolver ``supports_hidden``
    and the launch path both go through. The two must agree on a pinned block:
    the bench sweeps ``block=1024``, which hidden=4096 admits only via padding
    (its native blocks are 512/256/128). An earlier ``_geom_for`` sent every
    pinned block through the native-only ``fused_atoms_for_block`` and raised on
    exactly this pair, so ``supports_hidden`` said yes while the warm launch
    crashed.

    ``_geom_for`` reads only ``self.block``/``self._ladder``, so it
    is exercised on a bare instance -- host-side and GPU-free, no process group.
    """
    from aiter.ops.flydsl.kernels.comm.one_shot_allreduce import fused_block_options
    from aiter.ops.flydsl.one_shot_allreduce import OneShotAllReduceRMSNorm

    def geom(block, hidden, rung_atoms=1):
        eng = OneShotAllReduceRMSNorm.__new__(OneShotAllReduceRMSNorm)
        eng.block = block
        return OneShotAllReduceRMSNorm._geom_for(eng, hidden, rung_atoms)

    # 4096 admits 512/256/128 natively; 1024 only by padding to h_pad=8192.
    assert 1024 not in [b for b, _ in fused_block_options(4096)]

    # The regression: pinned non-native block -> resolves, no raise.
    atoms, h_pad, split = geom(1024, 4096)
    assert h_pad == 8192 and 1024 * atoms * 8 == h_pad and split == 1

    # A pinned block the width *does* admit natively stays native (h_pad==hidden).
    assert geom(512, 4096) == (1, 4096, 1)

    # supports_hidden and _geom_for must give the same yes/no on the pin. When
    # supports_hidden says yes, _geom_for must return a geometry that honours
    # the pin rather than raising.
    # A split rung pins block as the slice's width: 4096 has no unsplit b64
    # build, only k8/k4/k2 ones, and supports_hidden must still say yes -- the
    # launch's _check gates on it.
    for block, split in ((128, 1), (256, 1), (512, 1), (1024, 1), (64, 8)):
        eng = OneShotAllReduceRMSNorm.__new__(OneShotAllReduceRMSNorm)
        eng.block = block
        eng._ladder = (
            FusedOneShotRung(
                min_bytes=0, atoms=1, grid_cap=64, fanout="peer", split=split
            ),
        )
        if eng.supports_hidden(4096):
            atoms, h_pad, k = OneShotAllReduceRMSNorm._geom_for(eng, 4096, 1, split)
            assert block * atoms * 8 * k == h_pad, (block, atoms, h_pad, k)
            assert k == split, (block, split, k)
        else:
            assert split == 1, (block, split)


@pytest.mark.parametrize("world_size", SUPPORTED_WORLDS)
@pytest.mark.parametrize("atoms,hidden", FUSED_ATOMS_CASES)
def test_one_shot_allreduce_rmsnorm_atoms(atoms, hidden, world_size):
    """Narrower blocks: ``atoms`` sets BLOCK = hidden/(8*atoms) in a fused build.

    Covers the low end of the reduction ladder (atoms=4 at hidden=4096 is a
    two-wave block) and the per-atom fanout stride, which is identically zero
    at the atoms=1 default and so is untested by every case above.
    """
    pairs = [(m, hidden) for m in (1, 8)]
    # hidden is in the key as well as atoms: two cases can share an atoms value
    # and differ only in width, and they are different spawns.
    batch = _batch_cache_lookup((world_size, "fused", atoms, hidden), pairs)
    for m, _ in pairs:
        for rank, bad in enumerate(batch[(m, hidden)]):
            assert not bad, (
                f"{m}x{hidden} atoms={atoms}, tp={world_size}, rank {rank}: "
                + "; ".join(bad)
            )


@pytest.mark.parametrize("world_size", SUPPORTED_WORLDS)
@pytest.mark.parametrize("block,hidden", FUSED_BLOCK_CASES)
def test_one_shot_allreduce_rmsnorm_block(block, hidden, world_size):
    """Pinning ``block`` instead of ``atoms``, which is the tuner's currency.

    The engine has to solve ``block * atoms * 8 == hidden`` for atoms at this
    width and build the same kernel the equivalent atoms pin would. What this
    catches that ``..._atoms`` does not is the resolution itself: an off-by-one
    there produces a *valid* kernel of the wrong width, which still runs.
    """
    pairs = [(m, hidden) for m in (1, 8)]
    batch = _batch_cache_lookup(
        (world_size, "fused", f"b{block}", hidden),
        pairs,
        atoms=None,
        block=block,
    )
    for m, _ in pairs:
        for rank, bad in enumerate(batch[(m, hidden)]):
            assert not bad, (
                f"{m}x{hidden} block={block}, tp={world_size}, rank {rank}: "
                + "; ".join(bad)
            )


@pytest.mark.parametrize("world_size", SUPPORTED_WORLDS)
@pytest.mark.parametrize("split", FUSED_FRESH_SPLITS)
def test_one_shot_allreduce_rmsnorm_fresh_inputs(split, world_size):
    """The fused kernel on a new input every call, at varying sizes and under
    rank skew, checked against the reference call by call.

    The plain fresh-input test's argument applies unchanged: a static input
    cannot see a read of a stale inbox slot, because the stale data is the
    right answer. Every 7th call also carries the inbox sentinel.
    """
    ranks = _fused_spawn(
        world_size,
        [(1, FUSED_FRESH_HIDDEN)],
        atoms=None,
        grid_cap=None,
        fanout=None,
        split=split,
        mode="fused_fresh",
    )
    for rank, bad in enumerate(ranks):
        assert not bad, f"tp={world_size} split={split}, rank {rank}: " + "; ".join(bad)


@pytest.mark.parametrize("world_size", SUPPORTED_WORLDS)
def test_one_shot_allreduce_rmsnorm_run_ahead(world_size):
    """The fused kernel under deliberate rank skew.

    The epilogue rotates its LDS partials through one set per colour across
    the tokens of the grid-stride loop, with no barrier between tokens. A
    quiescent single call never puts weight on that argument; this does.
    """
    ranks = _fused_spawn(
        world_size, [(RUN_AHEAD_M, 4096)], mode="fused_run_ahead", iters=RUN_AHEAD_ITERS
    )
    for rank, bad in enumerate(ranks):
        assert not bad, f"tp={world_size}, rank {rank}: " + "; ".join(bad)


@pytest.mark.parametrize("world_size", SUPPORTED_WORLDS)
@pytest.mark.parametrize("split,hidden", FUSED_SPLIT_CASES)
def test_one_shot_allreduce_rmsnorm_split(split, hidden, world_size):
    """A token row spread over ``split`` workgroups.

    The same three checks as every fused case -- SQNR on ``out``, bit-exact
    ``residual_out``, both bit-identical across ranks -- which the split build
    has to meet through its HBM exchange: the sum of squares is joined with
    integer atomics precisely so the ranks cannot disagree. ``out`` may differ
    from the unsplit build's in the last ulp (a different reduction tree), which
    the SQNR floor allows.

    Each shape is called ``FUSED_SPLIT_REPEAT`` times and every call must
    reproduce the first one's bits: the exchange word is never reset and each
    workgroup's ``prev`` persists across launches.
    """
    pairs = [(m, hidden) for m in FUSED_SPLIT_M]
    batch = _batch_cache_lookup(
        (world_size, "fused", "split", split, hidden),
        pairs,
        # Explicit: key[2] is a label here, not an atoms value. atoms=None lets
        # the engine resolve the slice geometry from the ladder's atoms.
        atoms=None,
        split=split,
        repeat=FUSED_SPLIT_REPEAT,
    )
    for m, _ in pairs:
        for rank, bad in enumerate(batch[(m, hidden)]):
            assert not bad, (
                f"{m}x{hidden} split={split}, tp={world_size}, rank {rank}: "
                + "; ".join(bad)
            )


@pytest.mark.parametrize("world_size", SUPPORTED_WORLDS)
@pytest.mark.parametrize("scale", FUSED_SPLIT_SCALES)
def test_one_shot_allreduce_rmsnorm_split_magnitude(scale, world_size):
    """The fixed-point exchange at both ends of its range.

    A small row puts weight on the exchange's resolution (2**-16 per writer)
    and a large one on its range (the per-writer clamp). Both have to stay on
    the same SQNR floor. split=4 at 7168 also exercises the host's fallback to
    the nearest split the width has (2).
    """
    pairs = [(1, 7168), (2, 8192)]
    batch = _batch_cache_lookup(
        (world_size, "fused", "split_scale", scale),
        pairs,
        atoms=None,
        split=4,
        input_scale=scale,
    )
    for m, hidden in pairs:
        for rank, bad in enumerate(batch[(m, hidden)]):
            assert not bad, (
                f"{m}x{hidden} split=4 x{scale:g}, tp={world_size}, rank {rank}: "
                + "; ".join(bad)
            )


@pytest.mark.parametrize("world_size", SUPPORTED_WORLDS)
def test_one_shot_allreduce_rmsnorm_split_slice_lag(world_size):
    """The exchange's reuse argument under deliberate skew inside one row.

    Slice 0 of every row group stalls between contributing and reading
    (``debug_slice_delay``), so its siblings finish the row and run on into the
    next colour's word. That can only happen within a launch, so the grid cap
    is pinned to two row groups: at M=8 every workgroup handles four rows per
    call, crossing colours three times.
    """
    split = 4
    ranks = _fused_spawn(
        world_size,
        [(8, 8192)],
        atoms=None,
        grid_cap=2 * split,
        split=split,
        mode="fused_slice_lag",
        iters=RUN_AHEAD_ITERS,
    )
    for rank, bad in enumerate(ranks):
        assert not bad, f"tp={world_size}, rank {rank}: " + "; ".join(bad)


def test_one_shot_allreduce_rmsnorm_split_options():
    """``fused_split_options`` is the geometry every split build draws from.

    GPU-free. Pins the sets the shipped widths have -- 7168 splits only into
    {2, 7, 14} -- and the invariants each option must meet: the slices tile the
    row in whole waves, and every wave of the row fits the exchange word's
    arrival count.
    """
    from aiter.ops.flydsl.allreduce_policy import SUPPORTED_SPLITS
    from aiter.ops.flydsl.kernels.comm.one_shot_allreduce import (
        fused_block_options,
        fused_split_options,
    )
    from aiter.ops.flydsl.kernels.comm.quick_allreduce_fusions import XCHG_COUNT_BITS

    def splits(hidden):
        return {k for k, _b, _a in fused_split_options(hidden)}

    assert splits(7168) == {1, 2, 7, 14}
    assert splits(8192) == {1, 2, 4, 8, 16}
    assert splits(4096) == {1, 2, 4, 8}
    for hidden in range(512, 16385, 512):
        opts = fused_split_options(hidden)
        # split=1 is exactly the unsplit geometry, in the same order.
        assert [(b, a) for k, b, a in opts if k == 1] == list(
            fused_block_options(hidden)
        )
        for k, b, a in opts:
            assert k in SUPPORTED_SPLITS
            assert k * b * a * 8 == hidden and b % 64 == 0, (hidden, k, b, a)
            if k > 1:
                assert k * (b // 64) < 1 << XCHG_COUNT_BITS, (hidden, k, b, a)


def test_one_shot_allreduce_rmsnorm_split_factory():
    """The factory's split contract, GPU-free: what it rejects and what it
    reports.

    Building a spec traces nothing -- the kernel compiles on first launch --
    so every assertion here runs without a device.
    """
    from aiter.ops.flydsl.kernels.comm.one_shot_allreduce import (
        make_one_shot_allreduce_kernel as mk,
    )

    base = {
        "world_size": 4,
        "grid": 64,
        "inbox_memory": "uncached",
        "fanout": "peer",
        "rank": 0,
        "fusion": "rmsnorm",
        "hidden": 7168,
        "atoms": 1,
    }
    with pytest.raises(ValueError, match="only meaningful for a fused"):
        mk(
            world_size=4,
            atoms=1,
            grid=64,
            inbox_memory="uncached",
            fanout="peer",
            block=256,
            rank=0,
            split=2,
        )
    with pytest.raises(ValueError, match="split must be one of"):
        mk(**base, split=3)
    with pytest.raises(ValueError, match="needs a native geometry"):
        mk(**{**base, "hidden": 3072, "atoms": 2}, h_pad=4096, split=2)
    with pytest.raises(ValueError, match="does not divide"):
        mk(**{**base, "grid": 63, "hidden": 4096}, split=7)
    with pytest.raises(ValueError, match="not a multiple of split"):
        mk(**base, split=7)  # 64 % 7
    with pytest.raises(ValueError, match="needs a split build"):
        mk(**base, debug_slice_delay=2)
    # 16 slices of 16 waves: 256 writers, one more than the arrival count holds.
    with pytest.raises(ValueError, match="counts arrivals"):
        mk(**{**base, "hidden": 131072}, split=16)

    spec = mk(**base, split=2)
    assert (spec["split"], spec["block"], spec["tile_bytes"]) == (2, 448, 7168)
    # No LDS: a split build reduces across waves through HBM.
    assert spec["lds_bytes"] == 0
    # Words: 3 colours x 32 groups, a line each; prev: 64 workgroups x 3 x 8 B.
    assert spec["xchg_bytes"] == 3 * 32 * 128 + 64 * 3 * 8
    unsplit = mk(**base)
    assert (unsplit["split"], unsplit["xchg_bytes"]) == (1, 0)
    assert unsplit["lds_bytes"] > 0


def test_one_shot_allreduce_rmsnorm_split_host_resolution():
    """The host side of split, GPU-free: ``_geom_for``, ``_cfg_key`` and the
    launch geometry.

    A requested split the width lacks falls to the nearest one it has (ties to
    the smaller); padded widths run unsplit; the grid cap rounds down to whole
    row groups; the launch covers whole row groups and nothing past the cap.
    """
    from aiter.ops.flydsl.one_shot_allreduce import OneShotAllReduceRMSNorm as E

    def bare(block=None):
        eng = E.__new__(E)
        eng.block = block
        return eng

    def geom(hidden, rung_atoms, rung_split, block=None):
        return E._geom_for(bare(block), hidden, rung_atoms, rung_split)

    assert geom(7168, 2, 4) == (1, 7168, 2)  # 7168 has {2, 7, 14}
    assert geom(8192, 2, 4) == (2, 8192, 4)
    assert geom(7168, 1, 7, block=128) == (1, 7168, 7)  # block is the slice's
    assert geom(4096, 2, 1) == (2, 4096, 1)
    assert geom(3000, 2, 2)[2] == 1  # padded: unsplit

    key = E._cfg_key(
        bare(),
        7168,
        FusedOneShotRung(min_bytes=0, atoms=1, grid_cap=64, fanout="peer", split=7),
    )
    assert (key[2], key[5]) == (63, 7)  # cap 64 -> 9 groups of 7
    assert (
        E._cfg_key(
            bare(),
            7168,
            FusedOneShotRung(min_bytes=0, atoms=2, grid_cap=64, fanout="peer", split=1),
        )[2]
        == 64
    )

    row = lambda h: h * 2
    assert E._tiles_and_grid({"grid": 63, "split": 7}, 7168, 3 * row(7168)) == (21, 21)
    assert E._tiles_and_grid({"grid": 64, "split": 4}, 8192, 40 * row(8192)) == (
        160,
        64,
    )
    assert E._tiles_and_grid({"grid": 64}, 7168, 3 * row(7168)) == (3, 3)  # unsplit


def test_one_shot_allreduce_rmsnorm_split_exchange_word():
    """The exchange word's arithmetic, mirrored in Python.

    The kernel never resets the word and decodes ``cur - prev`` modulo 2**64,
    so this checks, for writer counts up to the limit and start values that
    wrap: a call is never seen complete before its last writer; a complete
    call decodes to exactly the sum of its contributions; and no call's sum
    can carry into the sign bit, whatever the partials -- the per-writer clamp
    bounds infinities and NaNs too.
    """
    import random

    from aiter.ops.flydsl.kernels.comm.quick_allreduce_fusions import (
        XCHG_COUNT_BITS,
        XCHG_FIX_BITS,
        XCHG_SUM_BITS,
        xchg_clamp_units,
    )

    mask = (1 << 64) - 1
    count_mask = (1 << XCHG_COUNT_BITS) - 1

    def contribution(partial, n):
        # Mirrors xchg_contribution: a select-based clamp (a NaN compares false
        # and clamps), then round to the nearest fixed-point unit.
        clamp = xchg_clamp_units(n) * 2.0**-XCHG_FIX_BITS
        u = min(clamp, partial)
        q = int(u * 2.0**XCHG_FIX_BITS + 0.5)
        return (q << XCHG_COUNT_BITS) + 1

    rng = random.Random(0)
    for n in (1, 2, 7, 14, 16, 64, 255):
        c = xchg_clamp_units(n)
        assert c & (c - 1) == 0 and n * c < 1 << XCHG_SUM_BITS
        for start in (0, (1 << 64) - (3 << 40), rng.getrandbits(64)):
            word = prev = start
            for _call in range(3):
                partials = [
                    rng.choice(
                        (
                            0.0,
                            rng.random() * 10.0 ** rng.randint(-6, 12),
                            1e30,
                            float("inf"),
                            float("nan"),
                        )
                    )
                    for _ in range(n)
                ]
                adds = [contribution(p, n) for p in partials]
                rng.shuffle(adds)  # arrival order must not matter
                for arrived, a in enumerate(adds):
                    assert ((word - prev) & mask) & count_mask == arrived
                    word = (word + a) & mask
                d = (word - prev) & mask
                assert d & count_mask == n and d < 1 << 63
                assert d >> XCHG_COUNT_BITS == sum(a >> XCHG_COUNT_BITS for a in adds)
                prev = word


def _summarize(name: str, rows: list[dict]) -> None:
    if rows:
        df = pd.DataFrame(rows).drop(columns=list(KNOBS), errors="ignore")
        aiter.logger.info(
            "%s summary (markdown):\n%s", name, df.to_markdown(index=False)
        )


def _verify(args, tps, dts, configs, given, link) -> None:
    ladder = configs == [dict.fromkeys(KNOBS)]
    # Register every row before running any, so each (tp, config) is one spawn.
    # Each entry of ``spawns`` is ``(tp, knobs, shapes)``.
    spawns = []
    coverage = []
    for tp, knobs in itertools.product(tps, configs):
        payloads, window = _production_payloads(tp, link)
        shapes = given or [(nbytes // _ROW_BYTES, ROW) for nbytes in payloads]
        if given is None and args.extended:
            shapes += [s for s in SPOT_SHAPES if s not in shapes]
        spawns.append((tp, knobs, shapes))
        if ladder and given is None:
            _WINDOWS[_key(tp, {})] = window
            coverage.append((tp, f"{window[0]}..{window[1]} B"))
    # Nothing pinned on the command line: with --extended, spot checks of the
    # widths the shipped ladder does not use, next to it.
    if ladder and args.extended:
        spot = [(tp, knobs) for tp in tps for _tp, knobs in SPOT_CONFIGS]
        spawns += [(tp, dict(knobs), given or SPOT_SHAPES) for tp, knobs in spot]
    rows = []
    for tp, knobs, shapes in spawns:
        _CASES[_key(tp, _engine_kw(**knobs))] = list(shapes)
        rows += [(tp, knobs, shape) for shape in shapes]

    for dtype in dts:
        _summarize(
            "flydsl one-shot allreduce",
            [
                test_one_shot_allreduce(t, h, dtype, tp, **knobs)
                for tp, knobs, (t, h) in rows
            ],
        )
    _summarize(
        "flydsl one-shot allreduce production rung coverage",
        [test_one_shot_allreduce_coverage(tp, window) for tp, window in coverage],
    )
    _summarize(
        "flydsl one-shot allreduce run-ahead",
        [
            test_one_shot_allreduce_run_ahead(
                RUN_AHEAD_M, HIDDEN, tp, RUN_AHEAD_ITERS, **knobs
            )
            for tp, knobs, _shapes in spawns
        ],
    )
    _summarize(
        "flydsl one-shot allreduce fresh inputs",
        [
            test_one_shot_allreduce_fresh_inputs(tp, FRESH_ITERS, **knobs)
            for tp, knobs, _shapes in spawns
        ],
    )
    if ladder and given is None:
        _summarize(
            "flydsl one-shot allreduce production dispatch",
            [test_one_shot_allreduce_dispatch(tp) for tp in tps],
        )


def _verify_fused(tps) -> None:
    """The fused verify sweep: the shipped engine on ``_FUSED_SHAPE_CASES``,
    and fresh inputs on the shipped ladder and on a split build."""
    for tp in tps:
        fused_pairs = [(m, h) for m, h, _ in _FUSED_SHAPE_CASES]
        fused_ranks = _fused_spawn(
            tp,
            fused_pairs,
            mode="fused",
        )
        fused_fails = [
            f"fused tp={tp} rank {r}: {bad}"
            for r, shape_bads in enumerate(fused_ranks)
            for bad in shape_bads
            if bad
        ]
        for split in FUSED_FRESH_SPLITS:
            fresh_ranks = _fused_spawn(
                tp,
                [(1, FUSED_FRESH_HIDDEN)],
                atoms=None,
                grid_cap=None,
                fanout=None,
                split=split,
                mode="fused_fresh",
            )
            fused_fails += [
                f"fused fresh tp={tp} split={split} rank {r}: {bad}"
                for r, bad in enumerate(fresh_ranks)
                if bad
            ]
        _FAILURES.extend(fused_fails)
        aiter.logger.info("fused tp=%s: %d failures", tp, len(fused_fails))
    _summarize(
        "fused allreduce + rmsnorm production dispatch",
        [
            test_fused_allreduce_rmsnorm_dispatch(tp, regime)
            for tp in tps
            for regime in ("NONE", "INT4")
        ],
    )


def _candidates(args, operation: str) -> tuple:
    """``(name, knobs)`` of every ``--candidates`` row for the one-shot."""
    if args.candidates is None:
        return ()
    return tuple(
        (c.name, tuple(sorted(c.knobs.items())))
        for c in fab.load_candidates(
            args.candidates,
            operation=operation,
            algorithms=("oneshot",),
            candidate_set=args.candidate_set,
        )
    )


def _bench(args, tps, configs, given, link) -> None:
    if args.operation == "ar_rmsnorm":
        _bench_fused(args, tps, given, link)
        return
    extra = _candidates(args, "ar")
    rows = []
    for tp, knobs in itertools.product(tps, configs):
        if given is None:
            payloads, _window = _production_payloads(tp, link)
            shapes = [(nbytes // _ROW_BYTES, ROW) for nbytes in payloads]
            shapes = sorted(
                dict.fromkeys(shapes + BENCH_SHAPES), key=lambda s: s[0] * s[1]
            )
        else:
            shapes = given
        _CASES[_bench_key(tp, _engine_kw(**knobs), extra)] = list(shapes)
        rows += [
            bench_one_shot_allreduce(t, h, tp, **knobs, extra=extra) for t, h in shapes
        ]
    rows = [{k: v for k, v in r.items() if k != "extra"} for r in rows]

    tables = fab.bench_tables(
        rows,
        candidates=("flydsl", *(name for name, _ in extra), *STOCK),
        drop=KNOBS,
        title="flydsl one-shot allreduce vs stock",
    )
    for title, df in tables:
        aiter.logger.info("%s (markdown):\n%s", title, fab.to_markdown(df))
    header = fab.provenance(
        args,
        link,
        [
            (
                "- prod: CudaCommunicator.all_reduce with AITER_FLY_AR=0 and "
                "AITER_QUICK_REDUCE_QUANTIZATION=NONE"
            ),
            "- `flydsl vs X` is `X us / flydsl us`: above 1.0 the one-shot is faster",
        ],
    )
    fab.write_outputs(args, "FlyDSL one-shot all-reduce bench", header, tables, rows)


def _fused_prod(ca, x, use_1stage: bool) -> str:
    """The stock candidate ``CudaCommunicator.fused_allreduce_rmsnorm`` runs on
    *x*, with the FlyDSL path off and quick reduce at its default (``NONE``).

    Mirrors its gates: ``CustomAllreduce``'s fused kernel where it accepts the
    payload, and otherwise the all-reduce production would run, then the norm.
    """
    nbytes = x.numel() * x.element_size()
    hidden = x.shape[-1]
    custom = ca is not None and ca.should_custom_ar(x)
    if custom and hidden <= 16384 and nbytes < 8 * 1024 * 8192:
        return "cdr_fused_1stage" if use_1stage else "cdr_fused_2stage"
    return "cdr+norm" if custom else "rccl+norm"


def _use_1stage(tp: int, nbytes: int) -> bool:
    """``use_1stage`` of ``CudaCommunicator.fused_allreduce_rmsnorm``."""
    from aiter.dist.device_communicators.communicator_cuda import CudaCommunicator

    override = CudaCommunicator._ar_1stage_override
    if override is not None:
        return override
    max_kb = CudaCommunicator._ar_1stage_max_kb
    limit = max_kb if max_kb >= 0 else 128 * 7168 * 2 // tp
    return nbytes <= limit


def _bench_fused_rank(
    rank: int,
    tp: int,
    init_method: str,
    cases: list[tuple],
    timing: tuple[int, int],
    extra: tuple = (),
) -> list[dict]:
    """One rank of one ``ar_rmsnorm`` bench spawn: every candidate on every case.

    ``flydsl`` is ``OneShotAllReduceRMSNorm`` on the shipped ladder, and
    *extra* the ``--candidates`` engines; the stock candidates are
    ``FUSED_STOCK``. A case is ``(tokens, hidden)``. Returns, per case, the
    production candidate's name and each candidate's per-rank result.
    """
    from aiter.ops.flydsl.one_shot_allreduce import OneShotAllReduceRMSNorm

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    tp_group = _init_dist(rank, tp, init_method, stock=True)
    cpu_group = tp_group.cpu_group
    comm = tp_group.device_communicator
    ca = comm.ca_comm
    if ca is not None and ca.disabled:
        ca = None
    nccl = comm.pynccl_comm
    if nccl is not None and nccl.disabled:
        nccl = None
    # Built in a fixed order: each engine's IPC exchange is a collective, and so
    # is each width's build (``preload`` below). The payload ceiling is a speed
    # policy, so it is lifted, as in verify.
    knobs_of = {"flydsl": {}, **{name: dict(knobs) for name, knobs in extra}}
    fly = {
        name: OneShotAllReduceRMSNorm(
            group=cpu_group,
            device=device,
            rank=rank,
            world_size=tp,
            max_bytes=1 << 30,
            **knobs,
        )
        for name, knobs in knobs_of.items()
    }
    reps, rounds = timing
    eps = fab.RMS_EPS

    # The inputs and every candidate's outputs, for the whole spawn. Refer to
    # flydsl_allreduce_bench for why the addresses must not change.
    names = (*fly, *FUSED_STOCK)
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
            parts, residual_src, weight = _fused_inputs(m, hidden, tp, device)
            x = _view("inp", m, hidden)
            x.copy_(parts[rank])
            residual = _view("residual", m, hidden)
            residual.copy_(residual_src)
            ar_buf = _view("ar", m, hidden)
            ref_out, ref_res = fab.fused_reference(
                torch.stack(parts).float().sum(0), residual, weight, eps
            )
            nbytes = m * hidden * 2
            use_1stage = _use_1stage(tp, nbytes)
            outs = {
                n: (_view(f"{n}:out", m, hidden), _view(f"{n}:res", m, hidden))
                for n in names
            }

            # A --candidates engine whose pinned geometry this width does not
            # have, or past its size cap, is left out: n/a.
            live = {
                name: e
                for name, e in fly.items()
                if name == "flydsl"
                or (
                    _fused_applies(knobs_of[name], hidden)
                    and nbytes <= _candidate_max_bytes()
                )
            }
            for e in live.values():
                e.preload(hidden)
            thunks = _fused_thunks(
                live, ca, nccl, outs, x, residual, weight, ar_buf, eps
            )
            graphs, res = {}, {}
            for name, thunk in thunks.items():
                graphs[name] = fab.capture(thunk)
                out, res_out = fab.graded_replay(graphs[name], outs[name], cpu_group)
                res[name] = {
                    "sqnr_db": min(
                        _sqnr_db(ref_out.float(), out.float()),
                        _sqnr_db(ref_res.float(), res_out.float()),
                    ),
                }
            times = fab.time_graphs(graphs, cpu_group, reps=reps, rounds=rounds)
            for name, us in times.items():
                res[name]["us"] = us
            for name, e in live.items():
                res[name]["variant"] = e.variant(hidden, nbytes)
            for name in ("cdr_fused_1stage", "cdr_fused_2stage", "cdr+norm"):
                if name in res and not ca.enable_register_for_capturing:
                    res[name]["variant"] = "staged"
            rows.append({"prod": _fused_prod(ca, x, use_1stage), "candidates": res})
            del graphs
    finally:
        for e in fly.values():
            e.close()
        _teardown_group(cpu_group)
    return rows


def _fused_thunks(fly, ca, nccl, outs, x, residual, weight, ar_buf, eps) -> dict:
    """The ``ar_rmsnorm`` bench thunks, by candidate name, for one case.

    Every thunk writes into its preallocated ``outs[name]``: ``fused_ar_rms``
    allocates whatever it is not given, and an allocation inside a capture is
    at best a surprise.
    """
    from aiter import rmsnorm2d_fwd_with_add

    thunks = {}
    for name, e in fly.items():

        def _fly(e=e, o=outs[name]):
            e.allreduce_rmsnorm(x, residual, weight, eps, out=o[0], residual_out=o[1])

        thunks[name] = _fly

    reg = ca is not None and ca.enable_register_for_capturing

    def _registered():
        # As CustomAllreduce: in place under capture, through its pool otherwise.
        return reg and torch.cuda.is_current_stream_capturing()

    custom = ca is not None and ca.should_custom_ar(x)
    if custom and x.shape[-1] <= 16384:
        for name, one in (("cdr_fused_1stage", True), ("cdr_fused_2stage", False)):

            def _cdr_fused(o=outs[name], one=one):
                ca.fused_ar_rms(
                    x,
                    residual,
                    res_out=o[1],
                    out=o[0],
                    w=weight,
                    eps=eps,
                    registered=_registered(),
                    use_1stage=one,
                )

            thunks[name] = _cdr_fused
    if custom:

        def _cdr_norm(o=outs["cdr+norm"]):
            ca.all_reduce(x, out=ar_buf, registered_input=_registered())
            rmsnorm2d_fwd_with_add(o[0], ar_buf, residual, o[1], weight, eps)

        thunks["cdr+norm"] = _cdr_norm
    if nccl is not None:

        def _rccl_norm(o=outs["rccl+norm"]):
            # What CudaCommunicator.all_reduce falls back to, then the norm.
            nccl.all_reduce(x, ar_buf)
            rmsnorm2d_fwd_with_add(o[0], ar_buf, residual, o[1], weight, eps)

        thunks["rccl+norm"] = _rccl_norm
    return thunks


def _teardown_group(group) -> None:
    import torch.distributed as dist

    torch.cuda.synchronize()
    dist.barrier(group=group)
    destroy_model_parallel()
    destroy_distributed_environment()


def _fused_route(tp: int, link: str, nbytes: int) -> tuple[str, str]:
    """The fused families the policy routes *nbytes* to, quantized and exact."""
    policy = fly_policy.resolve_fused_policy(world_size=tp, link=link)
    got = []
    for quantized in (True, False):
        alg = policy.select(nbytes, quantized=quantized)
        got.append("fallback" if alg is None else alg.value)
    return tuple(got)


@benchmark()
def bench_one_shot_allreduce_rmsnorm(tokens, hidden, tp, link, extra=()):
    """The fused one-shot against production's fused path and the best stock
    candidate. *extra* are the ``--candidates`` engines, timed in the same
    spawn."""
    key = _bench_key(tp, {}, extra, "ar_rmsnorm")
    i = _CASES[key].index((tokens, hidden))
    per_rank = [r[i] for r in _bench_ranks(key)]
    pinned = tuple(name for name, _knobs in extra)
    names = [
        n for n in ("flydsl", *pinned, *FUSED_STOCK) if n in per_rank[0]["candidates"]
    ]
    got = {n: fab.collapse([r["candidates"][n] for r in per_rank]) for n in names}

    label = f"ar_rmsnorm tp={tp} {tokens}x{hidden}"
    fails = []
    for name, c in got.items():
        if c["SQNR dB"] < FUSED_BENCH_SQNR_FLOOR_DB:
            fails.append(
                f"{name}: SQNR {c['SQNR dB']:.2f} dB below "
                f"{FUSED_BENCH_SQNR_FLOOR_DB} dB"
            )
        if c["variant"] is not None and c["variant"].startswith("MIXED"):
            fails.append(f"{name}: ranks ran different kernels: {c['variant']}")
    _check(label, fails)

    fly_us = got["flydsl"]["us"]
    prod = per_rank[0]["prod"]
    prod_us = got[prod]["us"] if prod in got else None
    passing = [
        n
        for n in FUSED_STOCK
        if n in got and got[n]["SQNR dB"] >= FUSED_BENCH_SQNR_FLOOR_DB
    ]
    best = min(passing, key=lambda n: got[n]["us"], default=None)
    best_us = got[best]["us"] if best else None
    route, route_exact = _fused_route(tp, link, tokens * hidden * 2)
    ret = {
        "gfx": ARCH,
        "operation": "ar_rmsnorm",
        "KiB": tokens * hidden * 2 / 1024,
        "route": route,
        "route exact": route_exact,
        "prod": prod,
        "prod us": prod_us,
        "flydsl vs prod": fab.speedup(prod_us, fly_us),
        "best stock": best,
        "best stock us": best_us,
        "flydsl vs best": fab.speedup(best_us, fly_us),
    }
    for name, c in got.items():
        ret.update({f"{name} {field}": v for field, v in c.items()})
    return ret


def _bench_fused(args, tps, given, link) -> None:
    extra = _candidates(args, "ar_rmsnorm")
    shapes = given or FUSED_BENCH_SHAPES
    rows = []
    for tp in tps:
        _CASES[_bench_key(tp, {}, extra, "ar_rmsnorm")] = list(shapes)
        rows += [
            bench_one_shot_allreduce_rmsnorm(t, h, tp, link, extra=extra)
            for t, h in shapes
        ]
    rows = [{k: v for k, v in r.items() if k not in ("extra", "link")} for r in rows]

    tables = fab.bench_tables(
        rows,
        candidates=("flydsl", *(name for name, _ in extra), *FUSED_STOCK),
        title="flydsl one-shot allreduce + rmsnorm vs stock",
    )
    for title, df in tables:
        aiter.logger.info("%s (markdown):\n%s", title, fab.to_markdown(df))
    header = fab.provenance(
        args,
        link,
        [
            (
                "- prod: CudaCommunicator.fused_allreduce_rmsnorm with "
                "AITER_FLY_AR=0 and AITER_QUICK_REDUCE_QUANTIZATION=NONE"
            ),
            (
                "- `route`: the fused policy's family with "
                "AITER_QUICK_REDUCE_QUANTIZATION=INT4, `route exact` without"
            ),
            "- `flydsl vs X` is `X us / flydsl us`: above 1.0 the one-shot is faster",
        ],
    )
    fab.write_outputs(
        args, "FlyDSL one-shot all-reduce + RMSNorm bench", header, tables, rows
    )


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
        "    e.g.: --tp 2",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=None,
        help="(tokens, hidden) pairs, on every engine. Default: derived from\n"
        "the dispatch policy for the shipped ladder, SPOT_SHAPES for the spot\n"
        "checks, and also BENCH_SHAPES in the bench scenario.\n"
        "    e.g.: -s 1,7168 8,7168",
    )
    # Unset by default, so the engine walks the shipped ladder, which is what
    # production runs. Any value pins every rung to it.
    parser.add_argument("--atoms", type=int, nargs="*", default=[None])
    parser.add_argument("--grid-cap", type=int, nargs="*", default=[None])
    parser.add_argument(
        "--fanout", nargs="*", default=[None], choices=("peer", "atom", None)
    )
    parser.add_argument(
        "--block",
        type=int,
        nargs="*",
        default=[None],
        choices=(*SUPPORTED_BLOCKS, None),
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="verify: also run what production does not: the spot-check\n"
        "configurations at every world size, and the spot-check shapes on the\n"
        "shipped ladder.",
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
        raise SystemExit(f"OneShotAllReduce is unsupported on {ARCH}")
    global _TIMING
    _TIMING = (args.reps, args.rounds)

    tps = fab.select_world_sizes(
        args.tp or list(SUPPORTED_WORLDS),
        explicit=bool(args.tp),
        n_gpu=n_gpu,
        supported=SUPPORTED_WORLDS,
    )
    dts = [d for d in args.dtype if d == dtypes.bf16]
    if len(dts) != len(args.dtype):
        aiter.logger.warning("OneShotAllReduce payload is bf16; skipping others")

    # Pinned knobs per configuration; unset (None) knobs are left to the engine,
    # which walks ONESHOT_LADDER when none of atoms/grid_cap/fanout/block is set.
    configs = [
        {
            "atoms": atoms,
            "grid_cap": grid_cap,
            "fanout": fanout,
            "block": block,
        }
        for atoms, grid_cap, fanout, block in itertools.product(
            args.atoms, args.grid_cap, args.fanout, args.block
        )
    ]
    given = None if args.mnk is None else [(int(t), int(h)) for t, h in args.mnk]
    link = fly_policy.detect_link()

    if args.scenario == "bench":
        if dts:
            _bench(args, tps, configs, given, link)
    else:
        if args.operation in (None, "ar"):
            _verify(args, tps, dts, configs, given, link)
        if args.operation in (None, "ar_rmsnorm"):
            _verify_fused(tps)

    if _FAILURES:
        raise SystemExit(
            f"{len(_FAILURES)} OneShotAllReduce check(s) failed:\n  "
            + "\n  ".join(_FAILURES)
        )


if __name__ == "__main__":
    freeze_support()
    # A subprocess rank worker of the fused tests (``_fused_spawn``) carries
    # ``--rank``; anything else is a user run, whose arguments main() parses.
    _rank_parser = argparse.ArgumentParser(add_help=False)
    _rank_parser.add_argument("--rank", type=int, default=None)
    _rank_parser.add_argument("--init-method", default=None)
    _rank_parser.add_argument("--tp", type=int, default=2)
    _rank_parser.add_argument("--atoms", type=int, default=None)
    _rank_parser.add_argument("--grid-cap", type=int, default=None)
    _rank_parser.add_argument("--fanout", default=None, choices=("peer", "atom"))
    _rank_parser.add_argument("--block", type=int, default=None)
    _rank_parser.add_argument("--split", type=int, default=None)
    _rank_parser.add_argument("--input-scale", type=float, default=1.0)
    _rank_parser.add_argument("--repeat", type=int, default=1)
    _rank_parser.add_argument(
        "--mode",
        default="fused",
        choices=("fused", "fused_run_ahead", "fused_slice_lag", "fused_fresh"),
    )
    _rank_parser.add_argument("--tokens", default="")
    _rank_parser.add_argument("--hiddens", default="")
    _rank_parser.add_argument("--iters", type=int, default=RUN_AHEAD_ITERS)
    _rank_parser.add_argument("--out", default=None)
    if "--rank" in sys.argv[1:]:
        known = _rank_parser.parse_args()
        import torch.distributed as dist

        known.tokens = [int(t) for t in known.tokens.split(",") if t]
        known.hiddens = [int(h) for h in known.hiddens.split(",") if h]
        if len(known.tokens) != len(known.hiddens):
            raise SystemExit("tokens and hiddens lists must match")
        rank = known.rank
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend="gloo",
            init_method=known.init_method,
            world_size=known.tp,
            rank=rank,
        )
        _run_rank_fused(known, rank, device, dist)
    else:
        from time import perf_counter

        start = perf_counter()
        main()
        end = perf_counter()
        aiter.logger.info(f"Test execution took {end - start:.2f}s")
