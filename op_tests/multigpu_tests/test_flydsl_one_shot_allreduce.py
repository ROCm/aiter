# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and timing for the exact one-shot (1-stage) all-reduce (``OneShotAllReduce``).

``--scenario`` selects what a run does: ``verify`` (the default) checks the
kernel, and ``bench`` compares it with the stock kernels production runs today.

**verify** runs four sweeps, each ending in a markdown table. A default run
drives the engine as production does -- no tuning knob pinned, so it walks
``ONESHOT_LADDER`` and picks a rung by payload size -- on payloads derived from
``allreduce_policy``: at every world size, both ends of each rung's slice of
the window the policy routes to the one-shot. Next to it, a few spot checks
run pinned configurations no shipped rung uses, one world size each.
``--atoms``, ``--grid-cap``, ``--fanout`` or ``--block`` pin a configuration
instead.

``test_one_shot_allreduce`` checks three things per shape, and the second
matters more than the first:

1. The sum is right, against an fp32 reference, at the bf16 rounding floor.
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

``--extended`` runs the spot-check configurations at every world size, and the
spot-check shapes on the production engine too.

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

``flydsl_allreduce_bench.py`` describes how the comparison is kept unbiased.
The default payloads are the verify sweep's plus the DeepSeek-V4 decode
shapes; ``-s`` or ``--shape-csv`` replace them, and the knob options pin the
one-shot's configuration. ``--report`` and ``--csv`` save the tables.

Every rank is a ``multiprocessing`` spawn worker; one spawn per engine
configuration and world size runs every sweep. OneShotAllReduce runs on
gfx942/gfx950 at TP in {2, 4, 8}; other archs skip, and ``main()`` skips a
world size when fewer GPUs are visible than TP.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
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
from aiter.ops.flydsl.allreduce_policy import oneshot_ladder
from aiter.ops.flydsl.kernels.comm.allreduce_shared import (
    SUPPORTED_ARCHS,
    SUPPORTED_BLOCKS,
    SUPPORTED_WORLDS,
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

# (tp, knobs): pinned configurations a default run spot-checks next to the
# shipped ladder, one world size each, for widths no shipped rung uses yet.
# Block 512 is the widest workgroup; at atoms=4 it is a 32 KiB tile, so every
# spot shape is a single partial tile. ``--extended`` runs each at every world
# size.
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


def _production_payloads(tp: int, link: str) -> tuple[list[int], tuple[int, int]]:
    """Whole-row payloads covering the window the policy routes to the
    one-shot, and that window, in bytes (inclusive).

    The ladder picks a rung by floor alone, so each rung serves one contiguous
    slice of the window; both ends of every slice are taken, which reaches
    every rung and the largest payload each is asked to move.
    """
    policy = fly_policy.resolve_oneshot_policy(link, tp)
    lo, hi = policy.min_bytes, policy.max_bytes
    ladder = oneshot_ladder(tp, link)
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


def _metrics(out, ref, tp: int, group) -> dict:
    """Accuracy against fp32, then bit-identity across ranks.

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


def _engine(rank: int, tp: int, group, device, engine_kw: dict):
    """The one-shot engine, every rung compiled."""
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
        **engine_kw,
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
            out = torch.zeros_like(inp)
            eng.allreduce(inp, out)
            torch.cuda.synchronize()
            res = _metrics(out, ref, tp, cpu_group)

            g = fab.capture(lambda e=eng, src=inp, dst=out: e.allreduce(src, dst))
            for _ in range(GRAPH_REPLAYS):
                out.zero_()
                g.replay()
                torch.cuda.synchronize()
                res = _worst(res, _metrics(out, ref, tp, cpu_group))
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
) -> list[dict]:
    """One rank of one bench spawn: every candidate on every case.

    A case is ``(tokens, hidden)``. Returns, per case, the production
    candidate's name and each candidate's per-rank result.
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
    reps, rounds = timing

    # The input and every candidate's output, for the whole spawn. Refer to
    # flydsl_allreduce_bench for why the addresses must not change.
    max_bytes = max(m * hidden * 2 for m, hidden in cases)
    bufs = {
        name: torch.empty(max_bytes, dtype=torch.uint8, device=device)
        for name in ("inp", "flydsl", *STOCK)
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
            outs = {name: _view(name, m, hidden) for name in ("flydsl", *STOCK)}

            thunks = {"flydsl": lambda x=x, o=outs["flydsl"]: eng.allreduce(x, o)}
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
            res["flydsl"]["variant"] = eng.variant(nbytes)
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
        _teardown(eng, cpu_group)
    return rows


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


def _bench_ranks(key: tuple) -> list[list[dict]]:
    if key not in _RESULTS:
        _RESULTS[key] = _spawn(
            _bench_rank,
            key[0],
            engine_kw=dict(key[1]),
            cases=_CASES[key],
            timing=_TIMING,
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
):
    """The one-shot against production's kernel and the best stock kernel."""
    engine_kw = _engine_kw(atoms, grid_cap, fanout, block)
    key = _key(tp, engine_kw)
    i = _CASES[key].index((tokens, hidden))
    per_rank = [r[i] for r in _bench_ranks(key)]
    names = [n for n in ("flydsl", *STOCK) if n in per_rank[0]["candidates"]]
    got = {n: fab.collapse([r["candidates"][n] for r in per_rank]) for n in names}

    label = f"tp={tp} {tokens}x{hidden} {engine_kw}"
    fails = []
    for name, c in got.items():
        floor = SQNR_FLOOR_DB if name == "flydsl" else STOCK_SQNR_FLOOR_DB
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
    # Nothing pinned on the command line: the shipped ladder, plus spot checks
    # of the widths it does not use yet.
    if ladder:
        spot = [(tp, knobs) for tp, knobs in SPOT_CONFIGS if tp in tps]
        if args.extended:
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


def _bench(args, tps, configs, given, link) -> None:
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
        _CASES[_key(tp, _engine_kw(**knobs))] = list(shapes)
        rows += [bench_one_shot_allreduce(t, h, tp, **knobs) for t, h in shapes]

    tables = fab.bench_tables(
        rows,
        candidates=("flydsl", *STOCK),
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


def main():
    if ARCH not in SUPPORTED_ARCHS:
        aiter.logger.warning("OneShotAllReduce unsupported on %s; skipping", ARCH)
        return
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
        default=list(SUPPORTED_WORLDS),
        help="World sizes to sweep (2, 4, 8). Default all; sizes with fewer\n"
        "visible GPUs are skipped.\n    e.g.: --tp 2",
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
        help="verify: run the spot-check configurations at every world size,\n"
        "and the spot-check shapes on the shipped ladder too.",
    )
    fab.add_bench_args(parser)
    args = parser.parse_args()
    fab.check_bench_args(parser, args)
    global _TIMING
    _TIMING = (args.reps, args.rounds)

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
        _verify(args, tps, dts, configs, given, link)

    if _FAILURES:
        raise SystemExit(
            f"{len(_FAILURES)} OneShotAllReduce check(s) failed:\n  "
            + "\n  ".join(_FAILURES)
        )


if __name__ == "__main__":
    freeze_support()
    from time import perf_counter

    start = perf_counter()
    main()
    end = perf_counter()
    aiter.logger.info(f"Test execution took {end - start:.2f}s")
