# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and timing for the exact one-shot (1-stage) all-reduce (``OneShotAllReduce``).

``python3`` this file runs three sweeps, each ending in a markdown table. A
default run drives the engine as production does -- no tuning knob pinned, so
it walks ``ONESHOT_LADDER`` and picks a rung by payload size -- on payloads
derived from ``allreduce_policy``: at every world size, both ends of each
rung's slice of the window the policy routes to the one-shot. Next to it, a
few spot checks run pinned configurations no shipped rung uses, one world size
each. ``--atoms``, ``--grid-cap``, ``--fanout``, ``--block`` or ``--skip-self``
pin a configuration instead.

``test_one_shot_allreduce`` checks three things per shape, and the second
matters more than the first:

1. The sum is right, against an fp32 reference, at the bf16 rounding floor.
2. The result is **bit-identical on every rank**. The kernel accumulates in a
   fixed rank order for exactly this reason, and an SQNR check cannot see an
   ordering bug -- both answers would be equally "accurate".
3. It is timed with ``run_perftest``. One row per world size captures the
   all-reduce into a CUDA graph and replays it, checking 1 and 2 after every
   replay.

``test_one_shot_allreduce_coverage`` checks that those rows ran every rung the
engine's own ``cfgs_for`` says the production window selects, so a retuned
ladder or policy the payload derivation does not follow fails rather than
silently leaving a rung untested.

``test_one_shot_allreduce_run_ahead`` makes repeated back-to-back calls under
deliberate rank skew. The inbox is double-buffered by ``colour & 1`` and the
safety argument depends on a straggler's read of call k finishing before
anyone's push for call k+2; a quiescent test never exercises that.

``--extended`` runs the spot-check configurations at every world size, and the
spot-check shapes on the production engine too.

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

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest

set_start_method("spawn", force=True)

from aiter.ops.flydsl import allreduce_policy as fly_policy
from aiter.ops.flydsl.kernels.one_shot_allreduce import (
    SUPPORTED_BLOCKS,
    oneshot_ladder,
)
from aiter.ops.flydsl.kernels.quick_allreduce_shared import SUPPORTED_WORLDS

try:
    ARCH = get_gfx_runtime()
except (KeyError, RuntimeError):
    ARCH = None
SUPPORTED_ARCHS = ("gfx942", "gfx950")

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

# Each replay advances the device-side colour and alternates the inbox parity
# slot, so only repeated replays show the captured launch advancing that state
# rather than freezing it.
GRAPH_REPLAYS = 4

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
            "skip_self": False,
        },
    ),
    (
        2,
        {
            "atoms": 4,
            "grid_cap": 64,
            "fanout": "peer",
            "block": 512,
            "skip_self": True,
        },
    ),
)

RUN_AHEAD_M = 5
RUN_AHEAD_ITERS = 200
SQNR_FLOOR_DB = 45.0

# Seconds to wait for each rank of a spawn. The kernels spin on flags written
# by peers, so a protocol bug or a dead rank hangs the rest; this fails the
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
    policy = fly_policy.resolve_oneshot(link, tp)
    lo, hi = policy.min_bytes, policy.max_bytes
    ladder = oneshot_ladder(tp, link)
    ends = [floor - 1 for floor, *_ in ladder[1:]] + [hi]
    payloads = set()
    for (floor, *_rung), end in zip(ladder, ends):
        first = -(-max(lo, floor, 1) // _ROW_BYTES) * _ROW_BYTES
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


def _metrics(out, ref, tp: int) -> dict:
    """Accuracy against fp32, then bit-identity across ranks."""
    import torch.distributed as dist

    # Widened to int32 because gloo rejects int16 ("Invalid scalar type"); the
    # widening is exact, so the comparison is still on bits.
    bits = out.view(torch.int16).to(torch.int32).cpu()
    gathered = [torch.empty_like(bits) for _ in range(tp)]
    dist.all_gather(gathered, bits)
    lanes_differing = max(int((g != gathered[0]).sum()) for g in gathered)
    err = checkAllclose(
        ref, out.float(), printLog=False, msg="one_shot_allreduce vs fp32"
    )
    return {
        "sqnr_db": _sqnr_db(ref, out),
        "lanes_differing": lanes_differing,
        "err": float(err),
    }


def _worst(a: dict, b: dict) -> dict:
    return {
        "sqnr_db": min(a["sqnr_db"], b["sqnr_db"]),
        "lanes_differing": max(a["lanes_differing"], b["lanes_differing"]),
        "err": max(a["err"], b["err"]),
    }


def _run_rank(
    rank: int,
    tp: int,
    init_method: str,
    engine_kw: dict,
    cases: list[tuple],
    window: tuple[int, int] | None = None,
) -> dict:
    """One rank of one spawn: every case, then the run-ahead loop.

    A case is ``(tokens, hidden, graph)``. The engine is built once, so every
    case shares its compiled rungs and IPC inboxes. *window* is the payload
    range production dispatch routes to this engine, if any; the result then
    carries the rungs the engine itself says that range selects.
    """
    import torch.distributed as dist

    from aiter.ops.flydsl.one_shot_allreduce import OneShotAllReduce

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="gloo", init_method=init_method, world_size=tp, rank=rank
    )

    eng = OneShotAllReduce(
        group=dist.group.WORLD,
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
    warm = torch.zeros(1, HIDDEN, dtype=torch.bfloat16, device=device)
    eng.compile_and_launch(warm, torch.empty_like(warm))
    production_cfgs = None if window is None else eng.cfgs_for(*window)

    rows = []
    try:
        for m, hidden, graph in cases:
            parts = _parts(m, hidden, tp, 1234 + m * 8191 + hidden, device)
            inp = parts[rank].contiguous()
            ref = torch.stack(parts).float().sum(0)
            out = torch.zeros_like(inp)
            eng.allreduce(inp, out)
            torch.cuda.synchronize()
            res = _metrics(out, ref, tp)

            if graph:
                # Captured on the current stream, which allreduce() launches on
                # when it is given none.
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    eng.allreduce(inp, out)
                for _ in range(GRAPH_REPLAYS):
                    out.zero_()
                    g.replay()
                    torch.cuda.synchronize()
                    res = _worst(res, _metrics(out, ref, tp))
                fn = g.replay
            else:

                def fn(e=eng, src=inp, dst=out):
                    e.allreduce(src, dst)
                    return dst

            dist.barrier()
            torch.cuda.synchronize()
            # cuda.Event timing, not run_perftest's default profiler timer:
            # `import aiter` creates a GPU context in the parent, and on some
            # ROCm/torch builds a child spawned after that records no GPU
            # events in torch.profiler, so the default timer fails reducing an
            # empty trace.
            _, us = run_perftest(fn, use_cuda_event=True)
            nbytes = inp.numel() * inp.element_size()
            res["us"] = float(us)
            res["variant"] = eng.variant(nbytes)
            res["cfg"] = eng._pick_cfg(nbytes)
            rows.append(res)

        # Run-ahead: many back-to-back calls with rank 0 deliberately late, so
        # the others get a chance to run ahead into the other parity slot.
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
    finally:
        dist.barrier()
        eng.close()
        dist.destroy_process_group()
    return {"rows": rows, "run_ahead": run_ahead, "production_cfgs": production_cfgs}


def _spawn(
    world_size: int,
    engine_kw: dict,
    cases: list[tuple],
    window: tuple[int, int] | None = None,
) -> list[dict]:
    if world_size not in SUPPORTED_WORLDS:
        raise ValueError(f"unsupported world_size={world_size}")
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    pool = Pool(processes=world_size)
    try:
        results = [
            pool.apply_async(
                _run_rank,
                kwds={
                    "rank": rank,
                    "tp": world_size,
                    "init_method": init_method,
                    "engine_kw": engine_kw,
                    "cases": cases,
                    "window": window,
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
# ``(tokens, hidden, graph)``.
_CASES: dict[tuple, list[tuple]] = {}
# Production dispatch window of an unpinned engine whose rung coverage is
# checked, per spawn key.
_WINDOWS: dict[tuple, tuple[int, int]] = {}
_RESULTS: dict[tuple, list[dict]] = {}
_FAILURES: list[str] = []

# Tuning knobs the command line can pin. They are test-function arguments, so
# they select the engine, but not table columns: the ``variant`` column names
# the binary that actually ran, which is what a pinned knob changes.
KNOBS = ("atoms", "grid_cap", "fanout", "block", "skip_self")


def _engine_kw(atoms, grid_cap, fanout, block, skip_self) -> dict:
    """OneShotAllReduce kwargs for the knobs that are pinned (not None)."""
    kw = {
        "atoms": atoms,
        "grid_cap": grid_cap,
        "fanout": fanout,
        "block": block,
        "skip_self": skip_self,
    }
    return {k: v for k, v in kw.items() if v is not None}


def _key(tp: int, engine_kw: dict) -> tuple:
    return (tp, tuple(sorted(engine_kw.items())))


def _ranks(key: tuple) -> list[dict]:
    if key not in _RESULTS:
        _RESULTS[key] = _spawn(
            key[0], dict(key[1]), _CASES[key], _WINDOWS.get(key)
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
    graph=False,
    atoms=None,
    grid_cap=None,
    fanout=None,
    block=None,
    skip_self=None,
):
    engine_kw = _engine_kw(atoms, grid_cap, fanout, block, skip_self)
    key = _key(tp, engine_kw)
    i = _CASES[key].index((tokens, hidden, graph))
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
    _check(f"tp={tp} {tokens}x{hidden} graph={graph} {engine_kw}", fails)
    nbytes = tokens * hidden * 2
    # (tp - 1) adds per element.
    flops = tokens * hidden * (tp - 1)
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
def test_one_shot_allreduce_run_ahead(
    tokens,
    hidden,
    tp,
    iters,
    atoms=None,
    grid_cap=None,
    fanout=None,
    block=None,
    skip_self=None,
):
    engine_kw = _engine_kw(atoms, grid_cap, fanout, block, skip_self)
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


def main():
    if ARCH not in SUPPORTED_ARCHS:
        aiter.logger.warning("OneShotAllReduce unsupported on %s; skipping", ARCH)
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
        "checks.\n    e.g.: -s 1,7168 8,7168",
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
        "--skip-self",
        type=int,
        nargs="*",
        default=[None],
        choices=(0, 1, None),
        help="Pin skip_self off (0) or on (1).",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Run the spot-check configurations at every world size, and the\n"
        "spot-check shapes on the shipped ladder too.",
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
            "skip_self": None if skip_self is None else bool(skip_self),
        }
        for atoms, grid_cap, fanout, block, skip_self in itertools.product(
            args.atoms, args.grid_cap, args.fanout, args.block, args.skip_self
        )
    ]
    ladder = configs == [dict.fromkeys(KNOBS)]
    given = None if args.mnk is None else [(int(t), int(h)) for t, h in args.mnk]
    link = fly_policy.detect_link()

    # Register every row before running any, so each (tp, config) is one spawn.
    # Each entry of ``spawns`` is ``(tp, knobs, shapes, graph)``.
    spawns = []
    coverage = []
    for tp, knobs in itertools.product(tps, configs):
        payloads, window = _production_payloads(tp, link)
        shapes = given or [(nbytes // _ROW_BYTES, ROW) for nbytes in payloads]
        if given is None and args.extended:
            shapes += [s for s in SPOT_SHAPES if s not in shapes]
        spawns.append((tp, knobs, shapes, True))
        if ladder and given is None:
            _WINDOWS[_key(tp, {})] = window
            coverage.append((tp, f"{window[0]}..{window[1]} B"))
    # Nothing pinned on the command line: the shipped ladder, plus spot checks
    # of the widths it does not use yet.
    if ladder:
        spot = [(tp, knobs) for tp, knobs in SPOT_CONFIGS if tp in tps]
        if args.extended:
            spot = [(tp, knobs) for tp in tps for _tp, knobs in SPOT_CONFIGS]
        spawns += [(tp, dict(knobs), given or SPOT_SHAPES, False) for tp, knobs in spot]
    rows = []
    for tp, knobs, shapes, graph in spawns:
        cases = [(t, h, False) for t, h in shapes]
        if graph:
            # Captured at every world size, on the smallest shape.
            cases.append((*shapes[0], True))
        _CASES[_key(tp, _engine_kw(**knobs))] = cases
        rows += [(tp, knobs, case) for case in cases]

    for dtype in dts:
        _summarize(
            "flydsl one-shot allreduce",
            [
                test_one_shot_allreduce(t, h, dtype, tp, graph=graph, **knobs)
                for tp, knobs, (t, h, graph) in rows
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
            for tp, knobs, _shapes, _graph in spawns
        ],
    )

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
