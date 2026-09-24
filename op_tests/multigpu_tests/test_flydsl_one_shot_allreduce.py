# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and timing for the exact one-shot (1-stage) all-reduce (``OneShotAllReduce``).

``python3`` this file runs two sweeps, each ending in a markdown table. Both
drive the engine as production does -- no tuning knob pinned, so it walks
``ONESHOT_LADDER`` and picks a rung by payload size -- unless ``--atoms``,
``--grid-cap``, ``--fanout``, ``--block`` or ``--skip-self`` pin one.

``test_one_shot_allreduce`` checks three things per shape, and the second
matters more than the first:

1. The sum is right, against an fp32 reference, at the bf16 rounding floor.
2. The result is **bit-identical on every rank**. The kernel accumulates in a
   fixed rank order for exactly this reason, and an SQNR check cannot see an
   ordering bug -- both answers would be equally "accurate".
3. It is timed with ``run_perftest``. One row captures the all-reduce into a
   CUDA graph and replays it, checking 1 and 2 after every replay.

``test_one_shot_allreduce_run_ahead`` makes repeated back-to-back calls under
deliberate rank skew. The inbox is double-buffered by ``colour & 1`` and the
safety argument depends on a straggler's read of call k finishing before
anyone's push for call k+2; a quiescent test never exercises that.

Every rank is a ``multiprocessing`` spawn worker; one spawn per world size runs
both sweeps. OneShotAllReduce runs on gfx942/gfx950 at TP in {2, 4, 8}; other
archs skip, and ``main()`` skips a world size when fewer GPUs are visible than
TP.
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

from aiter.ops.flydsl.kernels.one_shot_allreduce import SUPPORTED_BLOCKS
from aiter.ops.flydsl.kernels.quick_allreduce_shared import SUPPORTED_WORLDS

try:
    ARCH = get_gfx_runtime()
except (KeyError, RuntimeError):
    ARCH = None
SUPPORTED_ARCHS = ("gfx942", "gfx950")

HIDDEN = 7168
# m x HIDDEN spans 14 KiB to 224 KiB. m = 1, 3, 5 end in a partial last tile
# and m = 8, 16 on an exact multiple at every rung's tile width, and the range
# crosses the PCIe ladder's rung boundaries at 48 KiB (TP4) and 96 KiB (TP2,
# TP4), so switching rungs mid-stream is exercised. The narrow shapes reach the
# sub-tile, single-block corner HIDDEN cannot.
SHAPES = [(m, HIDDEN) for m in (1, 3, 5, 8, 16)] + [(1, 1024), (1, 2048), (1, 3072)]

# (tp, tokens, hidden) captured into a CUDA graph. Each replay advances the
# device-side colour and alternates the inbox parity slot, so only repeated
# replays show the captured launch advancing that state rather than freezing it.
GRAPH_CASES = ((8, 5, HIDDEN),)
GRAPH_REPLAYS = 4

# Pinned configurations a default run covers next to the shipped ladder, for
# widths no shipped rung uses yet. Block 512 is the widest workgroup; at atoms=4
# it is a 32 KiB tile, so every shape above is a single partial tile.
EXTRA_CONFIGS = (
    {"atoms": 1, "grid_cap": 64, "fanout": "peer", "block": 512, "skip_self": False},
    {"atoms": 4, "grid_cap": 64, "fanout": "peer", "block": 512, "skip_self": True},
)

RUN_AHEAD_M = 5
RUN_AHEAD_ITERS = 200
SQNR_FLOOR_DB = 45.0

# Seconds to wait for each rank of a spawn. The kernels spin on flags written
# by peers, so a protocol bug or a dead rank hangs the rest; this fails the
# spawn instead of leaving it to CI's per-file timeout. A full default run of
# either FlyDSL all-reduce test takes a few minutes, JIT included.
SPAWN_TIMEOUT_S = 600


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
) -> dict:
    """One rank of one spawn: every case, then the run-ahead loop.

    A case is ``(tokens, hidden, graph)``. The engine is built once, so every
    case shares its compiled rungs and IPC inboxes.
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
            res["us"] = float(us)
            res["variant"] = eng.variant(inp.numel() * inp.element_size())
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
    return {"rows": rows, "run_ahead": run_ahead}


def _spawn(world_size: int, engine_kw: dict, cases: list[tuple]) -> list[dict]:
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
        _RESULTS[key] = _spawn(key[0], dict(key[1]), _CASES[key])
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


def _summarize(name: str, rows: list[dict]) -> None:
    if rows:
        df = pd.DataFrame(rows).drop(columns=list(KNOBS))
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
        default=SHAPES,
        help="(tokens, hidden) pairs.\n    e.g.: -s 1,7168 8,7168",
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
    # Nothing pinned on the command line: the shipped ladder, plus the widths it
    # does not use yet.
    if configs == [dict.fromkeys(KNOBS)]:
        configs += [dict(c) for c in EXTRA_CONFIGS]

    # Register every row before running any, so each (tp, config) is one spawn.
    shapes = [(int(t), int(h)) for t, h in args.mnk]
    rows = []
    for tp, knobs in itertools.product(tps, configs):
        cases = [(t, h, False) for t, h in shapes]
        cases += [(t, h, True) for g_tp, t, h in GRAPH_CASES if g_tp == tp]
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
        "flydsl one-shot allreduce run-ahead",
        [
            test_one_shot_allreduce_run_ahead(
                RUN_AHEAD_M, HIDDEN, tp, RUN_AHEAD_ITERS, **knobs
            )
            for tp, knobs in itertools.product(tps, configs)
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
