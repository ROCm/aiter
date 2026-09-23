# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Runtime correctness and timing for FlyDSL quick all-reduce (``QuickAllReduceInt4``).

``python3`` this file runs four sweeps, each ending in a markdown table:

* ``test_quick_allreduce_int4`` -- the shipping configuration (codecs and
  super-tile left to the per-world defaults and ladders, as production dispatch
  constructs the engine), for both schedules at every world size, timed with
  ``run_perftest``. One row captures the all-reduce into a CUDA graph and
  replays it.
* ``test_quick_allreduce_int4_edge_inputs`` -- payloads that land on the E4M3
  scale's edge cases, and degenerate groups that must stay finite.
* ``test_quick_allreduce_int4_pinned_codec`` -- the ring with its wire formats
  pinned per lap: all-INT4 at TP8, and one lap lossless to isolate the other.
* ``test_quick_allreduce_transport`` -- the ``fp16`` wire format, a lossless
  passthrough, on an exactly representable input: the result must be
  bit-identical to the fp32 reference, which separates a chunk-addressing or
  flag-protocol bug from a codec one.

Every rank is a ``multiprocessing`` spawn worker that builds its own engine.
All rows that share an engine configuration ride one spawn. The oracle is an
untimed fp32 NCCL all-reduce of the same per-rank inputs. INT4/INT6 are lossy,
so those rows gate on SQNR, a calibrated mismatch ratio and a per-tile SQNR
floor.

hidden=5120 is the width the kernel was tuned on, not a shape the kernel
requires. QuickAllReduceInt4 runs on gfx942/gfx950 at TP in {2, 4, 8}; other
archs skip, and ``main()`` skips a world size when fewer GPUs are visible than
TP.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
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

from aiter.ops.flydsl.kernels.quick_allreduce_int4 import MESH_ST_LADDER
from aiter.ops.flydsl.kernels.quick_allreduce_int4_ring import ring_st_ladder
from aiter.ops.flydsl.kernels.quick_allreduce_shared import (
    SUPPORTED_WORLDS,
    TILE_BYTES,
    has_release_fence,
)

try:
    ARCH = get_gfx_runtime()
except (KeyError, RuntimeError):
    ARCH = None
SUPPORTED_ARCHS = ("gfx942", "gfx950")
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

# A 32 KiB tile the kernel never wrote scores ~0 dB; codec noise stays above 8.
# Per-tile rather than whole-payload, so one unwritten tile cannot be averaged
# away by the rest of a large message.
TILE_SQNR_MIN_DB = 8.0

# Calibrated to the INT4 group-16 codec vs fp32 all-reduce, not bit identity.
CLOSE_RTOL = 1e-1
CLOSE_ATOL = 1e-1
CLOSE_ERR_RATIO = 0.5

# Each replay advances the per-block colour and alternates the inbox parity
# slot, so a graph that froze either would pass the first replay and fail a
# later one. Four covers both parities twice.
GRAPH_REPLAYS = 4

# Shipping-configuration shapes, per world size.
SHAPES_PER_WORLD_SIZE = {
    8: [(8, 1024), (512, 5120), (9216, 4096), (32768, 5120)],
    4: [(512, 5120), (9216, 4096)],
    2: [(512, 5120), (9216, 4096)],
}

# (tp, algorithm, tokens, hidden) captured into a CUDA graph. 
GRAPH_CASES = ((8, "ring", 512, 5120),)

# (tp, algorithm, tokens, hidden, fill).
EDGE_CASES = (
    (2, "mesh", 16, 1024, "pos_underflow"),
    (2, "mesh", 16, 1024, "neg_underflow"),
    (2, "mesh", 16, 1024, "overflow_512"),
    (2, "mesh", 16, 1024, "zeros"),
    (2, "mesh", 512, 5120, "degenerate"),
    (8, "ring", 512, 5120, "degenerate"),
)

# (tp, tokens, hidden, rs_codec, ag_codec), ring algo only.
PINNED_CODEC_CASES = (
    (8, 512, 5120, "int4", "int4"),
    (8, 512, 5120, "fp16", "int4"),
    (8, 512, 5120, "int4", "fp16"),
)

# (tp, algorithm, tokens, hidden, super_tile) on the lossless fp16 wire.
TRANSPORT_CASES = (
    (8, "ring", 8, 1024, 1),
    (8, "ring", 512, 5120, 1),
    (8, "ring", 4096, 4096, 8),
    (4, "ring", 512, 5120, 1),
    (4, "ring", 4096, 4096, 8),
    (2, "ring", 512, 5120, 1),
    (8, "mesh", 512, 5120, 1),
    (8, "mesh", 4096, 4096, 8),
)
TRANSPORT_GRID_CAP = 64

# Seconds to wait for each rank of a spawn. The kernels spin on flags written
# by peers, so a protocol bug or a dead rank hangs the rest.
SPAWN_TIMEOUT_S = 600

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


def _num_tiles(nbytes: int) -> int:
    return max(1, (nbytes + TILE_BYTES - 1) // TILE_BYTES)


def _expected_st(
    nbytes: int,
    *,
    algorithm: str,
    world_size: int,
    inbox_memory: str,
    grid_by_st: dict[int, int],
) -> int:
    """Mirror of ``QuickAllReduceInt4._pick_st`` for an unpinned engine.

    Two rules compose. The payload one: the schedule's ladder assigns a
    super-tile by size -- publishes per rank are ``num_tiles / ST * 2(N-1)``,
    so a bigger payload wants a bigger one. The interconnect one: an inbox that
    needs a release fence makes each publish expensive enough to take the
    super-tile as soon as there is a whole one, while without a fence ST=1 is
    preferred until there are more tiles than blocks. Which applies is a
    property of the host, so it comes from the rank's reported
    ``inbox_memory`` rather than being assumed.

    *grid_by_st* must hold the engines' *clamped* grids, not the requested
    caps: the host reduces them to the measured resident workgroups per CU,
    and the clamped value is what the selection compares against.
    """
    ladder = (
        MESH_ST_LADDER[world_size]
        if algorithm == "mesh"
        else ring_st_ladder(world_size)
    )
    want = 1
    for floor, rung_st, _cap in ladder:
        if nbytes >= floor:
            want = rung_st
    if want == 1:
        return 1
    tiles = _num_tiles(nbytes)
    if has_release_fence(inbox_memory):
        return want if tiles >= want else 1
    return want if tiles > grid_by_st[want] else 1


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


def _min_tile_sqnr_db(got: torch.Tensor, reference: torch.Tensor) -> float:
    """Worst 32 KiB-tile SQNR, so one unwritten tile cannot be averaged away."""
    tile_elems = TILE_BYTES // 2
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


def _metrics(out: torch.Tensor, ref: torch.Tensor, rank: int) -> dict:
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
        "min_tile_sqnr_db": _min_tile_sqnr_db(got, ref),
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
    for key in ("err", "n_mismatch", "max_abs_err"):
        out[key] = max(a[key], b[key])
    # NaN must win, so compare with isfinite rather than max().
    if not math.isfinite(b["rel_mae"]) or b["rel_mae"] > a["rel_mae"]:
        out["rel_mae"] = b["rel_mae"]
    if a["first_bad"] < 0:
        out["first_bad"] = b["first_bad"]
    return out


def _run_rank(
    rank: int,
    tp: int,
    init_method: str,
    engine_kw: dict,
    cases: list[tuple],
) -> list[dict]:
    """One rank of one spawn: build the engine, then run every case on it.

    A case is ``(tokens, hidden, fill, graph, time_it)``. The engine is built
    once, so every case shares its compiled binaries and IPC inbox.
    """
    import torch.distributed as dist

    from aiter.ops.flydsl import QuickAllReduceInt4

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        world_size=tp,
        rank=rank,
        device_id=device,
    )
    # QuickAllReduceInt4 exchanges IPC metadata over a non-NCCL group; NCCL
    # stays for the fp32 reference all-reduce.
    gloo = dist.new_group(backend="gloo")
    group = dist.group.WORLD

    fly = QuickAllReduceInt4(
        group=gloo,
        device=device,
        rank=rank,
        world_size=tp,
        # The cases deliberately include sub-threshold shapes (8x1024 is
        # 16 KiB, well under the default floor) to cover the partial-tile path.
        min_bytes=0,
        **engine_kw,
    )
    # compile_and_launch() launches every ST binary on this shape and all ranks
    # must pass the same one, so keep the JIT buffer small at the widest hidden.
    compile_inp = torch.empty(
        (min(512, max(c[0] for c in cases)), max(c[1] for c in cases)),
        device=device,
        dtype=torch.bfloat16,
    )
    compile_out = torch.empty_like(compile_inp)
    dist.barrier()
    fly.compile_and_launch(compile_inp, compile_out)
    dist.barrier()
    del compile_inp, compile_out

    rows = []
    try:
        for ntok, hidden, fill, graph, time_it in cases:
            inp = _make_inp(ntok, hidden, fill, rank=rank, device=device)
            ref = inp.to(torch.float32)
            dist.all_reduce(ref, group=group)
            dist.barrier()

            out = torch.zeros_like(inp)
            fly.allreduce(inp, out)
            torch.cuda.synchronize()
            dist.barrier()
            m = _metrics(out, ref, rank)

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
                    m = _worst(m, _metrics(out, ref, rank))

            nbytes = int(inp.numel()) * int(inp.element_size())
            # Pass nbytes as well: with a ladder the super-tile is chosen by
            # payload size, and omitting it silently reports the fallback.
            st_used = fly._pick_st(_num_tiles(nbytes), nbytes)
            row = {
                **m,
                "inbox_memory": fly.inbox_memory,
                # Resolved, not requested: these come from the per-world-size
                # default unless the caller pinned a lap, and a regression
                # should name the codec that produced it.
                "rs_codec": fly.rs_codec,
                "ag_codec": fly.ag_codec,
                "st_used": int(st_used),
                "grid_by_st": {st: int(e.grid) for st, e in fly._by_st.items()},
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

                # use_cuda_event is mandatory here: run_perftest's default
                # timer wraps the iterations in torch.profiler, which collects
                # no device rows inside a spawn worker and then fails reducing
                # its empty trace. cuda.Event timing is unaffected.
                _, us = run_perftest(fn, use_cuda_event=True)
                row["us"] = float(us)
            rows.append(row)
            del inp, out, ref, g
            torch.cuda.empty_cache()
    finally:
        fly.close()
        dist.destroy_process_group()
    return rows


def _spawn(world_size: int, engine_kw: dict, cases: list[tuple]) -> list[list[dict]]:
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


# Rows are registered up front, grouped by the engine they need, so that each
# engine is built by exactly one spawn however many tables read from it.
# A spawn key is ``(tp, sorted engine kwargs)``; a case is
# ``(tokens, hidden, fill, graph, time_it)``.
_CASES: dict[tuple, list[tuple]] = {}
_RESULTS: dict[tuple, dict[tuple, list[dict]]] = {}
_FAILURES: list[str] = []


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
        ranks = _spawn(key[0], dict(key[1]), cases)
        _RESULTS[key] = {
            c: [rank_rows[i] for rank_rows in ranks] for i, c in enumerate(cases)
        }
    return _RESULTS[key][case]


def _check(label: str, fails: list[str]) -> bool:
    if fails:
        msg = f"{label}: " + "; ".join(fails)
        aiter.logger.error(msg)
        _FAILURES.append(msg)
    return not fails


def _check_sqnr(
    label: str,
    rows: list[dict],
    *,
    floor: float,
    expected_st: int | None,
) -> bool:
    fails = []
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


def _shipping_st(rows: list[dict], nbytes: int, algorithm: str, tp: int) -> int:
    return _expected_st(
        nbytes,
        algorithm=algorithm,
        world_size=tp,
        inbox_memory=rows[0]["inbox_memory"],
        grid_by_st=rows[0]["grid_by_st"],
    )


def _summary(rows: list[dict]) -> dict:
    return {
        "gfx": ARCH,
        "inbox_memory": rows[0]["inbox_memory"],
        "rs_codec": rows[0]["rs_codec"],
        "ag_codec": rows[0]["ag_codec"],
        "st_used": rows[0]["st_used"],
        "err": max(r["err"] for r in rows),
        "sqnr_db": min(r["sqnr_db"] for r in rows),
        "min_tile_sqnr_db": min(r["min_tile_sqnr_db"] for r in rows),
    }


def _ship_key(tp: int, algorithm: str, grid_cap: int | None) -> tuple:
    kw = {"algorithm": algorithm}
    if grid_cap is not None:
        kw["grid_cap"] = grid_cap
    return _key(tp, **kw)


def _transport_key(tp: int, algorithm: str, super_tile: int) -> tuple:
    return _key(
        tp,
        algorithm=algorithm,
        super_tile=super_tile,
        grid_cap=TRANSPORT_GRID_CAP,
        rs_codec="fp16",
        ag_codec="fp16",
    )


@benchmark()
def test_quick_allreduce_int4(
    tokens, hidden, dtype, tp, algorithm, grid_cap=None, graph=False
):
    """Shipping configuration: no codec or super-tile pinned."""
    rows = _result(
        _ship_key(tp, algorithm, grid_cap), (tokens, hidden, "normal", graph, True)
    )
    nbytes = tokens * hidden * 2
    _check_sqnr(
        f"tp={tp} {algorithm} {tokens}x{hidden} graph={graph}",
        rows,
        floor=SQNR_MIN_DB,
        expected_st=_shipping_st(rows, nbytes, algorithm, tp),
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
def test_quick_allreduce_int4_edge_inputs(tokens, hidden, tp, algorithm, fill):
    """Edge-case payloads on the shipping engine; correctness only."""
    rows = _result(_ship_key(tp, algorithm, None), (tokens, hidden, fill, False, False))
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
        )
    ret = _summary(rows)
    ret["rel_mae"] = max(r["rel_mae"] for r in rows)
    return ret


@benchmark()
def test_quick_allreduce_int4_pinned_codec(tokens, hidden, tp, rs_codec, ag_codec):
    """The ring with both laps' wire formats pinned; correctness only."""
    key = _key(tp, algorithm="ring", rs_codec=rs_codec, ag_codec=ag_codec)
    rows = _result(key, (tokens, hidden, "normal", False, False))
    label = f"tp={tp} ring {tokens}x{hidden} rs={rs_codec} ag={ag_codec}"
    _check_sqnr(
        label,
        rows,
        floor=SQNR_MIN_DB_TP8_INT4_RING,
        expected_st=_shipping_st(rows, tokens * hidden * 2, "ring", tp),
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
def test_quick_allreduce_transport(tokens, hidden, tp, algorithm, super_tile):
    """fp16 wire, exact-grid input: bit-identical to the fp32 reference.

    Exercises chunk/slot addressing, the flag protocol, the super-tile loop and
    the accumulate order with the codec taken out of the picture. super_tile is
    pinned, so there is no selection to check.
    """
    rows = _result(
        _transport_key(tp, algorithm, super_tile),
        (tokens, hidden, "exact", False, False),
    )
    _check(
        f"tp={tp} {algorithm} {tokens}x{hidden} st={super_tile} fp16-exact",
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
        "n_mismatch": max(r["n_mismatch"] for r in rows),
        "max_abs_err": max(r["max_abs_err"] for r in rows),
    }


def _summarize(name: str, rows: list[dict]) -> None:
    if rows:
        aiter.logger.info(
            "%s summary (markdown):\n%s",
            name,
            pd.DataFrame(rows).to_markdown(index=False),
        )


def main():
    if ARCH not in SUPPORTED_ARCHS:
        aiter.logger.warning("QuickAllReduceInt4 unsupported on %s; skipping", ARCH)
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
        "Default: a per-TP list; hidden=5120 is the tuned width.\n"
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
        "-o",
        "--out",
        default=None,
        help="Optional JSON output path for the shipping-sweep rows.",
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
        aiter.logger.warning("QuickAllReduceInt4 payload is bf16; skipping others")

    # Register every row before running any, so rows sharing an engine share
    # a spawn.
    ship = []
    for tp, algorithm, grid_cap in itertools.product(tps, algos, args.grid_cap):
        shapes = args.mnk if args.mnk is not None else SHAPES_PER_WORLD_SIZE[tp]
        for mnk in shapes:
            if not isinstance(mnk, tuple) or len(mnk) != 2:
                raise ValueError(f"-s expects tokens,hidden; got {mnk!r}")
            ship.append((int(mnk[0]), int(mnk[1]), tp, algorithm, grid_cap, False))
    for tp, algorithm, tokens, hidden in GRAPH_CASES:
        if tp in tps and algorithm in algos:
            ship.append((tokens, hidden, tp, algorithm, args.grid_cap[0], True))
    if dts:
        for tokens, hidden, tp, algorithm, grid_cap, graph in ship:
            _register(
                _ship_key(tp, algorithm, grid_cap),
                (tokens, hidden, "normal", graph, True),
            )
    edge = [c for c in EDGE_CASES if c[0] in tps and c[1] in algos]
    for tp, algorithm, tokens, hidden, fill in edge:
        _register(_ship_key(tp, algorithm, None), (tokens, hidden, fill, False, False))
    pinned = [c for c in PINNED_CODEC_CASES if c[0] in tps and "ring" in algos]
    for tp, tokens, hidden, rs, ag in pinned:
        _register(
            _key(tp, algorithm="ring", rs_codec=rs, ag_codec=ag),
            (tokens, hidden, "normal", False, False),
        )
    transport = [c for c in TRANSPORT_CASES if c[0] in tps and c[1] in algos]
    for tp, algorithm, tokens, hidden, st in transport:
        _register(
            _transport_key(tp, algorithm, st), (tokens, hidden, "exact", False, False)
        )

    for dtype in dts:
        rows = [
            test_quick_allreduce_int4(
                tokens, hidden, dtype, tp, algorithm, grid_cap=grid_cap, graph=graph
            )
            for tokens, hidden, tp, algorithm, grid_cap, graph in ship
        ]
        _summarize("flydsl quick allreduce INT4", rows)
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
            test_quick_allreduce_transport(tokens, hidden, tp, algorithm, st)
            for tp, algorithm, tokens, hidden, st in transport
        ],
    )

    if _FAILURES:
        raise SystemExit(
            f"{len(_FAILURES)} QuickAllReduceInt4 check(s) failed:\n  "
            + "\n  ".join(_FAILURES)
        )


if __name__ == "__main__":
    freeze_support()
    from time import perf_counter
    start = perf_counter()
    main()
    end = perf_counter()
    aiter.logger.info(f"Test execution took {end-start:.2f}s")
