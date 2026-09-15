# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Runtime correctness for FlyDSL INT6 quick all-reduce (``QuickAllReduceInt6``).

Pytest collects validity cases only (no timing). ``python3`` this file
runs an aiter-op-test ``@benchmark`` / markdown sweep. Every rank is a
``multiprocessing`` spawn worker that builds its own
``QuickAllReduceInt6`` engine, calls ``compile_and_launch()``, and in the
sweep times ``fly.allreduce`` with ``run_perftest``. The oracle is an
untimed fp32 NCCL all-reduce of the same per-rank inputs.

Mesh schedule only. INT6 is lossy, so those cases gate on SQNR, a
calibrated mismatch ratio and a per-tile SQNR floor. The ``fp16`` wire
format is a lossless passthrough, so the transport tests that use it gate
on bit-identity instead -- which is what isolates a chunk-addressing or
flag-protocol bug from a codec one.

hidden=5120 is the width the kernel was tuned on, not a shape the kernel
requires. QuickAllReduceInt6 runs on gfx942/gfx950 at TP∈{2,4,8}; other
archs skip, and pytest skips a world size when fewer GPUs are visible
than TP.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import statistics
import sys
from multiprocessing import Pool, freeze_support, set_start_method

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest

pytest.importorskip("flydsl")

set_start_method("spawn", force=True)

from aiter.ops.flydsl.kernels.quick_allreduce_int6 import MESH_ST_LADDER
from aiter.ops.flydsl.kernels.quick_allreduce_shared import (
    SUPPORTED_WORLDS,
    TILE_BYTES,
    WORLD,
)
from aiter.ops.flydsl.quick_allreduce_int6 import DEFAULT_GRID_CAP

try:
    ARCH = get_gfx_runtime()
except (KeyError, RuntimeError):
    ARCH = None
SUPPORTED_ARCHS = ("gfx942", "gfx950")

# Anything below 18.0 dB is a regression, not a known cost of the schedule.
SQNR_MIN_DB = 18.0

# A 32 KiB tile the kernel never wrote scores ~0 dB; codec noise stays above 8.
# Per-tile rather than whole-payload, so one unwritten tile cannot be averaged
# away by the rest of a large message.
TILE_SQNR_MIN_DB = 8.0

# Calibrated to the INT6 group-16 codec vs fp32 all-reduce, not bit identity.
CLOSE_RTOL = 1e-1
CLOSE_ATOL = 1e-1
CLOSE_ERR_RATIO = 0.5

SUPER_TILE = 8
TP = WORLD

_FILLS = (
    "normal",
    "degenerate",
    "exact",
    "pos_underflow",
    "neg_underflow",
    "overflow_512",
    "zeros",
)

pytestmark = pytest.mark.skipif(
    ARCH not in SUPPORTED_ARCHS,
    reason="QuickAllReduceInt6 requires an available gfx942 or gfx950 GPU",
)

# Distinct correctness branches, not a tokens x hidden product.
# hidden=5120 is the calibrated width; hidden=4096 covers a width the tuning
# was not fitted to. (8, 1024) is a payload smaller than one 32 KiB tile.
# Unpinned ``MESH_ST_LADDER``: ST=1 below 38 MiB, ST=8 at/above. Pytest skips
# a world size when fewer GPUs are visible than TP.
_PYTEST_CASES = (
    (8, 8, 1024, "partial-tile"),
    (8, 512, 5120, "st1-auto-calib"),
    (8, 9216, 4096, "st8-alt-hidden"),
    (8, 32768, 5120, "st8-calib-prefill"),
    (4, 512, 5120, "tp4-st1-auto-calib"),
    (4, 9216, 4096, "tp4-st8-alt-hidden"),
    (2, 512, 5120, "tp2-st1-auto-calib"),
    (2, 9216, 4096, "tp2-st8-alt-hidden"),
)


def _num_tiles(tokens: int, hidden: int) -> int:
    nbytes = tokens * hidden * 2
    return max(1, (nbytes + TILE_BYTES - 1) // TILE_BYTES)


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
        # codec="fp16" this makes the whole reduce lossless, so the result is
        # bit-identical to the fp32 reference regardless of accumulation order.
        src = torch.randint(-8, 8, shape, generator=gen).float() * (2.0**-4)
    elif fill in ("pos_underflow", "neg_underflow", "overflow_512", "zeros"):
        val = {
            "pos_underflow": 2.0**-8,
            "neg_underflow": -(2.0**-8),
            # Drives the E4M3 scale above its largest exponent, which the
            # encoder has to saturate. Only rank 0 carries the value: if every
            # rank sent 512 the reduced sum would also saturate the INT6 group
            # codec, and the case would fail on codec range rather than on
            # scale encoding.
            "overflow_512": 512.0 if rank == 0 else 0.0,
            "zeros": 0.0,
        }[fill]
        return torch.full(shape, val, device=device, dtype=torch.bfloat16)
    else:
        raise ValueError(f"unknown fill {fill!r}; expected one of {_FILLS}")
    return src.to(device=device, dtype=torch.bfloat16)


def _pick_st(
    tokens: int,
    hidden: int,
    requested: int = SUPER_TILE,
    *,
    world_size: int,
    pinned: bool = False,
) -> int:
    """Mirror of ``QuickAllReduceInt6._pick_st``.

    Unpinned: walk ``MESH_ST_LADDER``. Pinned: the requested ST, if there
    is a whole super-tile to take.
    """
    tiles = _num_tiles(tokens, hidden)
    nbytes = tokens * hidden * 2
    want = requested
    if not pinned:
        want = 1
        for floor, rung_st, _cap in MESH_ST_LADDER[world_size]:
            if nbytes >= floor:
                want = rung_st
    if want == 1:
        return 1
    return want if tiles >= want else 1


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


def _run_rank(
    rank: int,
    tp: int,
    init_method: str,
    tokens: list[int],
    hiddens: list[int],
    super_tile: int,
    grid_cap: int,
    algorithm: str,
    fill: str,
    rs_codec: str | None,
    ag_codec: str | None,
    pin_super_tile: bool,
    time_it: bool,
) -> list[dict]:
    import torch.distributed as dist

    from aiter.ops.flydsl import QuickAllReduceInt6

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        world_size=tp,
        rank=rank,
        device_id=device,
    )
    # QuickAllReduceInt6 exchanges IPC metadata over a non-NCCL group;
    # NCCL stays for the fp32 reference all-reduce.
    gloo = dist.new_group(backend="gloo")
    group = dist.group.WORLD

    fly = QuickAllReduceInt6(
        group=gloo,
        device=device,
        rank=rank,
        world_size=tp,
        algorithm=algorithm,
        **({"super_tile": super_tile} if pin_super_tile else {}),
        grid_cap=grid_cap,
        # The case list deliberately includes sub-threshold shapes (8x1024 is
        # 16 KiB, well under MIN_PAYLOAD_BYTES) to cover the partial-tile path.
        min_bytes=0,
        # None means "let the host default it" -- the env-var codec mechanism
        # (AITER_ALL_REDUCE_CODEC, set by _spawn's `codec` arg) still applies.
        rs_codec=rs_codec,
        ag_codec=ag_codec,
    )
    # compile_and_launch() launches every ST binary on this shape and all ranks
    # must pass the same one, so keep the JIT buffer small at the widest hidden.
    compile_tokens = min(512, max(tokens))
    compile_hidden = max(hiddens)
    compile_inp = torch.empty(
        (compile_tokens, compile_hidden), device=device, dtype=torch.bfloat16
    )
    compile_out = torch.empty_like(compile_inp)
    dist.barrier()
    fly.compile_and_launch(compile_inp, compile_out)
    dist.barrier()
    del compile_inp, compile_out

    rows = []
    try:
        for ntok, hidden in zip(tokens, hiddens, strict=True):
            inp = _make_inp(ntok, hidden, fill, rank=rank, device=device)
            ref = inp.to(torch.float32)
            dist.all_reduce(ref, group=group)
            dist.barrier()

            out = torch.empty_like(inp)
            out.zero_()
            fly.allreduce(inp, out)
            torch.cuda.synchronize()
            dist.barrier()
            got = out.to(torch.float32)

            nbytes = int(inp.numel()) * int(inp.element_size())
            n_tiles = max(1, (nbytes + TILE_BYTES - 1) // TILE_BYTES)
            st_used = fly._pick_st(n_tiles, nbytes)
            st1 = fly._by_st.get(1, fly._by_st[st_used])
            mismatch = got != ref
            n_mismatch = int(mismatch.sum().item())
            first_bad = -1
            if n_mismatch:
                first_bad = int(
                    torch.nonzero(mismatch.reshape(-1), as_tuple=False)[0].item()
                )
            diff = (got - ref).abs()
            close_err = checkAllclose(
                ref,
                got,
                rtol=CLOSE_RTOL,
                atol=CLOSE_ATOL,
                tol_err_ratio=CLOSE_ERR_RATIO,
                printLog=False,
                msg=f"quick_allreduce_int6 rank {rank}",
            )
            row = {
                "tokens": ntok,
                "hidden": hidden,
                "grid_cap": grid_cap,
                "algorithm": algorithm,
                "inbox_memory": fly.inbox_memory,
                # Resolved, not requested: both laps default to INT6 unless
                # AITER_ALL_REDUCE_CODEC overrode them, and a regression
                # should name the codec that produced it.
                "rs_codec": fly.rs_codec,
                "ag_codec": fly.ag_codec,
                "st1_grid": int(st1.grid),
                "st_used": int(st_used),
                "grid": int(fly._by_st[st_used].grid),
                "sqnr_db": _sqnr_db(got, ref),
                "min_tile_sqnr_db": _min_tile_sqnr_db(got, ref),
                "rel_mae": _rel_mae(got, ref),
                "n_mismatch": n_mismatch,
                "max_abs_err": float(diff.max().item()) if diff.numel() else 0.0,
                "first_bad": first_bad,
                "allclose": bool(torch.allclose(got, ref, rtol=1e-2, atol=8e-3)),
                "err": float(close_err),
                "us": None,
            }
            if time_it:
                dist.barrier(group=group)
                torch.cuda.synchronize()

                def _allreduce(eng=fly, src=inp, dst=out):
                    eng.allreduce(src, dst)
                    return dst

                # use_cuda_event is mandatory here: run_perftest's default
                # timer wraps the iterations in torch.profiler, which collects
                # no device rows inside a spawn worker and then fails reducing
                # its empty trace. cuda.Event timing is unaffected.
                _, us = run_perftest(_allreduce, use_cuda_event=True)
                row["us"] = float(us)
            rows.append(row)
            del inp, out, ref, got
            torch.cuda.empty_cache()
    finally:
        fly.close()
        dist.destroy_process_group()
    return rows


def _spawn(
    world_size: int,
    pairs: list[tuple[int, int]],
    *,
    time_it: bool,
    super_tile: int = SUPER_TILE,
    grid_cap: int = DEFAULT_GRID_CAP,
    algorithm: str = "mesh",
    codec: str | None = None,
    fill: str = "normal",
    rs_codec: str | None = None,
    ag_codec: str | None = None,
    pin_super_tile: bool = False,
) -> list[list[dict]]:
    if world_size not in SUPPORTED_WORLDS:
        raise ValueError(f"unsupported world_size={world_size}")
    n_gpu = torch.cuda.device_count()
    if n_gpu < world_size:
        pytest.skip(f"QuickAllReduceInt6 needs {world_size} GPUs, have {n_gpu}")
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    token_list = [t for t, _ in pairs]
    hidden_list = [h for _, h in pairs]
    timeout = float(os.environ.get("FLYDSL_QR_TIMEOUT", "3600"))

    # Pin the codec the way a deployment would, rather than through a private
    # test-only flag: this exercises the override path itself, while leaving it
    # unset exercises the host default (both laps INT6). It has to go through
    # the environment because the host parses it once at import, and a spawn
    # worker inherits the environment as it stood when the Pool was created --
    # hence setting it around the Pool construction rather than around the calls.
    prev_codec = os.environ.get("AITER_ALL_REDUCE_CODEC")
    if codec is not None:
        os.environ["AITER_ALL_REDUCE_CODEC"] = codec.upper()
    else:
        os.environ.pop("AITER_ALL_REDUCE_CODEC", None)
    try:
        pool = Pool(processes=world_size)
    finally:
        if prev_codec is None:
            os.environ.pop("AITER_ALL_REDUCE_CODEC", None)
        else:
            os.environ["AITER_ALL_REDUCE_CODEC"] = prev_codec

    try:
        results = [
            pool.apply_async(
                _run_rank,
                kwds={
                    "rank": rank,
                    "tp": world_size,
                    "init_method": init_method,
                    "tokens": token_list,
                    "hiddens": hidden_list,
                    "super_tile": super_tile,
                    "grid_cap": grid_cap,
                    "algorithm": algorithm,
                    "fill": fill,
                    "rs_codec": rs_codec,
                    "ag_codec": ag_codec,
                    "pin_super_tile": pin_super_tile,
                    "time_it": time_it,
                },
            )
            for rank in range(world_size)
        ]
        ranks = [fut.get(timeout=timeout) for fut in results]
    except Exception:
        pool.terminate()
        raise
    else:
        pool.close()
    finally:
        pool.join()
    if len(ranks) != world_size:
        raise RuntimeError(
            f"QuickAllReduceInt6 gathered {len(ranks)} ranks, expected {world_size}"
        )
    return ranks


# Run multiple cases in a single spawn and cache the results, so the first test
# in a group pays the process startup and JIT and the rest are nearly free.
_BATCH_CACHE: dict[tuple, dict[tuple[int, int], list[list[dict]]]] = {}


def _index_by_shape(
    ranks: list[list[dict]], pairs: list[tuple[int, int]]
) -> dict[tuple[int, int], list[list[dict]]]:
    """Reshape ``_spawn``'s per-rank row list into a per-shape view.

    ``ranks[r]`` holds one row per pair, in ``pairs`` order (``_run_rank``'s
    loop appends in that order). Slicing out one shape's row from every rank
    reproduces exactly the ``ranks`` shape a single-shape ``_spawn`` call
    would have returned, so the assert helpers need no changes.
    """
    return {
        pair: [[rank_rows[i]] for rank_rows in ranks] for i, pair in enumerate(pairs)
    }


def _batch_cache_lookup(
    key: tuple, pairs: list[tuple[int, int]], **spawn_kwargs
) -> dict:
    """One ``_spawn`` call per *key*, memoized for the rest of the session."""
    if key not in _BATCH_CACHE:
        ranks = _spawn(key[0], pairs, **spawn_kwargs)
        _BATCH_CACHE[key] = _index_by_shape(ranks, pairs)
    return _BATCH_CACHE[key]


def _assert_sqnr(
    ranks: list[list[dict]],
    *,
    tokens: int,
    hidden: int,
    world_size: int,
    label: str,
    floor: float | None = None,
) -> dict:
    expected_st = _pick_st(tokens, hidden, world_size=world_size)
    if len(ranks) != world_size:
        raise AssertionError(
            f"{label}: gathered {len(ranks)} ranks, expected {world_size}"
        )
    want = SQNR_MIN_DB if floor is None else floor
    fails = []
    for rank, rows in enumerate(ranks):
        if not rows:
            fails.append(f"rank {rank}: no rows")
            continue
        row = rows[0]
        if row["st_used"] != expected_st:
            fails.append(f"rank {rank}: ST={row['st_used']}, expected {expected_st}")
        if row["sqnr_db"] < want:
            fails.append(
                f"rank {rank}: SQNR {row['sqnr_db']:.2f} dB < {want} "
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
    if fails:
        codecs = ranks[0][0]
        raise AssertionError(
            f"{label} tp={world_size} tokens={tokens} hidden={hidden} "
            f"rs={codecs.get('rs_codec')} ag={codecs.get('ag_codec')}: "
            + "; ".join(fails)
        )
    return ranks[0][0]


@pytest.mark.parametrize("world_size,tokens,hidden,label", _PYTEST_CASES)
def test_quick_allreduce_int6_sqnr_vs_fp32_allreduce(
    world_size, tokens, hidden, label
):
    """Shipping configuration: no codec pinned, so both laps default to INT6.

    Every case here that shares world_size rides one spawn --
    see ``_batch_cache_lookup``.
    """
    group_pairs = [(t, h) for ws, t, h, _ in _PYTEST_CASES if ws == world_size]
    batch = _batch_cache_lookup(
        (world_size, "sqnr"), group_pairs, time_it=False
    )
    ranks = batch[(tokens, hidden)]
    row = _assert_sqnr(
        ranks,
        tokens=tokens,
        hidden=hidden,
        world_size=world_size,
        label=label,
    )
    assert row["rs_codec"] == "int6" and row["ag_codec"] == "int6", row


_CODEC_FILL_CASES = (
    ("pos_underflow", "pos-underflow-2^-8"),
    ("neg_underflow", "neg-underflow-2^-8"),
    ("overflow_512", "overflow-512"),
    ("zeros", "true-zero-scale"),
)


@pytest.mark.parametrize("fill,label", _CODEC_FILL_CASES)
def test_quick_allreduce_int6_e4m3_codec_fill(fill, label):
    """Uniform payloads that land on the E4M3 scale's edge cases.

    Each drives the group extremum somewhere the encoder has to special-case:
    below the magnitude floor, past the largest exponent, or exactly zero.
    """
    ranks = _spawn(2, [(16, 1024)], time_it=False, fill=fill)
    _assert_sqnr(ranks, tokens=16, hidden=1024, world_size=2, label=label)


def test_quick_allreduce_int6_degenerate_inputs():
    """All-zero and all-tiny groups must not produce NaN.

    A group whose extremum is zero decodes to a zero scale, so the encode
    reciprocal saturates; before it was clamped, that reached the codec as Inf
    and ``0 * Inf`` poisoned the tile.
    """
    ranks = _spawn(2, [(512, 5120)], time_it=False, fill="degenerate")
    for rank, rows in enumerate(ranks):
        assert rows, f"rank {rank}: no rows"
        for row in rows:
            assert math.isfinite(row["rel_mae"]), f"rank {rank}: {row}"


# Transport-in-isolation tests: codec="fp16" is a lossless passthrough wire
# format, so these gate on identity with the reference rather than on SQNR.
# They exercise chunk/slot addressing, the flag protocol, the super-tile loop
# and the accumulate order, with the codec taken out of the picture.
#
# (world_size, tokens, hidden, super_tile, label). ``super_tile`` is pinned
# explicitly so the transport path is exercised at both ST=1 and ST=8.
_EXACT_CASES = (
    (8, 8, 1024, 1, "partial-tile"),
    (8, 512, 5120, 1, "st1"),
    (8, 4096, 4096, 8, "st8-multi-tile"),
    (4, 512, 5120, 1, "tp4-st1"),
    (4, 4096, 4096, 8, "tp4-st8"),
    (2, 512, 5120, 1, "tp2-st1"),
)

# Same shapes, one per world size, for the randn/allclose variant.
_ALLCLOSE_CASES = tuple(c for c in _EXACT_CASES if c[3] == 1 and c[2] == 5120)


def _assert_exact(
    ranks: list[list[dict]],
    *,
    tokens: int,
    hidden: int,
    world_size: int,
    label: str,
    mode: str,
) -> dict:
    """Gate on bit-exactness (``mode="exact"``) or allclose (``mode="randn"``).

    Deliberately does not check ``st_used`` against a predicted value the way
    ``_assert_sqnr`` does -- these tests pin ``super_tile`` explicitly, so
    there is nothing to predict, and the point here is the transport, not the
    super-tile selection policy (already covered elsewhere).
    """
    if len(ranks) != world_size:
        raise AssertionError(
            f"{label}: gathered {len(ranks)} ranks, expected {world_size}"
        )
    fails = []
    for rank, rows in enumerate(ranks):
        if not rows:
            fails.append(f"rank {rank}: no rows")
            continue
        row = rows[0]
        if mode == "exact":
            if row["n_mismatch"] != 0:
                fails.append(
                    f"rank {rank}: {row['n_mismatch']} mismatched elements, "
                    f"max |err| {row['max_abs_err']:.3e}, "
                    f"first bad flat index {row['first_bad']}"
                )
        elif not row["allclose"]:
            fails.append(
                f"rank {rank}: not allclose, max |err| {row['max_abs_err']:.3e}"
            )
    if fails:
        codecs = ranks[0][0]
        raise AssertionError(
            f"{label} tp={world_size} tokens={tokens} hidden={hidden} "
            f"rs={codecs.get('rs_codec')} ag={codecs.get('ag_codec')}: "
            + "; ".join(fails)
        )
    return ranks[0][0]


@pytest.mark.parametrize("world_size,tokens,hidden,super_tile,label", _EXACT_CASES)
def test_quick_allreduce_transport_bit_exact(
    world_size, tokens, hidden, super_tile, label
):
    """fp16 wire, exact-grid input: bit-identical to the fp32 reference.

    Grouped by (world_size, super_tile) -- a pinned super_tile is a
    Python-level kernel constant here, so it has to be part of the batch key
    alongside world_size (see ``_batch_cache_lookup``).
    """
    group_pairs = [
        (t, h)
        for ws, t, h, st, _ in _EXACT_CASES
        if ws == world_size and st == super_tile
    ]
    batch = _batch_cache_lookup(
        (world_size, "bit_exact", super_tile),
        group_pairs,
        time_it=False,
        super_tile=super_tile,
        grid_cap=64,
        codec="fp16",
        fill="exact",
        pin_super_tile=True,
    )
    ranks = batch[(tokens, hidden)]
    _assert_exact(
        ranks,
        tokens=tokens,
        hidden=hidden,
        world_size=world_size,
        label=f"{label}/fp16-exact",
        mode="exact",
    )


@pytest.mark.parametrize("world_size,tokens,hidden,super_tile,label", _ALLCLOSE_CASES)
def test_quick_allreduce_transport_allclose(
    world_size, tokens, hidden, super_tile, label
):
    """fp16 wire, realistic randn input: allclose to the fp32 reference.

    Tolerance is dominated by the bf16 output rounding plus fp16 accumulation.
    """
    ranks = _spawn(
        world_size,
        [(tokens, hidden)],
        time_it=False,
        super_tile=super_tile,
        grid_cap=64,
        codec="fp16",
        fill="normal",
        pin_super_tile=True,
    )
    _assert_exact(
        ranks,
        tokens=tokens,
        hidden=hidden,
        world_size=world_size,
        label=f"{label}/fp16-randn",
        mode="randn",
    )


@benchmark()
def test_quick_allreduce_int6(tokens, hidden, dtype, tp, grid_cap=DEFAULT_GRID_CAP):
    ranks = _spawn(tp, [(tokens, hidden)], time_it=True, grid_cap=grid_cap)
    row = _assert_sqnr(
        ranks,
        tokens=tokens,
        hidden=hidden,
        world_size=tp,
        label="bench",
    )
    nbytes = tokens * hidden * 2
    # (tp - 1) adds per element; codec ALU work is not counted.
    flops = tokens * hidden * (tp - 1)
    us = statistics.median([r[0]["us"] for r in ranks])
    return {
        "gfx": ARCH,
        "tp": tp,
        "grid_cap": row["grid_cap"],
        "st_used": row["st_used"],
        "flydsl us": us,
        "flydsl TFLOPS": (flops / us / 1e6) if us else 0.0,
        "flydsl TB/s": (nbytes / us / 1e6) if us else 0.0,
        "flydsl err": max(r[0]["err"] for r in ranks),
        "flydsl sqnr_db": min(r[0]["sqnr_db"] for r in ranks),
    }


test_quick_allreduce_int6.__test__ = False


def main():
    if ARCH not in SUPPORTED_ARCHS:
        aiter.logger.warning("QuickAllReduceInt6 unsupported on %s; skipping", ARCH)
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
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[1],
        help="Not a QuickAllReduceInt6 dimension; only 1 runs, "
        "other values are skipped.",
    )
    parser.add_argument(
        "--tp",
        type=int,
        nargs="*",
        default=[TP],
        help="World sizes to sweep (2, 4, or 8). Default 8.\n    e.g.: --tp 8",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (512, 5120),
            (9216, 5120),
            (32768, 5120),
        ],
        help="(tokens, hidden) pairs; hidden is free, 5120 is the tuned width.\n"
        "    e.g.: -s 512,5120 9216,5120",
    )
    parser.add_argument(
        "-o",
        "--out",
        default=None,
        help="Optional JSON output path for the sweep rows.",
    )
    parser.add_argument(
        "--grid-cap",
        type=int,
        default=DEFAULT_GRID_CAP,
        help="Persistent-launch block cap; the engine clamps it to the\n"
        "    measured resident workgroups per CU.",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        if dtype != dtypes.bf16:
            aiter.logger.warning(
                "QuickAllReduceInt6 payload is bf16; skipping %s", dtype
            )
            continue
        df = []
        for tp, batch, mnk in itertools.product(args.tp, args.batch, args.mnk):
            if batch != 1:
                continue
            if tp not in SUPPORTED_WORLDS:
                aiter.logger.warning(
                    "QuickAllReduceInt6 unsupported world_size=%s; skipping", tp
                )
                continue
            if n_gpu < tp:
                aiter.logger.warning(
                    "QuickAllReduceInt6 needs %s GPUs, have %s; skipping tp=%s",
                    tp,
                    n_gpu,
                    tp,
                )
                continue
            if not isinstance(mnk, tuple) or len(mnk) < 2:
                raise ValueError(f"-s expects tokens,hidden; got {mnk!r}")
            tokens, hidden = int(mnk[0]), int(mnk[1])
            df.append(
                test_quick_allreduce_int6(
                    tokens,
                    hidden,
                    dtype,
                    tp,
                    grid_cap=args.grid_cap,
                )
            )
        if df:
            table = pd.DataFrame(df)
            aiter.logger.info(
                "flydsl quick allreduce INT6 summary (markdown):\n%s",
                table.to_markdown(index=False),
            )
            if args.out:
                out_dir = os.path.dirname(os.path.abspath(args.out))
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                with open(args.out, "w") as fh:
                    json.dump(
                        {
                            "meta": {
                                "gfx": ARCH,
                                "grid_cap": args.grid_cap,
                                "timer": "run_perftest cuda_event",
                            },
                            "rows": df,
                        },
                        fh,
                        indent=2,
                        default=str,
                    )
                aiter.logger.info("wrote %s", args.out)


if __name__ == "__main__":
    freeze_support()
    main()
