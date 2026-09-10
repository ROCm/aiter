# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Runtime correctness for FlyDSL INT4 QuickReduce (``QRInt4``).

Pytest collects validity cases only (no timing). ``python3`` this file
runs an aiter-op-test ``@benchmark`` / markdown sweep. Each rank times
``fly.allreduce`` with ``run_perftest`` after ``compile()``. The oracle
is an untimed fp32 NCCL all-reduce of the same per-rank inputs. INT4 is
lossy, so the gate is SQNR vs that oracle on **every** rank, not
bit-identity or tight ``checkAllclose``.

5120 is a measured calibration width, not an ABI requirement. The kernel
is gfx942/gfx950 TP∈{2,4,8}; other archs skip. Pytest skips a world size
when fewer GPUs are visible than TP.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import tempfile
import time

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
from aiter.test_common import benchmark, run_perftest

pytest.importorskip("flydsl")

from aiter.ops.flydsl.kernels.qr_int4 import DEFAULT_GRID_CAP
from aiter.ops.flydsl.kernels.qr_int4_ring_kernel import RING_ST_LADDER
from aiter.ops.flydsl.kernels.qr_int_shared import (
    SUPPORTED_WORLDS,
    TILE_BYTES,
    WORLD,
    has_release_fence,
)

ARCH = get_gfx_runtime()
SUPPORTED_ARCHS = ("gfx942", "gfx950")
# One SQNR floor for both schedules, in their shipping configuration.
#
# It used to take two. The ring's reduce-scatter lap requantizes N-1 times where
# the mesh requantizes once, and the partial sum it requantizes grows with the
# contributions folded in, so the ring's SQNR degrades with N where the mesh's
# does not: 22.2 dB at TP2, 18.7 at TP4, ~15 at TP8 on an all-INT4 wire. No
# single number covered that.
#
# Defaulting the ring's reduce-scatter lap to INT6 at TP8 lifts it to ~21 dB and
# removes the reason for the split. Anything that falls below 18.0 now is a
# regression, not a known cost of the schedule.
SQNR_MIN_DB = {"mesh": 18.0, "ring": 18.0}

# INT4 at TP8 is still a supported configuration -- AITER_ALL_REDUCE_CODEC=INT4
# reaches it -- and is covered by its own case rather than skipped. It is held
# to what it actually delivers, not to the shipping floor.
SQNR_MIN_DB_TP8_INT4_RING = 15.0
SUPER_TILE = 8
TP = WORLD

pytestmark = pytest.mark.skipif(
    ARCH not in SUPPORTED_ARCHS,
    reason="QRInt4 unsupported arch (need gfx942 or gfx950)",
)

# Distinct correctness branches, not a tokens x hidden product.
# hidden=5120 is calibration; 4096 proves the map is not width-locked.
# (8, 1024) is a half-tile tail (TILE_BYTES = 32 KiB).
# TP2/4: ST=1 calib plus one ST=8 case (num_tiles > grid_cap) each.
# Pytest skips a world size when fewer GPUs are visible than TP.
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


def _pick_st(
    tokens: int,
    hidden: int,
    requested: int = SUPER_TILE,
    *,
    grid_cap: int = DEFAULT_GRID_CAP,
    inbox_memory: str = "uncached",
    algorithm: str = "mesh",
) -> int:
    """Mirror of ``QRInt4._pick_st``, so the test asserts the rule not the code.

    Two rules compose. The interconnect one: an inbox that needs a release fence
    makes each publish expensive enough to take a super-tile as soon as there is
    one, while without a fence ST=1 is preferred for its parallelism. Which
    applies is a property of the host, so it comes from the rank's reported
    ``inbox_memory`` rather than being assumed.

    The payload one, ring only: publishes per rank are
    ``num_tiles / ST * 2(N-1)``, so a bigger payload wants a bigger super-tile.
    ``RING_ST_LADDER`` holds the sited rungs. The tests construct QRInt4 without
    pinning ``super_tile``, so the ring walks that ladder and ``requested`` does
    not apply to it.
    """
    tiles = _num_tiles(tokens, hidden)
    if algorithm == "ring":
        nbytes = tokens * hidden * 2
        requested = 1
        for floor, rung_st, _cap in RING_ST_LADDER:
            if nbytes >= floor:
                requested = rung_st
    if requested == 1:
        return 1
    if has_release_fence(inbox_memory):
        return requested if tiles >= requested else 1
    return requested if tiles > grid_cap else 1


def _sqnr_db(got: torch.Tensor, reference: torch.Tensor) -> float:
    mse = float(((got - reference) ** 2).mean().item())
    ref_pow = float((reference * reference).mean().item())
    if not math.isfinite(mse) or not math.isfinite(ref_pow):
        return float("-inf")
    if ref_pow <= 0.0:
        return float("inf") if mse <= 0.0 else float("-inf")
    if mse <= 0.0:
        return float("inf")
    return 10.0 * math.log10(ref_pow / mse)


def _rel_mae(got: torch.Tensor, reference: torch.Tensor) -> float:
    scale = float(reference.abs().mean().item())
    err = float((got - reference).abs().mean().item())
    return err / scale if scale else 0.0


def _run_rank(args) -> None:
    import torch.distributed as dist

    from aiter.ops.flydsl import QRInt4

    rank = args.rank
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=args.init_method,
        world_size=args.tp,
        rank=rank,
    )
    gloo = dist.new_group(backend="gloo")
    group = dist.group.WORLD

    fly = QRInt4(
        group=gloo,
        device=device,
        rank=rank,
        world_size=args.tp,
        algorithm=args.algorithm,
        # Left unpinned for the ring so QRInt4 walks RING_ST_LADDER -- that is
        # the configuration production runs, and the one worth testing. The
        # transport tests (fp16 codec) pin it explicitly instead.
        **(
            {"super_tile": args.super_tile}
            if (args.algorithm != "ring" or args.pin_super_tile)
            else {}
        ),
        grid_cap=args.grid_cap,
        # The case list deliberately includes sub-threshold shapes (8x1024 is
        # 16 KiB, well under MIN_PAYLOAD_BYTES) to cover the partial-tile path.
        min_bytes=0,
        # None means "let QRInt4 default it" -- the env-var codec mechanism
        # (AITER_ALL_REDUCE_CODEC, set by _spawn's `codec` arg) still applies.
        # Explicit here only for the lap-isolation test, which needs the two
        # laps to differ and the env var cannot express that.
        rs_codec=args.rs_codec,
        ag_codec=args.ag_codec,
    )
    compile_tokens = min(512, max(args.tokens))
    compile_hidden = max(args.hiddens)
    compile_inp = torch.empty(
        (compile_tokens, compile_hidden), device=device, dtype=torch.bfloat16
    )
    compile_out = torch.empty_like(compile_inp)
    dist.barrier()
    fly.compile(compile_inp, compile_out)
    dist.barrier()
    del compile_inp, compile_out

    rows = []
    for tokens, hidden in zip(args.tokens, args.hiddens, strict=True):
        gen = torch.Generator().manual_seed(1234 + rank)
        if args.fill == "exact":
            # Grid of 1/16, magnitude < 0.5: exact in both bf16 and fp16, and
            # every partial sum over up to 8 ranks stays exact in both too
            # (integer multiple of 1/16, magnitude <= 4 -- 7 significant bits).
            # Paired with codec="fp16" this makes the whole reduce lossless,
            # so the result is bit-identical to the fp32 reference regardless
            # of accumulation order.
            src = torch.randint(-8, 8, (tokens, hidden), generator=gen).float() * (
                2.0**-4
            )
        else:
            src = torch.randn(tokens, hidden, generator=gen, dtype=torch.float32) * 0.1
            if args.fill == "degenerate":
                # Half the rows exactly zero, half far below the E4M3
                # magnitude floor of 2^-7. Both drive the group extremum to
                # (or under) zero, which is where the encode reciprocal blows
                # up.
                src[0::2] = 0.0
                src[1::2] *= 1e-8
        inp = src.to(device=device, dtype=torch.bfloat16)
        out = torch.empty_like(inp)
        ref = inp.to(torch.float32)
        dist.all_reduce(ref, group=group)

        dist.barrier()
        out.zero_()
        fly.allreduce(inp, out)
        torch.cuda.synchronize()
        dist.barrier()
        got = out.to(torch.float32)
        nbytes = int(inp.numel()) * int(inp.element_size())
        tiles = max(1, (nbytes + TILE_BYTES - 1) // TILE_BYTES)
        mismatch = got != ref
        n_mismatch = int(mismatch.sum().item())
        first_bad = -1
        if n_mismatch:
            bad_idx = torch.nonzero(mismatch.reshape(-1), as_tuple=False)
            first_bad = int(bad_idx[0].item())
        diff = (got - ref).abs()
        row = {
            "tokens": tokens,
            "hidden": hidden,
            "grid_cap": args.grid_cap,
            "algorithm": args.algorithm,
            "inbox_memory": fly.inbox_memory,
            # Resolved, not requested: these come from the per-world-size
            # default unless AITER_ALL_REDUCE_CODEC overrode it, and a
            # regression should name the codec that produced it.
            "rs_codec": fly.rs_codec,
            "ag_codec": fly.ag_codec,
            # Pass nbytes as well: with a ladder the super-tile is chosen by
            # payload size, and omitting it silently reports the fallback.
            "st_used": fly._pick_st(tiles, nbytes),
            "sqnr_db": _sqnr_db(got, ref),
            "rel_mae": _rel_mae(got, ref),
            "n_mismatch": n_mismatch,
            "max_abs_err": float(diff.max().item()) if diff.numel() else 0.0,
            "first_bad": first_bad,
            "allclose": bool(torch.allclose(got, ref, rtol=1e-2, atol=8e-3)),
            "us": None,
        }
        if args.time_it:
            dist.barrier(group=group)
            torch.cuda.synchronize()

            def _allreduce(eng=fly, src=inp, dst=out):
                eng.allreduce(src, dst)
                return dst

            _, us = run_perftest(_allreduce)
            row["us"] = us
        rows.append(row)
        del inp, out, ref, got
        torch.cuda.empty_cache()

    gathered = [None] * args.tp
    dist.all_gather_object(gathered, rows, group=gloo)
    if rank == 0 and args.out:
        with open(args.out, "w") as fh:
            json.dump({"ranks": gathered}, fh)
    dist.barrier(group=group)
    fly.close()
    dist.destroy_process_group()


def _spawn(
    world_size: int,
    pairs: list[tuple[int, int]],
    *,
    time_it: bool,
    super_tile: int = SUPER_TILE,
    grid_cap: int = DEFAULT_GRID_CAP,
    algorithm: str = "mesh",
    codec: str | None = None,
    fill: str = "randn",
    rs_codec: str | None = None,
    ag_codec: str | None = None,
    pin_super_tile: bool = False,
) -> list[list[dict]]:
    # HIP QR/fused-AR use multiprocessing.Pool from ``python3`` __main__.
    # This file is also collected by pytest, and FlyDSL JIT needs a fresh
    # interpreter per rank, so ranks are Popen of this file with --rank
    # (not Pool / torchrun). Init method matches the HIP QR helpers.
    if world_size not in SUPPORTED_WORLDS:
        raise ValueError(f"unsupported world_size={world_size}")
    n_gpu = torch.cuda.device_count()
    if n_gpu < world_size:
        pytest.skip(f"QRInt4 needs {world_size} GPUs, have {n_gpu}")
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    out_path = os.path.join(tempfile.mkdtemp(prefix="flydsl_qr_int4_"), "rank0.json")
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        f"{_REPO_ROOT}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else _REPO_ROOT
    )
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("FLYDSL_GPU_ARCH", ARCH)
    # Pin the codec the way a deployment would, rather than through a private
    # test-only flag: this exercises the override path itself, while leaving it
    # unset exercises the per-world-size default.
    if codec is not None:
        env["AITER_ALL_REDUCE_CODEC"] = codec.upper()
    else:
        env.pop("AITER_ALL_REDUCE_CODEC", None)
    tokens = ",".join(str(t) for t, _ in pairs)
    hiddens = ",".join(str(h) for _, h in pairs)
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
            "--algorithm",
            algorithm,
            "--tokens",
            tokens,
            "--hiddens",
            hiddens,
            "--super-tile",
            str(super_tile),
            "--grid-cap",
            str(grid_cap),
            "--fill",
            fill,
        ]
        if rs_codec is not None:
            cmd += ["--rs-codec", rs_codec]
        if ag_codec is not None:
            cmd += ["--ag-codec", ag_codec]
        if pin_super_tile:
            cmd.append("--pin-super-tile")
        if time_it:
            cmd.append("--time-it")
        if rank == 0:
            cmd += ["--out", out_path]
        log = open(  # noqa: SIM115
            f"/tmp/flydsl_qr_int4_tp{world_size}_rank{rank}.log",
            "w",
        )
        procs.append(
            subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        )
        logs.append(log)
    rc = 0
    deadline = time.time() + float(os.environ.get("FLYDSL_QR_TIMEOUT", "3600"))
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
            path = f"/tmp/flydsl_qr_int4_tp{world_size}_rank{rank}.log"
            try:
                with open(path) as fh:
                    tails.append(f"===== rank {rank} =====\n{fh.read()[-4000:]}")
            except OSError:
                pass
        raise RuntimeError("QRInt4 ranks failed\n" + "\n".join(tails))
    with open(out_path) as fh:
        payload = json.load(fh)
    ranks = payload["ranks"]
    if len(ranks) != world_size:
        raise RuntimeError(f"QRInt4 gathered {len(ranks)} ranks, expected {world_size}")
    return ranks


def _assert_sqnr(
    ranks: list[list[dict]],
    *,
    tokens: int,
    hidden: int,
    world_size: int,
    label: str,
    algorithm: str = "mesh",
    floor: float | None = None,
) -> dict:
    expected_st = _pick_st(
        tokens,
        hidden,
        grid_cap=ranks[0][0]["grid_cap"],
        inbox_memory=ranks[0][0]["inbox_memory"],
        algorithm=algorithm,
    )
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
        if row["st_used"] != expected_st:
            fails.append(f"rank {rank}: ST={row['st_used']}, expected {expected_st}")
        want = SQNR_MIN_DB[algorithm] if floor is None else floor
        if row["sqnr_db"] < want:
            fails.append(
                f"rank {rank}: SQNR {row['sqnr_db']:.2f} dB < {want} "
                f"(rel MAE {row['rel_mae']:.3e})"
            )
    if fails:
        codecs = ranks[0][0]
        raise AssertionError(
            f"{label} tp={world_size} tokens={tokens} hidden={hidden} "
            f"rs={codecs.get('rs_codec')} ag={codecs.get('ag_codec')}: "
            + "; ".join(fails)
        )
    return ranks[0][0]


@pytest.mark.parametrize("algorithm", ("mesh", "ring"))
@pytest.mark.parametrize("world_size,tokens,hidden,label", _PYTEST_CASES)
def test_qr_int4_sqnr_vs_fp32_allreduce(world_size, tokens, hidden, label, algorithm):
    """Shipping configuration: no codec pinned, so the per-N default applies."""
    ranks = _spawn(world_size, [(tokens, hidden)], time_it=False, algorithm=algorithm)
    _assert_sqnr(
        ranks,
        tokens=tokens,
        hidden=hidden,
        world_size=world_size,
        label=f"{label}/{algorithm}",
        algorithm=algorithm,
    )


@pytest.mark.parametrize(
    "tokens,hidden,label",
    [(t, h, lbl) for ws, t, h, lbl in _PYTEST_CASES if ws == 8],
)
def test_qr_int4_ring_tp8_int4_codec(tokens, hidden, label):
    """TP8 ring forced back to an all-INT4 wire by the environment override.

    Two things at once: that ``AITER_ALL_REDUCE_CODEC`` actually reaches the
    kernel, and that the configuration it selects still produces a sane result.
    It is held to :data:`SQNR_MIN_DB_TP8_INT4_RING`, not to the shipping floor
    -- INT4 at TP8 is ~3 dB under that by construction, which is the whole
    reason the default is INT6 there.
    """
    ranks = _spawn(
        8, [(tokens, hidden)], time_it=False, algorithm="ring", codec="int4"
    )
    row = _assert_sqnr(
        ranks,
        tokens=tokens,
        hidden=hidden,
        world_size=8,
        label=f"{label}/ring-int4",
        algorithm="ring",
        floor=SQNR_MIN_DB_TP8_INT4_RING,
    )
    assert row["rs_codec"] == "int4" and row["ag_codec"] == "int4", row


@pytest.mark.parametrize("algorithm", ("mesh", "ring"))
def test_qr_int4_degenerate_inputs(algorithm):
    """All-zero and all-tiny groups must not produce NaN.

    A group whose extremum is zero decodes to a zero scale, so the encode
    reciprocal saturates; before it was clamped, that reached the codec as Inf
    and ``0 * Inf`` poisoned the tile. INT6 quadruples the reciprocal for a
    given extremum, so it has four times less headroom here than INT4.
    """
    ranks = _spawn(
        2, [(512, 5120)], time_it=False, algorithm=algorithm, fill="degenerate"
    )
    for rank, rows in enumerate(ranks):
        assert rows, f"rank {rank}: no rows"
        for row in rows:
            assert math.isfinite(row["rel_mae"]), f"rank {rank}: {row}"


# Transport-in-isolation tests: codec="fp16" is a lossless passthrough wire
# format, so these gate on closeness to the reference rather than on SQNR. 
# They exercise chunk/slot addressing, the flag protocol, the super-tile loop 
# and the accumulate order.
#
# (world_size, tokens, hidden, super_tile, label). ``super_tile`` is pinned
# explicitly (via pin_super_tile=True below) for both schedules, including
# the ring.
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


@pytest.mark.parametrize("algorithm", ("mesh", "ring"))
@pytest.mark.parametrize("world_size,tokens,hidden,super_tile,label", _EXACT_CASES)
def test_qr_transport_bit_exact(
    world_size, tokens, hidden, super_tile, label, algorithm
):
    """fp16 wire, exact-grid input: bit-identical to the fp32 reference.
    """
    ranks = _spawn(
        world_size,
        [(tokens, hidden)],
        time_it=False,
        algorithm=algorithm,
        super_tile=super_tile,
        grid_cap=64,
        codec="fp16",
        fill="exact",
        pin_super_tile=True,
    )
    _assert_exact(
        ranks,
        tokens=tokens,
        hidden=hidden,
        world_size=world_size,
        label=f"{label}/{algorithm}/fp16-exact",
        mode="exact",
    )


@pytest.mark.parametrize("algorithm", ("mesh", "ring"))
@pytest.mark.parametrize("world_size,tokens,hidden,super_tile,label", _ALLCLOSE_CASES)
def test_qr_transport_allclose(
    world_size, tokens, hidden, super_tile, label, algorithm
):
    """fp16 wire, realistic randn input: allclose to the fp32 reference.

    Tolerance is dominated by the bf16 output rounding plus fp16 accumulation.
    """
    ranks = _spawn(
        world_size,
        [(tokens, hidden)],
        time_it=False,
        algorithm=algorithm,
        super_tile=super_tile,
        grid_cap=64,
        codec="fp16",
        fill="randn",
        pin_super_tile=True,
    )
    _assert_exact(
        ranks,
        tokens=tokens,
        hidden=hidden,
        world_size=world_size,
        label=f"{label}/{algorithm}/fp16-randn",
        mode="randn",
    )


@pytest.mark.parametrize("rs_codec,ag_codec", (("fp16", "int4"), ("int4", "fp16")))
def test_qr_ring_lap_isolation(rs_codec, ag_codec):
    """Ring only, one lap fp16 and the other int4: names the guilty lap.

    With only one lap lossy, a healthy transport still clears the SQNR floor,
    so a failure here points at whichever lap is still quantized.
    """
    ranks = _spawn(
        8,
        [(512, 5120)],
        time_it=False,
        algorithm="ring",
        super_tile=1,
        grid_cap=64,
        rs_codec=rs_codec,
        ag_codec=ag_codec,
        pin_super_tile=True,
    )
    fails = []
    for rank, rows in enumerate(ranks):
        row = rows[0]
        if row["sqnr_db"] < SQNR_MIN_DB_TP8_INT4_RING:
            fails.append(
                f"rank {rank}: SQNR {row['sqnr_db']:.2f} dB < "
                f"{SQNR_MIN_DB_TP8_INT4_RING}"
            )
        if row["rs_codec"] != rs_codec or row["ag_codec"] != ag_codec:
            fails.append(
                f"rank {rank}: resolved codecs {row['rs_codec']}/{row['ag_codec']} "
                f"!= requested {rs_codec}/{ag_codec}"
            )
    assert not fails, "; ".join(fails)


@benchmark()
def test_qr_int4(tokens, hidden, dtype, tp, grid_cap=DEFAULT_GRID_CAP):
    ranks = _spawn(tp, [(tokens, hidden)], time_it=True, grid_cap=grid_cap)
    row = _assert_sqnr(
        ranks,
        tokens=tokens,
        hidden=hidden,
        world_size=tp,
        label="bench",
    )
    nbytes = tokens * hidden * 2
    # Reduce of tp ranks: (world-1) adds per element, plus INT4 codec ALU.
    flops = tokens * hidden * (tp - 1)
    us = row["us"] or 0.0
    return {
        "gfx": ARCH,
        "tp": tp,
        "grid_cap": row["grid_cap"],
        "st_used": row["st_used"],
        "flydsl us": us,
        "flydsl TFLOPS": (flops / us / 1e6) if us else 0.0,
        "flydsl TB/s": (nbytes / us / 1e6) if us else 0.0,
        "flydsl err": max(r[0]["rel_mae"] for r in ranks),
        "flydsl sqnr_db": min(r[0]["sqnr_db"] for r in ranks),
    }


test_qr_int4.__test__ = False


def main():
    if ARCH not in SUPPORTED_ARCHS:
        aiter.logger.warning("QRInt4 unsupported on %s; skipping", ARCH)
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
        help="Unused (not batch). Values other than 1 are skipped.",
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
        help="(tokens, hidden) pairs. 5120 is calibration, not ABI.\n"
        "    e.g.: -s 512,5120 9216,5120",
    )
    parser.add_argument(
        "-o",
        "--out",
        default=None,
        help="Optional JSON output path (rank-0 bench payload).",
    )
    parser.add_argument(
        "--grid-cap",
        type=int,
        default=DEFAULT_GRID_CAP,
        help="Persistent launch/inbox block cap (default 304*4=1216, matches HIP QR).",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        if dtype != dtypes.bf16:
            aiter.logger.warning("QRInt4 payload is bf16; skipping %s", dtype)
            continue
        df = []
        for tp, batch, mnk in itertools.product(args.tp, args.batch, args.mnk):
            if batch != 1:
                continue
            if tp not in SUPPORTED_WORLDS:
                aiter.logger.warning("QRInt4 unsupported world_size=%s; skipping", tp)
                continue
            if n_gpu < tp:
                aiter.logger.warning(
                    "QRInt4 needs %s GPUs, have %s; skipping tp=%s",
                    tp,
                    n_gpu,
                    tp,
                )
                continue
            if not isinstance(mnk, tuple) or len(mnk) < 2:
                raise ValueError(f"-s expects tokens,hidden; got {mnk!r}")
            tokens, hidden = int(mnk[0]), int(mnk[1])
            df.append(test_qr_int4(tokens, hidden, dtype, tp, grid_cap=args.grid_cap))
        if df:
            table = pd.DataFrame(df)
            aiter.logger.info(
                "flydsl QR INT4 summary (markdown):\n%s",
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
                                "algorithm": args.algorithm,
                                "timer": "run_perftest",
                            },
                            "rows": df,
                        },
                        fh,
                        indent=2,
                        default=str,
                    )
                aiter.logger.info("wrote %s", args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--init-method", default=None)
    parser.add_argument("--tokens", default="")
    parser.add_argument("--hiddens", default="")
    parser.add_argument("--algorithm", default="mesh")
    parser.add_argument("--super-tile", type=int, default=SUPER_TILE)
    parser.add_argument("--grid-cap", type=int, default=DEFAULT_GRID_CAP)
    parser.add_argument(
        "--fill", default="randn", choices=("randn", "degenerate", "exact")
    )
    parser.add_argument("--rs-codec", default=None)
    parser.add_argument("--ag-codec", default=None)
    parser.add_argument("--pin-super-tile", action="store_true")
    parser.add_argument("--time-it", action="store_true")
    parser.add_argument("--out", default=None)
    known, rest = parser.parse_known_args()
    if known.rank is not None:
        rank_parser = argparse.ArgumentParser()
        rank_parser.add_argument("--tp", type=int, default=TP)
        rank_args, _ = rank_parser.parse_known_args(rest)
        known.tp = rank_args.tp
        known.tokens = [int(t) for t in known.tokens.split(",") if t]
        known.hiddens = [int(h) for h in known.hiddens.split(",") if h]
        if len(known.tokens) != len(known.hiddens):
            raise SystemExit("tokens and hiddens lists must match")
        _run_rank(known)
    else:
        sys.argv = [sys.argv[0]] + rest
        main()
