# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness for the exact one-shot (1-stage) all-reduce and its fused form.

Two kernels, one schedule: ``OneShotAllReduce`` (plain) and
``OneShotAllReduceRMSNorm`` (all-reduce + residual add + RMSNorm, the drop-in
for ``aiter::allreduce_fusion_kernel_1stage<T, T, N, false>``).

Three things are checked per shape, and the second and third matter more than
the first:

1. The sum is right, against an fp32 reference, at the bf16 rounding floor.
2. The result is **bit-identical on every rank**. The kernel accumulates in a
   fixed rank order for exactly this reason, and an SQNR check cannot see an
   ordering bug -- both answers would be equally "accurate".
3. Repeated back-to-back calls stay correct under deliberate rank skew. The
   inbox is double-buffered by ``colour & 1`` and the safety argument depends
   on a straggler's read of call k finishing before anyone's push for call
   k+2; a quiescent test never exercises that. Covered by
   ``test_one_shot_allreduce_run_ahead`` and its fused counterpart.

The fused suites add a fourth, which is the strongest check here:
``residual_out`` must be **bit-exact** against the reference, not merely close.
Both sides sum in rank order in fp32 and round exactly once, so equality is the
correct expectation -- and it is the only cheap way to catch a missing bf16
round-trip before the residual add, an error SQNR on ``out`` is far too loose
to see and that compounds per layer in a real model.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
import torch

from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.jit.utils.chip_info import get_gfx_runtime

pytest.importorskip("flydsl")

from aiter.ops.flydsl.kernels.one_shot_allreduce import (
    DEFAULT_ATOMS,
    DEFAULT_FANOUT,
    DEFAULT_GRID_CAP,
    SUPPORTED_BLOCKS,
)
from aiter.ops.flydsl.kernels.quick_allreduce_shared import SUPPORTED_WORLDS
from aiter.ops.flydsl.quick_allreduce_int4 import _SUPPORTED_ARCHS

ARCH = get_gfx_runtime()

pytestmark = pytest.mark.skipif(
    ARCH not in _SUPPORTED_ARCHS,
    reason="OneShotAllReduce unsupported arch (need gfx942 or gfx950)",
)

HIDDEN = 7168
# Shapes chosen to straddle the interesting boundaries: a single 4 KiB tile,
# a partial last tile, an exact multiple, and enough tiles to force several
# per block at a small grid cap.
SHAPES = (1, 2, 3, 5, 8, 11, 16)

# The single-block corner, which HIDDEN=7168 cannot reach.
NARROW_SHAPES = ((1, 2048), (1, 1024), (1, 3072), (2, 2048))

_SHAPE_CASES = tuple((m, HIDDEN, f"m={m}") for m in SHAPES) + tuple(
    (m, hidden, f"{m}x{hidden}") for m, hidden in NARROW_SHAPES
)

RUN_AHEAD_M = 5
RUN_AHEAD_ITERS = 200
SQNR_FLOOR_DB = 45.0

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


def _sqnr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    ref = ref.double()
    err = ref - got.double()
    p = (ref * ref).mean().item()
    e = (err * err).mean().item()
    if e == 0:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(p / e)).item()


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


def _fused_inputs(m, hidden, tp, device):
    """Per-rank contributions, residual and gain. Same seed on every rank."""
    torch.manual_seed(1234 + m * 8191 + hidden)
    parts = [
        torch.randn(m, hidden, dtype=torch.bfloat16, device=device) * (r + 1)
        for r in range(tp)
    ]
    residual = torch.randn(m, hidden, dtype=torch.bfloat16, device=device)
    weight = torch.randn(hidden, dtype=torch.bfloat16, device=device)
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


def _run_rank(args) -> None:
    import torch.distributed as dist

    from aiter.ops.flydsl.one_shot_allreduce import OneShotAllReduce

    rank = args.rank
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="gloo", init_method=args.init_method, world_size=args.tp, rank=rank
    )

    if args.mode.startswith("fused"):
        _run_rank_fused(args, rank, device, dist)
        return

    eng = OneShotAllReduce(
        group=dist.group.WORLD,
        device=device,
        rank=rank,
        world_size=args.tp,
        atoms=args.atoms,
        grid_cap=args.grid_cap,
        fanout=args.fanout,
        skip_self=args.skip_self,
        block=args.block,
        # MAX_PAYLOAD_BYTES is a speed policy, not a correctness limit -- the
        # kernel is exact at every size -- so it must not decide what this
        # test covers. Lifted so the shape list stays free to include sizes
        # production would route elsewhere.
        max_bytes=1 << 30,
    )
    warm = torch.zeros(1, HIDDEN, dtype=torch.bfloat16, device=device)
    eng.compile_and_launch(warm, torch.empty_like(warm))

    def _check(m, hidden, tag):
        """One shape: accuracy against fp32, then bit-identity across ranks."""
        bad = []
        torch.manual_seed(1234 + m * 8191 + hidden)
        # Same seed on every rank, then a per-rank shift, so the reference
        # can be computed locally without another collective.
        parts = [
            torch.randn(m, hidden, dtype=torch.bfloat16, device=device) * (r + 1)
            for r in range(args.tp)
        ]
        inp = parts[rank].contiguous()
        out = torch.empty_like(inp)
        out.zero_()
        eng.allreduce(inp, out)
        torch.cuda.synchronize()

        ref = torch.zeros(m, hidden, dtype=torch.float32, device=device)
        for p in parts:
            ref += p.float()

        db = _sqnr_db(ref, out)
        if db < SQNR_FLOOR_DB:
            bad.append(
                f"{tag}: SQNR {db:.2f} dB below the {SQNR_FLOOR_DB} dB bf16 floor"
            )

        # Bit-identity across ranks: gather the raw bits, compare exactly.
        # Widened to int32 because gloo rejects int16 ("Invalid scalar
        # type"); the widening is exact, so the comparison is still on bits.
        bits = out.view(torch.int16).to(torch.int32).cpu()
        gathered = [torch.empty_like(bits) for _ in range(args.tp)]
        dist.all_gather(gathered, bits)
        for r, g in enumerate(gathered):
            if not torch.equal(g, gathered[0]):
                n = int((g != gathered[0]).sum())
                bad.append(
                    f"{tag}: rank {r} differs from rank 0 in {n} bf16 lanes "
                    "(accumulation order is not rank-stable)"
                )
                break
        return bad

    failures = []
    if args.mode == "run_ahead":
        m = args.tokens[0]
        hidden = args.hiddens[0]
        torch.manual_seed(99)
        parts = [
            torch.randn(m, hidden, dtype=torch.bfloat16, device=device) * (r + 1)
            for r in range(args.tp)
        ]
        inp = parts[rank].contiguous()
        out = torch.empty_like(inp)
        ref = torch.zeros(m, hidden, dtype=torch.float32, device=device)
        for p in parts:
            ref += p.float()
        drag = torch.randn(4096, 4096, device=device, dtype=torch.float32)
        bad = 0
        checks = 0
        for it in range(args.iters):
            # Rank 0 does unrelated work first, so it enters each call late and
            # the others get a chance to run ahead into the other parity slot.
            if rank == 0 and it % 3 == 0:
                for _ in range(3):
                    drag = drag @ drag.T * 1e-6
            eng.allreduce(inp, out)
            if it % 25 == 0:
                torch.cuda.synchronize()
                checks += 1
                if _sqnr_db(ref, out) < SQNR_FLOOR_DB:
                    bad += 1
        torch.cuda.synchronize()
        if _sqnr_db(ref, out) < SQNR_FLOOR_DB or bad:
            failures.append(f"run-ahead loop: {bad} bad checks of {checks}")
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
    """
    from aiter.ops.flydsl.one_shot_allreduce import OneShotAllReduceRMSNorm

    eng = OneShotAllReduceRMSNorm(
        group=dist.group.WORLD,
        device=device,
        rank=rank,
        world_size=args.tp,
        atoms=args.atoms,
        grid_cap=args.grid_cap,
        fanout=args.fanout,
        # As in the plain test: the payload ceiling is a speed policy, not a
        # correctness limit, so it must not decide what this test covers.
        max_bytes=1 << 30,
    )

    def _check(m, hidden, tag):
        bad = []
        parts, residual, weight = _fused_inputs(m, hidden, args.tp, device)
        inp = parts[rank].contiguous()
        # A real warm launch, so the timed/checked call below is never the one
        # that JITs. Writes into scratch, not into the buffers checked after.
        eng.compile_and_launch(inp, residual, weight, RMS_EPS)
        torch.cuda.synchronize()

        out, res_out = eng.allreduce_rmsnorm(inp, residual, weight, RMS_EPS)
        torch.cuda.synchronize()

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
    if args.mode == "fused_run_ahead":
        m, hidden = args.tokens[0], args.hiddens[0]
        parts, residual, weight = _fused_inputs(m, hidden, args.tp, device)
        inp = parts[rank].contiguous()
        out = torch.empty_like(inp)
        res_out = torch.empty_like(inp)
        eng.compile_and_launch(inp, residual, weight, RMS_EPS)
        out_ref, res_ref = _fused_reference(parts, residual, weight, RMS_EPS)
        drag = torch.randn(4096, 4096, device=device, dtype=torch.float32)
        bad = 0
        checks = 0
        for it in range(args.iters):
            # Rank 0 arrives late every third call, so the others run ahead into
            # the other parity slot -- the case the double-buffered inbox exists
            # for, and one a quiescent loop never reaches.
            if rank == 0 and it % 3 == 0:
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
            failures.append(f"fused run-ahead loop: {bad} bad checks of {checks}")
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


def _log_path(world_size: int, rank: int, mode: str, atoms: int) -> str:
    """One log per (world size, rank, mode, atoms) so concurrent spawn
    configurations cannot overwrite each other's failure tails."""
    return (
        f"/tmp/flydsl_one_shot_allreduce_tp{world_size}_{mode}_a{atoms}_rank{rank}.log"
    )


def _spawn(
    world_size: int,
    pairs: list[tuple[int, int]],
    *,
    atoms: int | None = DEFAULT_ATOMS,
    grid_cap: int | None = DEFAULT_GRID_CAP,
    fanout: str | None = DEFAULT_FANOUT,
    skip_self: bool | None = None,
    block: int | None = None,
    mode: str = "shapes",
    iters: int = RUN_AHEAD_ITERS,
) -> list:
    if world_size not in SUPPORTED_WORLDS:
        raise ValueError(f"unsupported world_size={world_size}")
    n_gpu = torch.cuda.device_count()
    if n_gpu < world_size:
        pytest.skip(f"OneShotAllReduce needs {world_size} GPUs, have {n_gpu}")
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    out_path = os.path.join(
        tempfile.mkdtemp(prefix="flydsl_one_shot_allreduce_"), "rank0.json"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        f"{_REPO_ROOT}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else _REPO_ROOT
    )
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("FLYDSL_GPU_ARCH", ARCH)
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
            "--mode",
            mode,
            "--tokens",
            tokens,
            "--hiddens",
            hiddens,
            "--iters",
            str(iters),
        ]
        # Omitted, not defaulted: OneShotAllReduce distinguishes an unset knob
        # ("walk ONESHOT_LADDER and pick by payload size") from a pinned one,
        # and only the unset form exercises _pick_cfg at all.
        if atoms is not None:
            cmd += ["--atoms", str(atoms)]
        if grid_cap is not None:
            cmd += ["--grid-cap", str(grid_cap)]
        if fanout is not None:
            cmd += ["--fanout", fanout]
        if skip_self is not None:
            cmd += ["--skip-self" if skip_self else "--no-skip-self"]
        if block is not None:
            cmd += ["--block", str(block)]
        if rank == 0:
            cmd += ["--out", out_path]
        log = open(  # noqa: SIM115
            _log_path(world_size, rank, mode, atoms),
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
            path = _log_path(world_size, rank, mode, atoms)
            try:
                with open(path) as fh:
                    tails.append(f"===== rank {rank} =====\n{fh.read()[-4000:]}")
            except OSError:
                pass
        raise RuntimeError("OneShotAllReduce ranks failed\n" + "\n".join(tails))
    with open(out_path) as fh:
        payload = json.load(fh)
    ranks = payload["ranks"]
    if len(ranks) != world_size:
        raise RuntimeError(
            f"OneShotAllReduce gathered {len(ranks)} ranks, expected {world_size}"
        )
    return ranks


# One `_spawn` call per group of shapes that share every axis baked
# into the compiled kernel as a Python-level constant.
_BATCH_CACHE: dict[tuple, dict[tuple[int, int], list]] = {}


def _index_by_shape(
    ranks: list, pairs: list[tuple[int, int]]
) -> dict[tuple[int, int], list]:
    """Reshape `_spawn`'s per-rank, per-shape bad-lists into a per-shape view.

    ``ranks[r]`` holds one bad-list per pair, in ``pairs`` order (``_run_rank``'s
    "shapes" branch appends in that order). Slicing out one shape's bad-list
    from every rank reproduces exactly what a single-shape ``_spawn`` call
    would have returned for that rank.
    """
    return {
        pair: [rank_shapes[i] for rank_shapes in ranks] for i, pair in enumerate(pairs)
    }


def _batch_cache_lookup(key: tuple, pairs: list[tuple[int, int]]) -> dict:
    """One ``_spawn`` call per `key`, all shapes computed at once.

    ``key`` is ``(world_size, mode)`` or ``(world_size, mode, atoms)``; every
    axis in it is one that forces a separate spawn.
    """
    if key not in _BATCH_CACHE:
        world_size, mode = key[0], key[1]
        atoms = key[2] if len(key) > 2 else DEFAULT_ATOMS
        ranks = _spawn(world_size, pairs, mode=mode, atoms=atoms)
        _BATCH_CACHE[key] = _index_by_shape(ranks, pairs)
    return _BATCH_CACHE[key]


@pytest.mark.parametrize("world_size", SUPPORTED_WORLDS)
@pytest.mark.parametrize("m,hidden,label", _SHAPE_CASES)
def test_one_shot_allreduce_sqnr_and_bitidentity(m, hidden, label, world_size):
    """Every shape in `_SHAPE_CASES` rides one spawn per world_size -- see `_batch_cache_lookup`."""
    group_pairs = [(mm, hh) for mm, hh, _ in _SHAPE_CASES]
    batch = _batch_cache_lookup((world_size, "shapes"), group_pairs)
    bad_per_rank = batch[(m, hidden)]
    for rank, bad in enumerate(bad_per_rank):
        assert not bad, f"{label}, tp={world_size}, rank {rank}: " + "; ".join(bad)


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
def test_one_shot_allreduce_rmsnorm_run_ahead(world_size):
    """The fused kernel under deliberate rank skew.

    The epilogue reuses one LDS buffer across every token of the grid-stride
    loop, ordered only by the barrier already inside ``_publish``. A quiescent
    single call never puts weight on that argument; this does.
    """
    ranks = _spawn(
        world_size, [(RUN_AHEAD_M, 4096)], mode="fused_run_ahead", iters=RUN_AHEAD_ITERS
    )
    for rank, bad in enumerate(ranks):
        assert not bad, f"tp={world_size}, rank {rank}: " + "; ".join(bad)


@pytest.mark.parametrize("world_size", SUPPORTED_WORLDS)
def test_one_shot_allreduce_run_ahead(world_size):
    """Many back-to-back calls with one rank deliberately late, to exercise
    the double-buffered inbox under skew rather than only at rest."""
    ranks = _spawn(world_size, [(RUN_AHEAD_M, HIDDEN)], mode="run_ahead")
    for rank, bad in enumerate(ranks):
        assert not bad, f"tp={world_size}, rank {rank}: " + "; ".join(bad)


@pytest.mark.parametrize("world_size", SUPPORTED_WORLDS)
def test_one_shot_allreduce_ladder(world_size):
    """The shipped ``ONESHOT_LADDER``, across a payload range that crosses its
    rung boundaries.

    Every other case here pins ``atoms``/``grid_cap``/``fanout``, which takes
    ``OneShotAllReduce`` down its single-rung path and leaves ``_pick_cfg``
    -- and therefore every multi-rung ladder -- uncovered. That was harmless
    while each world size had one rung; TP2 and TP4 now have two, so switching
    rungs mid-stream is live behaviour: a different engine, with its own IPC
    inbox and its own device-side colour counter, selected per payload.

    ``SHAPES`` spans 14 KiB to 224 KiB at HIDDEN=7168, which straddles both
    shipped boundaries (TP4 at 64 KiB, TP2 at 96 KiB). Run-ahead is included
    because the colour/parity state is per engine, so alternating across a
    boundary is what would expose a rung switch desynchronising the ranks.
    """
    pairs = [(m, HIDDEN) for m in SHAPES]
    ranks = _spawn(world_size, pairs, atoms=None, grid_cap=None, fanout=None)
    for rank, shape_bads in enumerate(ranks):
        bad = [b for b in shape_bads if b]
        assert not bad, f"tp={world_size}, rank {rank}: " + "; ".join(bad)

    ahead = _spawn(
        world_size,
        [(RUN_AHEAD_M, HIDDEN)],
        atoms=None,
        grid_cap=None,
        fanout=None,
        mode="run_ahead",
    )
    for rank, bad in enumerate(ahead):
        assert not bad, f"tp={world_size} run_ahead, rank {rank}: {bad}"


def main():
    if ARCH not in _SUPPORTED_ARCHS:
        print(f"OneShotAllReduce unsupported on {ARCH}; skipping")
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("-tp", type=int, default=2, choices=SUPPORTED_WORLDS)
    # Unset by default: a bare run then covers the shipped ladder, which is
    # what production walks. Pass any of them to pin a single rung instead.
    ap.add_argument("--atoms", type=int, default=None)
    ap.add_argument("--grid-cap", type=int, default=None)
    ap.add_argument("--fanout", default=None, choices=("peer", "atom"))
    ap.add_argument(
        "--skip-self",
        dest="skip_self",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    ap.add_argument("--block", type=int, default=None, choices=SUPPORTED_BLOCKS)
    ap.add_argument(
        "--plain-only",
        action="store_true",
        help="skip the fused (all-reduce + residual + RMSNorm) suites",
    )
    args = ap.parse_args()

    n = torch.cuda.device_count()
    if n < args.tp:
        raise SystemExit(f"need {args.tp} GPUs, saw {n}")

    pairs = [(m, HIDDEN) for m in SHAPES] + list(NARROW_SHAPES)
    ranks = _spawn(
        args.tp,
        pairs,
        atoms=args.atoms,
        grid_cap=args.grid_cap,
        fanout=args.fanout,
        skip_self=args.skip_self,
        block=args.block,
    )
    failures = [
        f"rank {r}: {bad}"
        for r, shape_bads in enumerate(ranks)
        for bad in shape_bads
        if bad
    ]

    run_ahead_ranks = _spawn(
        args.tp,
        [(RUN_AHEAD_M, HIDDEN)],
        atoms=args.atoms,
        grid_cap=args.grid_cap,
        fanout=args.fanout,
        skip_self=args.skip_self,
        block=args.block,
        mode="run_ahead",
    )
    failures += [f"rank {r}: {bad}" for r, bad in enumerate(run_ahead_ranks) if bad]

    if not args.plain_only:
        fused_pairs = [(m, h) for m, h, _ in _FUSED_SHAPE_CASES]
        fused_ranks = _spawn(
            args.tp,
            fused_pairs,
            atoms=args.atoms,
            grid_cap=args.grid_cap,
            fanout=args.fanout,
            mode="fused",
        )
        failures += [
            f"fused rank {r}: {bad}"
            for r, shape_bads in enumerate(fused_ranks)
            for bad in shape_bads
            if bad
        ]

        for fused_atoms, fused_hidden in FUSED_ATOMS_CASES:
            atom_ranks = _spawn(
                args.tp,
                [(m, fused_hidden) for m in (1, 8)],
                atoms=fused_atoms,
                grid_cap=args.grid_cap,
                fanout=args.fanout,
                mode="fused",
            )
            failures += [
                f"fused atoms={fused_atoms} hidden={fused_hidden} rank {r}: {bad}"
                for r, shape_bads in enumerate(atom_ranks)
                for bad in shape_bads
                if bad
            ]

        fused_ahead = _spawn(
            args.tp,
            [(RUN_AHEAD_M, 4096)],
            atoms=args.atoms,
            grid_cap=args.grid_cap,
            fanout=args.fanout,
            mode="fused_run_ahead",
        )
        failures += [
            f"fused run-ahead rank {r}: {bad}"
            for r, bad in enumerate(fused_ahead)
            if bad
        ]

    if failures:
        print("FAIL\n  " + "\n  ".join(failures))
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--init-method", default=None)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--atoms", type=int, default=None)
    parser.add_argument("--grid-cap", type=int, default=None)
    parser.add_argument("--fanout", default=None, choices=("peer", "atom"))
    parser.add_argument(
        "--skip-self",
        dest="skip_self",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--block", type=int, default=None)
    parser.add_argument(
        "--mode",
        default="shapes",
        choices=("shapes", "run_ahead", "fused", "fused_run_ahead"),
    )
    parser.add_argument("--tokens", default="")
    parser.add_argument("--hiddens", default="")
    parser.add_argument("--iters", type=int, default=RUN_AHEAD_ITERS)
    parser.add_argument("--out", default=None)
    known, rest = parser.parse_known_args()
    if known.rank is not None:
        known.tokens = [int(t) for t in known.tokens.split(",") if t]
        known.hiddens = [int(h) for h in known.hiddens.split(",") if h]
        if len(known.tokens) != len(known.hiddens):
            raise SystemExit("tokens and hiddens lists must match")
        _run_rank(known)
    else:
        sys.argv = [sys.argv[0]] + rest
        main()
