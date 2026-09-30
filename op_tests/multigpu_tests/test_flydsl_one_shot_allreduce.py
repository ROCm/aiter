# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and timing for the exact one-shot (1-stage) all-reduce and its
fused form (``OneShotAllReduceRMSNorm``).

Two kernels, one schedule: ``OneShotAllReduce`` (plain) and
``OneShotAllReduceRMSNorm`` (all-reduce + residual add + RMSNorm, the drop-in
for ``aiter::allreduce_fusion_kernel_1stage<T, T, N, false>``).

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

The fused suites add a fourth check: ``residual_out`` must be **bit-exact**
against the reference, not merely close. Both sides sum in rank order in fp32
and round exactly once, so equality is the correct expectation -- and it is the
only cheap way to catch a missing bf16 round-trip before the residual add, an
error SQNR on ``out`` is far too loose to see and that compounds per layer in a
real model.

``--extended`` runs the spot-check configurations at every world size, and the
spot-check shapes on the production engine too.

Every rank is a ``multiprocessing`` spawn worker (plain tests) or a subprocess
(fused tests); one spawn per engine configuration and world size runs every
sweep. OneShotAllReduce runs on gfx942/gfx950 at TP in {2, 4, 8}; other archs
skip, and ``main()`` skips a world size when fewer GPUs are visible than TP.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time
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
    DEFAULT_ATOMS,
    DEFAULT_FANOUT,
    DEFAULT_GRID_CAP,
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
FUSED_PAD_HIDDENS = (896, 2304, 2880)
_FUSED_PAD_SHAPE_CASES = tuple(
    (m, hidden, f"{m}x{hidden}pad")
    for hidden in FUSED_PAD_HIDDENS
    for m in (1, 5, 32)
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

# Self-skip on the fused kernel: this rank's contribution comes out of registers
# instead of out of its own inbox.
FUSED_SKIP_SELF_CASES = ((1, 4096), (8, 7168), (32, 8192))

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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Fused (subprocess) helpers -- used by OneShotAllReduceRMSNorm test suite
# ---------------------------------------------------------------------------


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
    atoms: int | None = DEFAULT_ATOMS,
    grid_cap: int | None = DEFAULT_GRID_CAP,
    fanout: str | None = DEFAULT_FANOUT,
    skip_self: bool | None = None,
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
    if skip_self is not None:
        tag += "_ss" if skip_self else "_noss"
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
        if skip_self is not None:
            cmd += ["--skip-self" if skip_self else "--no-skip-self"]
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
    deadline = time.time() + float(os.environ.get("FLYDSL_QR_TIMEOUT", str(SPAWN_TIMEOUT_S)))
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
        raise RuntimeError(
            "OneShotAllReduceRMSNorm ranks failed\n" + "\n".join(tails)
        )
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
    the exchange and reading it, so its siblings run ahead into the other
    parity's word.
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
        skip_self=args.skip_self,
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
            # the other parity slot -- the case the double-buffered inbox exists
            # for, and one a quiescent loop never reaches. (Slice lag skews the
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
        spawn_kwargs.setdefault("atoms", key[2] if len(key) > 2 else DEFAULT_ATOMS)
        ranks = _fused_spawn(world_size, pairs, mode=mode, **spawn_kwargs)
        _BATCH_CACHE[key] = _index_by_shape(ranks, pairs)
    return _BATCH_CACHE[key]


# ---------------------------------------------------------------------------
# Plain (pool-based) scaffolding -- used by OneShotAllReduce test suite
# ---------------------------------------------------------------------------


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
    eng.preload()
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


def _spawn_pool(
    world_size: int,
    engine_kw: dict,
    cases: list[tuple],
    window: tuple[int, int] | None = None,
) -> list[dict]:
    """Pool-based spawn for plain ``OneShotAllReduce`` tests.

    Returns a list of dicts (one per rank) each with ``rows``, ``run_ahead``
    and ``production_cfgs`` keys.
    """
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
        _RESULTS[key] = _spawn_pool(key[0], dict(key[1]), _CASES[key], _WINDOWS.get(key))
    return _RESULTS[key]


def _check(label: str, fails: list[str]) -> None:
    if fails:
        msg = f"{label}: " + "; ".join(fails)
        aiter.logger.error(msg)
        _FAILURES.append(msg)


# ---------------------------------------------------------------------------
# Plain test functions
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Fused test functions
# ---------------------------------------------------------------------------


import pytest

pytestmark = pytest.mark.skipif(
    ARCH not in SUPPORTED_ARCHS,
    reason="OneShotAllReduce unsupported arch (need gfx942 or gfx950)",
)


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
    batch = _batch_cache_lookup(
        (world_size, "fused", DEFAULT_ATOMS, "pad"), group_pairs
    )
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
    from aiter.ops.flydsl.kernels.one_shot_allreduce import (
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


def test_one_shot_allreduce_rmsnorm_geom_for_pinned_block_pads():
    """A pinned block a width lacks natively resolves through the padded set.

    ``OneShotAllReduceRMSNorm._geom_for`` is the resolver ``supports_hidden``
    and the launch path both go through. The two must agree on a pinned block:
    the bench sweeps ``block=1024``, which hidden=4096 admits only via padding
    (its native blocks are 512/256/128). An earlier ``_geom_for`` sent every
    pinned block through the native-only ``fused_atoms_for_block`` and raised on
    exactly this pair, so ``supports_hidden`` said yes while the warm launch
    crashed.

    ``_geom_for`` reads only ``self.block``/``self.pad``/``self._ladder``, so it
    is exercised on a bare instance -- host-side and GPU-free, no process group.
    """
    from aiter.ops.flydsl.kernels.one_shot_allreduce import fused_block_options
    from aiter.ops.flydsl.one_shot_allreduce import OneShotAllReduceRMSNorm

    def geom(block, pad, hidden, rung_atoms=1):
        eng = OneShotAllReduceRMSNorm.__new__(OneShotAllReduceRMSNorm)
        eng.block = block
        eng.pad = pad
        return OneShotAllReduceRMSNorm._geom_for(eng, hidden, rung_atoms)

    # 4096 admits 512/256/128 natively; 1024 only by padding to h_pad=8192.
    assert 1024 not in [b for b, _ in fused_block_options(4096)]

    # The regression: pinned non-native block, padding on -> resolves, no raise.
    atoms, h_pad, split = geom(1024, True, 4096)
    assert h_pad == 8192 and 1024 * atoms * 8 == h_pad and split == 1

    # A pinned block the width *does* admit natively stays native (h_pad==hidden).
    assert geom(512, True, 4096) == (1, 4096, 1)

    # supports_hidden and _geom_for must give the same yes/no on the pin. When
    # supports_hidden says yes, _geom_for must return a geometry that honours
    # the pin rather than raising.
    for block in (128, 256, 512, 1024):
        eng = OneShotAllReduceRMSNorm.__new__(OneShotAllReduceRMSNorm)
        eng.block = block
        eng.pad = True
        if eng.supports_hidden(4096):
            atoms, h_pad, _split = OneShotAllReduceRMSNorm._geom_for(eng, 4096, 1)
            assert block * atoms * 8 == h_pad, (block, atoms, h_pad)

    # Padding off: a pinned non-native block has no geometry, so _geom_for hands
    # back the rung atoms and lets the build raise -- it must not resolve to a
    # padded width behind the caller's back.
    assert geom(1024, False, 4096) == (1, 4096, 1)


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
@pytest.mark.parametrize("m,hidden", FUSED_SKIP_SELF_CASES)
def test_one_shot_allreduce_rmsnorm_skip_self(m, hidden, world_size):
    """Self-skip under the fused epilogue.

    The plain kernel's self-skip cases cover the wire -- one fewer store, one
    fewer flag, a remapped spin lane. What is specific here is that the fused
    epilogue consumes the *unrounded* fp32 accumulator, so it is the path where
    substituting the register copy for the inbox copy could change a rounding.
    ``residual_out`` is bit-exact against the reference, so it cannot.
    """
    pairs = [(m, hidden)]
    batch = _batch_cache_lookup(
        (world_size, "fused", "ss", m, hidden),
        pairs,
        # Explicit: key[2] is a label here, not an atoms value, so the
        # positional inference in _batch_cache_lookup must not be relied on.
        atoms=DEFAULT_ATOMS,
        skip_self=True,
    )
    for rank, bad in enumerate(batch[(m, hidden)]):
        assert (
            not bad
        ), f"{m}x{hidden} skip_self, tp={world_size}, rank {rank}: " + "; ".join(bad)


@pytest.mark.parametrize("world_size", SUPPORTED_WORLDS)
def test_one_shot_allreduce_rmsnorm_run_ahead(world_size):
    """The fused kernel under deliberate rank skew.

    The epilogue reuses one LDS buffer across every token of the grid-stride
    loop, ordered only by the barrier already inside ``_publish``. A quiescent
    single call never puts weight on that argument; this does.
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
    other parity's word. That can only happen within a launch, so the grid cap
    is pinned to two row groups: at M=8 every workgroup handles four rows per
    call, crossing parities three times.
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
    from aiter.ops.flydsl.kernels.one_shot_allreduce import (
        SUPPORTED_SPLITS,
        fused_block_options,
        fused_split_options,
    )
    from aiter.ops.flydsl.kernels.quick_allreduce_fusions import XCHG_COUNT_BITS

    def splits(hidden):
        return {k for k, _b, _a in fused_split_options(hidden)}

    assert splits(7168) == {1, 2, 7, 14}
    assert splits(8192) == {1, 2, 4, 8, 16}
    assert splits(4096) == {1, 2, 4, 8}
    for hidden in range(512, 16385, 512):
        opts = fused_split_options(hidden)
        # split=1 is exactly the unsplit geometry, in the same order.
        assert [(b, a) for k, b, a in opts if k == 1] == list(fused_block_options(hidden))
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
    from aiter.ops.flydsl.kernels.one_shot_allreduce import (
        make_one_shot_allreduce_kernel as mk,
    )

    base = dict(world_size=4, grid=64, fusion="rmsnorm", hidden=7168, atoms=1)
    with pytest.raises(ValueError, match="only meaningful for a fused"):
        mk(world_size=4, grid=64, split=2)
    with pytest.raises(ValueError, match="split must be one of"):
        mk(**base, split=3)
    with pytest.raises(ValueError, match="needs a native geometry"):
        mk(world_size=4, grid=64, fusion="rmsnorm", hidden=3072, h_pad=4096, atoms=2, split=2)
    with pytest.raises(ValueError, match="does not divide"):
        mk(world_size=4, grid=63, fusion="rmsnorm", hidden=4096, atoms=1, split=7)
    with pytest.raises(ValueError, match="not a multiple of split"):
        mk(**base, split=7)  # 64 % 7
    with pytest.raises(ValueError, match="needs a split build"):
        mk(**base, debug_slice_delay=2)
    # 16 slices of 16 waves: 256 writers, one more than the arrival count holds.
    with pytest.raises(ValueError, match="counts arrivals"):
        mk(world_size=4, grid=64, fusion="rmsnorm", hidden=131072, atoms=1, split=16)

    spec = mk(**base, split=2)
    assert (spec["split"], spec["block"], spec["tile_bytes"]) == (2, 448, 7168)
    # No LDS: a split build reduces across waves through HBM.
    assert spec["lds_bytes"] == 0
    # Words: 2 parities x 32 groups, a line each; prev: 64 workgroups x 2 x 8 B.
    assert spec["xchg_bytes"] == 2 * 32 * 128 + 64 * 2 * 8
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

    def bare(block=None, pad=True):
        eng = E.__new__(E)
        eng.block = block
        eng.pad = pad
        return eng

    def geom(hidden, rung_atoms, rung_split, block=None):
        return E._geom_for(bare(block), hidden, rung_atoms, rung_split)

    assert geom(7168, 2, 4) == (1, 7168, 2)  # 7168 has {2, 7, 14}
    assert geom(8192, 2, 4) == (2, 8192, 4)
    assert geom(7168, 1, 7, block=128) == (1, 7168, 7)  # block is the slice's
    assert geom(4096, 2, 1) == (2, 4096, 1)
    assert geom(3000, 2, 2)[2] == 1  # padded: unsplit

    key = E._cfg_key(bare(), 7168, (0, 1, 64, "peer", False, 7))
    assert (key[2], key[6]) == (63, 7)  # cap 64 -> 9 groups of 7
    assert E._cfg_key(bare(), 7168, (0, 2, 64, "peer", False, 1))[2] == 64

    row = lambda h: h * 2  # noqa: E731 -- bytes per token
    assert E._tiles_and_grid({"grid": 63, "split": 7}, 7168, 3 * row(7168)) == (21, 21)
    assert E._tiles_and_grid({"grid": 64, "split": 4}, 8192, 40 * row(8192)) == (160, 64)
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

    from aiter.ops.flydsl.kernels.quick_allreduce_fusions import (
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
        u = partial if partial < clamp else clamp
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
                        (0.0, rng.random() * 10.0 ** rng.randint(-6, 12), 1e30,
                         float("inf"), float("nan"))
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


def test_one_shot_allreduce_rmsnorm_ladder_rungs():
    """Every fused rung carries a split, and a legal one."""
    from aiter.ops.flydsl.kernels.one_shot_allreduce import (
        FUSED_ONESHOT_LADDER,
        SUPPORTED_SPLITS,
        fused_oneshot_ladder,
    )

    rungs = [r for v in FUSED_ONESHOT_LADDER.values() for r in v]
    rungs += list(fused_oneshot_ladder(3, "pcie"))  # the unlisted default
    for rung in rungs:
        assert len(rung) == 6 and rung[5] in SUPPORTED_SPLITS, rung


# ---------------------------------------------------------------------------
# Summary helper and main()
# ---------------------------------------------------------------------------


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
    parser.add_argument(
        "--plain-only",
        action="store_true",
        help="skip the fused (all-reduce + residual + RMSNorm) suites",
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

    if not args.plain_only:
        for tp in tps:
            fused_pairs = [(m, h) for m, h, _ in _FUSED_SHAPE_CASES]
            fused_ranks = _fused_spawn(
                tp,
                fused_pairs,
                atoms=DEFAULT_ATOMS,
                grid_cap=DEFAULT_GRID_CAP,
                fanout=DEFAULT_FANOUT,
                mode="fused",
            )
            fused_fails = [
                f"fused tp={tp} rank {r}: {bad}"
                for r, shape_bads in enumerate(fused_ranks)
                for bad in shape_bads
                if bad
            ]
            if fused_fails:
                _FAILURES.extend(fused_fails)
            aiter.logger.info("fused tp=%s: %d failures", tp, len(fused_fails))

    if _FAILURES:
        raise SystemExit(
            f"{len(_FAILURES)} OneShotAllReduce check(s) failed:\n  "
            + "\n  ".join(_FAILURES)
        )


if __name__ == "__main__":
    freeze_support()
    # When relaunched as a subprocess rank worker (fused tests), dispatch
    # directly to the rank entry point and exit. Otherwise run main().
    _rank_parser = argparse.ArgumentParser(add_help=False)
    _rank_parser.add_argument("--rank", type=int, default=None)
    _rank_parser.add_argument("--init-method", default=None)
    _rank_parser.add_argument("--tp", type=int, default=2)
    _rank_parser.add_argument("--atoms", type=int, default=None)
    _rank_parser.add_argument("--grid-cap", type=int, default=None)
    _rank_parser.add_argument("--fanout", default=None, choices=("peer", "atom"))
    _rank_parser.add_argument(
        "--skip-self",
        dest="skip_self",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    _rank_parser.add_argument("--block", type=int, default=None)
    _rank_parser.add_argument("--split", type=int, default=None)
    _rank_parser.add_argument("--input-scale", type=float, default=1.0)
    _rank_parser.add_argument("--repeat", type=int, default=1)
    _rank_parser.add_argument(
        "--mode",
        default="fused",
        choices=("fused", "fused_run_ahead", "fused_slice_lag"),
    )
    _rank_parser.add_argument("--tokens", default="")
    _rank_parser.add_argument("--hiddens", default="")
    _rank_parser.add_argument("--iters", type=int, default=RUN_AHEAD_ITERS)
    _rank_parser.add_argument("--out", default=None)
    known, rest = _rank_parser.parse_known_args()
    if known.rank is not None:
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
        sys.argv = [sys.argv[0]] + rest
        main()
        end = perf_counter()
        aiter.logger.info(f"Test execution took {end - start:.2f}s")
