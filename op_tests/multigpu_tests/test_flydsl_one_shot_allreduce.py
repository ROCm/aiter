# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness for the exact one-shot (1-stage) all-reduce (``OneShotAllReduce``).


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
   ``test_one_shot_allreduce_run_ahead``.

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


def _sqnr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    ref = ref.double()
    err = ref - got.double()
    p = (ref * ref).mean().item()
    e = (err * err).mean().item()
    if e == 0:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(p / e)).item()


def _run_rank(args) -> None:
    import torch.distributed as dist

    from aiter.ops.flydsl.one_shot_allreduce import OneShotAllReduce

    rank = args.rank
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="gloo", init_method=args.init_method, world_size=args.tp, rank=rank
    )

    eng = OneShotAllReduce(
        group=dist.group.WORLD,
        device=device,
        rank=rank,
        world_size=args.tp,
        atoms=args.atoms,
        grid_cap=args.grid_cap,
        fanout=args.fanout,
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


def _spawn(
    world_size: int,
    pairs: list[tuple[int, int]],
    *,
    atoms: int = DEFAULT_ATOMS,
    grid_cap: int = DEFAULT_GRID_CAP,
    fanout: str = DEFAULT_FANOUT,
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
            "--atoms",
            str(atoms),
            "--grid-cap",
            str(grid_cap),
            "--fanout",
            fanout,
            "--mode",
            mode,
            "--tokens",
            tokens,
            "--hiddens",
            hiddens,
            "--iters",
            str(iters),
        ]
        if rank == 0:
            cmd += ["--out", out_path]
        log = open(  # noqa: SIM115
            f"/tmp/flydsl_one_shot_allreduce_tp{world_size}_rank{rank}.log",
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
            path = f"/tmp/flydsl_one_shot_allreduce_tp{world_size}_rank{rank}.log"
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
    """One ``_spawn`` call per `key`, all shapes computed at once."""
    if key not in _BATCH_CACHE:
        ranks = _spawn(key[0], pairs)
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
def test_one_shot_allreduce_run_ahead(world_size):
    """Many back-to-back calls with one rank deliberately late, to exercise
    the double-buffered inbox under skew rather than only at rest."""
    ranks = _spawn(world_size, [(RUN_AHEAD_M, HIDDEN)], mode="run_ahead")
    for rank, bad in enumerate(ranks):
        assert not bad, f"tp={world_size}, rank {rank}: " + "; ".join(bad)


def main():
    if ARCH not in _SUPPORTED_ARCHS:
        print(f"OneShotAllReduce unsupported on {ARCH}; skipping")
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("-tp", type=int, default=2, choices=SUPPORTED_WORLDS)
    ap.add_argument("--atoms", type=int, default=DEFAULT_ATOMS)
    ap.add_argument("--grid-cap", type=int, default=DEFAULT_GRID_CAP)
    ap.add_argument("--fanout", default=DEFAULT_FANOUT, choices=("peer", "atom"))
    args = ap.parse_args()

    n = torch.cuda.device_count()
    if n < args.tp:
        raise SystemExit(f"need {args.tp} GPUs, saw {n}")

    pairs = [(m, HIDDEN) for m in SHAPES] + list(NARROW_SHAPES)
    ranks = _spawn(
        args.tp, pairs, atoms=args.atoms, grid_cap=args.grid_cap, fanout=args.fanout
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
        mode="run_ahead",
    )
    failures += [f"rank {r}: {bad}" for r, bad in enumerate(run_ahead_ranks) if bad]

    if failures:
        print("FAIL\n  " + "\n  ".join(failures))
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--init-method", default=None)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--atoms", type=int, default=DEFAULT_ATOMS)
    parser.add_argument("--grid-cap", type=int, default=DEFAULT_GRID_CAP)
    parser.add_argument("--fanout", default=DEFAULT_FANOUT, choices=("peer", "atom"))
    parser.add_argument("--mode", default="shapes", choices=("shapes", "run_ahead"))
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
