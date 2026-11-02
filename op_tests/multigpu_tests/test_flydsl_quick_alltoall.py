# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness, accuracy and timing of the FlyDSL all-to-all (``FlyQuickAllToAll``).

Equal-split semantics: rank ``r``'s output chunk ``i`` is rank ``i``'s input
chunk ``r``. Every rank generates every peer's input from the same seed, so the
reference is built locally with no collective; one case per spawn is also
checked against ``torch.distributed.all_to_all_single``.

``--scenario`` selects what a run does: ``verify`` (the default) checks the
kernels, and ``bench`` times them against RCCL.

**verify**, each sweep ending in a markdown table:

* ``test_quick_alltoall`` -- each schedule (mesh, ring) and wire format at
  payloads that hit the sub-tile path, a partial last tile, both ladder rungs,
  and the 64 MiB production size. ``none``/``fp16`` must be bit-exact; INT4 and
  INT6 gate on SQNR, on a bit-exact self chunk and on finite output.
* ``test_quick_alltoall_protocol`` -- the inbox reuse rules: back-to-back
  launches with one rank held back (``torch.cuda._sleep``), launches straddling
  the colour wrap, and CUDA-graph replay.
* ``test_quick_alltoall_dispatcher`` -- the production path,
  ``GroupCoordinator.all_to_all`` with ``AITER_FLY_A2A=1``: routing on both
  sides of the window floor; a first call made inside a HIP graph capture, with
  no warm-up, which only works if the engines were compiled and loaded at init;
  and one rank with ``AITER_FLY_A2A=0``, where every rank must decline the slot
  together rather than hang in an engine collective.
* ``test_quick_alltoall_vllm`` -- ``QuickAllToAll`` built directly on gloo
  subgroups with ``enable=True, codec="none"``, moving packed buffers of random
  bits, as vLLM's DCP all-to-all does.

**bench** times every candidate on the same per-rank input, as the all-reduce
benches do (``flydsl_allreduce_bench.py`` describes the method): each candidate
is captured into a CUDA graph of back-to-back calls, graded on a replay, and
timed in rotated rounds; a candidate's time is its slowest rank's. The
candidates are ``rccl`` (PyNccl grouped send/recv, the dispatcher's fallback),
``rccl_c10d`` (``dist.all_to_all_single``), the FlyDSL schedules pinned per
wire format, and ``fly_auto``, the production dispatcher. ``TB/s`` is bus
bandwidth in the nccl-tests all-to-all convention, ``algbw * (N-1) / N``.

FlyQuickAllToAll runs on gfx942/gfx950 at TP in {2, 4, 8}; ``main()`` skips
other archs, and world sizes with fewer GPUs than ranks.
"""

from __future__ import annotations

import argparse
import ctypes
import itertools
import os
import sys
from multiprocessing import Pool, set_start_method

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _path in (_REPO_ROOT, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import flydsl_allreduce_bench as fab
import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.dist.utils import (
    get_distributed_init_method,
    get_ip,
    get_open_port,
)
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.alltoall_policy import (
    A2A_QUANT_CODECS,
    SUPPORTED_ARCHS,
    SUPPORTED_WORLDS,
)
from aiter.ops.flydsl.kernels.comm.transport import ATOMS
from aiter.test_common import benchmark, run_perftest

# Rank workers initialize HIP, which a forked child cannot.
set_start_method("spawn", force=True)

ALGORITHMS = ("mesh", "ring")
CODECS = ("none", "fp16", "int4", "int6")
LOSSLESS = ("none", "fp16")

# Single quantization per element (no reduction), so a codec's floor is its own
# round-trip SQNR: INT4 lands ~19-20 dB, INT6 ~31 dB on Gaussian data.
SQNR_FLOOR_DB = {"int4": 15.0, "int6": 26.0}
# Worst tile, so one unwritten tile cannot be averaged away.
TILE_SQNR_FLOOR_DB = {"int4": 8.0, "int6": 18.0}

# The production message: 64 MiB of bf16.
TOP_PAYLOAD_BYTES = 64 << 20
PROTOCOL_LAUNCHES = 8
# Colour to seed every block with before the wrap case: -3, -2, -1, then the
# skip to 2. Parities 1, 0, 1, 0 -- the wrap must keep them alternating.
WRAP_SEED = -3
GRAPH_REPLAYS = 4
# "wide" stretches every group towards the codec's ceiling without crossing
# it: the group-16 scale is a signed E4M3, so a group whose extremum is past
# 480 saturates -- by design, and covered by the codec's own tests.
FILLS = ("randn", "zeros", "wide")
WIDE_SCALE = 64.0

# Seconds to wait for each rank of a spawn. The kernels spin on flags written
# by peers, so a protocol bug is a hang rather than an error; this turns it
# into a failure.
SPAWN_TIMEOUT_S = 600


# --------------------------------------------------------------------------- #
# Spawning, grading and reporting
# --------------------------------------------------------------------------- #
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


class SpawnRegistry:
    """Cases grouped by the engine they need, so each engine spawns once.

    A key is ``(tp, sorted engine kwargs)``; a case is whatever tuple the rank
    worker understands. Register every case first, then read results: the
    first read of a key spawns its engine with every case registered for it.
    """

    def __init__(self, rank_fn):
        self._rank_fn = rank_fn
        self.cases: dict[tuple, list] = {}
        self._ranks: dict[tuple, list] = {}

    @staticmethod
    def key(tp: int, **engine_kw) -> tuple:
        return (tp, tuple(sorted(engine_kw.items())))

    def register(self, key: tuple, case) -> None:
        cases = self.cases.setdefault(key, [])
        if case not in cases:
            cases.append(case)

    def result(self, key: tuple, case) -> list:
        """Per-rank rows for *case*, spawning the key's engine on first use."""
        if key not in self._ranks:
            self._ranks[key] = _spawn(
                self._rank_fn, key[0], engine_kw=dict(key[1]), cases=self.cases[key]
            )
        i = self.cases[key].index(case)
        return [rank_rows[i] for rank_rows in self._ranks[key]]


class FailureLog:
    """Collects gate failures so a sweep reports all of them, then fails once."""

    def __init__(self):
        self.failures: list[str] = []

    def check(self, label: str, fails: list[str]) -> None:
        if fails:
            msg = f"{label}: " + "; ".join(fails)
            aiter.logger.error(msg)
            self.failures.append(msg)

    def raise_if_failed(self, what: str) -> None:
        """Exit non-zero naming every failure, if there were any."""
        if self.failures:
            raise SystemExit(
                f"{len(self.failures)} {what} check(s) failed:\n  "
                + "\n  ".join(self.failures)
            )


failures = FailureLog()


def _sqnr(ref_pow: torch.Tensor, mse: torch.Tensor) -> torch.Tensor:
    """``10 log10(ref_pow / mse)``: +inf when both are zero, -inf on NaN."""
    score = torch.where(
        (ref_pow <= 0) & (mse <= 0),
        torch.full_like(mse, float("inf")),
        10.0 * torch.log10(ref_pow / mse),
    )
    return torch.nan_to_num(
        score, nan=float("-inf"), posinf=float("inf"), neginf=float("-inf")
    )


def _sqnr_db(got: torch.Tensor, reference: torch.Tensor) -> float:
    got = got.to(torch.float32)
    reference = reference.to(torch.float32)
    return float(
        _sqnr((reference * reference).mean(), ((got - reference) ** 2).mean()).item()
    )


def _min_tile_sqnr_db(got: torch.Tensor, reference: torch.Tensor, tile_bytes) -> float:
    """Worst per-tile SQNR, so one unwritten tile cannot be averaged away."""
    tile_elems = tile_bytes // got.element_size()
    g = got.reshape(-1).to(torch.float32)
    r = reference.reshape(-1).to(torch.float32)
    n_full = (g.numel() // tile_elems) * tile_elems
    vals = []
    if n_full:
        gt = g[:n_full].view(-1, tile_elems)
        rt = r[:n_full].view(-1, tile_elems)
        vals.append(
            float(_sqnr((rt * rt).mean(dim=1), ((gt - rt) ** 2).mean(dim=1)).min())
        )
    if g.numel() > n_full:
        vals.append(_sqnr_db(g[n_full:], r[n_full:]))
    return min(vals)


def _lanes_differing(got: torch.Tensor, reference: torch.Tensor) -> int:
    """16-bit lanes where *got* differs, bit for bit, from *reference*."""
    return int((got.view(torch.int16) != reference.view(torch.int16)).sum())


def _fmt_bytes(nbytes: int) -> str:
    for unit, shift in (("MiB", 20), ("KiB", 10)):
        if nbytes >= 1 << shift:
            return f"{nbytes / (1 << shift):g} {unit}"
    return f"{nbytes} B"


def _busbw_tbps(nbytes: int, tp: int, us: float | None) -> float | None:
    """Bus bandwidth in the nccl-tests all-to-all convention, in TB/s."""
    return None if us is None else nbytes * (tp - 1) / tp / us / 1e6


def summarize(name: str, rows: list[dict]) -> None:
    if rows:
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "%s summary (markdown):\n%s", name, df.to_markdown(index=False)
        )


# --------------------------------------------------------------------------- #
# Inputs and the reference
# --------------------------------------------------------------------------- #
def _block_tile_bytes(tp: int, block: int = 256) -> int:
    """HBM bytes of one tile of one chunk, at *block*."""
    return ATOMS // tp * block * 16


def payloads(tp: int) -> list[int]:
    """Sub-tile, partial tile, both ladder rungs, and the production size."""
    tile = _block_tile_bytes(tp)
    return [tp * 16, tp * (tile + 48), 1 << 20, 16 << 20, TOP_PAYLOAD_BYTES]


def _make_inp(rank: int, nbytes: int, fill: str, device) -> torch.Tensor:
    """Rank *rank*'s input, identical whichever rank generates it."""
    n = nbytes // 2
    gen = torch.Generator(device=device).manual_seed(1234 + rank)
    if fill == "zeros":
        return torch.zeros(n, dtype=torch.bfloat16, device=device)
    x = torch.randn(n, generator=gen, dtype=torch.float32, device=device)
    if fill == "wide":
        x = x * WIDE_SCALE
    return x.to(torch.bfloat16)


def _reference(rank: int, tp: int, nbytes: int, fill: str, device) -> torch.Tensor:
    """What rank *rank* must hold afterwards: chunk *rank* of every rank."""
    chunk = nbytes // 2 // tp
    return torch.cat(
        [
            _make_inp(src, nbytes, fill, device)[rank * chunk : (rank + 1) * chunk]
            for src in range(tp)
        ]
    )


def _grade(out, ref, rank, tp, tile_bytes) -> dict:
    chunk = out.numel() // tp
    own = slice(rank * chunk, (rank + 1) * chunk)
    return {
        "n_mismatch": _lanes_differing(out, ref),
        "self_mismatch": _lanes_differing(out[own], ref[own]),
        "finite": bool(torch.isfinite(out.float()).all()),
        "sqnr_db": _sqnr_db(out, ref),
        "min_tile_sqnr_db": _min_tile_sqnr_db(out, ref, tile_bytes),
    }


def _worst(a: dict | None, b: dict) -> dict:
    """Per-field worst of two grades of the same case."""
    if a is None:
        return b
    return {
        k: (
            min(a[k], v)
            if k.endswith("sqnr_db")
            else (a[k] and v) if k == "finite" else max(a[k], v)
        )
        for k, v in b.items()
    }


# --------------------------------------------------------------------------- #
# verify: the engine
# --------------------------------------------------------------------------- #
def _seed_colors(fly, value: int) -> None:
    """Overwrite every engine's per-block colours, to force the wrap."""
    from aiter.ops.flydsl.ipc_heap import UncachedIpcHeap

    torch.cuda.synchronize()
    for eng in fly._by_cfg.values():
        UncachedIpcHeap.copy_host_to_device(
            eng._colors,
            (ctypes.c_int32 * eng.grid)(*([value] * eng.grid)),
            eng.grid * 4,
        )


def _run_rank(rank, tp, init_method, engine_kw, cases) -> list[dict]:
    """One rank of one spawn: build the engine once, run every case on it.

    A case is ``(nbytes, fill, mode)``; *mode* is ``"plain"``, ``"skew"``,
    ``"wrap"`` or ``"graph"``.
    """
    import torch.distributed as dist

    from aiter.ops.flydsl.quick_alltoall import FlyQuickAllToAll

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        world_size=tp,
        rank=rank,
        device_id=device,
    )
    # IPC metadata goes over a non-NCCL group; NCCL stays for the cross-check.
    gloo = dist.new_group(backend="gloo")
    fly = FlyQuickAllToAll(
        group=gloo, device=device, rank=rank, world_size=tp, **engine_kw
    )
    fly.preload()

    rows = []
    try:
        checked_rccl = False
        for nbytes, fill, mode in cases:
            inp = _make_inp(rank, nbytes, fill, device)
            ref = _reference(rank, tp, nbytes, fill, device)
            cfg, _ = fly._pick_cfg(nbytes)
            tile_bytes = _block_tile_bytes(tp, cfg[1])
            outs = [torch.full_like(inp, float("nan"))]
            dist.barrier()

            if mode == "plain":
                fly.all_to_all(inp, outs[0])
            elif mode in ("skew", "wrap"):
                if mode == "wrap":
                    _seed_colors(fly, WRAP_SEED)
                    dist.barrier()
                outs = [
                    torch.full_like(inp, float("nan")) for _ in range(PROTOCOL_LAUNCHES)
                ]
                for i, out in enumerate(outs):
                    # The last rank falls behind on every other launch, so its
                    # peers run ahead into the other parity's slot.
                    if mode == "skew" and rank == tp - 1 and i % 2 == 0:
                        torch.cuda._sleep(2_000_000)
                    fly.all_to_all(inp, out)
            elif mode == "graph":
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    fly.all_to_all(inp, outs[0])
                for _ in range(GRAPH_REPLAYS):
                    outs[0].fill_(float("nan"))
                    g.replay()
            torch.cuda.synchronize()
            dist.barrier()

            m = None
            for out in outs:
                m = _worst(m, _grade(out, ref, rank, tp, tile_bytes))
            if not checked_rccl and mode == "plain" and fly.codec in LOSSLESS:
                rccl = torch.empty_like(inp)
                dist.all_to_all_single(rccl, inp)
                torch.cuda.synchronize()
                m["rccl_mismatch"] = _lanes_differing(outs[0], rccl)
                checked_rccl = True

            # A smoke check of the time; the bench scenario is the measurement.
            us = None
            if mode == "plain" and nbytes >= (1 << 20):
                dist.barrier()
                _, us = run_perftest(
                    lambda src=inp, dst=outs[0]: fly.all_to_all(src, dst),
                    use_cuda_event=True,
                )
            rows.append(
                {
                    **m,
                    "variant": fly.variant(nbytes),
                    "us": None if us is None else float(us),
                }
            )
            del inp, ref, outs
            torch.cuda.empty_cache()
    finally:
        fly.close()
        dist.destroy_process_group()
    return rows


registry = SpawnRegistry(_run_rank)


def _key(tp, algorithm, codec, order="peer"):
    return registry.key(tp, algorithm=algorithm, codec=codec, order=order)


def _gate(label: str, rows: list[dict], codec: str) -> None:
    fails = []
    for r, row in enumerate(rows):
        if codec in LOSSLESS:
            if row["n_mismatch"]:
                fails.append(f"rank {r}: {row['n_mismatch']} lanes differ")
        else:
            if row["sqnr_db"] < SQNR_FLOOR_DB[codec]:
                fails.append(f"rank {r}: SQNR {row['sqnr_db']:.2f} dB")
            if row["min_tile_sqnr_db"] < TILE_SQNR_FLOOR_DB[codec]:
                fails.append(f"rank {r}: tile SQNR {row['min_tile_sqnr_db']:.2f} dB")
            if row["self_mismatch"]:
                fails.append(f"rank {r}: self chunk not exact")
            if not row["finite"]:
                fails.append(f"rank {r}: non-finite output")
        if row.get("rccl_mismatch"):
            fails.append(f"rank {r}: {row['rccl_mismatch']} lanes differ from RCCL")
    failures.check(label, fails)


def _summary(rows: list[dict], nbytes: int, tp: int) -> dict:
    us = max((r["us"] for r in rows if r["us"] is not None), default=None)
    return {
        "gfx": get_gfx(),
        "payload": _fmt_bytes(nbytes),
        "variant": rows[0]["variant"],
        "mismatch": max(r["n_mismatch"] for r in rows),
        "SQNR dB": min(r["sqnr_db"] for r in rows),
        "tile SQNR dB": min(r["min_tile_sqnr_db"] for r in rows),
        "us": us,
        "TB/s": _busbw_tbps(nbytes, tp, us),
    }


@benchmark()
def test_quick_alltoall(nbytes, tp, algorithm, codec, order, fill):
    rows = registry.result(_key(tp, algorithm, codec, order), (nbytes, fill, "plain"))
    _gate(
        f"tp{tp} {algorithm} {codec} {order} {_fmt_bytes(nbytes)} {fill}", rows, codec
    )
    return _summary(rows, nbytes, tp)


@benchmark()
def test_quick_alltoall_protocol(nbytes, tp, algorithm, codec, mode):
    rows = registry.result(_key(tp, algorithm, codec), (nbytes, "randn", mode))
    _gate(f"tp{tp} {algorithm} {codec} {mode} {_fmt_bytes(nbytes)}", rows, codec)
    return _summary(rows, nbytes, tp)


# --------------------------------------------------------------------------- #
# verify: the production dispatcher
# --------------------------------------------------------------------------- #
# The dispatcher spawns run with the window floor pinned here, so routing is
# tested both ways whatever the shipped default is. The payloads straddle it,
# and 4 MiB is where PCIe TP4 hands over from the mesh to the ring.
DISPATCH_MIN_BYTES = 1 << 20
DISPATCH_PAYLOADS = (64 << 10, 1 << 20, 4 << 20)
# The payload of the first call, made inside a graph capture.
CAPTURE_FIRST_BYTES = 4 << 20
# Dispatcher spawn setups: "plain"; "capture_first", whose first call is made
# inside a HIP graph capture; "skewed", where the last rank sets
# AITER_FLY_A2A=0.
DISPATCH_SETUPS = ("plain", "capture_first", "skewed")


def _run_rank_dispatch(rank, tp, init_method, engine_kw, cases):
    """One rank of the production path: ``GroupCoordinator.all_to_all`` on the
    TP group, with ``AITER_FLY_A2A`` set by the parent before the spawn.

    A case is ``(nbytes, fill, mode)``; *mode* ``"capture"`` makes the call
    inside a graph capture and grades a replay. Under the ``capture_first``
    setup that case comes first, so nothing has run before the capture.
    """
    import torch.distributed as dist

    from aiter.dist.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        get_tp_group,
        graph_capture,
        init_distributed_environment,
        set_custom_all_reduce,
    )

    if engine_kw["setup"] == "skewed" and rank == tp - 1:
        # This rank declines; every rank has to decline with it.
        os.environ["AITER_FLY_A2A"] = "0"
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp, rank=rank, distributed_init_method=init_method
    )
    ensure_model_parallel_initialized(tp, 1)
    tp_group = get_tp_group()
    comm = tp_group.device_communicator.fly_a2a_comm
    rows = []
    try:
        for nbytes, fill, mode in cases:
            inp = _make_inp(rank, nbytes, fill, device)
            ref = _reference(rank, tp, nbytes, fill, device)
            routed = comm is not None and comm.should_all_to_all(inp)
            out = torch.full_like(inp, float("nan"))
            if mode == "capture":
                g = torch.cuda.CUDAGraph()
                with graph_capture() as gc, torch.cuda.graph(g, stream=gc.stream):
                    tp_group.all_to_all(inp, out)
                torch.cuda.synchronize()
                out.fill_(float("nan"))
                dist.barrier()
                g.replay()
            else:
                tp_group.all_to_all(inp, out)
            torch.cuda.synchronize()
            dist.barrier()
            rows.append(
                {
                    **_grade(out, ref, rank, tp, _block_tile_bytes(tp)),
                    "variant": comm.variant(nbytes) if routed else "rccl",
                    "built": comm is not None,
                    "us": None,
                }
            )
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
    return rows


dispatch_registry = SpawnRegistry(_run_rank_dispatch)


def _dispatch_cases(setup: str) -> list[tuple]:
    if setup == "capture_first":
        return [(CAPTURE_FIRST_BYTES, "randn", "capture")]
    return [(n, "randn", "plain") for n in DISPATCH_PAYLOADS]


@benchmark()
def test_quick_alltoall_dispatcher(nbytes, tp, setup, mode):
    key = dispatch_registry.key(tp, setup=setup)
    rows = dispatch_registry.result(key, (nbytes, "randn", mode))
    fails = [
        f"rank {r}: {row['n_mismatch']} lanes differ"
        for r, row in enumerate(rows)
        if row["n_mismatch"]
    ]
    from aiter.ops.flydsl import alltoall_policy

    if setup == "skewed":
        want = "rccl"
        if any(row["built"] for row in rows):
            fails.append("a rank kept FlyDSL while a peer declined it")
    else:
        policy = alltoall_policy.resolve(tp)
        want = (
            f"quick_alltoall_{policy.pick(nbytes)}_"
            if policy.routes(nbytes)
            else "rccl"
        )
    for r, row in enumerate(rows):
        if not row["variant"].startswith(want):
            fails.append(f"rank {r}: ran {row['variant']}, expected {want}*")
    failures.check(f"tp{tp} dispatcher {setup} {_fmt_bytes(nbytes)}", fails)
    return _summary(rows, nbytes, tp)


# --------------------------------------------------------------------------- #
# verify: the host-framework path
# --------------------------------------------------------------------------- #
# ``QuickAllToAll`` built directly on a gloo *subgroup* with ``enable=True,
# codec="none"`` and no ``AITER_FLY_A2A``, moving a packed ``(N, B, H/N, D +
# pack)`` buffer of random bits -- the shape vLLM's DCP all-to-all sends, with
# the fp32 LSE riding in the tail of each row.
VLLM_TP = 8
VLLM_PARTITIONS = {
    "contig4": [[0, 1, 2, 3], [4, 5, 6, 7]],
    "strided4": [[0, 2, 4, 6], [1, 3, 5, 7]],
    "pairs": [[0, 1], [2, 3], [4, 5], [6, 7]],
}
# (label, B, H/N, D, pack, dtype, routed, mode). Chunk bytes must be a multiple
# of 16; the "odd" row is 6 * 1028 B and has to fall through.
VLLM_SHAPES = (
    ("mla_bf16", 4, 16, 512, 2, "bfloat16", True, "plain"),
    ("mla_bf16_graph", 4, 16, 512, 2, "bfloat16", True, "graph"),
    ("gqa_fp32", 8, 8, 128, 1, "float32", True, "plain"),
    ("tiny", 1, 4, 128, 2, "bfloat16", True, "plain"),
    ("odd", 3, 2, 512, 2, "bfloat16", False, "plain"),
)


def _run_rank_vllm(rank, tp, init_method, engine_kw, cases):
    import torch.distributed as dist

    from aiter.dist.device_communicators.quick_all_to_all import QuickAllToAll

    # The host framework owns the switch; the environment must not matter.
    os.environ.pop("AITER_FLY_A2A", None)
    os.environ["AITER_FLY_A2A_CODEC"] = "int4"

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="gloo", init_method=init_method, world_size=tp, rank=rank
    )
    mine, comms = {}, {}
    for name, parts in VLLM_PARTITIONS.items():
        for ranks in parts:
            g = dist.new_group(ranks, backend="gloo")
            if rank in ranks:
                mine[name] = (g, ranks)

    rows = []
    try:
        for partition, _label, B, hpr, D, pack, dtype_name, _routed, mode in cases:
            group, ranks = mine[partition]
            if partition not in comms:
                comms[partition] = QuickAllToAll(
                    group, device, enable=True, codec="none"
                )
            comm = comms[partition]
            n, r = len(ranks), dist.get_rank(group)
            dtype = getattr(torch, dtype_name)
            int_dtype = torch.int16 if dtype.itemsize == 2 else torch.int32
            numel = n * B * hpr * (D + pack)

            def bits(global_rank, int_dtype=int_dtype, numel=numel):
                gen = torch.Generator(device=device).manual_seed(77 + global_rank)
                info = torch.iinfo(int_dtype)
                return torch.randint(
                    info.min,
                    info.max,
                    (numel,),
                    generator=gen,
                    dtype=int_dtype,
                    device=device,
                )

            inp = bits(ranks[r]).view(dtype)
            # Chunk r of every source, in source order.
            ref = torch.cat([bits(g).view(n, -1)[r] for g in ranks])
            out = torch.zeros_like(inp)
            got = comm.should_all_to_all(inp, out)
            dist.barrier(group)
            n_mismatch = -1
            if got:
                if mode == "graph":
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        comm.all_to_all(inp, out)
                    n_mismatch = 0
                    for _ in range(GRAPH_REPLAYS):
                        out.zero_()
                        g.replay()
                        torch.cuda.synchronize()
                        n_mismatch += int((out.view(int_dtype) != ref).sum())
                else:
                    comm.all_to_all(inp, out)
                    torch.cuda.synchronize()
                    n_mismatch = int((out.view(int_dtype) != ref).sum())
            dist.barrier(group)
            rows.append(
                {
                    "n_mismatch": n_mismatch,
                    "routed": bool(got),
                    "codec": comm._policy.codec if not comm.disabled else None,
                    "disabled": comm.disabled,
                    "variant": comm.variant(inp.numel() * inp.element_size()),
                }
            )
    finally:
        for comm in comms.values():
            comm.close()
        dist.destroy_process_group()
    return rows


vllm_registry = SpawnRegistry(_run_rank_vllm)


@benchmark()
def test_quick_alltoall_vllm(partition, label, B, hpr, D, pack, dtype, routed, mode):
    case = (partition, label, B, hpr, D, pack, dtype, routed, mode)
    rows = vllm_registry.result(vllm_registry.key(VLLM_TP), case)
    fails = []
    for r, row in enumerate(rows):
        if row["disabled"] or row["codec"] != "none":
            fails.append(f"rank {r}: engine disabled or codec {row['codec']}")
        if row["routed"] != routed:
            fails.append(f"rank {r}: routed={row['routed']}, expected {routed}")
        if routed and row["n_mismatch"]:
            fails.append(f"rank {r}: {row['n_mismatch']} lanes differ")
    failures.check(f"vllm {partition} {label} {mode}", fails)
    return {
        "gfx": get_gfx(),
        "variant": rows[0]["variant"],
        "mismatch": max(r["n_mismatch"] for r in rows),
    }


# --------------------------------------------------------------------------- #
# bench
# --------------------------------------------------------------------------- #
# Candidate -> FlyQuickAllToAll kwargs, or the family name of a stock path.
BENCH_CANDIDATES = {
    "rccl": "rccl",
    "rccl_c10d": "rccl_c10d",
    "fly_mesh": {"algorithm": "mesh", "codec": "none"},
    "fly_mesh_atom": {"algorithm": "mesh", "codec": "none", "order": "atom"},
    "fly_ring": {"algorithm": "ring", "codec": "none"},
    "fly_mesh_int4": {"algorithm": "mesh", "codec": "int4"},
    "fly_ring_int4": {"algorithm": "ring", "codec": "int4"},
    "fly_mesh_int6": {"algorithm": "mesh", "codec": "int6"},
    "fly_ring_int6": {"algorithm": "ring", "codec": "int6"},
    "fly_auto": "fly_auto",
}
# 64 MiB of bf16 is the production message; the smaller rows place the
# crossover with RCCL. Hidden 8192: 16 KiB per token.
BENCH_SHAPES = [(m, 8192) for m in (8, 32, 64, 256, 1024, 2048, 4096)]


def _bench_applicable(name: str, tp: int, dtype, nbytes: int) -> bool:
    if nbytes % (16 * tp):
        # FlyDSL and PyNccl both need whole 16 B per chunk.
        return name == "rccl_c10d"
    kw = BENCH_CANDIDATES[name]
    return not (
        isinstance(kw, dict)
        and kw["codec"] in A2A_QUANT_CODECS
        and dtype != torch.bfloat16
    )


def _bench_input(rank: int, tokens: int, hidden: int, dtype, device):
    gen = torch.Generator(device=device).manual_seed(4321 + rank)
    return torch.randn(
        tokens, hidden, generator=gen, dtype=torch.float32, device=device
    ).to(dtype)


def _bench_reference(rank, tp, tokens, hidden, dtype, device):
    chunk = tokens * hidden // tp
    return torch.cat(
        [
            _bench_input(src, tokens, hidden, dtype, device).view(-1)[
                rank * chunk : (rank + 1) * chunk
            ]
            for src in range(tp)
        ]
    ).view(tokens, hidden)


def _bench_rank(rank, tp, init_method, names, shapes, dtype, timing) -> list[dict]:
    """One rank of a bench spawn: build every engine once, then capture, grade
    and time every candidate at every shape."""
    import torch.distributed as dist

    from aiter.dist.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        get_tp_group,
        init_distributed_environment,
        set_custom_all_reduce,
    )
    from aiter.ops.flydsl.quick_alltoall import FlyQuickAllToAll

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp, rank=rank, distributed_init_method=init_method
    )
    ensure_model_parallel_initialized(tp, 1)
    tp_group = get_tp_group()
    group = tp_group.device_group
    pynccl = tp_group.device_communicator.pynccl_comm
    if pynccl is not None and pynccl.disabled:
        pynccl = None
    dispatcher = tp_group.device_communicator.fly_a2a_comm

    # One engine per pinned candidate, built in the same order on every rank:
    # each construction exchanges IPC handles, a collective.
    engines = {}
    for name in names:
        kw = BENCH_CANDIDATES[name]
        if isinstance(kw, dict) and not (
            kw["codec"] in A2A_QUANT_CODECS and dtype != torch.bfloat16
        ):
            engines[name] = FlyQuickAllToAll(
                group=tp_group.cpu_group,
                device=device,
                rank=rank,
                world_size=tp,
                **kw,
            )
            engines[name].preload()

    rows = []
    try:
        for tokens, hidden in shapes:
            x = _bench_input(rank, tokens, hidden, dtype, device)
            ref = _bench_reference(rank, tp, tokens, hidden, dtype, device)
            nbytes = x.numel() * x.element_size()
            thunks, outs, variants = {}, {}, {}
            for name in names:
                if not _bench_applicable(name, tp, dtype, nbytes):
                    continue
                family = BENCH_CANDIDATES[name]
                out = torch.empty_like(x)
                if family == "rccl":
                    if pynccl is None:
                        continue

                    def thunk(o=out, src=x):
                        pynccl.all_to_all(o, src)

                    variants[name] = "pynccl"
                elif family == "rccl_c10d":

                    def thunk(o=out, src=x):
                        dist.all_to_all_single(o.view(-1), src.view(-1), group=group)

                    variants[name] = "c10d"
                elif family == "fly_auto":

                    def thunk(o=out, src=x):
                        tp_group.all_to_all(src, o)

                    routed = dispatcher is not None and dispatcher.should_all_to_all(x)
                    variants[name] = dispatcher.variant(nbytes) if routed else "rccl"
                else:
                    if name not in engines:
                        continue
                    eng = engines[name]

                    def thunk(o=out, src=x, eng=eng):
                        eng.all_to_all(src, o)

                    variants[name] = eng.variant(nbytes)
                thunks[name], outs[name] = thunk, out

            graphs = {name: fab.capture(thunk) for name, thunk in thunks.items()}
            row = {"nbytes": nbytes}
            for name, g in graphs.items():
                got = fab.graded_replay(g, outs[name], group)
                row[name] = {
                    "sqnr_db": _sqnr_db(got, ref),
                    "mismatch": _lanes_differing(got, ref),
                    "variant": variants[name],
                }
            reps, rounds = timing
            for name, us in fab.time_graphs(
                graphs, group, reps=reps, rounds=rounds
            ).items():
                row[name]["us"] = us
            rows.append(row)
            del graphs, thunks, outs, x, ref
            torch.cuda.empty_cache()
    finally:
        for eng in engines.values():
            eng.close()
        destroy_model_parallel()
        destroy_distributed_environment()
    return rows


# Per ``(tp, dtype)``: every rank's rows, and the shapes they are for.
_BENCH_RESULTS: dict[tuple, list] = {}
_BENCH_SHAPES: dict[tuple, list] = {}


@benchmark()
def test_quick_alltoall_bench(tokens, hidden, tp, dtype):
    i = _BENCH_SHAPES[(tp, dtype)].index((tokens, hidden))
    rank_rows = [pr[i] for pr in _BENCH_RESULTS[(tp, dtype)]]
    nbytes = rank_rows[0]["nbytes"]
    ret = {"gfx": get_gfx(), "payload": _fmt_bytes(nbytes)}
    fails = []
    for name, kw in BENCH_CANDIDATES.items():
        if name not in rank_rows[0]:
            continue
        per = [r[name] for r in rank_rows]
        c = fab.collapse(per)
        mismatch = max(p["mismatch"] for p in per)
        codec = kw["codec"] if isinstance(kw, dict) else None
        ret[f"{name} us"] = c["us"]
        ret[f"{name} TB/s"] = _busbw_tbps(nbytes, tp, c["us"])
        ret[f"{name} SQNR dB"] = c["SQNR dB"]
        ret[f"{name} variant"] = c["variant"]
        # The dispatcher runs whichever codec AITER_FLY_A2A_CODEC names, so a
        # lossy fly_auto is graded like the loosest codec.
        if codec in A2A_QUANT_CODECS or (name == "fly_auto" and mismatch):
            floor = SQNR_FLOOR_DB.get(codec, min(SQNR_FLOOR_DB.values()))
            if c["SQNR dB"] < floor:
                fails.append(f"{name}: SQNR {c['SQNR dB']:.2f} dB")
        elif mismatch:
            fails.append(f"{name}: {mismatch} lanes differ")
    failures.check(f"bench tp{tp} {tokens}x{hidden}", fails)
    return ret


def _bench(args, tps) -> None:
    names = args.candidates
    if "fly_auto" in names:
        # The production slot is opt-in; the spawns inherit this.
        os.environ["AITER_FLY_A2A"] = "1"
    timing = (args.reps, args.rounds)
    for dtype in args.dtype:
        rows = []
        for tp in tps:
            _BENCH_SHAPES[(tp, dtype)] = list(args.mnk)
            _BENCH_RESULTS[(tp, dtype)] = _spawn(
                _bench_rank,
                tp,
                names=names,
                shapes=list(args.mnk),
                dtype=dtype,
                timing=timing,
            )
            rows += [
                test_quick_alltoall_bench(tokens, hidden, tp, dtype)
                for tokens, hidden in args.mnk
            ]
        summarize(f"quick_alltoall_bench {str(dtype).removeprefix('torch.')}", rows)
    failures.raise_if_failed("quick_alltoall bench")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _verify(args, tps) -> None:
    plan = []
    for tp, algorithm, codec in itertools.product(tps, args.algorithm, args.codec):
        sizes = args.nbytes or payloads(tp)
        # The register-direct store order only exists for the mesh's none codec.
        orders = (
            ("peer", "atom") if (algorithm, codec) == ("mesh", "none") else ("peer",)
        )
        fills = FILLS if codec not in LOSSLESS else ("randn",)
        # Edge fills at one multi-tile size only.
        edge = (1 << 20) if (1 << 20) in sizes else sizes[-1]
        for order, nbytes, fill in itertools.product(orders, sizes, fills):
            if fill != "randn" and nbytes != edge:
                continue
            plan.append((nbytes, tp, algorithm, codec, order, fill))
            registry.register(
                _key(tp, algorithm, codec, order), (nbytes, fill, "plain")
            )
    proto = []
    for tp, algorithm, codec, mode in itertools.product(
        tps, args.algorithm, ("none", "int4"), ("skew", "wrap", "graph")
    ):
        if codec not in args.codec:
            continue
        proto.append((4 << 20, tp, algorithm, codec, mode))
        registry.register(_key(tp, algorithm, codec), (4 << 20, "randn", mode))

    dispatch = []
    for tp, setup in itertools.product(tps, DISPATCH_SETUPS):
        for nbytes, fill, mode in _dispatch_cases(setup):
            dispatch.append((nbytes, tp, setup, mode))
            dispatch_registry.register(
                dispatch_registry.key(tp, setup=setup), (nbytes, fill, mode)
            )

    summarize("quick_alltoall", [test_quick_alltoall(*p) for p in plan])
    summarize(
        "quick_alltoall_protocol", [test_quick_alltoall_protocol(*p) for p in proto]
    )
    # The production path: AITER_FLY_A2A=1 with the default codec, and the
    # window floor pinned so routing is tested both ways. Set here so only the
    # dispatcher spawns inherit it.
    os.environ["AITER_FLY_A2A"] = "1"
    os.environ["AITER_FLY_A2A_MIN_BYTES"] = str(DISPATCH_MIN_BYTES)
    summarize(
        "quick_alltoall_dispatcher",
        [test_quick_alltoall_dispatcher(*d) for d in dispatch],
    )
    os.environ.pop("AITER_FLY_A2A")
    os.environ.pop("AITER_FLY_A2A_MIN_BYTES")
    if VLLM_TP in tps:
        vllm_cases = [(p, *shape) for p in VLLM_PARTITIONS for shape in VLLM_SHAPES]
        for case in vllm_cases:
            vllm_registry.register(vllm_registry.key(VLLM_TP), case)
        summarize(
            "quick_alltoall_vllm", [test_quick_alltoall_vllm(*c) for c in vllm_cases]
        )
    failures.raise_if_failed("quick_alltoall")


def main():
    if get_gfx() not in SUPPORTED_ARCHS:
        aiter.logger.warning("FlyQuickAllToAll unsupported on %s; skipping", get_gfx())
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "--scenario",
        choices=("verify", "bench"),
        default="verify",
        help="verify: correctness sweeps (default). bench: time against RCCL.",
    )
    parser.add_argument(
        "--tp",
        type=int,
        nargs="*",
        default=list(SUPPORTED_WORLDS),
        help="World sizes (2, 4, 8). Sizes with fewer visible GPUs are skipped.",
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        nargs="*",
        default=list(ALGORITHMS),
        choices=ALGORITHMS,
        help="verify: schedules to sweep.",
    )
    parser.add_argument(
        "--codec",
        nargs="*",
        default=list(CODECS),
        choices=CODECS,
        help="verify: wire formats to sweep.",
    )
    parser.add_argument(
        "-n",
        "--nbytes",
        type=int,
        nargs="*",
        default=None,
        help="verify: payload bytes per rank. Default: sub-tile, partial tile,\n"
        "1 MiB, 16 MiB and 64 MiB at each TP.",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=BENCH_SHAPES,
        help="bench: per-rank (tokens, hidden) shapes, e.g. -s 4096,8192.",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.d_dtypes["bf16"]],
        help="bench: payload dtypes. The INT4/INT6 candidates need bf16.",
    )
    parser.add_argument(
        "--candidates",
        nargs="*",
        default=list(BENCH_CANDIDATES),
        choices=list(BENCH_CANDIDATES),
        help="bench: what to time.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=fab.DEFAULT_REPS,
        help="bench: timed replays per sample.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=fab.DEFAULT_ROUNDS,
        help="bench: samples per candidate, one per rotated round.",
    )
    args = parser.parse_args()

    n_gpu = fab.gpu_count()
    tps = [tp for tp in args.tp if tp in SUPPORTED_WORLDS and tp <= n_gpu]
    if not tps:
        aiter.logger.warning("no world size in %s fits %d GPU(s)", args.tp, n_gpu)
        return
    if args.scenario == "bench":
        _bench(args, tps)
    else:
        _verify(args, tps)


if __name__ == "__main__":
    main()
