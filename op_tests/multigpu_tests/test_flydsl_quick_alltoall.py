# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and accuracy for the FlyDSL all-to-all (``FlyQuickAllToAll``).

Equal-split semantics: rank ``r``'s output chunk ``i`` is rank ``i``'s input
chunk ``r``. Every rank generates every peer's input from the same seed, so the
reference is built locally with no collective; one case per spawn is also
checked against ``torch.distributed.all_to_all_single``.

* ``test_quick_alltoall`` -- each schedule (mesh, ring) and wire format at
  payloads that hit the sub-tile path, a partial last tile, both ladder rungs,
  and the 64 MiB production size. ``none``/``fp16`` must be bit-exact; INT4 and
  INT6 gate on SQNR, on a bit-exact self chunk and on finite output.
* ``test_quick_alltoall_protocol`` -- the inbox reuse rules: back-to-back
  launches with one rank held back (``torch.cuda._sleep``), launches straddling
  the colour wrap, and CUDA-graph replay.

Performance is measured by ``op_benchmarks/flydsl/bench_comm.py --operation
a2a``, against RCCL; the ``us`` / ``TB/s`` columns here are a smoke check.
"""

from __future__ import annotations

import argparse
import ctypes
import itertools
import os

import torch
from flydsl_comm_test_utils import (
    ARCH,
    SUPPORTED_ARCHS,
    FailureLog,
    SpawnRegistry,
    fmt_bytes,
    min_tile_sqnr_db,
    sqnr_db,
    summarize,
)

import aiter
from aiter.ops.flydsl.kernels.collectives_shared import ATOMS, SUPPORTED_WORLDS
from aiter.test_common import benchmark, run_perftest

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

failures = FailureLog()


def _block_tile_bytes(tp: int, block: int = 256) -> int:
    """HBM bytes of one tile of one chunk, at *block*."""
    return ATOMS // tp * block * 16


def payloads(tp: int) -> list[int]:
    """Sub-tile, partial tile, both ladder rungs, and the production size."""
    tile = _block_tile_bytes(tp)
    return [
        tp * 16,
        tp * (tile + 48),
        1 << 20,
        16 << 20,
        TOP_PAYLOAD_BYTES,
    ]


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
        "n_mismatch": int((out.view(torch.int16) != ref.view(torch.int16)).sum()),
        "self_mismatch": int(
            (out[own].view(torch.int16) != ref[own].view(torch.int16)).sum()
        ),
        "finite": bool(torch.isfinite(out.float()).all()),
        "sqnr_db": sqnr_db(out, ref),
        "min_tile_sqnr_db": min_tile_sqnr_db(out, ref, tile_bytes),
    }


def _seed_colors(fly, value: int) -> None:
    """Overwrite every engine's per-block colours, to force the wrap."""
    from aiter.ops.flydsl.quick_allreduce_ipc import UncachedIpcHeap

    torch.cuda.synchronize()
    for eng in fly._by_cfg.values():
        UncachedIpcHeap.copy_host_to_device(
            eng._colors,
            (ctypes.c_int32 * eng.grid)(*([value] * eng.grid)),
            eng.grid * 4,
        )


def _run_rank(rank, tp, init_method, engine_kw, cases, window=None) -> list[dict]:
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
                g_m = _grade(out, ref, rank, tp, tile_bytes)
                m = (
                    g_m
                    if m is None
                    else {
                        k: (
                            min(m[k], v)
                            if k.endswith("sqnr_db")
                            else (m[k] and v) if k == "finite" else max(m[k], v)
                        )
                        for k, v in g_m.items()
                    }
                )
            if not checked_rccl and mode == "plain" and fly.codec in LOSSLESS:
                rccl = torch.empty_like(inp)
                dist.all_to_all_single(rccl, inp)
                torch.cuda.synchronize()
                m["rccl_mismatch"] = int(
                    (outs[0].view(torch.int16) != rccl.view(torch.int16)).sum()
                )
                checked_rccl = True

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
                    "link": fly.link,
                    "inbox_memory": fly.inbox_memory,
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


# The dispatcher spawns run with the window floor pinned here, so routing is
# tested both ways whatever the shipped default is. The payloads straddle it,
# and 4 MiB is where PCIe TP4 hands over from the mesh to the ring.
DISPATCH_MIN_BYTES = 1 << 20
DISPATCH_PAYLOADS = (64 << 10, 1 << 20, 4 << 20)


def _run_rank_dispatch(rank, tp, init_method, engine_kw, cases, window=None):
    """One rank of the production path: ``GroupCoordinator.all_to_all`` on the
    TP group, with ``AITER_FLY_A2A`` set by the parent before the spawn."""
    import torch.distributed as dist

    from aiter.dist.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        get_tp_group,
        init_distributed_environment,
        set_custom_all_reduce,
    )

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
        for nbytes, fill, _mode in cases:
            inp = _make_inp(rank, nbytes, fill, device)
            ref = _reference(rank, tp, nbytes, fill, device)
            routed = comm is not None and comm.should_all_to_all(inp)
            out = tp_group.all_to_all(inp)
            torch.cuda.synchronize()
            dist.barrier()
            rows.append(
                {
                    **_grade(out, ref, rank, tp, _block_tile_bytes(tp)),
                    "variant": comm.variant(nbytes) if routed else "rccl",
                    "us": None,
                }
            )
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
    return rows


dispatch_registry = SpawnRegistry(_run_rank_dispatch)


# Host-framework path: ``QuickAllToAll`` built directly on a gloo *subgroup*
# with ``enable=True, codec="none"`` and no ``AITER_FLY_A2A``, moving a packed
# ``(N, B, H/N, D + pack)`` buffer of random bits -- the shape vLLM's DCP
# all-to-all sends, with the fp32 LSE riding in the tail of each row.
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


def _run_rank_vllm(rank, tp, init_method, engine_kw, cases, window=None):
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
        for partition, _label, B, hpr, D, pack, dtype_name, routed, mode in cases:
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

            def bits(global_rank):
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
            ref = torch.cat(
                [bits(g).view(n, -1)[r] for g in ranks]
            )  # chunk r of every source, in source order
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
    worst_sqnr = min(r["sqnr_db"] for r in rows)
    us = max((r["us"] for r in rows if r["us"] is not None), default=None)
    return {
        "payload": fmt_bytes(nbytes),
        "variant": rows[0]["variant"],
        "mismatch": max(r["n_mismatch"] for r in rows),
        "SQNR dB": worst_sqnr,
        "tile SQNR dB": min(r["min_tile_sqnr_db"] for r in rows),
        "us": us,
        # Bus bandwidth in the nccl-tests all-to-all convention.
        "busbw TB/s": None if us is None else nbytes * (tp - 1) / tp / us / 1e6,
    }


@benchmark()
def test_quick_alltoall(nbytes, tp, algorithm, codec, order, fill):
    key = _key(tp, algorithm, codec, order)
    rows = registry.result(key, (nbytes, fill, "plain"))
    _gate(f"tp{tp} {algorithm} {codec} {order} {fmt_bytes(nbytes)} {fill}", rows, codec)
    return _summary(rows, nbytes, tp)


@benchmark()
def test_quick_alltoall_protocol(nbytes, tp, algorithm, codec, mode):
    key = _key(tp, algorithm, codec)
    rows = registry.result(key, (nbytes, "randn", mode))
    _gate(f"tp{tp} {algorithm} {codec} {mode} {fmt_bytes(nbytes)}", rows, codec)
    return _summary(rows, nbytes, tp)


@benchmark()
def test_quick_alltoall_dispatcher(nbytes, tp):
    key = dispatch_registry.key(tp)
    rows = dispatch_registry.result(key, (nbytes, "randn", "plain"))
    fails = [
        f"rank {r}: {row['n_mismatch']} lanes differ"
        for r, row in enumerate(rows)
        if row["n_mismatch"]
    ]
    from aiter.ops.flydsl import alltoall_policy

    policy = alltoall_policy.resolve(tp)
    want = f"quick_alltoall_{policy.pick(nbytes)}_" if policy.routes(nbytes) else "rccl"
    for r, row in enumerate(rows):
        if not row["variant"].startswith(want):
            fails.append(f"rank {r}: ran {row['variant']}, expected {want}*")
    failures.check(f"tp{tp} dispatcher {fmt_bytes(nbytes)}", fails)
    return _summary(rows, nbytes, tp)


@benchmark()
def test_quick_alltoall_vllm(partition, label, B, hpr, D, pack, dtype, routed, mode):
    key = vllm_registry.key(VLLM_TP)
    case = (partition, label, B, hpr, D, pack, dtype, routed, mode)
    rows = vllm_registry.result(key, case)
    fails = []
    for r, row in enumerate(rows):
        if row["disabled"] or row["codec"] != "none":
            fails.append(f"rank {r}: engine disabled or codec {row['codec']}")
        if row["routed"] != routed:
            fails.append(f"rank {r}: routed={row['routed']}, expected {routed}")
        if routed and row["n_mismatch"]:
            fails.append(f"rank {r}: {row['n_mismatch']} lanes differ")
    failures.check(f"vllm {partition} {label} {mode}", fails)
    return {"variant": rows[0]["variant"], "mismatch": max(r["n_mismatch"] for r in rows)}


def main():
    if ARCH not in SUPPORTED_ARCHS:
        aiter.logger.warning("FlyQuickAllToAll unsupported on %s; skipping", ARCH)
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "--tp",
        type=int,
        nargs="*",
        default=list(SUPPORTED_WORLDS),
        help="World sizes (2, 4, 8). Sizes with fewer visible GPUs are skipped.",
    )
    parser.add_argument(
        "-a", "--algorithm", nargs="*", default=list(ALGORITHMS), choices=ALGORITHMS
    )
    parser.add_argument(
        "-c", "--codec", nargs="*", default=list(CODECS), choices=CODECS
    )
    parser.add_argument(
        "-n",
        "--nbytes",
        type=int,
        nargs="*",
        default=None,
        help="Payload bytes per rank. Default: sub-tile, partial tile, 1 MiB,\n"
        "16 MiB and 64 MiB at each TP.",
    )
    args = parser.parse_args()

    n_gpu = torch.cuda.device_count()
    tps = [tp for tp in args.tp if tp in SUPPORTED_WORLDS and tp <= n_gpu]

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
    for tp, algorithm in itertools.product(tps, args.algorithm):
        for codec, mode in itertools.product(
            ("none", "int4"), ("skew", "wrap", "graph")
        ):
            if codec not in args.codec:
                continue
            proto.append((4 << 20, tp, algorithm, codec, mode))
            registry.register(_key(tp, algorithm, codec), (4 << 20, "randn", mode))

    # The production path: AITER_FLY_A2A=1 with the default codec and window.
    # Set here so only the dispatcher spawns inherit it.
    for tp in tps:
        for nbytes in DISPATCH_PAYLOADS:
            dispatch_registry.register(
                dispatch_registry.key(tp), (nbytes, "randn", "plain")
            )

    summarize("quick_alltoall", [test_quick_alltoall(*p) for p in plan])
    summarize(
        "quick_alltoall_protocol", [test_quick_alltoall_protocol(*p) for p in proto]
    )
    os.environ["AITER_FLY_A2A"] = "1"
    os.environ["AITER_FLY_A2A_MIN_BYTES"] = str(DISPATCH_MIN_BYTES)
    summarize(
        "quick_alltoall_dispatcher",
        [
            test_quick_alltoall_dispatcher(n, tp)
            for tp in tps
            for n in DISPATCH_PAYLOADS
        ],
    )
    if VLLM_TP in tps:
        vllm_cases = [
            (p, *shape) for p in VLLM_PARTITIONS for shape in VLLM_SHAPES
        ]
        for case in vllm_cases:
            vllm_registry.register(vllm_registry.key(VLLM_TP), case)
        summarize("quick_alltoall_vllm", [test_quick_alltoall_vllm(*c) for c in vllm_cases])
    failures.raise_if_failed("quick_alltoall")


if __name__ == "__main__":
    main()
