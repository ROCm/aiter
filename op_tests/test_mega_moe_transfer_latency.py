# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Measure one full MegaMoE scatter group and one full dispatch run on gfx1250.

Run from the repository root, on idle GPUs:
    PYTHONPATH=$PWD python3 -m torch.distributed.run --standalone \
        --nproc_per_node=4 op_tests/test_mega_moe_transfer_latency.py \
        --hidden 7168 --source-ranks 0 --destination-ranks 1 \
        --samples 256 --warmup 5 --iters 20

Only the selected source rank launches a transfer WG. Other ranks wait on CPU
Gloo barriers. CCO supplies the symmetric peer mapping; no SDMA queue is used.

Device samples exclude source staging, descriptors, launch, routing, ready flags,
and GEMMs. They include issue, completion waits, and timer instrumentation.
The separate single-operation event measurement includes launch, source staging,
descriptors and waits, but has no timestamps. Repeated samples overwrite the
same destination slots: these are warm-address, serialized completion latencies,
not bandwidth under concurrent MegaMoE load or receiver-observed ready latency.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import statistics
import traceback
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest

# The reference states the byte-layout contract independently of the kernel.
ROWS = 8
SCATTER_COLUMNS = 256
SCATTER_LDS_COLUMNS = 264
SCATTER_FIRST_SLOT = 3
SCATTER_SLOT_STEP = 17
SCATTER_COLUMN_OFFSET = 256
DISPATCH_FIRST_ROW = 16
SENTINEL = 0xA5
DISPATCH_ROWS = 32


def geometry(hidden):
    if hidden < 512 or hidden % 32:
        raise ValueError("hidden must be a multiple of 32 and at least 512")
    wire_row = ((hidden // 2 + hidden // 32 + 127) // 128) * 128
    if wire_row > 65535:
        raise ValueError("wire row exceeds the TDM tile dimension field")
    slot_bytes = 1 << (hidden * 2 - 1).bit_length()
    slot_count = SCATTER_FIRST_SLOT + (ROWS - 1) * SCATTER_SLOT_STEP + 2
    return (
        wire_row,
        slot_bytes,
        (
            ("scatter", slot_count * slot_bytes),
            ("wire", DISPATCH_ROWS * wire_row),
            ("rowmap", DISPATCH_ROWS * 8),
        ),
    )


def destination_rows(world, source_rank, destination_rank, scatter_mode):
    if not (0 <= source_rank < world and 0 <= destination_rank < world):
        raise ValueError("source/destination rank is outside WORLD_SIZE")
    if scatter_mode == "single-peer":
        return (destination_rank,) * ROWS
    if scatter_mode == "mixed-peer":
        return tuple((destination_rank + i) % world for i in range(ROWS))
    raise ValueError(f"unknown scatter mode: {scatter_mode}")


def tick_statistics(pairs, clock_hz, *, allow_zero=False):
    if not math.isfinite(clock_hz) or clock_hz <= 0:
        raise ValueError("clock_hz must be finite and positive")
    deltas = [int(stop) - int(start) for start, stop in pairs]
    if (
        not deltas
        or any(start < 0 or stop < 0 for start, stop in pairs)
        or any(delta < 0 or (delta == 0 and not allow_zero) for delta in deltas)
    ):
        raise RuntimeError(
            "timer samples were unwritten, zero, or non-monotonic; "
            f"first pairs: {pairs[:8]}"
        )
    ordered = sorted(deltas)

    def percentile(q):
        position = (len(ordered) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    scale = 1e6 / clock_hz
    return {
        "count": len(deltas),
        "p50 ticks": statistics.median(deltas),
        "min ticks": min(deltas),
        "p50 us": statistics.median(deltas) * scale,
        "p90 us": percentile(0.9) * scale,
        "p99 us": percentile(0.99) * scale,
    }


def make_inputs(op, hidden, source_rank, device):
    generator = torch.Generator().manual_seed(1234 + source_rank)
    if op == "scatter":
        cpu_bytes = torch.full(
            (ROWS, SCATTER_LDS_COLUMNS * 2),
            0x6D,
            dtype=torch.uint8,
            device="cpu",
        )
        data = torch.randn((ROWS, SCATTER_COLUMNS), generator=generator, device="cpu")
        data += torch.arange(ROWS, dtype=torch.float32, device="cpu")[:, None] * 3
        cpu_bytes.view(torch.bfloat16)[:, :SCATTER_COLUMNS].copy_(
            data.to(torch.bfloat16)
        )
    else:
        wire_row, _, _ = geometry(hidden)
        columns = torch.arange(wire_row, dtype=torch.int32, device="cpu")[None, :]
        rows = torch.arange(ROWS, dtype=torch.int32, device="cpu")[:, None]
        cpu_bytes = ((columns * 29 + rows * 43 + 7) % 256).to(torch.uint8)
        scale_start = hidden // 2
        scale_end = scale_start + hidden // 32
        cpu_bytes[:, scale_start:scale_end] = (
            121 + rows + columns[:, : hidden // 32] % 7
        ).to(torch.uint8)
        cpu_bytes[:, scale_end:] = (0xE0 + rows).to(torch.uint8)
    meta = (
        source_rank * 8192 * 9
        + (torch.arange(ROWS, dtype=torch.int32, device="cpu") * 17 + 3) * 9
        + torch.arange(ROWS, dtype=torch.int32, device="cpu") % 9
    )
    weights = (torch.arange(ROWS, dtype=torch.float32, device="cpu") + 1) / 16
    cpu_rowmap = torch.stack((meta, weights.view(torch.int32)), dim=1)
    source = cpu_bytes.flatten().view(torch.int32).to(device)
    rowmap = cpu_rowmap.flatten().view(torch.int64).to(device)
    return cpu_bytes, cpu_rowmap, source, rowmap


def run_torch(op, hidden, rank, peers, cpu_bytes, cpu_rowmap, layout):
    """Independent byte-exact reference, including untouched guards and slots."""
    expected = torch.full(
        (layout.total_bytes,), SENTINEL, dtype=torch.uint8, device="cpu"
    )
    wire_row, slot_bytes, _ = geometry(hidden)
    if op == "scatter":
        for row, peer in enumerate(peers):
            if peer == rank:
                slot = SCATTER_FIRST_SLOT + row * SCATTER_SLOT_STEP
                begin = layout.offsets["scatter"] + slot * slot_bytes
                begin += SCATTER_COLUMN_OFFSET * 2
                expected[begin : begin + SCATTER_COLUMNS * 2] = cpu_bytes[
                    row, : SCATTER_COLUMNS * 2
                ]
    elif peers[0] == rank:
        begin = layout.offsets["wire"] + DISPATCH_FIRST_ROW * wire_row
        expected[begin : begin + ROWS * wire_row] = cpu_bytes.flatten()
        begin = layout.offsets["rowmap"] + DISPATCH_FIRST_ROW * 8
        expected[begin : begin + ROWS * 8] = cpu_rowmap.flatten().view(torch.uint8)
    return expected


@dataclass(repr=False)
class Context:
    rank: int
    world: int
    device: torch.device
    arena: object
    local: torch.Tensor
    reports: list
    clocks: dict

    @property
    def offsets(self):
        return tuple(self.arena.offset(name) for name in ("scatter", "wire", "rowmap"))

    def launch(self, source, rowmap, times, op, hidden, peers, **kwargs):
        # Keep optional FlyDSL/Mori imports after main()'s architecture gate.
        from aiter.ops.flydsl.mega_moe_transfer_latency import launch_transfer

        launch_transfer(
            source,
            rowmap,
            times,
            arena_handle=self.arena.handle,
            op=op,
            hidden=hidden,
            world=self.world,
            peers=peers,
            offsets=self.offsets,
            **kwargs,
        )


def calibrate_timer(ctx, source, rowmap, hidden, peers):
    """Use the slope of long/short HIP-event intervals, cancelling fixed overhead."""
    output = torch.full((2,), -1, dtype=torch.int64, device=ctx.device)

    def probe(ticks):
        ctx.launch(source, rowmap, output, "clock", hidden, peers, clock_ticks=ticks)

    probe(1_000_000)  # Compilation also executes once; it is outside measurement.
    torch.cuda.synchronize(ctx.device)
    observations = []
    estimates = []
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(3):
        pair = []
        for target in (1_000_000, 4_000_000):
            start.record()
            probe(target)
            stop.record()
            stop.synchronize()
            event_us = start.elapsed_time(stop) * 1000
            start_tick, stop_tick = output.cpu().tolist()
            elapsed_ticks = stop_tick - start_tick
            if elapsed_ticks < target or event_us <= 0:
                raise RuntimeError("invalid steady-counter calibration probe")
            observation = {"ticks": elapsed_ticks, "event us": event_us}
            observations.append(observation)
            pair.append(observation)
        delta_us = pair[1]["event us"] - pair[0]["event us"]
        delta_ticks = pair[1]["ticks"] - pair[0]["ticks"]
        if delta_us <= 0 or delta_ticks <= 0:
            raise RuntimeError("clock calibration intervals are not increasing")
        estimates.append(delta_ticks / delta_us * 1e6)
    hz = statistics.median(estimates)
    spread = (max(estimates) - min(estimates)) / hz
    if not math.isfinite(hz) or hz <= 0 or spread > 0.02:
        raise RuntimeError(f"unstable clock calibration: {estimates}")
    return {"Hz": hz, "relative spread": spread, "probes": observations}


def collect_samples(fn, output, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(output.device)
    pairs = []
    for _ in range(iters):
        output.fill_(-1)
        fn()
        torch.cuda.synchronize(output.device)
        pairs.extend(output.cpu().reshape(-1, 2).tolist())
    return pairs


def synchronize_failure(error):
    failures = [None] * dist.get_world_size()
    dist.all_gather_object(failures, error)
    if any(failures):
        raise RuntimeError(
            "\n".join(
                f"rank {rank}: {failure}"
                for rank, failure in enumerate(failures)
                if failure
            )
        )


@benchmark()
def test_transfer(
    op, hidden, source_rank, destination_rank, scatter_mode, samples, warmup, iters, ctx
):
    peers = destination_rows(ctx.world, source_rank, destination_rank, scatter_mode)
    cpu_bytes, cpu_rowmap, source, rowmap = make_inputs(
        op, hidden, source_rank, ctx.device
    )
    times = torch.full((samples * 2,), -1, dtype=torch.int64, device=ctx.device)
    event_times = torch.full((2,), -1, dtype=torch.int64, device=ctx.device)
    payload_bytes = ROWS * SCATTER_COLUMNS * 2
    if op == "dispatch":
        payload_bytes = ROWS * (geometry(hidden)[0] + 8)
    remote_bytes = (
        sum(peer != source_rank for peer in peers) * SCATTER_COLUMNS * 2
        if op == "scatter"
        else payload_bytes * (destination_rank != source_rank)
    )
    ret = {
        "gfx": get_gfx_runtime(),
        "world": ctx.world,
        "payload B": payload_bytes,
        "remote payload B": remote_bytes,
        "peer rows": str(peers),
    }
    candidates = {"tdm": False}
    for name, empty in candidates.items():
        ctx.local.fill_(SENTINEL)
        torch.cuda.synchronize(ctx.device)
        dist.barrier()
        error = None
        report = None
        if ctx.rank == source_rank:
            try:
                if source_rank not in ctx.clocks:
                    ctx.clocks[source_rank] = calibrate_timer(
                        ctx, source, rowmap, hidden, peers
                    )
                clock = ctx.clocks[source_rank]

                def measured(empty=empty):
                    ctx.launch(
                        source,
                        rowmap,
                        times,
                        op,
                        hidden,
                        peers,
                        samples=samples,
                        instrument=True,
                        empty=empty,
                    )

                def control():
                    ctx.launch(
                        source,
                        rowmap,
                        times,
                        op,
                        hidden,
                        peers,
                        samples=samples,
                        instrument=True,
                        empty=True,
                    )

                # Warm the compiling launchers, then verify that the empty
                # control leaves the destination untouched on every rank.
                control()
                torch.cuda.synchronize(ctx.device)
            except Exception:  # noqa: BLE001 -- propagate failure to passive ranks
                error = traceback.format_exc()
        synchronize_failure(error)
        control_err = checkAllclose(
            torch.full_like(ctx.local, SENTINEL).float(),
            ctx.local.float(),
            rtol=0,
            atol=0,
            tol_err_ratio=0,
            msg=f"{op} empty-control rank {ctx.rank}",
        )
        synchronize_failure(
            f"empty control modified arena bytes: {control_err}"
            if control_err
            else None
        )

        error = None
        if ctx.rank == source_rank:
            try:
                empty_pairs = collect_samples(control, times, warmup, iters)
                pairs = collect_samples(measured, times, warmup, iters)

                def one_operation(empty=empty):
                    ctx.launch(
                        source,
                        rowmap,
                        event_times,
                        op,
                        hidden,
                        peers,
                        samples=1,
                        instrument=False,
                        empty=empty,
                    )
                    return event_times

                one_operation()  # Exclude the first compile-and-execute call.
                torch.cuda.synchronize(ctx.device)
                _, event_us = run_perftest(
                    one_operation,
                    num_warmup=warmup,
                    num_iters=iters,
                    num_rotate_args=1,
                    use_cuda_event=True,
                )
                stats = tick_statistics(pairs, clock["Hz"])
                # A valid empty interval can round to zero at REALTIME resolution.
                control_stats = tick_statistics(
                    empty_pairs, clock["Hz"], allow_zero=True
                )
                report = {
                    "op": op,
                    "hidden": hidden,
                    "source rank": source_rank,
                    "destination rank": destination_rank,
                    "scatter mode": scatter_mode,
                    "peers": peers,
                    "payload B": payload_bytes,
                    "remote payload B": remote_bytes,
                    "grid": [1, 1, 1],
                    "block": [128, 1, 1],
                    "clock": clock,
                    "device": stats,
                    "empty control": control_stats,
                    "single operation event us": event_us,
                    "device timestamp pairs": pairs,
                    "empty timestamp pairs": empty_pairs,
                }
                ret.update(
                    {
                        "clock Hz": clock["Hz"],
                        f"{name} device p50 ticks": stats["p50 ticks"],
                        f"{name} device p50 us": stats["p50 us"],
                        f"{name} device p90 us": stats["p90 us"],
                        f"{name} device p99 us": stats["p99 us"],
                        "empty control p50 us": control_stats["p50 us"],
                        f"{name} us": event_us,
                        f"{name} TFLOPS": 0.0,
                        f"{name} TB/s": payload_bytes / event_us / 1e6,
                    }
                )
            except Exception:  # noqa: BLE001 -- propagate failure to passive ranks
                error = traceback.format_exc()
        synchronize_failure(error)
        # Every destination initialized/drained before measurement. The source
        # completed its SYS-scoped stores before this CPU collective returned.
        expected = run_torch(
            op, hidden, ctx.rank, peers, cpu_bytes, cpu_rowmap, ctx.arena.layout
        )
        actual = ctx.local.cpu()
        err = checkAllclose(
            expected.float(),
            actual.float(),
            rtol=0,
            atol=0,
            tol_err_ratio=0,
            msg=f"{op} payload+guards rank {ctx.rank}",
        )
        synchronize_failure(f"byte-exact verification failed: {err}" if err else None)
        assert err == 0, f"{op}: byte-exact payload/guard mismatch"
        ret[f"{name} err"] = err
        packet = [ret if ctx.rank == source_rank else None]
        dist.broadcast_object_list(packet, src=source_rank)
        ret = packet[0]
        if ctx.rank == source_rank:
            report["err"] = err
            ctx.reports.append(report)
    return ret


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, nargs="+", default=[7168])
    parser.add_argument("--source-ranks", type=int, nargs="+", default=[0])
    parser.add_argument("--destination-ranks", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--scatter-mode",
        nargs="+",
        choices=("single-peer", "mixed-peer"),
        default=["single-peer"],
        help="mixed-peer cycles through all ranks, including any local rows",
    )
    parser.add_argument("--samples", type=int, nargs="+", default=[256])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--output", type=Path, help="optional JSON with all raw samples"
    )
    args = parser.parse_args()
    if (
        args.warmup < 1
        or args.iters < 1
        or any(not 1 <= n <= 4096 for n in args.samples)
    ):
        parser.error("warmup/iters must be positive; samples must be in [1, 4096]")
    requests = [dict(geometry(hidden)[2]) for hidden in args.hidden]
    regions = tuple(
        (name, max(request[name] for request in requests))
        for name in ("scatter", "wire", "rowmap")
    )
    if not torch.cuda.is_available():
        aiter.logger.warning("SKIP: gfx1250 GPU is required")
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if get_gfx_runtime() != "gfx1250":
        aiter.logger.warning("SKIP: transfer microbenchmark requires gfx1250")
        return
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        parser.error("launch with torch.distributed.run and at least two ranks")
    if int(os.environ.get("WORLD_SIZE", "1")) != int(
        os.environ.get("LOCAL_WORLD_SIZE", "1")
    ):
        parser.error("the LSA transfer test requires all ranks on one host")
    dist.init_process_group("gloo", timeout=timedelta(seconds=300))
    rank, world = dist.get_rank(), dist.get_world_size()
    if (
        max(args.source_ranks + args.destination_ranks) >= world
        or min(args.source_ranks + args.destination_ranks) < 0
    ):
        parser.error("source/destination rank is outside WORLD_SIZE")
    import flydsl
    from mori.cco import Communicator

    from aiter.ops.flydsl.kernels.mega_moe_gfx1250.arena import SymmetricArena
    from aiter.ops.flydsl.kernels.mega_moe_gfx1250.types import _from_gpu_ptr

    environment = {
        "gfx": get_gfx_runtime(),
        "torch": torch.__version__,
        "flydsl": flydsl.__version__,
        "flydsl module": flydsl.__file__,
        "world": world,
        "warmup": args.warmup,
        "iters": args.iters,
    }
    if rank == 0:
        aiter.logger.info("transfer environment: %s", json.dumps(environment))

    # A CPU UID broadcast and one CCO arena, matching the production mapping.
    uid = [Communicator.get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(uid, src=0)
    comm = Communicator.init(world, rank, uid[0], per_rank_vmm=64 * 1024**2)
    rows, reports, clocks = [], [], {}
    device = torch.device("cuda", local_rank)
    try:
        # One registered window for the whole sweep; closing and re-registering
        # windows between shapes is not needed and is unreliable on older ROCm.
        arena = SymmetricArena(comm, regions)
        local = _from_gpu_ptr(
            arena.local_ptr("scatter"), (arena.layout.total_bytes,), torch.uint8
        )
        ctx = Context(rank, world, device, arena, local, reports, clocks)
        try:
            for hidden in args.hidden:
                for op in ("scatter", "dispatch"):
                    modes = args.scatter_mode if op == "scatter" else ["single-peer"]
                    for src, dst, mode, samples in itertools.product(
                        args.source_ranks, args.destination_ranks, modes, args.samples
                    ):
                        result = test_transfer(
                            op,
                            hidden,
                            src,
                            dst,
                            mode,
                            samples,
                            args.warmup,
                            args.iters,
                            ctx,
                        )
                        result.pop("ctx", None)
                        if rank == 0:
                            rows.append(result)
        finally:
            torch.cuda.synchronize(device)
            dist.barrier()
            del ctx, local
            arena.close()
        gathered = [None] * world
        dist.all_gather_object(gathered, reports)
        if rank == 0:
            aiter.logger.info(
                "transfer summary (markdown; device raw includes timer overhead):\n%s",
                pd.DataFrame(rows).to_markdown(index=False),
            )
            aiter.logger.info(
                "tdm us is single-operation HIP-event time including launch/staging; "
                "device p50 us is issue-to-completion. TB/s uses logical payload "
                "bytes and is not physical link bandwidth. Empty control is "
                "reported separately and is not automatically subtracted."
            )
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(
                        {
                            "environment": environment,
                            "summary": rows,
                            "cases": [case for group in gathered for case in group],
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
    finally:
        comm.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
