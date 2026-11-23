# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mesh-lending all-reduce: bit-exact checks and per-call time.

No arguments: the checks at TP2 and TP4, skipped with a reason where the node
has no xGMI mesh, too few GPUs or KFD older than 1.17.

Per-call time of one configuration:
    python test_mesh_lend_all_reduce.py --bench --tp 2 --bits 4 --mb 16,80,256
    python test_mesh_lend_all_reduce.py --bench --tp 4 --max-lenders 0
"""

import argparse
import fcntl
import glob
import multiprocessing as mp
import statistics
import struct
import time
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist

from aiter.dist.device_communicators.mesh_lend_all_reduce import MeshLendAllReduce
from aiter.dist.utils import get_open_port

MB = 1024 * 1024


def unavailable_reason(tp_size: int, devices: list[int]) -> str:
    if torch.cuda.device_count() <= max(devices[:tp_size]):
        return f"needs GPUs {devices[:tp_size]}, found {torch.cuda.device_count()}"
    links = glob.glob("/sys/class/kfd/kfd/topology/nodes/*/io_links/*/properties")
    if not any("type 11\n" in Path(p).read_text() for p in links):
        return "no xGMI link in the KFD topology"
    try:
        with open("/dev/kfd", "rb", buffering=0) as kfd:
            # AMDKFD_IOC_GET_VERSION
            major, minor = struct.unpack("II", fcntl.ioctl(kfd, 0x80084B01, bytes(8)))
    except OSError as e:
        return f"/dev/kfd: {e}"
    if (major, minor) < (1, 17):
        return f"KFD ioctl {major}.{minor} < 1.17"
    return ""


def quantized(x: torch.Tensor, bits: int) -> torch.Tensor:
    """What a rank's input becomes on the wire and back, in float32.

    Groups of 16 values. Scale byte: sign of the extremum, 5-bit exponent
    (bias 12), 2-bit mantissa, rounded to nearest at 4 bits and up at 8.
    Codes in [-B, B - 1] with the extremum on -B.
    """
    B = 1 << (bits - 1)
    v = x.reshape(-1).float()
    n = v.numel()
    v = torch.cat([v, v.new_zeros((-n) % 16)])
    g = v.view(-1, 16)
    mp_, mn = g.max(dim=1).values, (-g).max(dim=1).values
    neg = mn > mp_
    b = torch.where(neg, mn, mp_).abs().view(torch.int32)
    e = (b >> 23) - 127
    m2 = ((b & 0x7FFFFF) + ((1 << 21) - 1 if bits > 4 else 1 << 20)) >> 21
    e = torch.where(m2 == 4, e + 1, e)
    m2 = torch.where(m2 == 4, torch.zeros_like(m2), m2)
    e5 = e + 12
    m2 = torch.where(
        e5 < 0, torch.zeros_like(m2), torch.where(e5 > 31, torch.full_like(m2, 3), m2)
    )
    e5 = e5.clamp(0, 31)
    sign = torch.where(neg, 1.0, -1.0).to(torch.float32)
    recip = torch.tensor(
        [1 / 4, 1 / 5, 1 / 6, 1 / 7], dtype=torch.float32, device=x.device
    )[m2.long()]
    one = torch.ones_like(recip)
    enc = sign * recip * torch.ldexp(one, 14 + (bits - 1) - e5)
    dec = sign * (4 + m2).float() * torch.ldexp(one, e5 - 14 - (bits - 1))
    q = torch.round(g * enc[:, None]).clamp(-B, B - 1)
    return (q * dec[:, None]).view(-1)[:n].view(x.shape)


def reference(xs: list[torch.Tensor], bits: int) -> torch.Tensor:
    """The result every rank must produce, bit for bit."""
    if bits:
        return (quantized(xs[0], bits) + quantized(xs[1], bits) + 0.0).to(
            torch.bfloat16
        )
    acc = xs[0].float()
    for x in xs[1:]:
        acc = acc + x.float()
    return acc.to(torch.bfloat16)


def make_input(rank: int, numel: int, salt: int, device) -> torch.Tensor:
    gen = torch.Generator(device=device).manual_seed(1000 * rank + salt)
    return torch.randn(numel, generator=gen, device=device).to(torch.bfloat16)


def mismatches(out: torch.Tensor, want: torch.Tensor) -> int:
    return int((out.view(torch.int16) != want.view(torch.int16)).sum().item())


def check_case(comm, rank, tp_size, case, salt, barrier) -> int:
    """Wrong elements over four calls issued back to back.

    Fails if a block reads a piece before its arrival flag, a piece or slice
    boundary is off, the tail or an unaligned tensor is written past its
    end, call k + 1 lands in an inbox a slower rank still reads for call k, or
    an in-place call writes over input that has not left.
    """
    numel, bits, offset = case["numel"], case["bits"], case["offset"]
    lo = 8 + offset  # odd offset: the data pointers are only 2-byte aligned
    calls = 4
    xs = [
        [make_input(r, numel, salt + call, comm.device) for r in range(tp_size)]
        for call in range(calls)
    ]
    bufs = [
        torch.full((numel + 16,), 3.0, dtype=torch.bfloat16, device=comm.device)
        for _ in range(calls)
    ]
    # Odd calls run in place.
    outs = [buf if call & 1 else buf.clone() for call, buf in enumerate(bufs)]
    torch.cuda.synchronize()
    barrier.wait()
    if rank == tp_size - 1:
        # The last rank is busy on another stream, so it reads its inbox late
        # while the other ranks move on to the next call.
        with torch.cuda.stream(torch.cuda.Stream(comm.device)):
            load = torch.ones(2048, 2048, device=comm.device)
            for _ in range(4):
                load = load @ load
    for call in range(calls):
        inp = bufs[call][lo : lo + numel]
        # Written by a kernel right before the call, with no host
        # synchronization in between.
        inp.copy_(xs[call][rank])
        comm.all_reduce(inp, outs[call][lo : lo + numel], bits=bits)
    bad = 0
    for call in range(calls):
        got = outs[call][lo : lo + numel] * 1.0
        bad += mismatches(got, reference(xs[call], bits))
        bad += int((outs[call][:lo] != 3.0).sum().item())
        bad += int((outs[call][lo + numel :] != 3.0).sum().item())
    return bad


def bench_case(comm, rank, case, reps, barrier, start_at) -> dict:
    numel, bits = case["numel"], case["bits"]
    xs = [make_input(rank, numel, salt, comm.device) for salt in range(2)]
    out = torch.empty_like(xs[0])
    torch.cuda.synchronize()
    # Several groups started together time the same case at the same moment.
    time.sleep(max(0.0, start_at - time.time()))
    times = []
    for i in range(reps + 5):
        barrier.wait()
        t0 = time.perf_counter()
        comm.all_reduce(xs[i & 1], out, bits=bits)
        torch.cuda.synchronize()
        if i >= 5:
            times.append(time.perf_counter() - t0)
    # Back to back, as a model forward issues them.
    barrier.wait()
    t0 = time.perf_counter()
    y = xs[0]
    for _ in range(20):
        y = comm.all_reduce(y, bits=bits)
    torch.cuda.synchronize()
    chained = (time.perf_counter() - t0) / 20
    return {
        "median_us": statistics.median(times) * 1e6,
        "best_us": min(times) * 1e6,
        "chained_us": chained * 1e6,
    }


def rank_main(
    rank, tp_size, devices, port, cases, max_lenders, reps, cycles, start_at, barrier, q
):
    try:
        device = torch.device(f"cuda:{devices[rank]}")
        torch.cuda.set_device(device)
        # As a vLLM worker: not the default stream.
        torch.cuda.set_stream(torch.cuda.Stream(device))
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=tp_size,
        )
        max_bytes = max(max(c["numel"] for c in cases) * 2, 4096)
        # Fails if destroy leaks an SDMA queue: an engine has 8.
        kwargs = {"max_bytes": max_bytes, "max_lenders": max_lenders}
        for _ in range(cycles):
            MeshLendAllReduce(dist.group.WORLD, device, **kwargs).close()
        comm = MeshLendAllReduce(dist.group.WORLD, device, **kwargs)
        rows = []
        for i, case in enumerate(cases):
            row = {"lenders": comm.num_lenders}
            if reps:
                at = start_at + 30 * i if start_at else 0.0
                row.update(bench_case(comm, rank, case, reps, barrier, at))
            row["mismatches"] = check_case(comm, rank, tp_size, case, 10 * i, barrier)
            rows.append(row)
        torch.cuda.synchronize()
        dist.barrier()
        comm.close()
        dist.destroy_process_group()
        q.put((rank, rows, None))
    except Exception as e:  # noqa: BLE001  the parent must hear of any failure
        barrier.abort()
        q.put((rank, None, repr(e)))


def test_mesh_lend_all_reduce(
    tp_size, cases, devices=None, max_lenders=7, reps=0, cycles=0, start_at=0.0
):
    """One TP group over `cases`; one row per case.

    Fails if any rank's output differs from the reference in any element, so
    also if two ranks disagree.
    """
    devices = devices or list(range(tp_size))
    ctx = mp.get_context("spawn")
    barrier, q = ctx.Barrier(tp_size), ctx.Queue()
    port = get_open_port()
    shared = (tp_size, devices, port, cases, max_lenders, reps, cycles, start_at)
    procs = [
        ctx.Process(target=rank_main, args=(r, *shared, barrier, q))
        for r in range(tp_size)
    ]
    for p in procs:
        p.start()
    results = sorted(q.get(timeout=900) for _ in procs)
    for p in procs:
        p.join(timeout=60)
    errors = [f"rank {rank}: {err}" for rank, _, err in results if err]
    assert not errors, "; ".join(errors)
    rows = []
    for i, case in enumerate(cases):
        per_rank = [r[1][i] for r in results]
        row = {
            "tp": tp_size,
            "lenders": per_rank[0]["lenders"],
            "bits": case["bits"],
            "numel": case["numel"],
            "MiB": round(case["numel"] * 2 / MB, 2),
            "offset": case["offset"],
        }
        if reps:
            for key in ("median_us", "best_us", "chained_us"):
                row[key] = round(max(r[key] for r in per_rank), 1)
        row["mismatches"] = sum(r["mismatches"] for r in per_rank)
        rows.append(row)
    bad = [r for r in rows if r["mismatches"]]
    assert not bad, f"mesh-lending all-reduce differs from the reference: {bad}"
    return rows


def default_cases(tp_size: int) -> list[dict]:
    """Shapes that reach every branch: below one vector, a tail, pieces that
    are empty, several pieces per path, and a tensor that is not 16-byte
    aligned. Large before small, so a stale inbox would show."""
    numels = [4096 * 2560, 4096 * 2560 + 5, 1000 * 2560, 300 * 4096 + 3, 517, 7, 1]
    cases = [{"numel": n, "bits": 0, "offset": o} for n in numels for o in (0, 1)]
    if tp_size == 2:
        cases += [
            {"numel": n, "bits": bits, "offset": o}
            for bits in (4, 8)
            for n in numels
            for o in (0, 1)
        ]
    return cases


if __name__ == "__main__":
    mp.freeze_support()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", action="store_true", help="time --mb sizes")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--devices", type=str, default=None, help="e.g. 0,1")
    parser.add_argument("--max-lenders", type=int, default=7)
    parser.add_argument("--bits", type=int, default=0, choices=(0, 4, 8))
    parser.add_argument("--mb", type=str, default="1,4,16,64,80,256")
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument(
        "--start-at",
        type=float,
        default=0.0,
        help="epoch seconds at which the first size is timed; each later size "
        "30 s after the one before. For groups that run at the same time.",
    )
    args = parser.parse_args()
    devices = [int(d) for d in args.devices.split(",")] if args.devices else None

    rows = []
    if args.bench:
        cases = [
            {"numel": int(float(mb) * MB) // 2, "bits": args.bits, "offset": 0}
            for mb in args.mb.split(",")
        ]
        rows = test_mesh_lend_all_reduce(
            args.tp,
            cases,
            devices,
            args.max_lenders,
            reps=args.reps,
            start_at=args.start_at,
        )
    else:
        for tp_size in (2, 4):
            why = unavailable_reason(tp_size, devices or list(range(tp_size)))
            if why:
                print(f"SKIP mesh-lending all-reduce TP{tp_size}: {why}")
                continue
            for max_lenders in (7, 0):
                rows += test_mesh_lend_all_reduce(
                    tp_size, default_cases(tp_size), devices, max_lenders, cycles=10
                )
    if rows:
        print(pd.DataFrame(rows).to_markdown(index=False))
    print("done")
