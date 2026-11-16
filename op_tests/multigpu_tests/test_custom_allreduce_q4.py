# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""INT4- / INT8-coded Lamport two-shot all-reduce (gfx1250, bf16, opt-in, lossy).

tensor_model_parallel_all_reduce takes it for bf16 inputs of
AITER_CUSTOM_AR_Q4_2S_MIN_SIZE < bytes <= AITER_CUSTOM_AR_Q4_2S_MAX_SIZE (the
test sets 8 MiB and 256 MiB unless they are given). Every 28 elements travel
as one 16-byte line (28 x int4 + a bf16 scale), or with --codec 8
(AITER_CUSTOM_AR_Q4_2S_CODEC=8) every 14 (14 x int8 + the scale). The result
is lossy (INT4: the class of the INT4 quick-reduce). The test bounds
max|out - ref| / max|ref| by --tol (0.2 for INT4, 0.03 for INT8) against an
fp32 sum of every rank's input (each rank regenerates all inputs, so the
reference needs no collective), and requires every rank to hold the same bits.

Cases: (m, n) with n = 7168 by default; sizes that leave a block's last
iteration partial (m = 600, 1638) are included. A mixed sequence then runs
seeded random sizes across the window boundary, eager and inside CUDA graphs
replayed with fresh inputs, with eager calls of other sizes in between.

Timing: a CUDA graph of CALLS calls replayed REPLAYS times; median per rank,
slowest rank reported; "fallback us" is the same input with the route off.
"""

import argparse
import os
import random
import statistics
from multiprocessing import Pool, set_start_method

import pandas as pd
import torch
import torch.distributed as dist

import aiter
from aiter import dtypes
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    graph_capture,
    init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.jit.utils.chip_info import get_gfx

SUPPORTED_GFX = ["gfx1250"]
CALLS = 10
REPLAYS = 10
# the fallback (RCCL past the custom all-reduce window) is timed up to this size only: on one MI455X tray
# RCCL fails with "invalid argument" at 224 MiB
FALLBACK_MAX_BYTES = 128 << 20


def inputs(tp, salt, m, n):
    """Every rank's input for one call (the same on every rank) and their fp32 sum."""
    xs = []
    for r in range(tp):
        g = torch.Generator(device="cuda").manual_seed(salt * 131 + r)
        xs.append((torch.randn(m, n, device="cuda", generator=g) * 0.5).to(dtypes.bf16))
    return xs, torch.stack([x.float() for x in xs]).sum(0)


def check(tp, rank, out, ref):
    """(relative error, whether all ranks hold the same bits)."""
    rel = ((out.float() - ref).abs().max() / ref.abs().max()).item()
    ck = torch.tensor(
        [out.view(torch.int16).double().sum().item(), rel], dtype=torch.float64
    )
    cks = [torch.zeros(2, dtype=torch.float64) for _ in range(tp)]
    dist.all_gather(cks, ck, group=get_tp_group().cpu_group)
    return max(c[1].item() for c in cks), all(c[0] == cks[0][0] for c in cks)


def graph_time(x):
    """Median per-call time of a CUDA graph of CALLS all-reduces of x, slowest rank."""
    with graph_capture() as gc:
        tensor_model_parallel_all_reduce(x)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=gc.stream):
            for _ in range(CALLS):
                tensor_model_parallel_all_reduce(x)
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPLAYS):
        get_tp_group().barrier()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        graph.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3 / CALLS)
    graph.reset()
    t = torch.tensor([statistics.median(ts)])
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=get_tp_group().cpu_group)
    return t.item()


def test_case(ca, tp, rank, m, n, tol):
    xs, ref = inputs(tp, m * 7 + n, m, n)
    x = xs[rank]
    routed = ca.should_lamport_q4_2s(x)
    out = tensor_model_parallel_all_reduce(x)
    torch.cuda.synchronize()
    rel, same = check(tp, rank, out, ref)
    row = {"tp": tp, "m": m, "n": n, "MiB": round(x.numel() * 2 / 2**20, 2)}
    row.update({"q4": routed, "rel err": rel, "ranks agree": same})
    row["q4 us"] = graph_time(x)
    row["fallback us"] = None
    if x.numel() * 2 <= FALLBACK_MAX_BYTES:
        cap = ca._lamportq4_max_bytes
        ca._lamportq4_max_bytes = 0  # the route off: what the call takes without it
        row["fallback us"] = graph_time(x)
        ca._lamportq4_max_bytes = cap
    row["ok"] = same and (rel < tol)
    return row


def mixed_sequence(ca, tp, rank, ms, n, tol, calls=60):
    """Seeded random sizes across the window, eager and in replayed graphs."""
    rng = random.Random(4242)
    worst, bad, routed = 0.0, 0, 0

    def one(salt, m, x=None, out=None, ref=None):
        nonlocal worst, bad, routed
        if out is None:
            xs, ref = inputs(tp, salt, m, n)
            x = xs[rank]
            out = tensor_model_parallel_all_reduce(x)
            torch.cuda.synchronize()
        routed += bool(ca.should_lamport_q4_2s(x))
        rel, same = check(tp, rank, out, ref)
        worst = max(worst, rel)
        bad += not (same and rel < tol)

    for i in range(calls):
        one(1000 + i, rng.choice(ms))
    for gi in range(3):
        seq = [rng.choice(ms) for _ in range(6)]
        xs = [inputs(tp, 2000 + gi * 10 + j, m, n)[0][rank] for j, m in enumerate(seq)]
        with graph_capture() as gc:
            for x in xs:
                tensor_model_parallel_all_reduce(x)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=gc.stream):
                outs = [tensor_model_parallel_all_reduce(x) for x in xs]
        for rep in range(4):
            refs = []
            for j, (m, x) in enumerate(zip(seq, xs)):
                nxs, ref = inputs(tp, 3000 + gi * 100 + rep * 10 + j, m, n)
                x.copy_(nxs[rank])
                refs.append(ref)
            graph.replay()
            torch.cuda.synchronize()
            for m, x, o, ref in zip(seq, xs, outs, refs):
                one(0, m, x, o, ref)
            one(4000 + gi * 10 + rep, rng.choice(ms))
        graph.reset()
    return {
        "tp": tp,
        "calls": calls + 3 * 4 * 7,
        "q4 calls": routed,
        "worst rel err": worst,
        "failures": bad,
    }


def worker(tp, rank, ms, n, tol, init_method):
    torch.cuda.set_device(rank)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp, rank=rank, distributed_init_method=init_method
    )
    ensure_model_parallel_initialized(tp, 1)
    ca = get_tp_group().device_communicator.ca_comm
    assert ca is not None and not ca.disabled, "custom allreduce is disabled"
    assert getattr(ca, "_lamportq4_max_bytes", 0) > 0, "the INT4 two-shot inbox is off"
    dist.all_reduce(torch.zeros(1, device="cuda"), group=get_tp_group().device_group)
    rows = [test_case(ca, tp, rank, m, n, tol) for m in ms]
    mixed = mixed_sequence(ca, tp, rank, ms, n, tol)
    torch.cuda.synchronize()
    get_tp_group().barrier()
    destroy_model_parallel()
    destroy_distributed_environment()
    return rows, mixed


def run_tp(tp, ms, n, tol):
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with Pool(processes=tp) as pool:
        jobs = [
            pool.apply_async(worker, (tp, r, ms, n, tol, init_method))
            for r in range(tp)
        ]
        # a protocol bug hangs instead of failing
        results = [j.get(timeout=3600) for j in jobs]
    return results[0]  # every row is already reduced over the ranks


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "INT4 Lamport two-shot unsupported on %s; skipping", get_gfx()
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="INT4-coded Lamport two-shot all-reduce (gfx1250, bf16, lossy)",
    )
    parser.add_argument("-t", "--tp", type=int, nargs="*", default=[2, 4])
    parser.add_argument(
        "-m",
        type=int,
        nargs="*",
        default=[300, 545, 600, 1024, 1638, 2048, 4096, 8192, 16384],
    )
    parser.add_argument("-n", type=int, default=7168)
    parser.add_argument("--codec", type=int, choices=[4, 8], default=4)
    parser.add_argument("--tol", type=float, default=None)
    args = parser.parse_args()
    tol = args.tol if args.tol is not None else (0.2 if args.codec == 4 else 0.03)
    os.environ["AITER_CUSTOM_AR_Q4_2S_CODEC"] = str(args.codec)
    os.environ.setdefault("AITER_CUSTOM_AR_Q4_2S_MIN_SIZE", str(8 << 20))
    os.environ.setdefault("AITER_CUSTOM_AR_Q4_2S_MAX_SIZE", str(256 << 20))
    ngpus = torch.cuda.device_count()
    set_start_method("spawn", force=True)
    ok = True
    for tp in args.tp:
        if tp > ngpus:
            aiter.logger.warning("tp %d > %d GPUs; skipping", tp, ngpus)
            continue
        rows, mixed = run_tp(tp, args.m, args.n, tol)
        print(pd.DataFrame(rows).to_string(index=False), flush=True)
        print(f"mixed sequence: {mixed}", flush=True)
        ok = ok and all(r["ok"] for r in rows) and mixed["failures"] == 0
    print("ALL PASSED" if ok else "FAILED", flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
