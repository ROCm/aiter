# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness test for the fused TP MoE collectives.

Compares the one-launch P2P push-AllGather against
``dist.all_gather_into_tensor`` for every region it moves (MXFP4 payload, E8M0
scales, routing ids and weights), and the one-launch pull-ReduceScatter against
``dist.reduce_scatter_tensor``.  Each case is replayed several times so the
epoch/flag protocol is exercised across calls.

    torchrun --standalone --nproc_per_node=8 \
        op_tests/multigpu_tests/test_mega_moe_tp_allgather.py
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist

from aiter import dtypes
from aiter.ops.flydsl.kernels.mega_moe_tp.collectives import TpMoeCollectives


def setup():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)
    return rank, world, device


def _check(name, actual, expected, *, exact, rank, total, repeat):
    if exact:
        if torch.equal(actual, expected):
            return 0
        detail = f"{int((actual != expected).sum().item())} elems differ"
    else:
        # The pull-reduce accumulates in FP32 and NCCL reduces in BF16, so the
        # two disagree by rounding; gate on relative L2 rather than bitwise.
        err = torch.linalg.vector_norm((actual.float() - expected.float()).flatten())
        ref = torch.linalg.vector_norm(expected.float().flatten())
        rel = float(err / ref) if float(ref) else 0.0
        if rel < 5e-3:
            return 0
        detail = f"rel_l2={rel:.6f}"
    print(
        f"[FAIL] rank={rank} M={total} repeat={repeat} {name}: {detail}",
        flush=True,
    )
    return 1


def _time(fn, args, device) -> float:
    """Mean microseconds per call, maximised across ranks and minimised over rounds.

    Both legs end (AllGather) or begin (ReduceScatter) with a rank barrier, so a
    single rank's number is meaningless on its own: the slowest rank is what the
    group actually waits for.
    """
    for _ in range(args.warmup):
        fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(3):
        torch.cuda.synchronize()
        dist.barrier()
        start.record()
        for _ in range(args.iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        local = torch.tensor(
            start.elapsed_time(end) / args.iters * 1000.0,
            dtype=torch.float32,
            device=device,
        )
        dist.all_reduce(local, op=dist.ReduceOp.MAX)
        best = min(best, float(local.item()))
    return best


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dim", type=int, default=3584)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="*",
        default=[8, 32, 128, 1024, 8192],
        help="GLOBAL token counts.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--perf", action="store_true", help="Also time vs NCCL.")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args(argv)

    rank, world, device = setup()
    try:
        h, k = args.model_dim, args.topk
        max_local = max(args.tokens) // world
        comm = TpMoeCollectives(
            rank=rank,
            world_size=world,
            model_dim=h,
            topk=k,
            max_local_tokens=max_local,
            device=device,
        )
        failures = 0
        for total in args.tokens:
            m = total // world
            gen = torch.Generator(device=device).manual_seed(1234 + rank)
            xq = torch.randint(
                0, 256, (m, h // 2), dtype=torch.uint8, device=device, generator=gen
            )
            xs = torch.randint(
                0, 256, (m, h // 32), dtype=torch.uint8, device=device, generator=gen
            )
            ids = torch.randint(
                0, 896, (m, k), dtype=torch.int32, device=device, generator=gen
            )
            wts = torch.rand((m, k), dtype=torch.float32, device=device, generator=gen)
            partial = torch.randn(
                (total, h), dtype=dtypes.bf16, device=device, generator=gen
            )

            ref_q = torch.empty((total, h // 2), dtype=torch.uint8, device=device)
            ref_s = torch.empty((total, h // 32), dtype=torch.uint8, device=device)
            ref_i = torch.empty((total, k), dtype=torch.int32, device=device)
            ref_w = torch.empty((total, k), dtype=torch.float32, device=device)
            ref_y = torch.empty((m, h), dtype=dtypes.bf16, device=device)
            dist.all_gather_into_tensor(ref_q, xq)
            dist.all_gather_into_tensor(ref_s, xs)
            dist.all_gather_into_tensor(ref_i, ids)
            dist.all_gather_into_tensor(ref_w, wts)
            dist.reduce_scatter_tensor(ref_y, partial)

            for repeat in range(args.repeats):
                got = comm.all_gather(xq, xs, ids, wts)
                for name, actual, expected in (
                    ("payload", got.payload[:total], ref_q),
                    ("scale", got.scale[:total], ref_s),
                    ("ids", got.topk_ids[:total], ref_i),
                    ("weights", got.topk_weights[:total], ref_w),
                ):
                    failures += _check(
                        name,
                        actual,
                        expected,
                        exact=True,
                        rank=rank,
                        total=total,
                        repeat=repeat,
                    )
                comm.partial_buffer(total).copy_(partial)
                y = comm.reduce_scatter(m)
                failures += _check(
                    "reduce_scatter",
                    y,
                    ref_y,
                    exact=False,
                    rank=rank,
                    total=total,
                    repeat=repeat,
                )
            if rank == 0 and not failures:
                print(f"[OK] M={total} m={m}", flush=True)

            if args.perf:
                nccl_ag = _time(
                    lambda: (
                        dist.all_gather_into_tensor(ref_q, xq),
                        dist.all_gather_into_tensor(ref_s, xs),
                        dist.all_gather_into_tensor(ref_i, ids),
                        dist.all_gather_into_tensor(ref_w, wts),
                    ),
                    args,
                    device,
                )
                push_ag = _time(
                    lambda: comm.all_gather(xq, xs, ids, wts), args, device
                )
                nccl_rs = _time(
                    lambda: dist.reduce_scatter_tensor(ref_y, partial), args, device
                )
                pull_rs = _time(lambda: comm.reduce_scatter(m), args, device)
                if rank == 0:
                    print(
                        f"[PERF] M={total:6d} "
                        f"AG nccl={nccl_ag:8.2f}us fused={push_ag:8.2f}us "
                        f"({nccl_ag / push_ag:4.2f}x) | "
                        f"RS nccl={nccl_rs:8.2f}us fused={pull_rs:8.2f}us "
                        f"({nccl_rs / pull_rs:4.2f}x)",
                        flush=True,
                    )

        flag = torch.tensor(failures, dtype=torch.int32, device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            print(
                "TP_COMM_UT_OK" if int(flag.item()) == 0 else "TP_COMM_UT_FAILED",
                flush=True,
            )
        return 0 if int(flag.item()) == 0 else 1
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
