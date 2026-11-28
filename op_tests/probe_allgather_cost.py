# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Probe: how much of MoonEP's per-layer all-gather is transfer vs waiting?

The prefill trace attributes 1760 ms / 1178 calls (1494 us/call) to
``ncclDevKernel_Generic``, zero on the mori side -- it is the ``[E]`` int32
tokens-per-expert histogram MoonEP all-gathers once per MoE layer to balance
against the *global* expert load (``moonep_dispatch_combine_op.py:223``).

Before replacing it with a symmetric-memory push, settle what that 1494 us is:

* **Barrier-aligned** -- every rank enters at the same instant, so the number is
  the bare RCCL cost with no skew to absorb.  ~20 us here means the trace figure
  is almost entirely *waiting for stragglers*, and moving the exchange into the
  planning kernel just relocates the wait to dispatch's exit barrier: not worth
  doing.  Hundreds of us here means RCCL itself is slow on 12 KB (small-message
  protocol, communicator churn) and the replacement pays for itself.
* **Unaligned** -- ranks enter as they arrive, like the real serving path.  The
  gap between the two is the skew MoonEP is absorbing, and it is *not* recovered
  by changing the transport.

Also times a mori symmetric-heap push of the same payload, which is what the
replacement would actually do, so the comparison is against a real alternative
rather than against upstream's 60 us microbenchmark (a different machine, a
different measurement -- apples to oranges).

Run:
    torchrun --nproc-per-node 8 op_tests/probe_allgather_cost.go.py
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import torch
import torch.distributed as dist


def timed(fn, iters, warmup, align, dev):
    """Median wall time per call, in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(dev)
    samples = []
    for _ in range(iters):
        if align:
            dist.barrier()
            torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(dev)
        samples.append((time.perf_counter() - t0) * 1e6)
    return samples


def report(rank, name, s):
    if rank != 0:
        return
    s = sorted(s)
    print(
        f"  {name:<34s} median={statistics.median(s):8.1f} us   "
        f"p10={s[len(s)//10]:8.1f}   p90={s[9*len(s)//10]:8.1f}   "
        f"max={s[-1]:8.1f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=256, help="E; payload is E int32")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--skew-us", type=int, default=0,
                    help="rank r sleeps r*skew before entering (fake straggler)")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = rank % torch.cuda.device_count()
    torch.cuda.set_device(dev)

    E = args.experts
    payload = E * 4 * world
    tpe = torch.ones(E, dtype=torch.int32, device="cuda")
    gathered = [torch.empty_like(tpe) for _ in range(world)]

    if rank == 0:
        print("=" * 84)
        print(f"E={E}  world={world}  payload={payload} B ({payload/1024:.1f} KiB)  "
              f"iters={args.iters}")
        print("trace 里这一项是 1494 us/call (1760 ms / 1178 calls)")
        print("=" * 84)

    def ag():
        dist.all_gather(gathered, tpe)

    report(rank, "all_gather, barrier 对齐", timed(ag, args.iters, args.warmup, True, dev))
    report(rank, "all_gather, 不对齐", timed(ag, args.iters, args.warmup, False, dev))

    if args.skew_us:
        def skewed():
            if rank:
                t = time.perf_counter() + rank * args.skew_us / 1e6
                while time.perf_counter() < t:
                    pass
            dist.all_gather(gathered, tpe)
        report(rank, f"all_gather, 人为偏斜 {args.skew_us}us/rank",
               timed(skewed, args.iters, args.warmup, True, dev))

    # The actual alternative: push our row into every peer over the symmetric
    # heap, then one device barrier. This is what the replacement would cost.
    try:
        import mori.shmem as ms
        from mori.shmem import (
            mori_shmem_create_tensor,
            mori_shmem_free_tensor,
            symm_mori_shmem_tensor,
        )

        ms.shmem_torch_process_group_init("default")
        board = mori_shmem_create_tensor((world, E), torch.int32)
        board.zero_()
        ms.shmem_barrier_all()
        peers = [symm_mori_shmem_tensor(board, p) for p in range(world)]

        def push():
            for p in range(world):
                peers[p][rank].copy_(tpe)
            ms.shmem_barrier_on_stream()

        report(rank, "对称堆 push + barrier (替代方案)",
               timed(push, args.iters, args.warmup, True, dev))
        ms.shmem_barrier_all()
        mori_shmem_free_tensor(board)
    except Exception as e:  # noqa: BLE001
        if rank == 0:
            print(f"  对称堆方案跳过: {type(e).__name__}: {e}")

    if rank == 0:
        print("=" * 84)
        print("判读：对齐后 ~20us  => 1494us 几乎全是等待，换传输白改")
        print("      对齐后 几百us => RCCL 小消息路径问题，换传输才有意义")
        print("      对齐 vs 不对齐的差 = MoonEP 正在吸收的 rank 间偏斜（换传输不会消失）")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
