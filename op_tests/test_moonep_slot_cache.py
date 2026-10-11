# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Multi-rank test for MoonEPWeightPool.prefetch with a resident table.

The pool holds two parts per expert, a bf16 weight and a 48-byte block that
leaves the segment off the VMM granularity.  Asserts:
  1. without ``resident`` every live slot is a byte copy of its owner's home,
     in every part
  2. with ``resident`` naming what a slot holds, that slot is not touched
     (proved by corrupting it first)
  3. a slot whose resident entry names another expert is refreshed
  4. pools of two 4-rank EP groups bootstrap side by side inside one world,
     prefetch within their own group, and hold no fds once built

Run:
    torchrun --standalone --nproc-per-node=8 op_tests/test_moonep_slot_cache.py
"""

import os
import sys

import torch
import torch.distributed as dist

EPN, B = 4, 3
PARTS = [((256, 1024), torch.bfloat16), ((48,), torch.uint8)]


def expert_parts(e, dev):
    g = torch.Generator(device="cpu").manual_seed(500 + e)
    w = (torch.randn(*PARTS[0][0], generator=g) * 0.05).to(torch.bfloat16)
    s = torch.randint(0, 256, PARTS[1][0], generator=g, dtype=torch.uint8)
    return [w.to(dev), s.to(dev)]


def home_of(pool_rank, base, dev):
    return [
        torch.stack(t)
        for t in zip(
            *(expert_parts(base + pool_rank * EPN + k, dev) for k in range(EPN))
        )
    ]


def main() -> int:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)

    import mori

    cpu_group = dist.new_group(backend="gloo")
    torch._C._distributed_c10d._register_process_group("mori", cpu_group)
    mori.shmem.shmem_torch_process_group_init("mori")

    from aiter.ops.flydsl.kernels.moonep_weights import MoonEPWeightPool

    pool = MoonEPWeightPool(
        rank=rank,
        world_size=world,
        experts_per_rank=EPN,
        prefetch_slots=B,
        parts=PARTS,
        block_num=256,
    )
    pool.stage_home(home_of(rank, 0, dev))

    nxt, far = (rank + 1) % world, (rank + 3) % world
    sel = torch.tensor([nxt * EPN, -1, far * EPN + 2], dtype=torch.int32, device=dev)
    failures = []

    def check(tag, slot, expected, target=pool):
        for i, want in enumerate(expected):
            if not torch.equal(target.prefetched[i][slot], want):
                failures.append(f"rank{rank} {tag}: slot {slot} part {i} mismatch")

    # 1. no resident table: plain copy.
    pool.prefetch(sel)
    torch.cuda.synchronize()
    check("plain", 0, expert_parts(nxt * EPN, dev))
    check("plain", 2, expert_parts(far * EPN + 2, dev))

    # 2. resident names exactly what was selected: nothing may move.
    resident = sel.clone()
    poison = [
        torch.full(PARTS[0][0], 7.0, dtype=torch.bfloat16, device=dev),
        torch.full(PARTS[1][0], 7, dtype=torch.uint8, device=dev),
    ]
    for slot in (0, 2):
        for i, p in enumerate(poison):
            pool.prefetched[i][slot].copy_(p)
    pool.prefetch(sel, resident)
    torch.cuda.synchronize()
    check("hit", 0, poison)
    check("hit", 2, poison)

    # 3. slot 2 claims another expert: it must refresh, slot 0 must not.
    resident[2] = far * EPN + 1
    pool.prefetch(sel, resident)
    torch.cuda.synchronize()
    check("partial", 0, poison)
    check("partial", 2, expert_parts(far * EPN + 2, dev))

    # 4. two EP groups: positions are group-local, experts never cross groups.
    half = world // 2
    groups = [
        dist.new_group(list(range(g * half, (g + 1) * half)), backend="gloo")
        for g in range(2)
    ]
    gid, local = rank // half, rank % half
    fds_before = len(os.listdir("/proc/self/fd"))
    sub_pools = []
    for _ in range(4):
        sub_pool = MoonEPWeightPool(
            rank=local,
            world_size=half,
            experts_per_rank=EPN,
            prefetch_slots=B,
            parts=PARTS,
            block_num=256,
            group=groups[gid],
        )
        sub_pool.stage_home(home_of(local, 1000 * gid, dev))
        sub_pools.append(sub_pool)
    fds_grown = len(os.listdir("/proc/self/fd")) - fds_before
    if fds_grown > len(sub_pools):
        failures.append(f"rank{rank} groups: {fds_grown} fds left open by 4 pools")
    peer = (local + 1) % half
    sel = torch.tensor([peer * EPN + 1, -1, -1], dtype=torch.int32, device=dev)
    sub_pools[-1].prefetch(sel)
    torch.cuda.synchronize()
    check("groups", 0, expert_parts(1000 * gid + peer * EPN + 1, dev), sub_pools[-1])
    for sub_pool in sub_pools:
        sub_pool.close()

    flags = torch.tensor([len(failures)], device=dev)
    dist.all_reduce(flags)
    for line in failures:
        print(line, flush=True)
    if rank == 0:
        print("PASS" if flags.item() == 0 else f"FAIL ({flags.item()})", flush=True)
    pool.close()
    dist.destroy_process_group()
    return 0 if flags.item() == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
