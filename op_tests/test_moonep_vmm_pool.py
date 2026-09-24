# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Multi-rank test for MoonEPVmmPool: row addressing + fused_moe consumption.

Asserts (not prints) the two properties the [E+B] layout rests on:

  1. every rank reads global expert ``e`` at row ``global_row(e)`` and gets the
     owning rank's bytes, and its own prefetch tail is private
  2. a ``fused_moe`` over the stitched range is bit-identical to one over a
     fully local copy of the same weights

Run:
    torchrun --nproc-per-node 8 op_tests/test_moonep_vmm_pool.py
"""

from __future__ import annotations

import sys

import torch
import torch.distributed as dist

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
from aiter.ops.flydsl.moonep_vmm_pool import MoonEPVmmPool, pad_experts, vmm_granularity

EPN, B, H, I, TOKENS = 2, 2, 1024, 512, 256


def expert_w(e, shape, dev):
    g = torch.Generator(device=dev).manual_seed(9000 + e)
    return (torch.randn(*shape, generator=g, device=dev) * 0.05).to(torch.bfloat16)


def main() -> int:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = rank % torch.cuda.device_count()
    torch.cuda.set_device(dev)
    devs = f"cuda:{dev}"

    E = world * EPN
    w1_shape, w2_shape = (2 * I, H), (H, I)
    w1_row = 2 * I * H * 2
    w2_row = H * I * 2

    if rank == 0:
        g = vmm_granularity(dev)
        print(f"granularity={g}  epn={EPN} -> {pad_experts(EPN, w1_row, g)}  "
              f"E={E} B={B}")

    p1 = MoonEPVmmPool(row_bytes=w1_row, experts_per_rank=EPN, prefetch_slots=B,
                       rank=rank, world_size=world)
    p2 = MoonEPVmmPool(row_bytes=w2_row, experts_per_rank=EPN, prefetch_slots=B,
                       rank=rank, world_size=world)
    assert p1.epn_padded == EPN, f"unexpected padding {p1.epn_padded}"
    assert p1.rows == (world + 1) * EPN

    t1 = p1.tensor(torch.bfloat16, w1_shape)
    t2 = p2.tensor(torch.bfloat16, w2_shape)
    assert t1.is_contiguous() and t2.is_contiguous()
    assert not getattr(t1, "is_shuffled", False), "adopted pointer must not claim shuffled"

    ref1 = torch.zeros(p1.rows, *w1_shape, dtype=torch.bfloat16, device=devs)
    ref2 = torch.zeros(p2.rows, *w2_shape, dtype=torch.bfloat16, device=devs)
    for e in range(E):
        ref1[p1.global_row(e)] = expert_w(e, w1_shape, devs)
        ref2[p2.global_row(e)] = expert_w(1000 + e, w2_shape, devs)

    # Write ONLY our own home experts; every other row must arrive over XGMI.
    for k in range(EPN):
        e = rank * EPN + k
        t1[p1.global_row(e)].copy_(ref1[p1.global_row(e)])
        t2[p2.global_row(e)].copy_(ref2[p2.global_row(e)])
    for b in range(B):
        t1[p1.prefetch_row(b)].fill_(float(rank + 1))
        t2[p2.prefetch_row(b)].zero_()
    torch.cuda.synchronize()
    dist.barrier()

    # (1) addressing
    for e in range(E):
        r = p1.global_row(e)
        assert torch.equal(t1[r], ref1[r]), f"rank {rank}: expert {e} row {r} wrong"
    for b in range(B):
        assert torch.equal(
            t1[p1.prefetch_row(b)],
            torch.full(w1_shape, float(rank + 1), dtype=torch.bfloat16, device=devs),
        ), f"rank {rank}: prefetch slot {b} is not private"

    # (2) the GEMM
    g = torch.Generator(device=devs).manual_seed(4242)
    x = (torch.randn(TOKENS, H, generator=g, device=devs) * 0.1).to(torch.bfloat16)
    ids = (torch.arange(TOKENS, device=devs, dtype=torch.int32) % E)
    ids = torch.tensor([p1.global_row(int(v)) for v in ids],
                       dtype=torch.int32, device=devs).view(TOKENS, 1)
    tw = torch.ones(TOKENS, 1, dtype=torch.float32, device=devs)

    def run(a, b):
        return fused_moe(x, a, b, tw, ids, None, ActivationType.Silu,
                         quant_type=QuantType.No)

    out_ref, out_vmm = run(ref1, ref2), run(t1, t2)
    torch.cuda.synchronize()
    assert torch.equal(out_ref, out_vmm), (
        f"rank {rank}: fused_moe over the stitched pool differs from local; "
        f"max|d|={(out_ref.float() - out_vmm.float()).abs().max():.3e}"
    )

    ok = torch.tensor([1], dtype=torch.int32, device=devs)
    dist.all_reduce(ok)
    if rank == 0:
        print(f"PASS: {int(ok.item())}/{world} ranks -- addressing + fused_moe")
    dist.barrier()

    del t1, t2
    p2.close()
    p1.close()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
