# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Eight ranks: FlyDSL DCP mailbox LSE combine and query gather."""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aiter.ops.flydsl.dcp_mailbox import (
    CombineWorkspace,
    GatherWorkspace,
    dcp_lse_combine,
    dcp_quant_query_gather,
    dcp_query_gather,
)


class _Group:
    def __init__(self, cpu_group, rank, world):
        self.cpu_group = cpu_group
        self.rank_in_group = rank
        self.world_size = world


def _reference(parts, lses):
    stacked = torch.stack(parts, 0).float()
    lse = torch.stack(lses, 0).float()
    lse = torch.where(torch.isnan(lse) | torch.isposinf(lse), float("-inf"), lse)
    m = lse.max(0).values
    m = torch.where(torch.isneginf(m), 0.0, m)
    g = torch.log(torch.exp(lse - m).sum(0)) + m
    w = torch.exp(lse - g)
    w = torch.where(torch.isnan(w), 0.0, w)
    return (stacked * w[..., None]).sum(0).to(torch.bfloat16), g


def worker(rank, world):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ.setdefault("MASTER_PORT", "29741")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    group = _Group(dist.new_group(backend="gloo"), rank, world)
    try:
        tokens, hl, dim = 8, 12, 512
        gen = torch.Generator(device="cuda")
        gen.manual_seed(100 + rank)
        cws = CombineWorkspace(group, torch.device("cuda"), hl, dim, 64)
        for _ in range(3):
            out = torch.randn(
                tokens,
                world * hl,
                dim,
                device="cuda",
                dtype=torch.bfloat16,
                generator=gen,
            )
            lse = torch.randn(
                tokens, world * hl, device="cuda", dtype=torch.float32, generator=gen
            )
            if rank == 0:
                lse[0, 0] = float("nan")
            res, res_lse = dcp_lse_combine(cws, out, lse, return_lse=True)
            torch.cuda.synchronize()
            go = [torch.empty_like(out) for _ in range(world)]
            gl = [torch.empty_like(lse) for _ in range(world)]
            dist.all_gather(go, out)
            dist.all_gather(gl, lse)
            sl = slice(rank * hl, (rank + 1) * hl)
            exp, exp_lse = _reference([g[:, sl] for g in go], [g[:, sl] for g in gl])
            torch.testing.assert_close(res.float(), exp.float(), rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(res_lse, exp_lse, rtol=1e-4, atol=1e-4)

        gws = GatherWorkspace(
            group, torch.device("cuda"), hl, dim, torch.bfloat16, world * hl, 64
        )
        query = torch.randn(
            tokens, hl, dim, device="cuda", dtype=torch.bfloat16, generator=gen
        )
        got = dcp_query_gather(gws, query)
        torch.cuda.synchronize()
        parts = [torch.empty_like(query) for _ in range(world)]
        dist.all_gather(parts, query)
        exp_q = torch.cat(parts, dim=1)
        torch.testing.assert_close(got, exp_q)

        nope_w, pe_w = 512, 64
        qdim = nope_w + pe_w
        qws = GatherWorkspace(
            group, torch.device("cuda"), hl, qdim, torch.float8_e4m3fn, world * hl, 64
        )
        nope = torch.randn(
            tokens, hl, nope_w, device="cuda", dtype=torch.bfloat16, generator=gen
        )
        pe = torch.randn(
            tokens, hl, pe_w, device="cuda", dtype=torch.bfloat16, generator=gen
        )
        scale = torch.tensor([2.0], device="cuda", dtype=torch.float32)
        qgot = dcp_quant_query_gather(qws, nope, pe, scale)
        torch.cuda.synchronize()
        local = torch.cat([nope, pe], dim=-1)
        ref = (local.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        refs = [torch.empty_like(ref) for _ in range(world)]
        dist.all_gather(refs, ref)
        exp_fp8 = torch.cat(refs, dim=1)
        torch.testing.assert_close(qgot.float(), exp_fp8.float(), rtol=0.1, atol=1.0)
        if rank == 0:
            print("DCP_MAILBOX OK", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    world = int(os.environ.get("WORLD", "8"))
    mp.spawn(worker, args=(world,), nprocs=world, join=True)
