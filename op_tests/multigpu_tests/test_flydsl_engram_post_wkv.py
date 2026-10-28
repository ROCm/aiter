# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Run with torchrun --standalone --nproc_per_node={2,4} on gfx950.

Independent FP32 reference, changing graph inputs/masks, rank arrival skew,
and both signed-int32 and uint32 epoch wrap boundaries. No ATOM dependency.
"""

import os
import time

import torch
import torch.distributed as dist

from aiter.dist.parallel_state import get_tp_group
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.communication import destroy_dist_env, init_dist_env
from aiter.ops.flydsl.engram_post_wkv import FusedEngramPost


def reference(h, kv, weight, mask):
    h = h.float()
    k = kv[:, :20480].view(-1, 4, 5120).float()
    rstd = torch.rsqrt(h.square().mean(-1) + 1e-20) * torch.rsqrt(
        k.square().mean(-1) + 1e-20
    )
    dot = ((h * weight) * k).sum(-1) * rstd * 5120**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    if mask is not None:
        gate = gate.masked_fill(~mask.unsqueeze(-1), 0)
    return (h + gate.unsqueeze(-1) * kv[:, 20480:].float().unsqueeze(1)).bfloat16()


def main():
    rank, tp = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    if get_gfx_runtime() != "gfx950":
        raise RuntimeError("This kernel targets gfx950")
    init_dist_env(tp, rank, local_rank=rank)
    group = get_tp_group()
    post = FusedEngramPost(group.cpu_group)
    keys, rotation = 20480 // tp, rank * (5120 // tp)
    count, max_nrmse = 0, 0.0
    for m in (1, 8, 13, 16, 33, 64, 96, 128, 7, 2):
        torch.manual_seed(7900 + m)
        h = torch.randn(m, 4, 5120, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(m, 25600, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(4, 5120, device="cuda")

        def local_kv(kv=kv):
            return torch.cat(
                (
                    kv[:, rank * keys : (rank + 1) * keys],
                    torch.roll(kv[:, 20480:], -rotation, -1),
                ),
                dim=1,
            )

        local = local_kv()
        for masked in (False, True):
            mask = (torch.arange(m, device="cuda") % 3 != 0) if masked else None
            # Only M=1 is warmed: capture unseen M using the same binary.
            if m == 1:
                post(h, local, weight, mask, value_start=rotation)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = [
                    post(h, local, weight, mask, value_start=rotation) for _ in range(4)
                ]
            starts = (None, 2147483646, -2) if m == 33 else (None,)
            for start in starts:
                if start is not None:
                    # Only active slots have a valid message history. Leave
                    # never-used slots at their initial epoch for later M.
                    post.epoch[: m * 4].fill_(start)
                    torch.cuda.synchronize()
                    dist.barrier(group=group.cpu_group)
                for repeat in range(3):
                    h.normal_()
                    kv.normal_()
                    local.copy_(local_kv())
                    if masked:
                        mask.copy_(torch.arange(m, device="cuda") % 3 != repeat)
                    torch.cuda.synchronize()
                    dist.barrier(group=group.cpu_group)
                    if repeat == 1 and rank == tp - 1:
                        time.sleep(0.02)
                    graph.replay()
                    torch.cuda.synchronize()
                    post.check()
                    expected = reference(h, kv, weight, mask)
                    for actual in outputs:
                        a, b = actual.float(), expected.float()
                        nrmse = (
                            (
                                (a - b).square().mean()
                                / b.square().mean().clamp_min(1e-30)
                            )
                            .sqrt()
                            .item()
                        )
                        outliers = (
                            (~torch.isclose(a, b, rtol=0.01, atol=0.01))
                            .float()
                            .mean()
                            .item()
                        )
                        assert (
                            torch.isfinite(a).all()
                            and nrmse < 0.001
                            and outliers < 0.001
                        ), (
                            rank,
                            m,
                            masked,
                            start,
                            repeat,
                            nrmse,
                            outliers,
                            (a - b).abs().max().item(),
                        )
                        if masked:
                            assert torch.equal(actual[~mask], h[~mask])
                        max_nrmse = max(max_nrmse, nrmse)
                        count += 1
            del graph, outputs
    assert len(post.compiled) == 2, len(post.compiled)
    metrics = [None] * tp
    dist.all_gather_object(metrics, (count, max_nrmse), group=group.cpu_group)
    if rank == 0:
        print(
            "PASS",
            tp,
            "ranks;",
            sum(x[0] for x in metrics),
            "comparisons; max NRMSE",
            max(x[1] for x in metrics),
            flush=True,
        )
    torch.cuda.synchronize()
    dist.barrier(group=group.cpu_group)
    post.close()
    destroy_dist_env()


if __name__ == "__main__":
    with torch.inference_mode():
        main()
