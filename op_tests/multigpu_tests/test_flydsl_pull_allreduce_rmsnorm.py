# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""TP4 check of the ``pull`` schedule of ``FlyQuickAllReduceRMSNorm``: exact bf16
all-reduce + residual add + Gemma RMSNorm on graph-registered inputs (hidden 6144).

Every tier (one-shot with acks, one-row pull with block / wave hand-offs, streamed
pull at R2 and R4) is captured into a CUDA graph and replayed on fresh inputs. The
reference is the two-stage all-reduce of AITER in fp32 (the rows of chunk ``c`` are
summed by rank ``c`` in the order ``c, c + 1, c + 2, c + 3``, one bf16 rounding), then
the residual add and the Gemma norm ``y = s * rstd * (1 + w)``:

* ``residual_out`` is compared bit for bit;
* ``out`` may differ by one bf16 ulp where the fp32 sum of squares of the kernel
  (Triton's ``tl.sum`` order) and the reference round differently.

    python3 op_tests/multigpu_tests/test_flydsl_pull_allreduce_rmsnorm.py
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 4
HIDDEN = 6144
EPS = 1e-6
ITERS = 50  # graph replays per timing
# One row count per tier edge: one-shot (<= 11), pull (12-256; block / wave hand-offs
# below / above 37 and 96), streamed (257-512 at R2, 513-1024 at R4).
ROWS = (
    1,
    8,
    11,
    12,
    36,
    37,
    96,
    97,
    256,
    257,
    287,
    288,
    384,
    480,
    512,
    513,
    640,
    768,
    1024,
)


def _reference(xs, residual, weight, eps):
    """fp32 two-stage all-reduce + residual add + Gemma RMSNorm of the ranks' ``xs``."""
    rows = residual.shape[0]
    q = -(-rows // WORLD)
    reduced = torch.empty(rows, HIDDEN, dtype=torch.float32, device=residual.device)
    for c in range(WORLD):
        lo, hi = c * q, min(rows, (c + 1) * q)
        if lo >= hi:
            continue
        acc = xs[c][lo:hi].float()
        for k in range(1, WORLD):
            acc = acc + xs[(c + k) % WORLD][lo:hi].float()
        reduced[lo:hi] = acc
    s = reduced.to(torch.bfloat16).float() + residual.float()
    res_out = s.to(torch.bfloat16)
    rstd = torch.rsqrt((s * s).sum(-1, keepdim=True) / HIDDEN + eps)
    out = (s * rstd * (weight.float() + 1.0)).to(torch.bfloat16)
    return out, res_out


def _data(rows, seed, rank, device):
    g = torch.Generator(device="cpu").manual_seed(seed * 100003 + rank * 131 + rows)
    return (torch.randn(rows, HIDDEN, generator=g) * 2.0).to(torch.bfloat16).to(device)


def _worker(rank, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    group = dist.group.WORLD
    from aiter.ops.flydsl.quick_allreduce import FlyQuickAllReduceRMSNorm

    eng = FlyQuickAllReduceRMSNorm(
        group=group, device=device, rank=rank, world_size=WORLD, algorithm="pull"
    )
    assert (eng.rs_codec, eng.ag_codec) == ("bf16", "bf16")
    assert eng.supports_hidden(HIDDEN) and not eng.supports_hidden(4096)
    eng.preload(HIDDEN)
    weight = (
        (torch.randn(HIDDEN, generator=torch.Generator().manual_seed(7)) * 0.3)
        .to(torch.bfloat16)
        .to(device)
    )

    # outside a registered capture the call is declined, not guessed
    x = _data(8, 1, rank, device)
    r = _data(8, 2, rank, device)
    assert not eng.accepts(x, r, weight, gemma_norm=True)
    with pytest.raises(ValueError):
        eng.allreduce_rmsnorm(x, r, weight, EPS, gemma_norm=True)
    with pytest.raises(ValueError, match="Gemma"):
        eng.allreduce_rmsnorm(x, r, weight, EPS, gemma_norm=False)

    graphs = []
    with eng.capture():
        for rows in ROWS:
            x = _data(rows, 1, rank, device)
            r = _data(rows, 2, rank, device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                # served only while a graph is being captured
                assert eng.accepts(x, r, weight, gemma_norm=True)
                assert not eng.accepts(x, r, weight, gemma_norm=False)
                out, res_out = eng.allreduce_rmsnorm(x, r, weight, EPS, gemma_norm=True)
            graphs.append((rows, x, r, graph, out, res_out))
        too_many = _data(1025, 1, rank, device)
        # destinations the kernels cannot write correctly are declined, not launched
        x8, r8 = _data(8, 1, rank, device), _data(8, 2, rank, device)
        other = torch.device("cuda", (rank + 1) % WORLD)
        cpu_out = torch.empty(8, HIDDEN, dtype=torch.bfloat16)
        far_out = torch.empty(8, HIDDEN, dtype=torch.bfloat16, device=other)
        same = torch.empty(8, HIDDEN, dtype=torch.bfloat16, device=device)
        a = torch.empty(2 * 8, HIDDEN, dtype=torch.bfloat16, device=device)
        lap_a, lap_b = a[:8], a[7:15]  # rows 7 of the two views are the same memory
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            assert not eng.accepts(too_many, too_many, weight, gemma_norm=True)
            assert not eng.accepts(x8, r8, weight, out=cpu_out, gemma_norm=True)
            assert not eng.accepts(
                x8, r8, weight, residual_out=far_out, gemma_norm=True
            )
            assert not eng.accepts(
                x8, r8, weight, out=same, residual_out=same, gemma_norm=True
            )
            assert not eng.accepts(
                x8, r8, weight, out=lap_a, residual_out=lap_b, gemma_norm=True
            )
            assert eng.accepts(
                x8, r8, weight, out=a[:8], residual_out=a[8:16], gemma_norm=True
            )

    bad = []
    for rows, x, r, graph, out, res_out in graphs:
        for it in range(3):
            x.copy_(_data(rows, 10 + it, rank, device))
            r.copy_(_data(rows, 20 + it, rank, device))
            torch.cuda.synchronize()
            dist.barrier(group)
            graph.replay()
            torch.cuda.synchronize()
            # every rank's partials, to build the reference locally
            mine = x.cpu()
            allx = [None] * WORLD
            dist.all_gather_object(allx, mine, group=group)
            ref_out, ref_res = _reference([a.to(device) for a in allx], r, weight, EPS)
            d_res = (
                (res_out.view(torch.int16) != ref_res.view(torch.int16)).sum().item()
            )
            ulp = (out.view(torch.int16).int() - ref_out.view(torch.int16).int()).abs()
            if d_res or int(ulp.max()) > 1 or (ulp != 0).float().mean().item() > 1e-3:
                bad.append((rows, it, d_res, int(ulp.max()), (ulp != 0).sum().item()))
    # graph-replay time per tier edge (all ranks replay together; rank 0 reports)
    table = []
    for rows, x, r, graph, out, res_out in graphs:
        torch.cuda.synchronize()
        dist.barrier(group)
        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(True)
        t0.record()
        for _ in range(ITERS):
            graph.replay()
        t1.record()
        t1.synchronize()
        table.append(
            (
                rows,
                eng._registered_engine.tier_name(rows),
                t0.elapsed_time(t1) * 1e3 / ITERS,
            )
        )
    if rank == 0:
        print("\n| rows | tier | graph replay us |\n|---|---|---|")
        for rows, tier, us in table:
            print(f"| {rows} | {tier} | {us:.1f} |")
    eng.close()
    dist.barrier(group)
    dist.destroy_process_group()
    assert (
        not bad
    ), f"rank {rank}: (rows, replay, residual_out mismatches, max ulp, out mismatches) {bad}"


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < WORLD,
    reason="the pull all-reduce check needs 4 gfx950 GPUs",
)
def test_pull_allreduce_rmsnorm_tp4():
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx950":
        pytest.skip("the pull schedule needs gfx950")
    mp.spawn(_worker, args=(29700 + os.getpid() % 200,), nprocs=WORLD)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
