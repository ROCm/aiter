# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import multiprocessing as mp

import pytest
import torch
import torch.distributed as dist

import aiter
from aiter.dist.parallel_state import get_tp_group
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.ops.triton.comms.all_reduce import (
    all_reduce,
    all_reduce_free,
    all_reduce_init,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.test_common import ensure_spawn_method

WORLD_SIZE = 4
HIDDEN = 2048
# Token counts of decode and prefill, then 2 sizes with a partial last chunk.
NUMELS = [t * HIDDEN for t in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)]
NUMELS += [3 * HIDDEN + 8, 600 * HIDDEN + 8]
MIXED = [8, 1, 2048, 2, 64, 4, 512, 8, 16, 1024, 32, 128, 256, 1, 8]

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < WORLD_SIZE or arch_info.get_arch() != "gfx1250",
    reason="needs 4 gfx1250 GPUs",
)


def _inputs(numel, seed, rank, device):
    """This rank's input and the fp32 sum of all inputs in rank order, as bf16."""
    gen = torch.Generator(device=device).manual_seed(seed)
    xs = [
        torch.randn(numel, generator=gen, dtype=torch.bfloat16, device=device)
        for _ in range(WORLD_SIZE)
    ]
    ref = xs[0].float() + xs[1].float() + xs[2].float() + xs[3].float()
    return xs[rank], ref.to(torch.bfloat16)


def _bad(out, ref):
    return int((out.view(torch.int16) != ref.view(torch.int16)).sum())


def _worker(rank, init_method):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    aiter.init_dist_env(WORLD_SIZE, rank, distributed_init_method=init_method)
    tp = get_tp_group()
    group = tp.cpu_group
    inboxes, calls = all_reduce_init(group, 2 * max(NUMELS))
    failures = []
    try:
        for i, numel in enumerate(NUMELS):
            x, ref = _inputs(numel, i, rank, device)
            out = all_reduce(x, group, inboxes, calls)
            outs = [torch.empty_like(out) for _ in range(WORLD_SIZE)]
            dist.all_gather(outs, out, group=tp.device_group)
            if _bad(out, ref) or any(_bad(o, out) for o in outs):
                failures.append(f"rank {rank} numel {numel}: wrong or rank-dependent")

        # Mixed sizes back to back, then the same calls in a HIP graph.
        cases = [
            _inputs(t * HIDDEN, 100 + i, rank, device) for i, t in enumerate(MIXED)
        ]
        outs = [all_reduce(x, group, inboxes, calls) for x, _ in cases]
        if sum(_bad(o, ref) for o, (_, ref) in zip(outs, cases)):
            failures.append(f"rank {rank}: mixed sizes are wrong")
        xs = [torch.empty_like(x) for x, _ in cases]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = [all_reduce(x, group, inboxes, calls) for x in xs]
        for rep in range(20):
            refs = []
            for i, (t, x) in enumerate(zip(MIXED, xs)):
                new_x, ref = _inputs(t * HIDDEN, 1000 * rep + i, rank, device)
                x.copy_(new_x)
                refs.append(ref)
            graph.replay()
            if sum(_bad(o, ref) for o, ref in zip(outs, refs)):
                failures.append(f"rank {rank}: graph replay {rep} is wrong")
        if calls[-1].item():
            failures.append(f"rank {rank}: {calls[-1].item()} polls timed out")
        return failures
    finally:
        all_reduce_free(group, inboxes)
        aiter.destroy_dist_env()


def test_all_reduce():
    ensure_spawn_method()
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with mp.Pool(processes=WORLD_SIZE) as pool:
        results = [
            pool.apply_async(_worker, (rank, init_method)) for rank in range(WORLD_SIZE)
        ]
        failures = [f for r in results for f in r.get(timeout=1800)]
    assert not failures, "\n".join(failures)
