# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from aiter.ops.triton.comms.all_reduce import (
    _VARIANTS,
    all_reduce,
    all_reduce_free,
    all_reduce_init,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from op_tests.multigpu_tests.triton_test.test_all_reduce import _inputs


@pytest.mark.skipif(
    int(os.getenv("WORLD_SIZE", "1")) != 4 or get_arch() != "gfx1250",
    reason="run with torchrun --nproc-per-node=4 on gfx1250",
)
def test_all_reduce_decode_variants():
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("gloo", timeout=timedelta(seconds=180))
    group = dist.group.WORLD
    inboxes, calls = all_reduce_init(group, 256 * 7168 * 2)
    try:
        sizes = [
            h * m
            for h in (5120, 6144, 7168)
            for m in (1, 2, 4, 8, 16, 32, 64, 128, 256)
        ]
        boundaries = [minimum for minimum, _, _ in _VARIANTS[inboxes[rank]]]
        sizes += [n + offset for n in boundaries for offset in (-4, 0, 4)]
        sizes += [12, 256 * 7168 - 4]
        for i, n in enumerate(sizes):
            x, ref = _inputs(n, 700 + i, rank, device)
            torch.testing.assert_close(
                all_reduce(x, group, inboxes, calls), ref, atol=0, rtol=0
            )

        # Each variant wraps its own counters; switching variants must not reuse tags.
        torch.cuda.synchronize()
        dist.barrier(group=group)
        all_counters = [calls] + [c for _, _, c in _VARIANTS[inboxes[rank]]]
        for c in all_counters:
            c[:-1].fill_(2**30 - 2)
        mixed = [12, 256 * 7168, 64 * 6144, 128 * 5120, 7168]
        mixed += [n + 4 for n in boundaries]
        for rep in range(4):
            for i, n in enumerate(mixed):
                x, ref = _inputs(n, 2000 + rep * 100 + i, rank, device)
                torch.testing.assert_close(
                    all_reduce(x, group, inboxes, calls), ref, atol=0, rtol=0
                )

        xs = [torch.empty(n, dtype=torch.bfloat16, device=device) for n in mixed]
        # Warm every shape before capture, including the partial last chunks.
        for i, x in enumerate(xs):
            source, _ = _inputs(x.numel(), 3000 + i, rank, device)
            x.copy_(source)
            all_reduce(x, group, inboxes, calls)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = [all_reduce(x, group, inboxes, calls) for x in xs]
        for rep in range(20):
            refs = []
            for i, x in enumerate(xs):
                source, ref = _inputs(x.numel(), 4000 + rep * 100 + i, rank, device)
                x.copy_(source)
                refs.append(ref)
            graph.replay()
            for output, ref in zip(outputs, refs):
                torch.testing.assert_close(output, ref, atol=0, rtol=0)
        assert calls[-1].item() == 0, "peer readiness wait timed out"
    finally:
        all_reduce_free(group, inboxes)
        dist.destroy_process_group()
