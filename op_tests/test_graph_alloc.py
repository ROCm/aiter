# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Coverage for aiter.utility.graph_alloc.persistent_alloc.

Scratch buffers that a kernel caches and reuses across launches may first be
allocated while a CUDA graph is capturing. Memory obtained there comes from that
graph's private mempool, which recycles addresses the same capture freed -- so
the cached buffer can land on an intermediate tensor whose producing kernel is a
node in the graph, and every replay overwrites it.

Run:
    python3 -m pytest op_tests/test_graph_alloc.py -v
"""

import pytest
import torch

from aiter.utility.graph_alloc import (
    ROUTES_INSIDE_CAPTURE,
    _persistent_pool,
    persistent_alloc,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a GPU to capture a graph"
)

BIG, CANARY, NUM_GRAPHS = 1 << 20, 777.0, 4


@pytest.mark.skipif(
    not ROUTES_INSIDE_CAPTURE,
    reason=f"torch {torch.__version__} ignores use_mem_pool inside a capture",
)
def test_buffer_survives_replay_of_graphs_sharing_a_pool():
    device = torch.device("cuda:0")
    pool = torch.cuda.graph_pool_handle()
    stream = torch.cuda.Stream(device=device)
    src = torch.full((BIG,), 3.0, device=device)

    graphs, buffers, intermediates = [], [], []
    with torch.cuda.stream(stream):
        for i in range(NUM_GRAPHS):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                # An intermediate freed inside the capture: its address goes
                # back to the graph's private pool for reuse.
                scratch = src * float(i + 2)
                intermediates.append(scratch.data_ptr())
                scratch.sum()
                del scratch
                with persistent_alloc(device):
                    buffers.append(torch.empty(BIG, dtype=torch.float32, device=device))
            graphs.append(graph)
    torch.cuda.synchronize()

    aliased = [i for i, b in enumerate(buffers) if b.data_ptr() in intermediates]
    assert not aliased, f"buffers aliased graph intermediates: {aliased}"

    for buf in buffers:
        buf.fill_(CANARY)
    torch.cuda.synchronize()
    for _ in range(5):
        for graph in graphs:
            graph.replay()
    torch.cuda.synchronize()

    clobbered = [i for i, b in enumerate(buffers) if int((b != CANARY).sum())]
    assert not clobbered, f"buffers overwritten by replay: {clobbered}"


def test_pool_is_one_per_device():
    index = torch.cuda.current_device()
    assert _persistent_pool(index) is _persistent_pool(index)
