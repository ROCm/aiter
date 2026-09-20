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


def test_every_cached_scratch_comes_from_the_private_pool():
    """Needs no capture, so it also runs on the torch floor where the test above
    is skipped, and it names any site that loses its persistent_alloc."""
    from aiter.ops.flydsl.gemm_a16w16_gfx1250 import _split_k_counters
    from aiter.ops.flydsl.gemm_kernels import _get_preshuffle_split_buffers
    from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 import get_split_k_buffers
    from aiter.ops.gemm_op_a8w8 import get_zero_bias_buf_keyed
    from aiter.ops.gemm_op_a16w16 import _get_semaphore_workspace_keyed
    from aiter.ops.topk import _get_topk_mb_workspace_keyed

    dev = torch.device("cuda:0")
    s = torch.cuda.Stream(device=dev)
    sid = s.cuda_stream
    sites = {
        "preshuffle_split_k": _get_preshuffle_split_buffers(dev, s),
        "a16w16_gfx950": get_split_k_buffers(s, dev),
        "a16w16_asm": _get_semaphore_workspace_keyed(dev, sid),
        "a16w16_gfx1250": _split_k_counters(dev, s),
        "a8w8_zero_bias": get_zero_bias_buf_keyed(dev, sid, 64),
        "topk_workspace": _get_topk_mb_workspace_keyed(dev, sid, 4096),
    }
    want = _persistent_pool(dev.index).id
    segments = torch.cuda.memory_snapshot()

    def pool_of(t):
        return next(
            seg["segment_pool_id"]
            for seg in segments
            if seg["address"] <= t.data_ptr() < seg["address"] + seg["total_size"]
        )

    wrong = {
        name: pool_of(t)
        for name, got in sites.items()
        for t in (got if isinstance(got, tuple) else (got,))
        if pool_of(t) != want
    }
    assert not wrong, f"not allocated from the private pool {want}: {wrong}"
