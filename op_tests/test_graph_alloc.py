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

import subprocess
import sys

import pytest
import torch

from aiter.utility.graph_alloc import (
    ROUTES_INSIDE_CAPTURE,
    _persistent_pool,
    capture_scoped_cache,
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
        "a16w16_gfx1250": _split_k_counters(dev, s),
        "a16w16_asm": _get_semaphore_workspace_keyed(dev, sid),
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


@pytest.mark.skipif(
    not ROUTES_INSIDE_CAPTURE,
    reason=f"torch {torch.__version__} ignores use_mem_pool inside a capture",
)
def test_capture_scoped_cache_keeps_capture_built_buffers_private():
    """A buffer first built inside a capture is zeroed by a node of that graph,
    so no other graph or eager caller may receive it."""
    device = torch.device("cuda:0")

    @capture_scoped_cache
    def zeros(key):
        with persistent_alloc(device):
            return torch.zeros(256, dtype=torch.int64, device=device)

    stream = torch.cuda.Stream(device=device)
    seen = []
    for _ in range(2):
        with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=stream):
            first = zeros("a")
            assert zeros("a") is first, "one buffer per capture"
            seen.append(first)
    assert seen[0] is not seen[1], "captures must not share capture-built buffers"
    eager = zeros("a")
    assert all(eager is not b for b in seen), "eager must not get a capture buffer"
    with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=stream):
        assert zeros("a") is eager, "eagerly built buffers are shared as before"
        eager.sum()  # keep the graph non-empty


# Runs in a subprocess: on failure the GEMM spins forever, and only a process
# kill gets the GPU back.
_SPLIT_K_FIRST_USE_IN_CAPTURE = """
import torch
from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm
from aiter.utility.graph_alloc import persistent_alloc

dev = torch.device("cuda:0")
cfg = dict(block_m=32, block_n=32, block_k=128, stages=5, split_k=8,
           m_waves=2, n_waves=2, k_waves=1, group_m=0, policy="ft")
a = torch.randn(64, 8192, dtype=torch.bfloat16, device=dev)
b = torch.randn(4096, 8192, dtype=torch.bfloat16, device=dev)
ref = a.float() @ b.float().t()
flydsl_hgemm(a, b, **cfg)  # compile and cache the eager stream's semaphores
stream = torch.cuda.Stream(dev)
# Freed memory is recycled without clearing, as on a GPU that ran something
# before: make the capture stream's next semaphore allocation start dirty.
with torch.cuda.stream(stream), persistent_alloc(dev):
    torch.full((1 << 16,), 7, dtype=torch.int64, device=dev)
torch.cuda.synchronize()
graphs, outs = [], []
for _ in range(2):
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        outs.append(flydsl_hgemm(a, b, **cfg))
    graphs.append(g)
graphs[1].replay()  # before the graph that recorded the semaphore zero fill
graphs[0].replay()
torch.cuda.synchronize()
for out in outs:
    err = ((out.float() - ref).abs().max() / ref.abs().max()).item()
    assert err < 3e-2, err
print("OK")
"""


@pytest.mark.skipif(
    not ROUTES_INSIDE_CAPTURE,
    reason=f"torch {torch.__version__} ignores use_mem_pool inside a capture",
)
def test_split_k_gemm_whose_semaphores_are_first_used_inside_a_capture():
    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() != "gfx950":
        pytest.skip("FlyDSL a16w16 split-K kernel is gfx950-only")
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _SPLIT_K_FIRST_USE_IN_CAPTURE],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("split-K GEMM hung on semaphores never zeroed outside a graph")
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stderr[-2000:]
