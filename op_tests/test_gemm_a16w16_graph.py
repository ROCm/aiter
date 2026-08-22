# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Split-K semaphore behaviour under CUDA graph capture.

The defect is that the graph contains no zero-fill for the split-K counter; the
deadlock is only its consequence. These tests check the cause directly, so they
fail in seconds instead of wedging a GPU, and they launch no kernel.
"""

import pytest
import torch


def _warm_capture_stream(G, device):
    """Cache the eager workspace for the stream the graph will capture on.

    Without this the capture misses the cache, `torch.zeros` runs inside the
    capture region, and its zero-fill is recorded -- hiding the defect.
    """
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        G.get_semaphore_workspace(device)
    stream.synchronize()
    return stream


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a ROCm GPU")
def test_capture_records_semaphore_zero_fill() -> None:
    import aiter.ops.gemm_op_a16w16 as G

    device = torch.device("cuda:0")
    stream = _warm_capture_stream(G, device)

    sink = torch.zeros(1, device=device)
    graph = torch.cuda.CUDAGraph()
    eager_allocs = G._get_semaphore_workspace_keyed.cache_info().misses
    with torch.cuda.graph(graph, stream=stream):
        sema = G.get_semaphore_workspace(device)
        sink.add_(1)  # a graph needs at least one node
    torch.cuda.synchronize()

    # If this fires the warm-up missed and the defect under test is hidden.
    assert G._get_semaphore_workspace_keyed.cache_info().misses == eager_allocs

    sema.view(torch.int32).fill_(7)  # only a recorded zero-fill clears it
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    assert int(sink.item()) == 1, "the graph did not record the expected node"
    assert int(sema.view(torch.int32).max().item()) == 0, (
        "replay left the split-K counter dirty: the graph records no zero-fill, "
        "so the next launch starts from a non-zero counter and the kernel spins"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a ROCm GPU")
def test_capture_uses_distinct_preallocated_slots() -> None:
    """Capture must not allocate, and each recorded launch needs its own slot."""
    import aiter.ops.gemm_op_a16w16 as G

    device = torch.device("cuda:0")
    # Independent of whatever ran before, so the baseline cannot be poisoned.
    G._capture_rings.clear()
    G._capture_ring_next.clear()

    stream = _warm_capture_stream(G, device)
    assert device in G._capture_rings, "an eager call must pre-allocate the pool"
    pool = G._capture_rings[device]

    sink = torch.zeros(1, device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        pointers = [G.get_semaphore_workspace(device).data_ptr() for _ in range(5)]
        sink.add_(1)
    torch.cuda.synchronize()

    # Sharing a counter deadlocks any graph holding two split-K GEMMs.
    assert len(set(pointers)) == 5, "recorded launches share a semaphore slot"

    # Allocating under capture is what forced #4494 to retain every workspace.
    assert G._capture_rings[device] is pool
    assert pool.shape == (G.CAPTURE_SEMAPHORE_POOL_SLOTS, *G._SEMA_SHAPE)
    assert pool.dtype == torch.uint32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a ROCm GPU")
def test_capture_pool_exhaustion_raises() -> None:
    """Reusing a slot would let two graphs share a counter; refuse instead."""
    import aiter.ops.gemm_op_a16w16 as G

    device = torch.device("cuda:0")
    G._capture_rings.clear()
    G._capture_ring_next.clear()
    stream = _warm_capture_stream(G, device)

    # Start one slot short of the end so the second request must fail.
    G._capture_ring_next[device] = G.CAPTURE_SEMAPHORE_POOL_SLOTS - 1
    sink = torch.zeros(1, device=device)
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match="pool exhausted"),
        torch.cuda.graph(graph, stream=stream),
    ):
        G.get_semaphore_workspace(device)  # takes the last slot
        G.get_semaphore_workspace(device)  # must raise
        sink.add_(1)
    G._capture_ring_next.clear()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
