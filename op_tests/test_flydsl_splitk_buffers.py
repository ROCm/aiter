# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter.ops.flydsl.gemm_kernels as gemm_kernels
from aiter import dtypes
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_bpreshuffle
from aiter.ops.shuffle import shuffle_weight

# Small M with a wide N: a shape the tuned configs give split_k > 1.
M, N, K = 8, 3584, 7168
DIRT = 0x0DEFACED


def _inputs(device):
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16).to(dtypes.fp8)
    w = torch.randn(N, K, device=device, dtype=torch.bfloat16).to(dtypes.fp8)
    return (
        x,
        shuffle_weight(w, layout=(16, 16)),
        torch.ones(M, 1, device=device, dtype=torch.float32),
        torch.ones(N, 1, device=device, dtype=torch.float32),
    )


def _gemm(args):
    return gemm_a8w8_bpreshuffle(*args, dtype=torch.bfloat16)


def _dirty_pool(device, pool, stream):
    """Leave residue in the graph pool, as a loaded server would."""
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool, stream=stream):
        blocks = [
            torch.empty(4096, dtype=torch.int32, device=device) for _ in range(16)
        ]
        for block in blocks:
            block.fill_(DIRT)
    graph.replay()
    torch.cuda.synchronize()
    del blocks
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_graphs_share_a_zeroed_semaphore():
    """Every graph must start split-K on a semaphore that was really zeroed.

    Capture on a stream other than the warmup stream, which is what a framework
    does when it calls ``torch.cuda.graph()`` without ``stream=``. Before the
    fix the first captured graph allocated the semaphore and only *recorded*
    its memset; the rest shared that never-zeroed buffer, so replaying any of
    them first reduced garbage.
    """
    device = torch.device("cuda:0")
    args = _inputs(device)
    gemm_kernels._get_preshuffle_split_buffers.cache_clear()
    reference = _gemm(args).clone()
    torch.cuda.synchronize()

    pool = torch.cuda.graph_pool_handle()
    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    _dirty_pool(device, pool, capture_stream)

    _gemm(args)  # warmup, on the default stream
    torch.cuda.synchronize()

    graphs = []
    for _ in range(4):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool, stream=capture_stream):
            out = _gemm(args)
        graphs.append((graph, out))
    torch.cuda.synchronize()

    _, semaphore = gemm_kernels._get_preshuffle_split_buffers(device)
    assert int((semaphore != 0).sum()) == 0

    for graph, out in reversed(graphs):
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, reference, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_concurrent_streams_do_not_share_a_semaphore():
    """Eager launches on different streams must not share the atomic counter,
    or arrival counts mix and the reduction never fires."""
    device = torch.device("cuda:0")
    gemm_kernels._get_preshuffle_split_buffers.cache_clear()

    streams = [torch.cuda.Stream(device=device) for _ in range(2)]
    semaphores = []
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            _gemm(_inputs(device))
            semaphores.append(gemm_kernels._get_preshuffle_split_buffers(device)[1])
        torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize()

    assert semaphores[0].data_ptr() != semaphores[1].data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_capture_without_preallocation_raises():
    """A buffer allocated under capture cannot be shared, so refuse instead."""
    device = torch.device("cuda:0")
    gemm_kernels._get_preshuffle_split_buffers.cache_clear()

    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="before CUDA graph capture"):
        with torch.cuda.graph(graph, stream=capture_stream):
            _gemm(_inputs(device))
    torch.cuda.synchronize()
