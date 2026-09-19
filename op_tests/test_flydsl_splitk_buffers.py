# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Coverage for the FlyDSL split-K preshuffle scratch buffers.

The split-K a8w8 preshuffle GEMM reduces through a workspace guarded by a
semaphore, both cached and reused across launches. Where that memory is
allocated matters: memory obtained while a CUDA graph is capturing comes from
that graph's private mempool, and a later capture into the same pool may be
handed the same block. The cached buffers then alias another graph's tensors,
whose replays overwrite the semaphore, and the reduction reads workspace slots
the launch never wrote.

These tests pin the two properties that keep the buffers out of a graph pool:
the cache is keyed on device alone (so it can be warmed before any capture
stream exists), and an allocation attempted during capture is refused rather
than silently poisoned.

Run:
    python3 -m pytest op_tests/test_flydsl_splitk_buffers.py -v
"""

from unittest import mock

import pytest
import torch

from aiter.ops.flydsl.gemm_kernels import (
    _get_preshuffle_split_buffers,
    preallocate_preshuffle_split_buffers,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a GPU to allocate the buffers"
)


@pytest.fixture(autouse=True)
def _clear_buffer_cache():
    _get_preshuffle_split_buffers.cache_clear()
    yield
    _get_preshuffle_split_buffers.cache_clear()


def test_buffers_are_shared_across_streams():
    """One buffer set per device, not per stream.

    A stream-keyed cache cannot be warmed ahead of a capture, because the
    capture stream does not exist yet; the capture would allocate its own set
    from the graph pool.
    """
    device = torch.device("cuda", 0)
    workspace, semaphore = _get_preshuffle_split_buffers(device)

    with torch.cuda.stream(torch.cuda.Stream(device=device)):
        other_workspace, other_semaphore = _get_preshuffle_split_buffers(device)

    assert other_workspace.data_ptr() == workspace.data_ptr()
    assert other_semaphore.data_ptr() == semaphore.data_ptr()


def test_allocation_during_capture_is_refused():
    """A cold cache must not allocate inside a capture region."""
    device = torch.device("cuda", 0)
    with (
        mock.patch.object(torch.cuda, "is_current_stream_capturing", return_value=True),
        pytest.raises(RuntimeError, match="preallocate_preshuffle_split_buffers"),
    ):
        _get_preshuffle_split_buffers(device)


def test_preallocation_lets_capture_reuse_the_buffers():
    """Warmed ahead of time, a call made during capture is a hit, not an alloc."""
    device = torch.device("cuda", 0)
    preallocate_preshuffle_split_buffers(device)
    workspace, semaphore = _get_preshuffle_split_buffers(device)

    with mock.patch.object(
        torch.cuda, "is_current_stream_capturing", return_value=True
    ):
        captured_workspace, captured_semaphore = _get_preshuffle_split_buffers(device)

    assert captured_workspace.data_ptr() == workspace.data_ptr()
    assert captured_semaphore.data_ptr() == semaphore.data_ptr()


def test_real_graph_capture_allocates_nothing():
    """End-to-end: with the cache warm, capture touches no new memory."""
    device = torch.device("cuda", 0)
    preallocate_preshuffle_split_buffers(device)
    workspace, _ = _get_preshuffle_split_buffers(device)

    # Warm up on a side stream, as torch.cuda.graph requires.
    side = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        torch.zeros(8, device=device).add_(1)
    torch.cuda.current_stream(device).wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    scratch = torch.zeros(8, device=device)
    with torch.cuda.graph(graph):
        captured_workspace, _ = _get_preshuffle_split_buffers(device)
        scratch.add_(1)

    assert captured_workspace.data_ptr() == workspace.data_ptr()
