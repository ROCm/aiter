# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Capture a Python call into a CUDA graph for back-to-back replay."""


def capture(fn, repeats: int = 1, warmups: int = 1):
    """A ``torch.cuda.CUDAGraph`` that runs ``fn`` ``repeats`` times per replay.

    ``fn`` runs ``warmups`` times on a side stream first, so lazy setup
    (allocations, JIT loads, library handles) happens outside the capture, as
    ``torch.cuda.graph`` requires. Raises when ``fn`` cannot be captured, for
    example when it synchronizes with the host.
    """
    import torch

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmups):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            fn()
    return graph
