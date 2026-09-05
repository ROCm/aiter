# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

import aiter
from aiter.mla import _fold_seqlen_indptr

torch.set_default_device("cuda")


def _reference_fold(indptr, fold_factor):
    """Host-side reference for the head-folded indptr."""
    lens = (indptr[1:] - indptr[:-1]).tolist()
    out, acc = [0], 0
    for n in lens:
        for _ in range(fold_factor):
            acc += n
            out.append(acc)
    return torch.tensor(out, dtype=indptr.dtype, device=indptr.device)


def test_fold_seqlen_indptr_values():
    for lens in ([3, 4, 5], [0, 7, 0, 2], [1]):
        indptr = torch.tensor(
            [0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32
        )
        for fold_factor in (1, 2, 4, 8):
            got = _fold_seqlen_indptr(indptr, fold_factor)
            want = _reference_fold(indptr, fold_factor)
            assert got.dtype == indptr.dtype, f"{got.dtype=} != {indptr.dtype=}"
            assert torch.equal(got, want), f"{lens=} {fold_factor=} {got=} {want=}"
    aiter.logger.info("_fold_seqlen_indptr values: passed")


def test_fold_seqlen_indptr_cuda_graph_capture():
    """Capture _fold_seqlen_indptr under a real torch.cuda.graph().

    Regression test for the capture-safety fix: seeding the leading zero with
    ``out[0] = 0`` staged a host scalar into device memory, which raises
    "Cannot copy between CPU and CUDA tensors during CUDA graph capture unless
    the CPU tensor is pinned" and made persistent MLA decode uncapturable for
    any head count that takes the head-folding path (nhead 32..128 step 16).
    """
    indptr = torch.tensor([0, 3, 7, 12], dtype=torch.int32)
    fold_factor = 4

    # Warm-up on a side stream (standard torch.cuda.graph() prerequisite).
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _fold_seqlen_indptr(indptr, fold_factor)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _fold_seqlen_indptr(indptr, fold_factor)

    graph.replay()
    torch.cuda.synchronize()
    want = _reference_fold(indptr, fold_factor)
    assert torch.equal(out, want), f"{out=} {want=}"
    aiter.logger.info("_fold_seqlen_indptr cuda-graph capture/replay: passed")


if __name__ == "__main__":
    test_fold_seqlen_indptr_values()
    test_fold_seqlen_indptr_cuda_graph_capture()
