# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Launcher for the gfx1250 Gluon DSA prefill top-k (_gluon_kernels/gfx1250/topk/topk_prefill)."""
import torch
import triton

from aiter.ops.triton._gluon_kernels.gfx1250.topk.topk_prefill import _topk_prefill_kernel

_VPT = 8  # keys per lane per tile
_WARPS = 8  # 256 lanes per row: 2048-key tiles, four rows per CU
_CAP = 4096  # LDS list for the bracket around the threshold
_SAMPLE_BLOCKS = 8  # contiguous sample blocks spread over long rows
_HI_NUM, _LO_NUM = 12, 22  # bracket: [LO/16, HI/16] x the sample's expected winners (tuned)


def top_k_per_row_prefill(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    indices: torch.Tensor,
    k: int = 2048,
) -> torch.Tensor:
    """Per-row top-k of ``logits[r, row_starts[r]:row_ends[r]]`` into ``indices[r, :k]``.

    Same contract as vLLM's ``torch.ops._C.top_k_per_row_prefill``: int32 indices relative to
    ``row_starts``, unordered, -1 padded for rows with fewer than k entries (ends below starts
    are treated as empty). NaN ranks below every number; ties are resolved deterministically.
    """
    assert logits.dtype == torch.float32 and logits.dim() == 2
    assert indices.dtype == torch.int32 and indices.stride(1) == 1
    assert k & (k - 1) == 0 and k <= _CAP, "k must be a power of two <= 4096"
    rows, width = logits.shape
    assert width < 1 << 18, "rows longer than 2**18 - 1 entries are not supported"
    if rows == 0:
        return indices
    lanes = triton.next_power_of_2(max(1, triton.cdiv(width, _VPT)))
    num_warps = min(_WARPS, max(4, lanes // 32))
    block = 32 * num_warps * _VPT
    _topk_prefill_kernel[(rows,)](
        logits, row_starts, row_ends, indices,
        logits.stride(0), logits.stride(1), indices.stride(0),
        K=k, VPT=_VPT, NUM_CHUNKS=triton.cdiv(width, block), CAP=_CAP,
        BUF=triton.next_power_of_2(2 * k + _CAP), SAMPLE_MIN=4 * block,
        CONTIG=logits.stride(1) == 1, SAMPLE_BLOCKS=_SAMPLE_BLOCKS, HI_NUM=_HI_NUM,
        LO_NUM=_LO_NUM, num_warps=num_warps,
    )
    return indices
