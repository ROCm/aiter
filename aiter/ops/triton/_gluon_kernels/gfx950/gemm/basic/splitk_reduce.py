# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 OpenAI

"""Exact split-K reductions shared by registered FP8 GEMM specializations."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _flat_splitk_reduce(
    workspace_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    ACTUAL_KSPLIT: tl.constexpr,
    MAX_KSPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    partitions = tl.arange(0, MAX_KSPLIT)
    values = tl.load(
        workspace_ptr + partitions[:, None] * TOTAL + offsets[None, :],
        mask=partitions[:, None] < ACTUAL_KSPLIT,
        other=0.0,
    )
    reduced = tl.sum(values, axis=0)
    tl.store(
        out_ptr + offsets, reduced.to(out_ptr.dtype.element_ty), cache_modifier=".wt"
    )


def reduce_n2112_k7168_m32(workspace: torch.Tensor, out: torch.Tensor) -> None:
    """Reduce exact FP32 `[7,32,2112]` partials to BF16 `[32,2112]`."""
    _flat_splitk_reduce[(66,)](
        workspace,
        out,
        TOTAL=32 * 2112,
        ACTUAL_KSPLIT=7,
        MAX_KSPLIT=8,
        BLOCK=1024,
        num_warps=4,
        num_stages=2,
        waves_per_eu=1,
    )


__all__ = ["reduce_n2112_k7168_m32"]
