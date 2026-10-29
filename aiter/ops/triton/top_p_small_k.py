# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Top-p sampling over a short row: the last step of the top-k-first fast path.

There the row is the k survivors of a top-k (k <= 64 here), and the generic
rejection kernel (`TopPSamplingFromProbKernel`, 1024-thread CTAs) spills to
scratch: on gfx1250 every launch then waits for the runtime to grow the queue's
scratch, about 100 us a call at decode batch sizes. One program a row keeps the
row in registers instead.

Same accepted set as that kernel: element i survives when the mass of the
elements strictly larger than it is below p. The draw is proportional over the
survivors, from one uniform a row (the same distribution, not the same draws).
"""
import torch
import triton
import triton.language as tl

MAX_K = 64


@triton.jit
def _top_p_small_k_kernel(
    probs_ptr, top_p_ptr, u_ptr, out_ptr, K, stride_row, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < K
    p = tl.load(probs_ptr + row * stride_row + offs, mask=mask, other=0.0)
    top_p = tl.load(top_p_ptr + row)
    u = tl.load(u_ptr + row)
    # mass of the elements strictly larger than each element
    gt = tl.sum(tl.where(p[None, :] > p[:, None], p[None, :], 0.0), axis=1)
    keep = (gt < top_p) & mask
    q = tl.where(keep, p, 0.0)
    cdf = tl.cumsum(q, axis=0)
    total = tl.sum(q, axis=0)
    hit = keep & (cdf > u * total)
    idx = tl.min(tl.where(hit, offs, BLOCK), axis=0)
    # rounding left no element past u * total (or the row is all zero): the
    # last survivor, which keeps the index in range
    last = tl.max(tl.where(keep, offs, -1), axis=0)
    idx = tl.where(idx == BLOCK, last, idx)
    tl.store(out_ptr + row, idx.to(tl.int32))


def top_p_sampling_small_k(probs: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    """Index in [0, k) sampled from each row of `probs` [rows, k] (k <= MAX_K)
    under that row's `top_p` [rows]."""
    rows, k = probs.shape
    assert k <= MAX_K and probs.stride(1) == 1
    out = torch.empty(rows, dtype=torch.int32, device=probs.device)
    u = torch.rand(rows, dtype=torch.float32, device=probs.device)
    _top_p_small_k_kernel[(rows,)](
        probs,
        top_p.float().contiguous(),
        u,
        out,
        k,
        probs.stride(0),
        BLOCK=max(16, triton.next_power_of_2(k)),
    )
    return out
