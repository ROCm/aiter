# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host side of the candidate gather for the paged MXFP4 MQA-logits kernel.

The kernel's addressing change is one broadcast add, and this is what makes it
one. Both stored byte maps are separable -- A(d) + B(n) for the values,
As(s) + Bs(n) for the scales -- and a candidate block of GATHER_BLOCK tokens
never straddles a shuffle group, so for a block starting at in-page token `t0`

    preshuffled:  B(t0 + c) = B(t0) + c * K_WIDTH     Bs(t0 + c) = Bs(t0) + c * s_hi
    token major:  B(t0 + c) = B(t0) + c * head_bytes  Bs(t0 + c) = Bs(t0) + c * n_sc

for c < GATHER_BLOCK. The `c` term is loop invariant and stays in the offsets
tensor the kernel builds once. Everything else -- the page address and B(t0) --
is per block and is resolved here into a single int32 per candidate block per
stream, so the walk carries no `block -> position -> block table -> page` chain.

Two streams, because the value and scale regions have different byte maps.

The block-to-byte map is not linear in the block index. At page 64, head_size
128, preshuffled, the eight 8-token blocks of a page start at

    0, 128, 256, 384, 2048, 2176, 2304, 2432

because the shuffle group is 32 tokens. Multiplying a block index by a stride
produces plausible garbage that no tolerance check catches, which is why the
gate on this path is bit-identity against the contiguous kernel rather than a
tolerance.
"""

import torch

from aiter.ops.triton.attention.pa_mqa_logits_mxfp4 import (
    K_WIDTH, SCALE_GROUP, _split_cache, mfma_nonk_dim)


def cache_strides(kv_cache, head_size, kv_scale_cache=None):
    """(page_size, value page stride, scale page stride), in bytes.

    In the packed form the values and the scales share one page-strided buffer,
    so both regions step by the *whole* page and only their base differs -- not
    `page_size * head_bytes`. Getting this wrong stays inside the cache and
    addresses the wrong page, which is silent.
    """
    if kv_scale_cache is None:
        _, _, page_size, page_bytes = _split_cache(kv_cache, head_size)
        return page_size, page_bytes, page_bytes
    num_pages = kv_cache.shape[0]
    head_bytes = head_size // 2
    page_size = kv_cache.reshape(num_pages, -1).shape[1] // head_bytes
    return (page_size, page_size * head_bytes,
            page_size * (head_size // SCALE_GROUP))


def block_offsets(pos0, block_table, page_size, head_size, n_per_tile,
                  kv_stride, kvs_stride, preshuffle=1, scale_mode=1):
    """Resolved (value, scale) offsets for the candidate blocks starting at pos0.

    `pos0` is [R, K] int64 KV positions, each a multiple of the candidate block
    size; `block_table` is [R, max_blocks] int32, already expanded to one row
    per *query* row rather than per sequence.
    """
    head_bytes = head_size // 2
    num_scales = head_size // SCALE_GROUP
    s_lo = 64 // n_per_tile
    s_hi = num_scales // s_lo

    page = pos0 // page_size
    t0 = pos0 % page_size
    pid = torch.gather(block_table.long(), 1, page).long()

    if preshuffle:
        bn = (t0 % n_per_tile) * K_WIDTH + (t0 // n_per_tile) * (n_per_tile
                                                                 * head_bytes)
        # Mode 1 puts the token at stride s_hi inside a group, mode 0 at
        # stride 1. Both are constant, which is all a gather needs.
        tok_stride = s_hi if scale_mode == 1 else 1
        bs = (t0 % n_per_tile) * tok_stride + (t0 // n_per_tile) * (n_per_tile
                                                                    * num_scales)
    else:
        bn = t0 * head_bytes
        bs = t0 * num_scales

    # The value offset is stored in K_WIDTH units, not bytes. It is always a
    # whole number of them -- the page stride is 16-byte aligned (asserted in
    # the launcher) and every `bn` above is a multiple of 16 -- and the kernel
    # multiplies it back. That multiply is not arithmetic for its own sake: an
    # offset arriving from memory carries no provable alignment, and without
    # one Triton refuses to vectorise the KV load and emits one
    # buffer_load_ubyte per byte instead of buffer_load_dwordx4.
    voff = pid * kv_stride + bn
    soff = pid * kvs_stride + bs
    assert int((voff % K_WIDTH).max()) == 0, "value offsets must be k_width aligned"
    voff = voff // K_WIDTH
    assert int(voff.max()) < 2 ** 31 and int(soff.max()) < 2 ** 31, (
        "resolved offsets are i32: the reachable cache is 2 GiB of scale bytes "
        "and 32 GiB of value bytes on this path")
    return voff.to(torch.int32), soff.to(torch.int32)


def build_gather(positions, block_table, kv_cache, num_heads, head_size,
                 block=8, kv_scale_cache=None, preshuffle=1, scale_mode=1):
    """[R, K] int64 block-start positions -> the kernel's `gather=` argument.

    `positions[r]` must be sorted ascending, unique, every entry a multiple of
    `block`, and causally legal for row r: the kernel applies no candidate mask
    and no causal limit of its own.

    How many of the expanded `K * block` slots are real is the launch's
    `cu_ends` -- the same per-row exclusive bound the dense path takes, read in
    slot space rather than key space. Slots past it are dropped by store_hi.
    """
    n_per_tile = mfma_nonk_dim(num_heads, head_size)
    page_size, kv_stride, kvs_stride = cache_strides(kv_cache, head_size,
                                                     kv_scale_cache)
    assert page_size % block == 0 and block <= n_per_tile
    voff, soff = block_offsets(positions, block_table, page_size, head_size,
                               n_per_tile, kv_stride, kvs_stride, preshuffle,
                               scale_mode)
    return dict(voff=voff.contiguous(), soff=soff.contiguous(), block=block,
                positions=positions)


def expand(positions, block=8):
    """The slot -> KV position map the kernel walks, for a reference."""
    r, k = positions.shape
    c = torch.arange(block, device=positions.device, dtype=positions.dtype)
    return (positions[:, :, None] + c[None, None, :]).reshape(r, k * block)
