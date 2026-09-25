# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host side of the candidate gather for the paged MXFP4 MQA-logits kernel.

Both stored byte maps are separable and a candidate block never straddles a
shuffle group, so B(t0 + c) = B(t0) + c * stride for c < GATHER_BLOCK. The `c`
term is loop invariant and lives in the kernel; everything per block -- the page
address and B(t0) -- is resolved here into one int32 per block per stream, which
is why the walk carries no block -> position -> table -> page chain. Two streams,
because values and scales have different maps.

The block-to-byte map is not linear in the block index. At page 64, head_size
128, preshuffled, a page's eight 8-token blocks start at

    0, 128, 256, 384, 2048, 2176, 2304, 2432

because the shuffle group is 32 tokens. Multiplying a block index by a stride
gives plausible garbage no tolerance check catches, which is why this path is
gated on bit-identity against the contiguous kernel.
"""

import torch

from aiter.ops.triton.attention.pa_mqa_logits_mxfp4 import (
    K_WIDTH,
    SCALE_GROUP,
    _split_cache,
    mfma_nonk_dim,
)


def cache_strides(kv_cache, head_size, kv_scale_cache=None):
    """(page_size, value page stride, scale page stride), in bytes.

    In the packed form the values and the scales share one page-strided buffer,
    so both regions step by the *whole* page and only their base differs -- not
    `page_size * head_bytes`. Getting this wrong stays inside the cache and
    addresses the wrong page, which is silent.
    """
    if kv_scale_cache is None:
        _, _, page_size, page_stride = _split_cache(kv_cache, head_size)
        return page_size, page_stride, page_stride
    page_size = kv_cache[0].numel() // (head_size // 2)
    return page_size, kv_cache.stride(0), kv_scale_cache.stride(0)


def offset_dtype(num_pages, kv_stride, kvs_stride, s_unit):
    """The width the resolved offsets need.

    int32 while both streams still reach: values are stored in K_WIDTH units
    and scales in s_unit, so the scales bind first -- 4 GiB against 32 GiB.
    """
    fits = (
        num_pages * kv_stride <= 2**31 * K_WIDTH
        and num_pages * kvs_stride <= 2**31 * s_unit
    )
    return torch.int32 if fits else torch.int64


def gather_s_unit(block, num_scales):
    """The unit the scale list is stored in, in e8m0 bytes.

    Two whenever every resolved scale offset is even, which is what lets the
    kernel's `* U` hand the 2-byte alignment back to the vectorizer -- without
    it each of the tile's 2-byte runs splits into two buffer_load_ubyte. The
    kernel derives the same value from GATHER_BLOCK and NUM_SCALES; the two
    lines must stay in step.
    """
    return 2 if (block % 2 == 0 and num_scales % 2 == 0) else 1


def block_offsets(
    pos0,
    block_table,
    page_size,
    head_size,
    n_per_tile,
    kv_stride,
    kvs_stride,
    block,
    preshuffle=1,
    scale_mode=1,
    dtype=torch.int32,
):
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
        bn = (t0 % n_per_tile) * K_WIDTH + (t0 // n_per_tile) * (
            n_per_tile * head_bytes
        )
        # Mode 1 puts the token at stride s_hi inside a group, mode 0 at
        # stride 1. Both are constant, which is all a gather needs.
        tok_stride = s_hi if scale_mode == 1 else 1
        bs = (t0 % n_per_tile) * tok_stride + (t0 // n_per_tile) * (
            n_per_tile * num_scales
        )
    else:
        bn = t0 * head_bytes
        bs = t0 * num_scales

    # Stored in K_WIDTH units, not bytes, and multiplied back in the kernel:
    # an offset arriving from memory carries no provable alignment, and without
    # one Triton emits a buffer_load_ubyte per byte. Always whole -- the page
    # stride is 16-byte aligned and every `bn` is a multiple of 16.
    voff = pid * kv_stride + bn
    soff = pid * kvs_stride + bs
    s_unit = gather_s_unit(block, num_scales)
    # Every bound this function checks, in one device-to-host transfer. They
    # were a sync apiece, and the resolver already costs more than the launch
    # it feeds.
    v_mod, s_mod, p_mod, v_max, s_max = torch.stack(
        [
            (voff % K_WIDTH).max(),
            (soff % s_unit).max(),
            (pos0 % block).max(),
            voff.max() // K_WIDTH,
            soff.max(),
        ]
    ).tolist()
    assert v_mod == 0, "value offsets must be k_width aligned"
    assert s_mod == 0, "scale offsets must be unit aligned"
    # With page_size % block this keeps a block inside one shuffle group and one
    # page, which is what makes this file's opening `c` term loop invariant. A
    # misaligned start satisfies both checks above and reads the wrong bytes.
    assert p_mod == 0, "positions must be block-aligned"
    assert dtype == torch.int64 or (
        v_max < 2**31 and s_max < 2**31
    ), "resolved offsets do not fit i32; pass dtype=torch.int64"
    return (voff // K_WIDTH).to(dtype), (soff // s_unit).to(dtype)


def build_gather(
    positions,
    block_table,
    kv_cache,
    num_heads,
    head_size,
    block=8,
    kv_scale_cache=None,
    preshuffle=1,
    scale_mode=1,
    dtype=None,
):
    """[R, K] int64 block-start positions -> the kernel's `gather=` argument.

    `positions[r]` must be sorted ascending, unique, every entry a multiple of
    `block`, and causally legal for row r: the kernel applies no candidate mask
    and no causal limit of its own.

    How many of the expanded `K * block` slots are real is the launch's
    `row_ends` -- the same per-row exclusive bound the dense path takes, read in
    slot space rather than key space. Slots past it are dropped by store_hi.
    """
    n_per_tile = mfma_nonk_dim(num_heads, head_size)
    page_size, kv_stride, kvs_stride = cache_strides(
        kv_cache, head_size, kv_scale_cache
    )
    assert page_size % block == 0 and block <= n_per_tile
    assert scale_mode in (0, 1), "scale_mode must be 0 or 1"
    if dtype is None:
        dtype = offset_dtype(
            kv_cache.shape[0],
            kv_stride,
            kvs_stride,
            gather_s_unit(block, head_size // SCALE_GROUP),
        )
    voff, soff = block_offsets(
        positions,
        block_table,
        page_size,
        head_size,
        n_per_tile,
        kv_stride,
        kvs_stride,
        block,
        preshuffle,
        scale_mode,
        dtype,
    )
    return {
        "voff": voff.contiguous(),
        "soff": soff.contiguous(),
        "block": block,
        "positions": positions,
    }


def expand(positions, block=8):
    """The slot -> KV position map the kernel walks, for a reference."""
    r, k = positions.shape
    c = torch.arange(block, device=positions.device, dtype=positions.dtype)
    return (positions[:, :, None] + c[None, None, :]).reshape(r, k * block)
