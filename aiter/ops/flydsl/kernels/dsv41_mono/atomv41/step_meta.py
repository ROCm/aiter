# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The per-token integers of a decode step the mono kernels read, each in one
launch: each token's window ring row (K1 writes its KV row there) and its window key
metadata (K2a's attention, ``attention._keys``): a target verify step's
(``write_step_meta``; as a torch chain these were ~19 elementwise kernels a
step) and a DSpark draft block's (``write_draft_step_meta``, with its rows'
positions)."""

import triton
import triton.language as tl

# a token's window key metadata words: first window position, window count,
# block table offset, slot row base
KMETA = 4


@triton.jit
def _step_meta_kernel(
    batch_ids,
    slots,
    positions,
    ring_rel,
    key_meta,
    table_stride,
    slot_rows,
    ring_slots,
    window_size,
    n,
    WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    m = i < n
    batch = tl.load(batch_ids + i, mask=m, other=-1).to(tl.int64)
    live = batch >= 0
    row = tl.maximum(batch, 0)
    slot = tl.load(slots + row, mask=m, other=0).to(tl.int64)
    pos = tl.load(positions + i, mask=m, other=0).to(tl.int64)
    # the bf16 pool's WindowParams: slot * slot_rows + pos % ring_slots; -1 a pad
    rel = slot * slot_rows + pos % ring_slots
    tl.store(ring_rel + i, tl.where(live, rel, -1).to(tl.int32), mask=m)
    # the original index build's (indices._indices) integers for a decode
    first = tl.maximum(pos - window_size + 1, 0)
    count = tl.where(live, pos + 1 - first, 0)
    tl.store(key_meta + i * WORDS, first.to(tl.int32), mask=m)
    tl.store(key_meta + i * WORDS + 1, count.to(tl.int32), mask=m)
    tl.store(key_meta + i * WORDS + 2, (row * table_stride).to(tl.int32), mask=m)
    tl.store(key_meta + i * WORDS + 3, (slot * slot_rows).to(tl.int32), mask=m)


def write_step_meta(step, geometry, window, ring_rel, key_meta) -> None:
    """``ring_rel`` [S] int32: each token's window row before the layer's ring
    start, -1 for a pad row. ``key_meta`` [S, KMETA] int32: its first window
    position, window count (0 for a pad), block table offset and slot row base."""
    n = ring_rel.shape[0]
    assert key_meta.shape == (n, KMETA) and key_meta.is_contiguous()
    _step_meta_kernel[(1,)](
        step.batch_ids,
        step.slots,
        step.positions,
        ring_rel,
        key_meta,
        step.block_tables.stride(0),
        window.slot_rows,
        window.ring_slots,
        geometry.window_size,
        n,
        WORDS=KMETA,
        BLOCK=triton.next_power_of_2(n),
    )


@triton.jit
def _draft_meta_kernel(
    anchors,
    slots,
    positions,
    ring_rel,
    key_meta,
    slot_rows,
    ring_slots,
    window_size,
    width,
    num_slots,
    n,
    WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    m = i < n
    # row i: request i // width's block row i % width
    request = i // width
    anchor = tl.load(anchors + request, mask=m, other=0).to(tl.int64)
    slot = tl.load(slots + request, mask=m, other=0).to(tl.int64)
    # a slot outside the pool (a warmup batch's rows past the ones its metadata
    # wrote) writes no key and reads none
    live = (slot >= 0) & (slot < num_slots)
    slot = tl.where(live, slot, 0)
    pos = anchor + 1 + i % width
    tl.store(positions + i, pos, mask=m)
    rel = tl.where(live, slot * slot_rows + pos % ring_slots, -1)
    tl.store(ring_rel + i, rel.to(tl.int32), mask=m)
    # every row: its request's window up to the anchor and whole block, both ways
    first = tl.maximum(anchor - window_size + 1, 0)
    count = tl.where(live, anchor + width - first + 1, 0)
    tl.store(key_meta + i * WORDS, first.to(tl.int32), mask=m)
    tl.store(key_meta + i * WORDS + 1, count.to(tl.int32), mask=m)
    # no block table offset: the draft has no indexer, so nothing selects through it
    tl.store(key_meta + i * WORDS + 2, tl.zeros([BLOCK], dtype=tl.int32), mask=m)
    tl.store(key_meta + i * WORDS + 3, (slot * slot_rows).to(tl.int32), mask=m)


def write_draft_step_meta(
    anchors, slots, num_slots, geometry, window, positions, ring_rel, key_meta
) -> None:
    """The DSpark draft blocks of ``anchors``' requests, request after request,
    ``width`` = rows / requests rows each: a block's rows sit at anchor + 1 ..
    anchor + width (``positions`` int64), their keys go to its slot's ring at
    those positions (``ring_rel``: the ring has window + width slots, so they
    never overwrite the window), and every row sees its request's window up to
    the anchor and all of its block (``key_meta``). A request whose slot is not
    one of the pool's ``num_slots`` writes and reads no key (``ring_rel`` -1,
    count 0); a padded request's slot, a real one, would be another's."""
    n = ring_rel.shape[0]
    requests = anchors.numel()
    width = n // requests
    assert width * requests == n and slots.numel() >= requests
    assert key_meta.shape == (n, KMETA) and key_meta.is_contiguous()
    assert window.ring_slots >= geometry.window_size + width
    _draft_meta_kernel[(1,)](
        anchors,
        slots,
        positions,
        ring_rel,
        key_meta,
        window.slot_rows,
        window.ring_slots,
        geometry.window_size,
        width,
        num_slots,
        n,
        WORDS=KMETA,
        BLOCK=triton.next_power_of_2(n),
    )
