# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exact split-candidate TopK merge with optional paged-index mapping."""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr

from .candidate_topk_common import (
    BLOCK_THREADS,
    NUM_HIST_BINS,
    NUM_KEY_BYTES,
    NUM_WAVES,
    RADIX_SIGN_BIT,
    block_exclusive_prefix_i32,
    f32_to_ordered_i32,
    key_is_better,
    key_matches_prefix,
    prefix_byte_mask,
    radix_byte,
)
from .kernels_common import atomic_add_i32

_SCORE_PREFIX = 0
_SCORE_MASK = 1
_INDEX_PREFIX = 2
_INDEX_MASK = 3
_REMAINING = 4
_WRITE_CURSOR = 5
_EQUAL_SEEN = 6
_TARGET = 7
_STATE_SIZE = 8

# A scan processes at most 256 predicates, so its low 16-bit count cannot carry
# into the packed high count.
_PACK_SHIFT = 16
_PACK_MASK = (1 << _PACK_SHIFT) - 1


def _make_storage():
    @fx.struct
    class Storage:
        histogram: fx.Array[fx.Int32, NUM_HIST_BINS, 16]
        scan: fx.Array[fx.Int32, NUM_WAVES + 1, 16]
        state: fx.Array[fx.Int32, _STATE_SIZE, 16]

    return Storage


@cache
def build_candidate_topk_merge_module(topk: int, page_size: int):
    """Build one-block-per-row exact merge for fixed-width CTA planes.

    Candidate planes in ``[row_offsets[r], row_offsets[r + 1])`` may contain up
    to ``topk`` live entries each. Selection is lexicographic
    ``(score descending, raw index ascending)`` over the union.  Output order is
    stable plane/slot order; membership, including the cutoff tie, follows the
    total order exactly.
    """

    if topk <= 0:
        raise ValueError("topk must be positive")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    candidate_steps = (topk + BLOCK_THREADS - 1) // BLOCK_THREADS

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def merge_kernel(
        candidate_values: fx.Tensor,
        candidate_indices: fx.Tensor,
        candidate_counts: fx.Tensor,
        row_offsets: fx.Tensor,
        row_to_batch: fx.Tensor,
        block_table: fx.Tensor,
        out_values: fx.Tensor,
        out_raw_indices: fx.Tensor,
        out_physical_indices: fx.Tensor,
        out_counts: fx.Tensor,
    ):
        row = fx.block_idx.x
        tid = fx.thread_idx.x
        cta_begin = row_offsets[row]
        cta_end = row_offsets[row + fx.Int32(1)]

        storage = fx.SharedAllocator().allocate(_make_storage())
        histogram = storage.histogram.peek().view(fx.make_layout(NUM_HIST_BINS, 1))
        scan = storage.scan.peek().view(fx.make_layout(NUM_WAVES + 1, 1))
        state = storage.state.peek().view(fx.make_layout(_STATE_SIZE, 1))

        if tid == 0:
            total = fx.Int32(0)
            for cta in range(cta_begin, cta_end, fx.Int32(1)):
                count = candidate_counts[cta]
                count = (count < 0).select(fx.Int32(0), count)
                count = (count > fx.Int32(topk)).select(fx.Int32(topk), count)
                total = total + count
            target = (total < fx.Int32(topk)).select(total, fx.Int32(topk))
            state[_SCORE_PREFIX] = 0
            state[_SCORE_MASK] = 0
            state[_INDEX_PREFIX] = 0
            state[_INDEX_MASK] = 0
            state[_REMAINING] = target
            state[_TARGET] = target
        gpu.barrier()

        # Eight stable radix decisions: four score bytes in descending signed
        # ordinal order, then four non-negative raw-index bytes ascending.
        for key_byte in range_constexpr(NUM_KEY_BYTES):
            histogram[tid] = 0
            gpu.barrier()

            score_prefix = state[_SCORE_PREFIX]
            score_mask = state[_SCORE_MASK]
            index_prefix = state[_INDEX_PREFIX]
            index_mask = state[_INDEX_MASK]
            if const_expr(key_byte < 4):
                byte_position = key_byte
                xor_value = RADIX_SIGN_BIT if key_byte == 0 else 0
            else:
                byte_position = key_byte - 4
                xor_value = 0

            for cta in range(cta_begin, cta_end, fx.Int32(1)):
                count = candidate_counts[cta]
                count = (count < 0).select(fx.Int32(0), count)
                count = (count > fx.Int32(topk)).select(fx.Int32(topk), count)
                for step in range_constexpr(candidate_steps):
                    slot = fx.Int32(step * BLOCK_THREADS) + tid
                    live = slot < count
                    safe_slot = live.select(slot, fx.Int32(0))
                    score_ord = f32_to_ordered_i32(candidate_values[cta, safe_slot])
                    raw_index = candidate_indices[cta, safe_slot]
                    prefix_match = live & key_matches_prefix(
                        score_ord,
                        raw_index,
                        score_prefix,
                        score_mask,
                        index_prefix,
                        index_mask,
                    )
                    if prefix_match:
                        key = score_ord if const_expr(key_byte < 4) else raw_index
                        bucket = radix_byte(key, byte_position) ^ fx.Int32(xor_value)
                        atomic_add_i32(histogram, 1, bucket, "workgroup")
            gpu.barrier()

            selected_bucket = (
                fx.Int32(NUM_HIST_BINS - 1) - tid if const_expr(key_byte < 4) else tid
            )
            count = histogram[selected_bucket]
            before, _ = block_exclusive_prefix_i32(tid, count, scan)
            remaining = state[_REMAINING]
            # Keep every lane on the old remaining value before the unique hit
            # lane publishes the next prefix.
            gpu.barrier()
            if (before < remaining) & (before + count >= remaining):
                actual_bucket = selected_bucket ^ fx.Int32(xor_value)
                byte_mask, shift = prefix_byte_mask(byte_position)
                if const_expr(key_byte < 4):
                    state[_SCORE_PREFIX] = score_prefix | (
                        actual_bucket << fx.Int32(shift)
                    )
                    state[_SCORE_MASK] = score_mask | byte_mask
                else:
                    state[_INDEX_PREFIX] = index_prefix | (
                        actual_bucket << fx.Int32(shift)
                    )
                    state[_INDEX_MASK] = index_mask | byte_mask
                state[_REMAINING] = remaining - before
            gpu.barrier()

        threshold_score = state[_SCORE_PREFIX]
        threshold_index = state[_INDEX_PREFIX]
        threshold_equal_needed = state[_REMAINING]
        target = state[_TARGET]

        row_values = fx.slice(out_values, (row, None))
        row_raw = fx.slice(out_raw_indices, (row, None))
        row_physical = fx.slice(out_physical_indices, (row, None))
        for step in range_constexpr(candidate_steps):
            out_pos = fx.Int32(step * BLOCK_THREADS) + tid
            if out_pos < fx.Int32(topk):
                row_values[out_pos] = fx.Float32(float("-inf"))
                row_raw[out_pos] = fx.Int32(-1)
                row_physical[out_pos] = fx.Int32(-1)
        if tid == 0:
            state[_WRITE_CURSOR] = 0
            state[_EQUAL_SEEN] = 0
            out_counts[row] = target
        gpu.barrier()

        batch = row_to_batch[row]
        row_blocks = fx.slice(block_table, (batch, None))
        for cta in range(cta_begin, cta_end, fx.Int32(1)):
            count = candidate_counts[cta]
            count = (count < 0).select(fx.Int32(0), count)
            count = (count > fx.Int32(topk)).select(fx.Int32(topk), count)
            for step in range_constexpr(candidate_steps):
                slot = fx.Int32(step * BLOCK_THREADS) + tid
                live = slot < count
                safe_slot = live.select(slot, fx.Int32(0))
                value = candidate_values[cta, safe_slot]
                score_ord = f32_to_ordered_i32(value)
                raw_index = candidate_indices[cta, safe_slot]
                better = live & key_is_better(
                    score_ord,
                    raw_index,
                    threshold_score,
                    threshold_index,
                )
                equal = (
                    live
                    & (score_ord == threshold_score)
                    & (raw_index == threshold_index)
                )
                better_i32 = better.select(fx.Int32(1), fx.Int32(0))
                equal_i32 = equal.select(fx.Int32(1), fx.Int32(0))
                packed = (better_i32 << fx.Int32(_PACK_SHIFT)) + equal_i32
                packed_before, packed_total = block_exclusive_prefix_i32(
                    tid,
                    packed,
                    scan,
                )
                better_before = packed_before >> fx.Int32(_PACK_SHIFT)
                equal_before = packed_before & fx.Int32(_PACK_MASK)
                better_total = packed_total >> fx.Int32(_PACK_SHIFT)
                equal_total = packed_total & fx.Int32(_PACK_MASK)

                equal_seen = state[_EQUAL_SEEN]
                room = threshold_equal_needed - equal_seen
                room = (room < 0).select(fx.Int32(0), room)
                admit_equal = equal & (equal_before < room)
                keep = better | admit_equal
                admitted_equal_before = (equal_before < room).select(
                    equal_before,
                    room,
                )
                destination = (
                    state[_WRITE_CURSOR] + better_before + admitted_equal_before
                )
                if keep:
                    physical = row_blocks[raw_index // fx.Int32(page_size)] * fx.Int32(
                        page_size
                    ) + raw_index % fx.Int32(page_size)
                    row_values[destination] = value
                    row_raw[destination] = raw_index
                    row_physical[destination] = physical
                gpu.barrier()

                if tid == 0:
                    admitted_equal_total = (equal_total < room).select(
                        equal_total,
                        room,
                    )
                    state[_WRITE_CURSOR] = (
                        state[_WRITE_CURSOR] + better_total + admitted_equal_total
                    )
                    state[_EQUAL_SEEN] = state[_EQUAL_SEEN] + equal_total
                gpu.barrier()

    @flyc.jit
    def launch(
        candidate_values: fx.Tensor,
        candidate_indices: fx.Tensor,
        candidate_counts: fx.Tensor,
        row_offsets: fx.Tensor,
        row_to_batch: fx.Tensor,
        block_table: fx.Tensor,
        out_values: fx.Tensor,
        out_raw_indices: fx.Tensor,
        out_physical_indices: fx.Tensor,
        out_counts: fx.Tensor,
        rows: fx.Int32,
        stream: fx.Stream,
    ):
        merge_kernel(
            candidate_values,
            candidate_indices,
            candidate_counts,
            row_offsets,
            row_to_batch,
            block_table,
            out_values,
            out_raw_indices,
            out_physical_indices,
            out_counts,
        ).launch(
            grid=(rows, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch
