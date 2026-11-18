# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Split-aware merge for the fused indexer select.

The scorer already counted the top 8 bits of each candidate key. This kernel
sums those histograms across the row's spans, walks them from the high bin,
and runs the second radix pass on the one bin that straddles 512. Exact ties
on that final key may resolve to either block. It then expands the packed
block ids to token ids and clears the histograms it read.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import Int32, gpu, range_constexpr

from aiter.ops.flydsl.kernels.kernels_common import div_nonneg, kernel_signature
from aiter.ops.flydsl.kernels.qsa.indexer_fused_select import HIST_BINS
from aiter.ops.flydsl.kernels.qsa.indexer_select import _K, _R

_TOKEN_WIDTH = _K * _R + _R - 1
_MERGE_THREADS = HIST_BINS


def build_qsa_indexer_fused_merge_module():
    """One workgroup merges one row's span candidates and writes its token ids."""
    if _MERGE_THREADS != 256 or _K != 512:
        raise ValueError("fused merge is built for 256 histogram bins and 512 blocks")

    @fx.struct
    class MergeStorage:
        counts: fx.Array[Int32, HIST_BINS, 16]
        higher: fx.Array[Int32, HIST_BINS, 16]
        low_counts: fx.Array[Int32, HIST_BINS, 16]
        low_higher: fx.Array[Int32, HIST_BINS, 16]
        selected: fx.Array[Int32, _K, 16]

    @flyc.kernel(
        name="qsa_indexer_fused_merge_"
        + kernel_signature(k=_K, bins=HIST_BINS, r=_R, tok=_TOKEN_WIDTH),
        known_block_size=[_MERGE_THREADS, 1, 1],
    )
    def qsa_indexer_fused_merge_kernel(
        candidates: fx.Tensor,
        histogram: fx.Tensor,
        split_begins: fx.Tensor,
        block_ids: fx.Tensor,
        token_ids: fx.Tensor,
        token_to_req: fx.Tensor,
        query_positions: fx.Tensor,
        context_lens: fx.Tensor,
        n_splits: Int32,
        rows: Int32,
        n_req: Int32,
    ):
        row = fx.block_idx.x
        tid = fx.thread_idx.x
        zero = Int32(0)
        one = Int32(1)
        neg = Int32(-1)
        budget = Int32(_K)
        ratio = Int32(_R)
        width = Int32(_TOKEN_WIDTH)
        storage = fx.SharedAllocator().allocate(MergeStorage).peek()
        counts = storage.counts.view(fx.make_layout(HIST_BINS, 1))
        higher = storage.higher.view(fx.make_layout(HIST_BINS, 1))
        low_counts = storage.low_counts.view(fx.make_layout(HIST_BINS, 1))
        low_higher = storage.low_higher.view(fx.make_layout(HIST_BINS, 1))
        selected = storage.selected.view(fx.make_layout(_K, 1))

        if row < rows:
            total = zero
            for split in range(n_splits):
                total = total + histogram[Int32(split), row, tid]
            counts[tid] = total
            low_counts[tid] = zero
            selected[tid] = neg
            selected[tid + Int32(_MERGE_THREADS)] = neg
        gpu.barrier()

        # ``higher[bin]`` is how many candidates have a strictly better high
        # byte. The bin that crosses 512 is the second radix pass.
        if (row < rows) & (tid == zero):
            running = zero
            partial = neg
            quota = zero
            for step in range(Int32(HIST_BINS)):
                bin_id = Int32(HIST_BINS - 1) - Int32(step)
                count = counts[bin_id]
                higher[bin_id] = running
                crossed = (
                    (partial < zero) & (running < budget) & ((running + count) > budget)
                )
                if crossed:
                    partial = bin_id
                    quota = budget - running
                running = running + count

            if partial >= zero:
                for split_c in range(n_splits):
                    for slot_c in range(budget):
                        packed_c = candidates[Int32(split_c), row, Int32(slot_c)]
                        if packed_c >= zero:
                            key_c = (packed_c >> Int32(16)) & Int32(0xFFFF)
                            if (key_c >> Int32(8)) == partial:
                                low_c = key_c & Int32(0xFF)
                                low_counts[low_c] = low_counts[low_c] + one
            low_running = zero
            low_partial = neg
            low_take = zero
            for low_step in range(Int32(HIST_BINS)):
                low_id = Int32(HIST_BINS - 1) - Int32(low_step)
                low_count = low_counts[low_id]
                low_higher[low_id] = low_running
                low_crossed = (
                    (partial >= zero)
                    & (low_partial < zero)
                    & (low_running < quota)
                    & ((low_running + low_count) > quota)
                )
                if low_crossed:
                    low_partial = low_id
                    low_take = quota - low_running
                low_running = low_running + low_count

            written = zero
            ties = low_take
            for split_w in range(n_splits):
                begin_w = split_begins[Int32(split_w)]
                for slot_w in range(budget):
                    packed_w = candidates[Int32(split_w), row, Int32(slot_w)]
                    if packed_w >= zero:
                        key_w = (packed_w >> Int32(16)) & Int32(0xFFFF)
                        high_w = key_w >> Int32(8)
                        low_w = key_w & Int32(0xFF)
                        above = higher[high_w]
                        take = zero
                        if above < budget:
                            if (partial >= zero) & (high_w == partial):
                                low_above = low_higher[low_w]
                                if low_above < quota:
                                    if (low_partial >= zero) & (low_w == low_partial):
                                        if ties > zero:
                                            ties = ties - one
                                            take = one
                                    else:
                                        take = one
                            else:
                                take = one
                        if (take == one) & (written < budget):
                            selected[written] = begin_w + (packed_w & Int32(0xFFFF))
                            written = written + one
        gpu.barrier()

        if row < rows:
            block_ids[row, tid] = selected[tid]
            block_ids[row, tid + Int32(_MERGE_THREADS)] = selected[
                tid + Int32(_MERGE_THREADS)
            ]
            req = token_to_req[row]
            req_ok = (req >= zero) & (req < n_req)
            safe_req = req_ok.select(req, zero)
            slen = context_lens[safe_req]
            slen = req_ok.select(slen, zero)
            qpos = query_positions[row]
            qpos_len = qpos + one
            visible = fx.min(div_nonneg(qpos_len, ratio), div_nonneg(slen, ratio))
            complete = fx.min(visible, budget)
            expanded_count = complete * ratio
            tail_start = div_nonneg(qpos_len, ratio) * ratio
            tail_count = qpos_len - tail_start
            for part in range_constexpr(9):
                col = tid + Int32(part * _MERGE_THREADS)
                if col < width:
                    is_expanded = col < expanded_count
                    # Columns past the expanded prefix still form a rank. Clamp
                    # it so the select's unused side stays inside the 512 ids.
                    rank = fx.min(div_nonneg(col, ratio), Int32(_K - 1))
                    block = is_expanded.select(selected[rank], neg)
                    offset = col - div_nonneg(col, ratio) * ratio
                    expanded = block * ratio + offset
                    tail_offset = col - expanded_count
                    past_prefix = col >= expanded_count
                    is_tail = (
                        past_prefix
                        & (tail_offset < tail_count)
                        & (tail_offset < Int32(_R - 1))
                    )
                    token = is_expanded.select(expanded, tail_start + tail_offset)
                    valid = (is_expanded | is_tail) & (token >= zero) & (token < slen)
                    token_ids[row, col] = valid.select(token, neg)
            for split_z in range(n_splits):
                histogram[Int32(split_z), row, tid] = zero

    @flyc.jit
    def launch_qsa_indexer_fused_merge(
        candidates: fx.Tensor,
        histogram: fx.Tensor,
        split_begins: fx.Tensor,
        block_ids: fx.Tensor,
        token_ids: fx.Tensor,
        token_to_req: fx.Tensor,
        query_positions: fx.Tensor,
        context_lens: fx.Tensor,
        n_splits: Int32,
        rows: Int32,
        n_req: Int32,
        stream: fx.Stream,
    ):
        qsa_indexer_fused_merge_kernel(
            candidates,
            histogram,
            split_begins,
            block_ids,
            token_ids,
            token_to_req,
            query_positions,
            context_lens,
            n_splits,
            rows,
            n_req,
        ).launch(
            grid=(rows, 1, 1),
            block=(_MERGE_THREADS, 1, 1),
            stream=stream,
        )

    return launch_qsa_indexer_fused_merge
