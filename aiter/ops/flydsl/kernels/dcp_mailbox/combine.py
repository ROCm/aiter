# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL kernel body for the DCP LSE combine."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Int32, Int64

from aiter.ops.flydsl.kernels.communication_ops_utils import (
    CACHE_DEVICE,
    EPOCH_MASK,
    TaggedMailbox,
    buffer_load_i32,
    load_i64_from_buffer,
    tagged_peer_address,
    tagged_shift,
    wave_uniform_i32,
)
from aiter.ops.flydsl.kernels.dcp_mailbox.common import (
    bf2_f32,
    bf16_pair,
    finish,
    push_epoch,
)
from aiter.ops.flydsl.kernels.kernels_common import LOG2E
from aiter.ops.flydsl.kernels.tensor_shim import (
    buf_copy_load,
    buf_copy_store,
    ptr_buf_tensor,
)

NEG_INF = float("-inf")
POS_INF = float("inf")


def _row_empty(row, seqs, seq_lens, query_start_loc):
    """Return whether a graph row has no local KV data."""
    rows = buffer_load_i32(query_start_loc, seqs)
    seq_index = fx.Int32(0)
    for index in range_constexpr(seqs):
        seq_index = seq_index + (
            buffer_load_i32(query_start_loc, index + 1) <= row
        ).select(1, 0)
    seq_index = fx.min(seq_index, seqs - 1)
    return (row >= rows) | (buffer_load_i32(seq_lens, seq_index) == 0)


@flyc.jit
def _poll_combine(
    poll,
    own_lse,
    own_data,
    ranks,
    max_tokens,
    heads,
    words,
    row,
    local_head,
    tid,
    base_e,
    return_lse,
    result,
    result_stride_batch,
    result_stride_head,
    result_lse,
):
    """Poll one head from all ranks and store its LSE combine."""
    word = tid * 2
    own_slot = [
        (source * max_tokens + row) * heads + local_head for source in range(ranks)
    ]
    got = poll(
        [(own_lse, own_slot[source], 1) for source in range(ranks)]
        + [(own_data, own_slot[source] * words + word, 2) for source in range(ranks)],
        batch=2 * ranks,
    )
    lse_values = []
    for source in range_constexpr(ranks):
        value = got[source][0].bitcast(fx.Float32)
        invalid = (value != value) | (value == fx.Float32(POS_INF))  # noqa: PLR0124
        lse_values.append(invalid.select(fx.Float32(NEG_INF), value))
    maximum = lse_values[0]
    for source in range_constexpr(1, ranks):
        maximum = fx.max(maximum, lse_values[source])
    maximum = (maximum == fx.Float32(NEG_INF)).select(fx.Float32(0.0), maximum)
    scale = LOG2E if base_e else 1.0
    exponentials = [
        fx.exp2((lse_values[source] - maximum) * fx.Float32(scale))
        for source in range(ranks)
    ]
    total = exponentials[0]
    for source in range_constexpr(1, ranks):
        total = total + exponentials[source]
    live = total > fx.Float32(0.0)
    inverse = fx.Float32(1.0) / live.select(total, fx.Float32(1.0))
    weights = [
        live.select(exponentials[source] * inverse, fx.Float32(0.0))
        for source in range(ranks)
    ]
    accumulator = [fx.Float32(0.0)] * 4
    for source in range_constexpr(ranks):
        pair = got[ranks + source]
        zero = weights[source] == fx.Float32(0.0)
        for index in range_constexpr(2):
            low, high = bf2_f32(pair[index])
            accumulator[2 * index] = accumulator[2 * index] + zero.select(
                fx.Float32(0.0), low * weights[source]
            )
            accumulator[2 * index + 1] = accumulator[2 * index + 1] + zero.select(
                fx.Float32(0.0), high * weights[source]
            )
    result_words = fx.Vector.from_elements(
        [
            bf16_pair(accumulator[0], accumulator[1]).bitcast(fx.Int32),
            bf16_pair(accumulator[2], accumulator[3]).bitcast(fx.Int32),
        ],
        fx.Int32,
    )
    buf_copy_store(
        ptr_buf_tensor(result, fx.Int32, unit_elems=2, unit_stride=1),
        (row * result_stride_batch + local_head * result_stride_head) // 2 + word,
        result_words,
        fx.Int32,
        unit_elems=2,
    )
    if const_expr(return_lse):  # noqa: SIM102
        if tid == 0:
            if const_expr(base_e):
                combined = fx.log(total) + maximum
            else:
                combined = fx.log2(total) + maximum
            buf_copy_store(
                ptr_buf_tensor(result_lse, fx.Float32),
                row * heads + local_head,
                fx.Float32(combined),
                fx.Float32,
            )


def make_kernel(key, name, block_count, threads, layout, half_span, epoch_base):
    """Create the specialized DCP combine kernel."""
    ranks = key.ranks
    heads = key.heads
    words = key.dim // 2
    max_tokens = key.max_tokens
    tile = key.token_tile

    @flyc.kernel(name=name, known_block_size=[threads, 1, 1])
    def combine(
        out: Int64,
        out_sb: Int32,
        out_sh: Int32,
        lse: Int64,
        lse_sb: Int32,
        lse_sh: Int32,
        res: Int64,
        res_sb: Int32,
        res_sh: Int32,
        res_lse: Int64,
        seq_lens: Int64,
        qsl: Int64,
        row_counts: Int64,
        peers: Int64,
        rank: Int32,
        state: Int64,
    ):
        tid = fx.thread_idx.x
        block = fx.block_idx.x
        first_row = (block // heads) * tile
        local_head = block % heads
        epoch = wave_uniform_i32(buffer_load_i32(state, 0, cache_modifier=CACHE_DEVICE))
        parity_offset = fx.Int64(epoch % 2) * half_span
        peer_bases = [load_i64_from_buffer(peers, source) for source in range(ranks)]
        own = load_i64_from_buffer(peers, rank)
        epoch_targets = [
            tagged_peer_address(peer, epoch_base + rank * 4) for peer in peer_bases
        ]
        own_epoch = tagged_peer_address(own, epoch_base)
        epoch_sources = [
            tagged_peer_address(own_epoch, source * 4) for source in range(ranks)
        ]
        if const_expr(key.debug):
            push_epoch(block, tid, epoch, epoch_targets)
        mailbox = TaggedMailbox(
            epoch & fx.Int32(EPOCH_MASK),
            debug=key.debug,
            epoch_addresses=epoch_sources,
            expected_epoch=epoch + 1,
            diagnostic=state,
        )
        put, put_words, poll = mailbox.put, mailbox.put_words, mailbox.poll
        data_targets = [
            tagged_shift(tagged_peer_address(peer, layout["data"][0]), parity_offset)
            for peer in peer_bases
        ]
        lse_targets = [
            tagged_shift(tagged_peer_address(peer, layout["lse"][0]), parity_offset)
            for peer in peer_bases
        ]
        own_data = tagged_shift(
            tagged_peer_address(own, layout["data"][0]), parity_offset
        )
        own_lse = tagged_shift(
            tagged_peer_address(own, layout["lse"][0]), parity_offset
        )
        word = tid * 2
        for token_offset in range_constexpr(tile):
            row = first_row + token_offset
            if row < key.tokens:
                slot = (rank * max_tokens + row) * heads + local_head
                out_buffer = ptr_buf_tensor(out, fx.Int32, unit_elems=2, unit_stride=1)
                out_words = [
                    buf_copy_load(
                        out_buffer,
                        (row * out_sb + (destination * heads + local_head) * out_sh)
                        // 2
                        + word,
                        fx.Int32,
                        unit_elems=2,
                    )
                    for destination in range(ranks)
                ]
                lse_buffer = ptr_buf_tensor(lse, fx.Float32)
                lse_values = [
                    fx.Float32(
                        buf_copy_load(
                            lse_buffer,
                            row * lse_sb + (destination * heads + local_head) * lse_sh,
                            fx.Float32,
                        )
                    )
                    for destination in range(ranks)
                ]
                for destination in range_constexpr(ranks):
                    put_words(
                        data_targets[destination],
                        slot * words + word,
                        [out_words[destination][0], out_words[destination][1]],
                    )
                if const_expr(key.seqs > 0):
                    empty = _row_empty(row, key.seqs, seq_lens, qsl)
                    for destination in range_constexpr(ranks):
                        lse_values[destination] = empty.select(
                            fx.Float32(NEG_INF), lse_values[destination]
                        )
                if const_expr(key.row_mask):
                    empty = buffer_load_i32(row_counts, row) == 0
                    for destination in range_constexpr(ranks):
                        lse_values[destination] = empty.select(
                            fx.Float32(NEG_INF), lse_values[destination]
                        )
                for destination in range_constexpr(ranks):
                    if tid == destination:
                        put(
                            lse_targets[destination],
                            slot,
                            lse_values[destination],
                        )
                _poll_combine(
                    poll,
                    own_lse,
                    own_data,
                    ranks,
                    max_tokens,
                    heads,
                    words,
                    row,
                    local_head,
                    tid,
                    key.base_e,
                    key.ret_lse,
                    res,
                    res_sb,
                    res_sh,
                    res_lse,
                )
        finish(tid, state, epoch, block_count)

    return combine
