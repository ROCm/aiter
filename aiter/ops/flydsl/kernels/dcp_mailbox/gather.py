# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL kernel body for the DCP query gather."""

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
from aiter.ops.flydsl.kernels.dcp_mailbox.common import bf2_f32, finish, push_epoch
from aiter.ops.flydsl.kernels.quant_utils import emit_f32_to_fp8
from aiter.ops.flydsl.kernels.tensor_shim import (
    buf_copy_load,
    buf_copy_store,
    ptr_buf_tensor,
)

FP8_MAX = 448.0


def make_kernel(key, name, block_count, layout, half_span, epoch_base, threads):
    """Create the specialized DCP gather kernel."""
    ranks = key.ranks
    heads = key.heads
    words = key.words
    max_tokens = key.max_tokens
    tile = key.token_tile

    @flyc.kernel(name=name, known_block_size=[threads, 1, 1])
    def gather(
        q: Int64,
        q_sb: Int32,
        q_sh: Int32,
        pe: Int64,
        pe_sb: Int32,
        pe_sh: Int32,
        scale: Int64,
        gq: Int64,
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
        put_words, poll = mailbox.put_words, mailbox.poll
        targets = [
            tagged_shift(tagged_peer_address(peer, layout["q"][0]), parity_offset)
            for peer in peer_bases
        ]
        own_data = tagged_shift(tagged_peer_address(own, layout["q"][0]), parity_offset)
        word = fx.min(tid, words // 2 - 1) * 2
        push = key.phase != 2
        for token_offset in range_constexpr(tile):
            row = first_row + token_offset
            if row < key.tokens:
                slot = (rank * max_tokens + row) * heads + local_head
                if const_expr(push and key.nope > 0):
                    element = word * 4
                    in_nope = element < key.nope
                    nope_words = buf_copy_load(
                        ptr_buf_tensor(q, fx.Int32, unit_elems=4, unit_stride=1),
                        (row * q_sb + local_head * q_sh + fx.min(element, key.nope - 8))
                        // 2,
                        fx.Int32,
                        unit_elems=4,
                    )
                    pe_words = buf_copy_load(
                        ptr_buf_tensor(pe, fx.Int32, unit_elems=4, unit_stride=1),
                        (
                            row * pe_sb
                            + local_head * pe_sh
                            + fx.max(element - key.nope, 0)
                        )
                        // 2,
                        fx.Int32,
                        unit_elems=4,
                    )
                    inverse = fx.Float32(1.0) / fx.Float32(
                        buf_copy_load(
                            ptr_buf_tensor(scale, fx.Float32),
                            0,
                            fx.Float32,
                        )
                    )
                    values = []
                    for index in range_constexpr(4):
                        low, high = bf2_f32(
                            in_nope.select(nope_words[index], pe_words[index])
                        )
                        values += [low, high]
                    quantized = [
                        fx.max(
                            fx.min(value * inverse, fx.Float32(FP8_MAX)),
                            fx.Float32(-FP8_MAX),
                        )
                        for value in values
                    ]
                    packed = [
                        emit_f32_to_fp8(quantized[0], quantized[1])
                        | (emit_f32_to_fp8(quantized[2], quantized[3]) << 16),
                        emit_f32_to_fp8(quantized[4], quantized[5])
                        | (emit_f32_to_fp8(quantized[6], quantized[7]) << 16),
                    ]
                if const_expr(push and key.nope == 0):
                    query_words = buf_copy_load(
                        ptr_buf_tensor(q, fx.Int32, unit_elems=2, unit_stride=1),
                        row * q_sb + local_head * q_sh + word,
                        fx.Int32,
                        unit_elems=2,
                    )
                    packed = [query_words[0], query_words[1]]
                if const_expr(push):
                    for destination in range_constexpr(ranks):
                        put_words(
                            targets[destination],
                            slot * words + word,
                            [packed[0], packed[1]],
                        )
                if const_expr(key.phase != 1):
                    received = poll(
                        [
                            (
                                own_data,
                                ((source * max_tokens + row) * heads + local_head)
                                * words
                                + word,
                                2,
                            )
                            for source in range(ranks)
                        ],
                        batch=ranks,
                    )
                    output = ptr_buf_tensor(gq, fx.Int32, unit_elems=2, unit_stride=1)
                    for source in range_constexpr(ranks):
                        if const_expr(key.fold > 0):
                            head = source * heads + local_head
                            output_row = (
                                (row // key.qlen) * key.fold + head // 16
                            ) * key.qlen + row % key.qlen
                            output_index = (output_row * 16 + head % 16) * words + word
                        else:
                            output_index = (
                                row * key.pad_heads + source * heads + local_head
                            ) * words + word
                        buf_copy_store(
                            output,
                            output_index,
                            fx.Vector.from_elements(
                                [received[source][0], received[source][1]],
                                fx.Int32,
                            ),
                            fx.Int32,
                            unit_elems=2,
                        )
        if const_expr(key.phase != 1):
            finish(tid, state, epoch, block_count)

    return gather
