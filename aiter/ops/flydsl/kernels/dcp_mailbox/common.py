# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared device helpers for DCP mailbox kernels."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr

from aiter.ops.flydsl.kernels.comm.one_shot_allreduce import (
    _sentinel_mask as _bf16_sentinel_mask,
)
from aiter.ops.flydsl.kernels.communication_ops_utils import (
    CACHE_DEVICE,
    CACHE_SYSTEM,
    EPOCH_MASK,
    atomic_add_agent,
    spin_mailbox,
)
from aiter.ops.flydsl.kernels.tensor_shim import (
    buf_copy_load,
    buf_copy_store,
    ptr_buf_tensor,
)


def bf16_pair(low, high):
    """Pack two f32 values as one pair of bf16 values."""
    return (
        fx.Vector.from_elements([low, high], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Float32)[0]
    )


def bf2_f32(word):
    """Unpack one bf16 pair into two f32 values."""
    return (word << 16).bitcast(fx.Float32), (word & fx.Int32(-65536)).bitcast(
        fx.Float32
    )


@flyc.jit
def push_epoch(bid, tid, epoch, targets):
    """Debug builds publish ``epoch + 1`` so zero means never arrived."""
    for dest in range_constexpr(len(targets)):
        if (bid == 0) & (tid == dest):
            buf_copy_store(
                ptr_buf_tensor(targets[dest], fx.Int32),
                0,
                epoch + 1,
                fx.Int32,
                cache_modifier=CACHE_SYSTEM,
            )


@flyc.jit
def finish(tid, state, epoch, block_count):
    """The last CTA advances the epoch for the next launch."""
    gpu.barrier()
    if tid == 0:
        old = atomic_add_agent(state + fx.Int64(4), 1)
        if old == block_count - 1:
            state_buffer = ptr_buf_tensor(state, fx.Int32)
            buf_copy_store(
                state_buffer,
                1,
                fx.Int32(0),
                fx.Int32,
                cache_modifier=CACHE_DEVICE,
            )
            buf_copy_store(
                state_buffer,
                0,
                (epoch + 1) & fx.Int32(EPOCH_MASK),
                fx.Int32,
                cache_modifier=CACHE_DEVICE,
            )


# A mailbox word holds the sentinel until its payload lands: all ones, which is
# a bf16 NaN (0xFFFF) in each half and an fp32 NaN as a whole. Senders rewrite a
# payload equal to it to another NaN, so a landed word never reads as the
# sentinel and needs no tag beside it. The 0xFFFF test is one-shot all-reduce's.
SENTINEL = -1
BF16 = 0
F32 = 1


def _canonical(word, kind):
    """``word`` with the sentinel rewritten: a 0xFFFF bf16 half becomes 0x7FFF
    and an all-ones fp32 becomes 0x7FFFFFFF, both still NaN."""
    if kind == BF16:
        return word ^ _bf16_sentinel_mask(word)
    return (word == fx.Int32(SENTINEL)).select(fx.Int32(0x7FFFFFFF), word)


def _pending(word, kind):
    """Whether ``word`` still reads as the sentinel (its payload has not landed)."""
    if kind == BF16:
        return _bf16_sentinel_mask(word) != 0
    return word == fx.Int32(SENTINEL)


class SentinelMailbox:
    """Transfer 32-bit words into a peer's sentinel-filled mailbox, untagged.

    Unlike one-shot all-reduce, which rotates three buffers and rearms one a
    launch later, a slot here is rearmed in the launch that reads it: its owner
    polls it until no word reads as the sentinel, then writes the sentinel back
    (``rearm``). The parity flips every launch, so a peer writes the slot again
    two launches later. That write cannot overtake the rearm: the peer reaches
    launch e + 2 only after every rank finished launch e + 1, which this rank
    starts only after its launch e, rearm included, completed; and the buffer is
    uncached, so the rearm is in memory, not in an L2 the peer bypasses. This
    needs every rank to launch with the same token count, as the combine
    already requires, and every slot written in a launch to be read in it.
    """

    def __init__(
        self,
        *,
        debug=False,
        epoch_addresses=None,
        expected_epoch=None,
        diagnostic=None,
    ):
        self.debug = debug
        self.epoch_addresses = epoch_addresses
        self.expected_epoch = expected_epoch
        self.diagnostic = diagnostic

    def put(self, address, index, words, kind):
        values = [_canonical(fx.Int32(word), kind) for word in words]
        buf_copy_store(
            ptr_buf_tensor(address, fx.Int32, unit_elems=len(values), unit_stride=1),
            index,
            (
                values[0]
                if len(values) == 1
                else fx.Vector.from_elements(values, fx.Int32)
            ),
            fx.Int32,
            unit_elems=len(values),
            cache_modifier=CACHE_SYSTEM,
        )

    def rearm(self, address, index, count):
        buf_copy_store(
            ptr_buf_tensor(address, fx.Int32, unit_elems=count, unit_stride=1),
            index,
            (
                fx.Int32(SENTINEL)
                if count == 1
                else fx.Vector.filled((count,), SENTINEL, fx.Int32)
            ),
            fx.Int32,
            unit_elems=count,
            cache_modifier=CACHE_SYSTEM,
        )

    def poll(self, specs, batch=12):
        """Poll ``(address, word index, word count, kind)`` until every word landed."""
        if len(specs) == 0:
            return []
        if len(specs) > batch:
            return self.poll(specs[:batch], batch) + self.poll(specs[batch:], batch)
        kinds = [kind for _, _, count, kind in specs for _ in range(count)]

        def load_all():
            words = []
            for address, index, count, _ in specs:
                loaded = buf_copy_load(
                    ptr_buf_tensor(address, fx.Int32, unit_elems=count, unit_stride=1),
                    fx.Int32(index),
                    fx.Int32,
                    unit_elems=count,
                    cache_modifier=CACHE_SYSTEM,
                )
                words += (
                    [loaded] if count == 1 else [loaded[item] for item in range(count)]
                )
            return fx.Vector.from_elements(words, fx.Int32)

        def pending(values):
            missing = _pending(values[0], kinds[0])
            for index in range_constexpr(1, len(kinds)):
                missing = missing | _pending(values[index], kinds[index])
            return missing

        values = spin_mailbox(
            load_all,
            pending,
            self.debug,
            self.epoch_addresses,
            self.expected_epoch,
            self.diagnostic,
        )
        out, index = [], 0
        for _, _, count, _ in specs:
            out.append([values[index + item] for item in range(count)])
            index += count
        return out
