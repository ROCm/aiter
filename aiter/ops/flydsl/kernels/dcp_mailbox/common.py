# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared device helpers for DCP mailbox kernels."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr

from aiter.ops.flydsl.kernels.communication_ops_utils import (
    CACHE_DEVICE,
    CACHE_SYSTEM,
    EPOCH_MASK,
    atomic_add_agent,
)
from aiter.ops.flydsl.kernels.tensor_shim import (
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
