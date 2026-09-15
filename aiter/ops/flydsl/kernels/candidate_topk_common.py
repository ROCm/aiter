# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Wave64 helpers for experimental exact candidate TopK kernels.

The radix and block-scan primitives are adapted from ROCm/aiter PR #5282.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu
from flydsl.expr import rocdl as fly_rocdl
from flydsl.expr.typing import T

from .kernels_common import uint32_to_int32

BLOCK_THREADS = 256
WAVE_SIZE = 64
NUM_WAVES = BLOCK_THREADS // WAVE_SIZE
KEY_BITS = 32
RADIX_BITS = 11
NUM_RADIX_PASSES = (KEY_BITS + RADIX_BITS - 1) // RADIX_BITS
FINAL_RADIX_BITS = KEY_BITS - (NUM_RADIX_PASSES - 1) * RADIX_BITS
RADIX_SIGN_BIT = 1 << (RADIX_BITS - 1)
NUM_HIST_BINS = 1 << RADIX_BITS
INT32_MIN = -(1 << 31)

_DPP_ROW_SHR_1 = 0x111
_DPP_ROW_SHR_2 = 0x112
_DPP_ROW_SHR_4 = 0x114
_DPP_ROW_SHR_8 = 0x118
_DPP_ROW_BCAST_15 = 0x142
_DPP_ROW_BCAST_31 = 0x143
_DPP_ROW_MASK = 0xF
_DPP_BANK_MASK = 0xF


def make_streaming_topk_storage(
    pool_capacity,
    state_size,
    index_type=fx.Int32,
    num_waves=NUM_WAVES,
):
    """Create a statically sized LDS carrier outside postponed annotations."""

    @fx.struct
    class StreamingTopKStorage:
        pool_values: fx.Array[fx.Float32, pool_capacity, 16]
        pool_indices: fx.Array[index_type, pool_capacity, 16]
        histogram: fx.Array[fx.Int32, NUM_HIST_BINS, 16]
        scan: fx.Array[fx.Int32, num_waves + 1, 16]
        state: fx.Array[fx.Int32, state_size, 16]

    return StreamingTopKStorage


def f32_to_ordered_i32(value):
    """Map float32 to signed-order int32, canonicalizing NaNs to the bottom."""
    bits = value.bitcast(fx.Int32)
    ordered = bits ^ ((bits >> fx.Int32(31)) & fx.Int32(0x7FFFFFFF))
    abs_bits = bits & fx.Int32(0x7FFFFFFF)
    is_nan = arith.cmpi(
        arith.CmpIPredicate.ugt,
        abs_bits,
        fx.Int32(0x7F800000),
    )
    return arith.select(is_nan, fx.Int32(INT32_MIN), ordered)


def radix_pass_bits(pass_index):
    return FINAL_RADIX_BITS if pass_index == NUM_RADIX_PASSES - 1 else RADIX_BITS


def radix_shift(pass_index):
    return max(0, KEY_BITS - (pass_index + 1) * RADIX_BITS)


def radix_mask(pass_index):
    return (1 << radix_pass_bits(pass_index)) - 1


def radix_bucket(key, pass_index):
    shift = radix_shift(pass_index)
    return (key >> fx.Int32(shift)) & fx.Int32(radix_mask(pass_index))


def prefix_radix_mask(pass_index):
    shift = radix_shift(pass_index)
    mask = radix_mask(pass_index)
    return fx.Int32(uint32_to_int32(mask << shift)), shift


def _unwrap(value):
    return value.ir_value() if hasattr(value, "ir_value") else arith.unwrap(value)


def warp_inclusive_prefix_i32(value, lane):
    value_raw = _unwrap(value)
    zero_raw = _unwrap(0)
    for dpp_op, threshold in (
        (_DPP_ROW_SHR_1, 1),
        (_DPP_ROW_SHR_2, 2),
        (_DPP_ROW_SHR_4, 4),
        (_DPP_ROW_SHR_8, 8),
    ):
        remote = fly_rocdl.update_dpp(
            T.i32,
            zero_raw,
            value_raw,
            dpp_op,
            _DPP_ROW_MASK,
            _DPP_BANK_MASK,
            True,
        )
        value = (lane >= fx.Int32(threshold)).select(
            value + fx.Int32(remote),
            value,
        )
        value_raw = _unwrap(value)

    remote16 = fly_rocdl.update_dpp(
        T.i32,
        zero_raw,
        value_raw,
        _DPP_ROW_BCAST_15,
        0xA,
        _DPP_BANK_MASK,
        False,
    )
    value = value + fx.Int32(remote16)
    remote32 = fly_rocdl.update_dpp(
        T.i32,
        zero_raw,
        _unwrap(value),
        _DPP_ROW_BCAST_31,
        0xC,
        _DPP_BANK_MASK,
        False,
    )
    return value + fx.Int32(remote32)


def make_block_exclusive_prefix_i32(num_waves):
    """Build a block exclusive prefix over ``num_waves`` wave-64 groups."""

    @flyc.jit
    def block_exclusive_prefix_i32(tid, value, scan):
        """Return ``(exclusive_prefix, block_total)`` for one i32 per thread."""
        lane = tid % fx.Int32(WAVE_SIZE)
        wave = tid // fx.Int32(WAVE_SIZE)
        inclusive = warp_inclusive_prefix_i32(value, lane)
        exclusive = inclusive - value
        if lane == fx.Int32(WAVE_SIZE - 1):
            scan[wave] = inclusive
        gpu.barrier()

        cross_wave = fx.Int32(0)
        total = fx.Int32(0)
        for wave_index in range(num_waves):
            wave_total = scan[wave_index]
            cross_wave = (wave > fx.Int32(wave_index)).select(
                cross_wave + wave_total,
                cross_wave,
            )
            total = total + wave_total
        result = cross_wave + exclusive
        gpu.barrier()
        return result, total

    return block_exclusive_prefix_i32


block_exclusive_prefix_i32 = make_block_exclusive_prefix_i32(NUM_WAVES)
