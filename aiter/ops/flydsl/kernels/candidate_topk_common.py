# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Wave64 helpers shared by experimental exact candidate TopK kernels."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu
from flydsl.expr import rocdl as fly_rocdl
from flydsl.expr.typing import T

from .kernels_common import uint32_to_int32

BLOCK_THREADS = 256
WAVE_SIZE = 64
NUM_WAVES = BLOCK_THREADS // WAVE_SIZE
RADIX_BITS = 8
RADIX_MASK = (1 << RADIX_BITS) - 1
RADIX_SIGN_BIT = 1 << (RADIX_BITS - 1)
NUM_KEY_BYTES = 8  # score ordinal (descending), then raw index (ascending)
NUM_HIST_BINS = 1 << RADIX_BITS
INT32_MIN = -(1 << 31)

_DPP_ROW_SHR_1 = 0x111
_DPP_ROW_SHR_2 = 0x112
_DPP_ROW_SHR_4 = 0x114
_DPP_ROW_SHR_8 = 0x118
_DPP_ROW_MASK = 0xF
_DPP_BANK_MASK = 0xF


def make_streaming_topk_storage(topk, pool_capacity, state_size):
    """Create a statically sized LDS carrier outside postponed annotations."""

    @fx.struct
    class StreamingTopKStorage:
        pool_values: fx.Array[fx.Float32, pool_capacity, 16]
        pool_indices: fx.Array[fx.Int32, pool_capacity, 16]
        selected_values: fx.Array[fx.Float32, topk, 16]
        selected_indices: fx.Array[fx.Int32, topk, 16]
        histogram: fx.Array[fx.Int32, NUM_HIST_BINS, 16]
        scan: fx.Array[fx.Int32, NUM_WAVES + 1, 16]
        state: fx.Array[fx.Int32, state_size, 16]

    return StreamingTopKStorage


def f32_to_ordered_i32(value):
    """Map float32 to signed-order int32; NaNs tie at the very bottom.

    The normal sign-fold also keeps ``+0`` one ordinal above ``-0``.  Raw index
    is the secondary key, so all NaNs are ordered by increasing index.
    """

    bits = value.bitcast(fx.Int32)
    ordered = bits ^ ((bits >> fx.Int32(31)) & fx.Int32(0x7FFFFFFF))
    abs_bits = bits & fx.Int32(0x7FFFFFFF)
    is_nan = arith.cmpi(
        arith.CmpIPredicate.ugt,
        abs_bits,
        fx.Int32(0x7F800000),
    )
    return arith.select(is_nan, fx.Int32(INT32_MIN), ordered)


def key_is_better(score_ord, raw_index, threshold_score, threshold_index):
    """Whether ``(score desc, index asc)`` precedes the threshold key."""

    return (score_ord > threshold_score) | (
        (score_ord == threshold_score) & (raw_index < threshold_index)
    )


def key_matches_prefix(
    score_ord,
    raw_index,
    score_prefix,
    score_mask,
    index_prefix,
    index_mask,
):
    return ((score_ord & score_mask) == score_prefix) & (
        (raw_index & index_mask) == index_prefix
    )


def radix_byte(key, byte_position):
    shift = (3 - byte_position) * RADIX_BITS
    return (key >> fx.Int32(shift)) & fx.Int32(RADIX_MASK)


def prefix_byte_mask(byte_position):
    shift = (3 - byte_position) * RADIX_BITS
    return fx.Int32(uint32_to_int32(RADIX_MASK << shift)), shift


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

    src16 = (lane & fx.Int32(0x30)) - fx.Int32(1)
    remote16 = fly_rocdl.ds_bpermute(T.i32, src16 * fx.Int32(4), value)
    value = (lane >= fx.Int32(16)).select(value + fx.Int32(remote16), value)
    src32 = (lane & fx.Int32(0x30)) - fx.Int32(17)
    remote32 = fly_rocdl.ds_bpermute(T.i32, src32 * fx.Int32(4), value)
    return (lane >= fx.Int32(32)).select(value + fx.Int32(remote32), value)


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

    if wave == 0:
        wave_value = fx.Int32(0)
        if lane < fx.Int32(NUM_WAVES):
            wave_value = scan[lane]
        wave_inclusive = warp_inclusive_prefix_i32(wave_value, lane)
        if lane < fx.Int32(NUM_WAVES):
            scan[lane] = wave_inclusive - wave_value
        if lane == fx.Int32(NUM_WAVES - 1):
            scan[NUM_WAVES] = wave_inclusive
    gpu.barrier()

    result = scan[wave] + exclusive
    total = scan[NUM_WAVES]
    # The next scan may reuse ``scan`` immediately.
    gpu.barrier()
    return result, total
