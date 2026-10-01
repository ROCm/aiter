# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Marlowe AI
"""Consumer-local M8 grouping, derived from Marlowe's MIT route ballot kernel.

Each workgroup is named by one of the 64 routed slots or the shared slot 64.
Only the first occurrence of an expert executes a native BM16 matrix tile.
Native top-8 has distinct IDs within a token, so an expert occupies at most
eight rows. G1/G2 independently recover the same stable row order; no published
route map, counter, polling, or inter-workgroup synchronization is needed.
"""

import flydsl.expr as fx
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from .mxfp4_gemm_common import _global_i32_at


def expert_group(ids, slot, lane):
    """Return uniform expert, 64-bit matching-slot ballot and leader predicate."""
    candidate = _global_i32_at(ids, lane + lane // fx.Int32(8))
    source = fx.min(slot, fx.Int32(63))
    target = _global_i32_at(ids, source + source // fx.Int32(8))
    expert = (slot == fx.Int32(64)).select(fx.Int32(256), target)
    expert = fx.Int32(rocdl.readfirstlane(T.i32, as_ir_value(expert)))
    matches = fx.Int64(rocdl.ballot(T.i64, as_ir_value(candidate == expert)))
    # Bit 63 supplies a defined cttz operand even for the shared/no-match case.
    first = fx.Int32(fx.cttz(matches | fx.Int64(-9223372036854775808)))
    active = (slot == fx.Int32(64)) | (first == slot)
    return expert, matches, active


def route_row(matches, slot, row):
    """Return native packed token/choice and its index in the supplied Mx9 map."""
    count = fx.Int32(fx.ctpop(matches))
    remaining = matches
    source = fx.Int32(0)
    for rank in range_constexpr(8):
        bit = fx.Int32(fx.cttz(remaining | fx.Int64(-9223372036854775808)))
        source = (row == fx.Int32(rank)).select(bit, source)
        remaining = remaining & (remaining - fx.Int64(1))
    shared = slot == fx.Int32(64)
    token = shared.select(row, source // fx.Int32(8))
    choice = shared.select(fx.Int32(8), source % fx.Int32(8))
    valid = shared.select(row < fx.Int32(8), row < count)
    packed = valid.select(token | (choice << fx.Int32(24)), fx.Int32((9 << 24) | 8))
    index = valid.select(token * fx.Int32(9) + choice, fx.Int32(0))
    return packed, index, valid
