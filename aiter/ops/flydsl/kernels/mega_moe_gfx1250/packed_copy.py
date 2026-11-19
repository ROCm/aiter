"""Load destination-ordered, preshuffled source frames with regular TDM."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, arith
from flydsl.expr.rocdl import ballot, readfirstlane
from flydsl.expr.typing import T
from . import tdm_prims as tdm
from .dispatch_pull import pull_lds_row_pitch


def make_packed_copy(*, model_dim, wire_row, max_routes, world, rank, cap, off_packed):
    payload_bytes, scale_bytes = model_dim // 2, model_dim // 32
    frame_bytes = 16 * (payload_bytes + scale_bytes)
    pitch = pull_lds_row_pitch(wire_row)
    assert cap % 16 == 0 and frame_bytes % 128 == 0
    assert frame_bytes <= 16 * pitch

    @flyc.jit
    def copy(
        lds_rows: fx.Pointer,
        output: fx.Int64,
        scales: fx.Pointer,
        local_map: fx.Int64,
        window_base: fx.Int64,
        window_stride: fx.Int64,
        row0: fx.Int32,
        end: fx.Int32,
    ):
        lane, wave = fx.thread_idx.x % 32, fx.thread_idx.x // 32
        i32ptr = fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4)
        vptr = fx.PointerType.get(T.i32, fx.AddressSpace.Global, 16)
        lptr = fx.PointerType.get(T.i32, fx.AddressSpace.Shared, 16)
        mapped = (row0 + lane % 16 < end).select(row0 + lane % 16, row0)
        owner = (
            fx.generic_load(fx.inttoptr(i32ptr, local_map + fx.Int64(mapped) * 8))
            // max_routes
        )
        first_owner = fx.Int32(readfirstlane(T.i32, arith.unwrap(owner)))
        mismatch = fx.Int32(ballot(T.i32, arith.unwrap(owner != first_owner)))
        single = mismatch == 0
        first = single.select(first_owner, 0)
        last = single.select(first_owner + 1, world)
        lds_base = fx.Int64(fx.ptrtoint(lds_rows)) + wave * 16 * pitch
        for source in range(first, last):
            present = fx.Int32(ballot(T.i32, arith.unwrap(owner == source)))
            if present != 0:
                address = window_base + fx.Int64(source) * window_stride + off_packed
                address += (fx.Int64(rank) * cap + row0) // 16 * frame_bytes
                tdm.tdm_load(
                    tdm.tdm_group0(fx.Int32(lds_base), address),
                    tdm.tdm_group1(frame_bytes, 1, 1),
                )
                tdm.tdm_wait(0)
                # The source frame already has [K/32, M16, 16B] order.
                # Each lane writes adjacent vectors without a shuffle.
                for item in range(lane, payload_bytes, 32):
                    if owner == source:
                        value = fx.generic_load(
                            fx.inttoptr(lptr, lds_base + item * 16), count=4
                        )
                        if row0 + item % 16 >= end:
                            value = fx.Vector.from_elements(
                                [fx.Int32(0)] * 4, dtype=fx.Int32
                            )
                        offset = fx.Int64(row0) * payload_bytes + fx.Int64(item) * 16
                        fx.generic_store(fx.inttoptr(vptr, output + offset), value)
                for item in range(lane, 16 * (scale_bytes // 4), 32):
                    row = row0 + item // (scale_bytes // 4)
                    scale_owner = first_owner
                    if ~single:
                        mapped_scale = (row < end).select(row, row0)
                        scale_owner = (
                            fx.generic_load(
                                fx.inttoptr(
                                    i32ptr, local_map + fx.Int64(mapped_scale) * 8
                                )
                            )
                            // max_routes
                        )
                    if scale_owner == source:
                        value = lds_rows[
                            wave * 16 * (pitch // 4) + 16 * (payload_bytes // 4) + item
                        ]
                        value = (row < end).select(value, 0)
                        scales[row0 * (scale_bytes // 4) + item] = value

    return copy
