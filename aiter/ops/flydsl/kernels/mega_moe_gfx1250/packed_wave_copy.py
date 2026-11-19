# SPDX-License-Identifier: MIT
"""Copy a source-homogeneous M64 with one ordinary TDM per K segment."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.rocdl import readfirstlane
from flydsl.expr.typing import T

from . import tdm_prims as tdm


def packed_wave_lds_bytes(model_dim: int, k_bytes: int) -> int:
    """Scratch size for four independent waves, including final scales."""
    payload_bytes, scale_bytes = model_dim // 2, model_dim // 32
    if k_bytes <= 0 or k_bytes % 16:
        raise ValueError("packed wave K bytes must be a positive multiple of 16")
    rounds = (payload_bytes + k_bytes - 1) // k_bytes
    last = payload_bytes - (rounds - 1) * k_bytes
    row_bytes = max(k_bytes if rounds > 1 else 0, last + scale_bytes) * 16
    return 4 * 4 * row_bytes


def make_packed_wave_copy(
    *, model_dim, max_routes, rank, cap, off_packed, k_bytes, control_gap=False
):
    """Load four M16 frames together, with payload/scales already shuffled."""
    payload_bytes, scale_bytes = model_dim // 2, model_dim // 32
    frame_bytes = 16 * (payload_bytes + scale_bytes)
    rounds = (payload_bytes + k_bytes - 1) // k_bytes
    wave_bytes = packed_wave_lds_bytes(model_dim, k_bytes) // 4
    if cap % 64 or frame_bytes > tdm.TDM_MAX_DIM:
        raise ValueError("packed wave copy requires aligned supported frames")

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
        vector_ptr = fx.PointerType.get(T.i32, fx.AddressSpace.Global, 16)
        shared_ptr = fx.PointerType.get(T.i32, fx.AddressSpace.Shared, 16)
        route = fx.generic_load(fx.inttoptr(i32ptr, local_map + fx.Int64(row0) * 8))
        # TDM descriptors require uniform SGPR bases; readfirstlane is the
        # existing raw boundary because no typed uniform-cast API is available.
        source = fx.Int32(readfirstlane(T.i32, (route // max_routes).ir_value()))
        address = window_base + fx.Int64(source) * window_stride + off_packed
        address += (fx.Int64(rank) * cap + row0) // 16 * frame_bytes
        wave_slot = wave
        if control_gap:
            # Leave one 64KiB slot between waves 1 and 2. In the full planar
            # GEMM this contains its existing control gap; no consumer data
            # layout or synchronization changes.
            wave_slot = (wave >= 2).select(wave + 1, wave)
        lds_base = fx.Int64(fx.ptrtoint(lds_rows)) + wave_slot * wave_bytes
        for part in range_constexpr(rounds):
            width = min(k_bytes, payload_bytes - part * k_bytes)
            final = part == rounds - 1
            row_bytes = 16 * (width + (scale_bytes if final else 0))
            # Four consecutive M16 frames, strided at their full-frame pitch.
            # One descriptor loads this wave's entire M64 K segment.
            tdm.tdm_load(
                tdm.tdm_group0(fx.Int32(lds_base), address + part * k_bytes * 16),
                tdm.tdm_group1(row_bytes, 4, 1, stride=frame_bytes),
            )
            tdm.tdm_wait(0)
            for item in range(lane, 4 * width, 32):
                frame, element = item // width, item % width
                value = fx.generic_load(
                    fx.inttoptr(
                        shared_ptr, lds_base + frame * row_bytes + element * 16
                    ),
                    count=4,
                )
                if row0 + frame * 16 + element % 16 >= end:
                    value = fx.Vector.from_elements([fx.Int32(0)] * 4, dtype=fx.Int32)
                destination = (fx.Int64(row0) + frame * 16) * payload_bytes
                destination += part * k_bytes * 16 + element * 16
                fx.generic_store(fx.inttoptr(vector_ptr, output + destination), value)
            if final:
                for item in range(lane, 64 * (scale_bytes // 4), 32):
                    frame = item // (16 * (scale_bytes // 4))
                    element = item % (16 * (scale_bytes // 4))
                    offset = frame * (row_bytes // 4) + 16 * (width // 4) + element
                    value = lds_rows[wave_slot * (wave_bytes // 4) + offset]
                    row = row0 + item // (scale_bytes // 4)
                    value = (row < end).select(value, 0)
                    scales[row0 * (scale_bytes // 4) + item] = value

    return copy
