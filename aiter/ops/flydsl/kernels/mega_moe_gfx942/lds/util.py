# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Device helpers of the LDS mega MoE kernels and the loader of a8w4's fd2 W."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import rocdl
from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace
from flydsl.expr import range_constexpr
from flydsl.expr.meta import dsl_loc_tracing
from flydsl.expr.typing import T, is_target_address_space
from flydsl.expr.typing import Vector as Vec

from ..atomic import _atomic_load_at, _inv_l1
from ..cvt import _channel_factor, _fp4x8_to_fp8x8_fast, _fp8_lut_predelta, _perm_u32
from ..hw import _sleep


@flyc.jit
def spin_wait_ge(addr, target, scope):
    """Poll the i32 at `addr` until it is >= target, then invalidate the L1 (`buffer_inv sc0`).

    On gfx942 a scoped acquire also walks the L2 for non-local (MTYPE NC) lines; local memory lines are kept coherent
    by probes, so data in local memory that was written through (or written back) before the flag needs only the L1
    invalidated.
    """
    seen = _atomic_load_at(addr, scope)
    while seen < target:
        _sleep(2)
        seen = _atomic_load_at(addr, scope)
    _inv_l1()


def buffer_rsrc_at(addr, num_bytes):
    """Buffer resource for `num_bytes` at an i64 address (e.g. a peer's buffer from a pointer table)."""
    ptr = fx.inttoptr(
        fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 16), fx.Int64(addr)
    )
    return fx.rocdl.get_buffer_rsrc(
        fx.rocdl.make_buffer_ptr(ptr, num_records_bytes=num_bytes)
    )


def _offset_i32(value):
    """An i32 SSA value or a 32-bit compile-time int as is; truncation and unit conversion come before."""
    if isinstance(value, (fx.Int32, fx.Uint32)):
        return value.ir_value()
    if (
        isinstance(value, ir.Value)
        and isinstance(value.type, ir.IntegerType)
        and value.type.width == 32
    ):
        return value
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and -(1 << 31) <= value < (1 << 32)
    ):
        return fx.Uint32(value & 0xFFFFFFFF).ir_value()
    raise TypeError("offset: expected a precomputed 32-bit SSA value or 32-bit int")


class BufferTensor(fx.Tensor):
    """Buffer tensor whose load is one raw buffer load; keeps no extra dynamic Python fields."""

    @dsl_loc_tracing
    def __getitem__(self, coord):
        result = super().__getitem__(coord)
        return type(self)(result) if isinstance(result, fx.Tensor) else result

    def _packet_bits(self):
        if not is_target_address_space(
            self.address_space, TargetAddressSpace.BufferDesc
        ):
            raise TypeError("explicit buffer loads need a BufferDesc tensor")
        if (
            not self.layout.is_static
            or fx.rank(self.layout) != 1
            or self.stride.to_py_value() not in (1, (1,))
        ):
            raise ValueError(
                "explicit buffer loads take a static contiguous 1-D packet"
            )
        bits = fx.size(self).to_py_value() * self.dtype.width
        if bits not in (32, 64, 128):
            raise ValueError(
                "packets are 32/64/128 bits; issue 16 B + 8 B as two loads"
            )
        return bits

    @dsl_loc_tracing
    def load(self, *, voffset_bytes=None, soffset_bytes=0, aux=0):
        if voffset_bytes is None:
            if not isinstance(soffset_bytes, int) or soffset_bytes != 0 or aux != 0:
                raise ValueError("an explicit soffset / aux needs voffset_bytes too")
            return super().load()
        bits = self._packet_bits()
        if not isinstance(aux, int):
            raise TypeError("aux: expected a compile-time int")
        resource = fx.rocdl.get_buffer_rsrc(fx.get_iter(self))
        # Raw buffer load, for the soffset (see buffer_load).
        result = rocdl.RawPtrBufferLoadOp(
            ir.VectorType.get([bits // 32], fx.Uint32.ir_type),
            resource,
            _offset_i32(voffset_bytes),
            _offset_i32(soffset_bytes),
            aux=ir.IntegerAttr.get(fx.Int32.ir_type, aux),
        ).result
        return Vec(result).bitcast(self.dtype)


def pack_bf16x2_rne(values):
    """Four FP32 -> two BF16x2 dwords, rounded to nearest even (pack_bf16x2 truncates normal values)."""
    bits = [fx.Float32(values[index]).bitcast(fx.Int32) for index in range(4)]
    rounded = [bit + (0x7FFF + (bit >> 16 & 1)) for bit in bits]
    return [
        rounded[pair * 2] >> 16 & 0xFFFF | rounded[pair * 2 + 1] & -65536
        for pair in range(2)
    ]


def _rsrc_of(ptr, num_records_bytes):
    tensor = fx.rocdl.make_buffer_tensor(
        fx.make_view(ptr, fx.make_layout(1, 1)),
        max_size=False,
        num_records_bytes=num_records_bytes,
    )
    return fx.rocdl.get_buffer_rsrc(fx.get_iter(tensor))


def buffer_load(ty, rsrc, voffset, soffset=0):
    """`ty` at byte offset voffset (per lane) + soffset (wave uniform) of the buffer resource `rsrc`; a lane whose
    offset is past the resource's range reads 0."""
    # Raw buffer load: the offsets are found at run time (sorted ids, routes, masked lanes at OOB_OFFSET), which has
    # no layout form for fx.copy, and FlyDSL 0.5.2's buffer copy atoms take no soffset (wave-uniform offsets in
    # soffset keep them out of the VGPRs).
    return fx.rocdl.raw_ptr_buffer_load(
        ty, rsrc, fx.Int32(voffset).ir_value(), fx.Int32(soffset).ir_value(), None
    )


def buffer_store(value, rsrc, voffset, soffset=0, aux=0):
    """Store the SSA `value` at byte offset voffset (per lane) + soffset (wave uniform) of `rsrc`; a lane whose
    offset is past the range stores nothing. aux: cache bits (sc0 1, nt 2, sc1 16)."""
    # Raw buffer store: as buffer_load, and FlyDSL 0.5.2's buffer copy atoms set no cache bits but nt, while these
    # stores write through (sc1: another XCD sees the data after a vmcnt wait) or to a peer (sc0 sc1 nt).
    rocdl.raw_ptr_buffer_store(
        value,
        rsrc,
        fx.Int32(voffset).ir_value(),
        fx.Int32(soffset).ir_value(),
        aux=ir.IntegerAttr.get(T.i32, aux),
    )


def _load_i32(rsrc, voffset):
    return fx.Int32(buffer_load(T.i32, rsrc, voffset))


def readlane_f32(value, lane):
    """The f32 `value` of the wave-uniform, run-time `lane`, in an SGPR."""
    # Raw readlane, for an f32: as hw._readlane (fx.gpu.shuffle_idx gives v_readlane only for a compile-time lane).
    return fx.Float32(
        fx.rocdl.readlane(T.f32, value.ir_value(), fx.Int32(lane).ir_value())
    )


def set_priority(level):
    """s_setprio `level` for the waves of this workgroup."""
    # Raw: FlyDSL 0.5.2 has no stable s_setprio (not in fx.rocdl.__all__).
    fx.rocdl.s_setprio(level)


def _fp8_fragment(word):
    """MFMA operand fragment: the 8 FP8 values of an i64."""
    fragment = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.Float8E4M3FNUZ)
    fragment.store(Vec.from_elements([word], fx.Int64).bitcast(fx.Float8E4M3FNUZ))
    return fragment


def mfma_fp8(target, a, b, accumulate):
    """target (4 x f32 registers) = a x b + (target if accumulate else 0) with one 16x16x32 FP8 MFMA; a and b are
    i64 words of 8 FP8 values."""
    if not accumulate:
        target.store(Vec.filled(4, 0.0, fx.Float32))
    atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.Float8E4M3FNUZ))
    fx.gemm(atom, target, _fp8_fragment(a), _fp8_fragment(b), target)


# a8w4's fd2 W (weights.prepare_weights): a 16-row group's 128-K chunk is 1 KiB, lane q * 16 + r
# holding row r's 32-K group q, 8 nibbles per dword in K order. The scales of 32-row pair p and
# 256-K block b are dword p * (K / 4) + b * 64 + lane, byte (k_block % 2) * 2 + group % 2.
FD2_CHUNK_BYTES = 1024


def fd2_lane_offsets(wave, lane):
    """Byte voffsets (weights, scales) of the fd2 data replacing the 16 B fp8 unit a loader thread
    loads: row lane % 16 of 32-K group wave % 2 * 2 + lane // 32, half lane // 16 % 2.
    """
    unit_lane = (wave % 2 * 2 + lane // 32) * 16 + lane % 16
    return unit_lane * 16 + lane // 16 % 2 * 8, unit_lane * 4


def load_fd2(w_packet, scale_rsrc, voffsets, soffsets, k_block):
    """[w0, w1, scale dword, scale dword] of a packet in K block `k_block`; soffsets: the packet's
    wave-uniform (weights, scales) byte offsets. The 4th dword only keeps the stage 4 dwords.
    """
    w = Vec(
        w_packet.load(
            voffset_bytes=voffsets[0],
            soffset_bytes=k_block * FD2_CHUNK_BYTES + soffsets[0],
        )
    )
    s = fx.Int32(
        buffer_load(T.i32, scale_rsrc, voffsets[1], (k_block >> 1) * 256 + soffsets[1])
    )
    return Vec.from_elements([fx.Int32(w[0]), fx.Int32(w[1]), s, s], fx.Int32)


def decode_fd2(value, selector):
    """load_fd2's [w0, w1, scale dword, -] -> 16 B of fp8 with the group scale folded in;
    `selector` replicates the index of the scale byte."""
    v = Vec(value)
    p = fx.Uint32(fx.Int32(v[2]))
    lut = _fp8_lut_predelta(_perm_u32(p, p, fx.Uint32(selector)))
    words = []
    for i in range_constexpr(2):
        pair = Vec(_fp4x8_to_fp8x8_fast(fx.Int32(v[i]), lut)).bitcast(fx.Int32)
        words += [fx.Int32(pair[0]), fx.Int32(pair[1])]
    return Vec.from_elements(words, fx.Int32)


def fd2_channel_factors(scale_ptr, *, rows, K, first_columns):
    """f32 2**(exp - 132) of columns c .. c + 3 for each c in first_columns; the int32 exponents
    of all `rows` weight rows follow their block scales."""
    exp_rsrc = _rsrc_of(scale_ptr + fx.Int64(rows * K // 128), rows * 4)
    return [
        [
            _channel_factor(_load_i32(exp_rsrc, (column + element) * 4))
            for element in range_constexpr(4)
        ]
        for column in first_columns
    ]
