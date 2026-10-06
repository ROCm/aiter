# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL single-tensor intranode push all-to-all kernel."""

from __future__ import annotations

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.backends import current_target
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Stream

from aiter.ops.mha_v4 import AttentionPack
from aiter.utility.mx_types import MxDtypeInt, MxScaleRoundModeInt

from .communication_ops_utils import (
    spin_until_ge_i64,
    wait_lds_wave,
    wave_uniform_i64,
)
from .kernels_common import ceildiv
from .quant_utils import (
    emit_f32_to_e2m1,
    emit_f32_to_e2m3,
    emit_f32_to_e2m3_native,
    emit_f32_to_fp8,
    emit_fp8_to_f32,
    emit_mx_e8m0_scale,
)
from .tensor_shim import buf_copy_load, buf_copy_store, ptr_buf_tensor

_JIT_CACHE_TAG = "attention_a2a_intranode"
_TRANSPORT_CHUNK_BYTES = 16
_PUSH_PIPELINE_DEPTH = 16
# Rows of one e4m3_pc partial-maximum tile. The tile count sets both the
# private partial workspace (one channel row per tile and destination) and the
# number of independent work items, so smaller tiles trade workspace for
# parallelism.
V_PC_TOKEN_TILE = 32
# Outer extent for an idx2crd mode the index can never wrap. FlyDSL lowers every
# mode as mod(div(i, stride), shape), so a bounded outer extent leaves a redundant
# remainder that defeats loop scalarization and load pipelining in hot loops.
# Use only where the index is guaranteed below outer_extent * outer_stride.
UNBOUNDED_OUTER_EXTENT = 2**31 - 1


def v_pc_partial_elems(seq_len, heads, head_dim):
    """FP32 elements of the private e4m3_pc partial-maximum workspace."""
    return ceildiv(seq_len, V_PC_TOKEN_TILE) * heads * head_dim


def needs_split_v_exchange(v_pack, seq_len):
    """FP6-P V splits a 64-token frame across two senders unless seq_len % 64 == 0."""
    return v_pack == AttentionPack.V_FOR_FP6_P and seq_len % 64 != 0


def make_attention_a2a_reuse_jit(*, rank, npes):
    """Gate parity reuse on stream-ordered consumer completion at every rank."""

    @flyc.kernel(known_block_size=[64, 1, 1])
    def attention_a2a_reuse_barrier(
        addr_ready_mem: fx.Int64,
        addr_p2p_ready_mem: fx.Int64,
        addr_ready_flag: fx.Int64,
        publish: fx.Constexpr[bool],
    ):
        tid = fx.thread_idx.x
        ready_flag = ptr_buf_tensor(addr_ready_flag, fx.Int64, num_records_bytes=8)
        generation = fx.Int64(buf_copy_load(ready_flag, 0, fx.Int64))
        if tid < npes:
            if const_expr(publish):
                p2p_ready = ptr_buf_tensor(
                    addr_p2p_ready_mem, fx.Int64, num_records_bytes=npes * 8
                )
                remote = fx.inttoptr(
                    fx.PointerType.get(
                        fx.Int64.ir_type,
                        address_space=fx.AddressSpace.Global,
                        alignment=8,
                    ),
                    buf_copy_load(p2p_ready, tid, fx.Int64),
                )
                remote_slot = fx.add_offset(remote, rank)
                # Reaching this launch drains earlier consumers on this stream.
                fx.generic_store(
                    remote_slot,
                    generation,
                    memory_order=fx.AtomicOrdering.Release,
                    syncscope=fx.rocdl.SyncScope.OneAs,
                )
            else:
                ready = fx.inttoptr(
                    fx.PointerType.get(
                        fx.Int64.ir_type,
                        address_space=fx.AddressSpace.Global,
                        alignment=8,
                    ),
                    addr_ready_mem,
                )
                spin_until_ge_i64(fx.ptrtoint(fx.add_offset(ready, tid)), generation)
                fx.memory_fence(
                    ordering=fx.AtomicOrdering.Acquire,
                    syncscope=fx.rocdl.SyncScope.OneAs,
                )
        fx.barrier()
        # Nested rather than combined: const_expr must gate the runtime test.
        if const_expr(not publish):  # noqa: SIM102
            if tid == 0:
                fx.atomic_add(
                    fx.inttoptr(
                        fx.PointerType.get(
                            fx.Int64.ir_type,
                            address_space=fx.AddressSpace.Global,
                            alignment=8,
                        ),
                        addr_ready_flag,
                    ),
                    fx.Int64(1),
                    ordering=fx.AtomicOrdering.Monotonic,
                    syncscope=fx.rocdl.SyncScope.OneAs,
                )

    key = (rank, npes, _JIT_CACHE_TAG)

    @flyc.jit
    def launch(
        addr_ready_mem: fx.Int64,
        addr_p2p_ready_mem: fx.Int64,
        addr_ready_flag: fx.Int64,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        _ = key
        for publish in range_constexpr(2):
            attention_a2a_reuse_barrier(
                addr_ready_mem, addr_p2p_ready_mem, addr_ready_flag, publish == 0
            ).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    return launch


def _fp6_output_views(address, num_records_bytes=None):
    return tuple(
        ptr_buf_tensor(
            address,
            elem,
            unit_elems=width,
            unit_stride=1,
            num_records_bytes=num_records_bytes,
        )
        for elem, width in ((fx.Int32, 1), (fx.Int32, 2), (fx.Int32, 4), (fx.Int8, 1))
    )


def _int8_e8m0_scale(amax):
    # Match the symmetric INT8 codec's floor and exponent cap, not MXFP8's.
    need = (amax / fx.Float32(127.0)).maximumf(fx.Float32(1.0e-30))
    exponent = (fmath.ceil(fmath.log2(need)) + fx.Float32(127.0)).to(fx.Int32)
    exponent = (exponent < 0).select(fx.Int32(0), exponent)
    return (exponent > 254).select(fx.Int32(254), exponent)


def _pack_int8_pair(first, second):
    packed = []
    for value in (first, second):
        rounded = fmath.roundeven(value)
        clipped = rounded.maximumf(fx.Float32(-127.0)).minimumf(fx.Float32(127.0))
        packed.append(clipped.to(fx.Int32) & 255)
    return (packed[0] | (packed[1] << 8)).to(fx.Int16)


def _unpack_int8_pair(word, high):
    # Arithmetic shifts sign-extend each byte without an i8 vector memory op.
    shift = 16 if high else 0
    first = ((word << (24 - shift)) >> 24).to(fx.Float32)
    second = ((word << (16 - shift)) >> 24).to(fx.Float32)
    return fx.Vector.from_elements([first, second], fx.Float32)


def _row_field_offset(row, field, fields):
    row_layout = fx.make_layout((UNBOUNDED_OUTER_EXTENT, fields), (fields, 1))
    field_layout = fx.slice(row_layout, (row, None))
    row_base = fx.Int32(fx.get_scalar(fx.crd2idx((row, fx.Int32(0)), row_layout)))
    return row_base + fx.Int32(fx.get_scalar(fx.crd2idx(field, field_layout)))


def _field_view(buffer, base, fields):
    return fx.slice(
        fx.make_view(
            fx.get_iter(buffer),
            fx.make_layout((UNBOUNDED_OUTER_EXTENT, fields), (1, 1)),
        ),
        (base, None),
    )


def _transport_bytes(numel, codec):
    return numel * {"mxfp4": 4, "mxfp6": 6}.get(codec, 8) // 8


def _transport_scale(amax, codec, fp8_fnuz):
    if codec == "int8":
        return _int8_e8m0_scale(amax)
    if codec == "mxfp6":
        bits = (amax * fx.Float32(1.0 / 7.5)).bitcast(fx.Int32)
        exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(fx.Int32)
        return (exponent > 255).select(fx.Int32(255), exponent)
    return fx.Int32(
        emit_mx_e8m0_scale(
            amax.ir_value(),
            mode=MxScaleRoundModeInt.RoundUp,
            dtype=(
                MxDtypeInt.FP4_E2M1
                if codec == "mxfp4"
                else (MxDtypeInt.FP8_E4M3_FNUZ if fp8_fnuz else MxDtypeInt.FP8_E4M3)
            ),
        )
    )


def _pack_transport_pair(first, second, codec):
    if codec == "int8":
        return _pack_int8_pair(first, second)
    if codec == "mxfp6":
        low = fx.Uint32(emit_f32_to_e2m3(first.ir_value()))
        high = fx.Uint32(emit_f32_to_e2m3(second.ir_value()))
        return low | (high << 6)
    if codec == "mxfp4":
        low = fx.Int32(emit_f32_to_e2m1(first.ir_value()))
        high = fx.Int32(emit_f32_to_e2m1(second.ir_value()))
        return low | (high << 4)
    return emit_f32_to_fp8(first, second).to(fx.Int16)


def _v4_fp8_scale(amax, fp8_fnuz):
    # Match rotate_activation_mxfp8_quant, not native transport's RoundUp.
    exponent = ((amax.bitcast(fx.Int32) + 0x00200000) & 0x7F800000) >> 23
    scale = (exponent > (7 if fp8_fnuz else 8)).select(
        exponent - (7 if fp8_fnuz else 8), fx.Int32(0)
    )
    return (exponent == 255).select(fx.Int32(254), scale)


def _v4_fp6_scale(amax):
    exponent = ((amax.bitcast(fx.Int32) >> 23) & 255) - 2
    return (amax == 0.0).select(fx.Int32(127), exponent)


def _pack_v4_fp6(values, reciprocal):
    codes = [
        fx.Uint32(emit_f32_to_e2m3((values[i] * reciprocal).ir_value()))
        for i in range(8)
    ]
    peer = [code.shuffle_xor(2, 64) for code in codes]
    # Lanes 0/1 interleave their eight fields with lanes 2/3, respectively.
    triples = [
        codes[i] | (peer[i] << 6) | (codes[i + 1] << 12) | (peer[i + 1] << 18)
        for i in range(0, 8, 2)
    ]
    return fx.Vector.from_elements(
        [
            triples[0] | (triples[1] << 24),
            (triples[1] >> 8) | (triples[2] << 16),
            (triples[2] >> 16) | (triples[3] << 8),
        ],
        fx.Uint32,
    )


@flyc.jit
def _store_v4_fp6_q(words, dst_words, offset, lane):
    # Slice the shared row before selecting its unrolled fields.
    row_words = fx.slice(
        fx.make_view(
            fx.get_iter(dst_words), fx.make_layout((UNBOUNDED_OUTER_EXTENT, 3), (1, 1))
        ),
        (offset, None),
    )
    if lane % 4 < 2:
        for i in range_constexpr(3):
            buf_copy_store(row_words, i, words[i], fx.Int32)


def fp6p_source_token(token):
    frame_token = fx.Int32(
        fx.get_scalar(
            fx.crd2idx(token & 63, fx.make_layout((4, 2, 4, 2), (1, 32, 8, 4)))
        )
    )
    return (token & ~63) + frame_token


def _k_fp6_scale_slot(token):
    return fx.Int32(
        fx.get_scalar(fx.crd2idx(token & 127, fx.make_layout((16, 2, 4), (16, 256, 4))))
    )


def _k_fp6_predecessor(global_token, channel):
    layout = fx.make_layout((128, 4), (4, 1))
    flat = fx.Int32(fx.get_scalar(fx.crd2idx((global_token, channel), layout)))
    # Guard zero before signed decomposition; the token mode repeats every tile.
    previous = (flat > 0).select(flat - 1, fx.Int32(0))
    coord = fx.idx2crd(previous, layout)
    return (
        fx.Int32(fx.get_scalar(fx.get(coord, 0))),
        fx.Int32(fx.get_scalar(fx.get(coord, 1))),
    )


def _k_sequence_coord(seq):
    return fx.idx2crd(
        fx.Int32(seq),
        fx.make_layout((UNBOUNDED_OUTER_EXTENT, 128), (128, 1)),
    )


def _k_fp6_band_view(dst_words, tile_word, token, channel, words_per_token):
    tile = fx.slice(
        fx.make_view(
            fx.get_iter(dst_words),
            fx.make_layout((UNBOUNDED_OUTER_EXTENT, 4352), (1, 1)),
        ),
        (tile_word, None),
    )
    band = fx.make_view(
        fx.get_iter(tile),
        fx.make_layout(
            (4, 4, 32, words_per_token),
            (128 * words_per_token, 32 * words_per_token, words_per_token, 1),
        ),
    )
    if words_per_token == 2:
        band = fx.make_view(fx.add_offset(fx.get_iter(tile), 2048), fx.get_layout(band))
    # Explicit token // 32, % 32: layout forms drain vmcnt(0) before the first input load.
    return fx.slice(band, (token // 32, channel, token % 32, None))


def _k_fp6_scale_image(dst_bytes, tile_byte, image):
    tail = _field_view(dst_bytes, tile_byte, 17408)
    images = fx.make_view(
        fx.add_offset(fx.get_iter(tail), 16384),
        fx.make_layout((2, 512), (512, 1)),
    )
    return fx.slice(images, (image, None))


def _k_fp6_scale_field(slot, channel):
    return fx.Int32(
        fx.get_scalar(fx.crd2idx((slot, channel), fx.make_layout((512, 4), (1, 1))))
    )


@flyc.jit
def _store_v4_fp6_k(words, scale, dst_words, dst_bytes, tile_word, seq, lane):
    # Band selection and predecessor replication are codec ownership, not affine rows.
    # Explicit lane coordinates: layout forms drain vmcnt(0) before the first input load.
    group = lane % 16 // 4
    seq_coord = _k_sequence_coord(seq)
    token = fx.Int32(fx.get_scalar(fx.get(seq_coord, 1)))
    # Two lanes own the packed six-dword row; unrolled fields stay outside the layout.
    if lane % 4 < 2:
        band_a = _k_fp6_band_view(dst_words, tile_word, token, group, 4)
        band_b = _k_fp6_band_view(dst_words, tile_word, token, group, 2)
        for i in range_constexpr(3):
            word = lane % 2 * 3 + i
            if word < 4:
                buf_copy_store(band_a, word, words[i], fx.Int32)
            else:
                buf_copy_store(band_b, word - 4, words[i], fx.Int32)
    if lane % 4 == 0:
        tile_byte = _row_field_offset(tile_word, fx.Int32(0), 4)
        image_a = _k_fp6_scale_image(dst_bytes, tile_byte, 0)
        image_b = _k_fp6_scale_image(dst_bytes, tile_byte, 1)
        scale_slot = _k_fp6_scale_slot(token)
        buf_copy_store(
            image_a,
            _k_fp6_scale_field(scale_slot, group),
            scale,
            fx.Int8,
        )
        previous_token, previous_channel = _k_fp6_predecessor(seq, group)
        if group > 0:
            buf_copy_store(
                image_b,
                _k_fp6_scale_field(scale_slot, previous_channel),
                scale,
                fx.Int8,
            )
        elif seq > 0:
            previous_slot = _k_fp6_scale_slot(previous_token)
            previous_tile = _row_field_offset(tile_word, fx.Int32(0), 4) - (
                token == 0
            ).select(fx.Int32(17408), fx.Int32(0))
            buf_copy_store(
                _k_fp6_scale_image(dst_bytes, previous_tile, 1),
                _k_fp6_scale_field(previous_slot, previous_channel),
                scale,
                fx.Int8,
            )


def _pack_v4_v_pair(first, second):
    # The V4 Triton packer nudges nonzero magnitudes down one FP32 ULP before
    # conversion, so exact FP4 midpoints round toward zero rather than to even.
    codes = []
    for value in (first, second):
        magnitude = fmath.absf(value)
        code = fx.Int32(0)
        for midpoint in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
            code = code + (magnitude > midpoint).to(fx.Int32)
        codes.append(code | ((value.bitcast(fx.Int32) >> 28) & 8))
    return codes[0] | (codes[1] << 4)


def _pack_transport_words(pairs, codec):
    if codec == "mxfp6":
        lo = pairs[0] | (pairs[1] << 12) | (pairs[2] << 24)
        hi = (pairs[2] >> 8) | (pairs[3] << 4)
        # Two adjacent lanes own one aligned 96-bit span. Shuffle before predication.
        peer_lo = lo.shuffle_xor(1, 64)
        peer_hi = hi.shuffle_xor(1, 64)
        return fx.Vector.from_elements(
            [lo, hi | (peer_lo << 16), (peer_lo >> 16) | (peer_hi << 16)],
            fx.Uint32,
        )
    if codec == "mxfp4":
        word = pairs[0]
        for i in range(1, 4):
            word = word | (pairs[i] << (8 * i))
        return word
    return fx.Vector.from_elements(pairs, fx.Int16).bitcast(fx.Int32)


@flyc.jit
def _store_fp6(words, dst_words, chunk):
    # Adjacent chunks share a 96-bit span; only its even owner writes all three words.
    record_coord = fx.idx2crd(
        fx.Int32(chunk), fx.make_layout((UNBOUNDED_OUTER_EXTENT, 2), (2, 1))
    )
    record = fx.Int32(fx.get_scalar(fx.get(record_coord, 0)))
    record_words = fx.slice(
        fx.make_view(
            fx.get_iter(dst_words), fx.make_layout((UNBOUNDED_OUTER_EXTENT, 3), (3, 1))
        ),
        (record, None),
    )
    if chunk % 2 == 0:
        for i in range_constexpr(3):
            buf_copy_store(record_words, i, words[i], fx.Int32)


def _load_fp6(payload, chunk):
    # Odd chunks start one dword into the shared 96-bit span and need realignment.
    record_coord = fx.idx2crd(
        fx.Int32(chunk), fx.make_layout((UNBOUNDED_OUTER_EXTENT, 2), (2, 1))
    )
    word = fx.Int32(
        fx.get_scalar(
            fx.crd2idx(
                record_coord, fx.make_layout((UNBOUNDED_OUTER_EXTENT, 2), (3, 1))
            )
        )
    )
    words = buf_copy_load(payload, word, fx.Int32, 2)
    first, second = fx.Uint32(words[0]), fx.Uint32(words[1])
    odd = chunk % 2 != 0
    lo = odd.select((first >> 16) | (second << 16), first)
    hi = odd.select(second >> 16, second & 65535)
    return fx.Vector.from_elements([lo, hi], fx.Uint32)


def _unpack_fp6_pair(words, pair_index):
    values = []
    for i in range(2):
        shift = (pair_index * 2 + i) * 6
        code = words[0] >> shift if shift < 32 else words[1] >> (shift - 32)
        if shift < 32 and shift + 6 > 32:
            code = code | (words[1] << (32 - shift))
        code = code & 63
        magnitude = code & 31
        exponent = magnitude >> 3
        normal = (((exponent + 126) << 23) | ((magnitude & 7) << 20)).bitcast(
            fx.Float32
        )
        value = (exponent == 0).select(magnitude.to(fx.Float32) * 0.125, normal)
        values.append(((code & 32) != 0).select(-value, value))
    return fx.Vector.from_elements(values, fx.Float32)


def _unpack_transport_pair(word, high, codec):
    if codec == "int8":
        return _unpack_int8_pair(word, high)
    if codec == "mxfp4":
        values = []
        for i in range(2):
            nibble = (word >> (int(high) * 8 + i * 4)) & 15
            magnitude = nibble & 7
            # E2M1's positive levels are 0, .5, 1, 1.5, 2, 3, 4, 6.
            exponent = magnitude >> 1
            normal = ((exponent + 126) << 23) | ((magnitude & 1) << 22)
            bits = (magnitude < 2).select(magnitude * 0x3F000000, normal)
            bits = bits | ((nibble & 8) << 28)
            values.append(bits.bitcast(fx.Float32))
        return fx.Vector.from_elements(values, fx.Float32)
    return emit_fp8_to_f32(word, high)


def _hadamard_head(values, lane, head_dim, registers=8):
    # Adjacent channels live in registers; the rest of the head is in
    # aligned lane groups. XOR never crosses a head boundary.
    result = [values[i] for i in range(registers)]
    for stage in range(registers.bit_length() - 1):
        shift = 1 << stage
        result = [
            (
                result[i ^ shift] - result[i]
                if i & shift
                else result[i] + result[i ^ shift]
            )
            for i in range(registers)
        ]
    for stage in range((head_dim // registers).bit_length() - 1):
        shift = 1 << stage
        result = [
            ((lane & shift) == 0).select(
                value + value.shuffle_xor(shift, 64),
                value.shuffle_xor(shift, 64) - value,
            )
            for value in result
        ]
    return fx.Vector.from_elements(result, fx.Float32) * (head_dim**-0.5)


def make_attention_a2a_kernel(
    *,
    rank,
    npes,
    heads,
    seq_len,
    head_dim,
    block_num,
    warp_num_per_block,
    quant=False,
    codec="e4m3",
    hadamard=False,
    v4_output="",
    v_pack=AttentionPack.DEFAULT,
    q_multiplier=1.0,
    v4_amax=False,
    fp8_fnuz=False,
    partial_only=False,
    pc_reduce=False,
):
    target_arch = current_target().arch
    native_fp6 = (
        target_arch == "gfx950"
        and os.environ.get("AITER_A2A_FP6_FORCE_SOFTWARE", "0") != "1"
    )
    heads_local = heads // npes
    seq_full = seq_len * npes
    k_tiles = (seq_full + 127) // 128
    split_v_exchange = needs_split_v_exchange(v_pack, seq_len)
    split_frame = ((rank // 2 * 2 + 1) * seq_len) // 64
    # Tiles wholly inside this sender's sequence range are emitted by one wave so
    # their scale tail images can be staged in LDS.
    k_first_tile = (rank * seq_len + 127) // 128
    k_last_tile = ((rank + 1) * seq_len) // 128
    k_own_tiles = max(0, k_last_tile - k_first_tile)

    def input_offset(seq, head, channel, elems_per_unit=1):
        row = fx.Int32(
            fx.get_scalar(
                fx.crd2idx((seq, head), fx.make_layout((seq_len, heads), (heads, 1)))
            )
        )
        return _row_field_offset(row, channel, head_dim // elems_per_unit)

    def row_offset(seq, head, channel, channels, head_major=False):
        layout = fx.make_layout(
            (seq_full, heads_local),
            (1, seq_full) if head_major else (heads_local, 1),
        )
        row = fx.Int32(fx.get_scalar(fx.crd2idx((seq, head), layout)))
        return _row_field_offset(row, channel, channels)

    def wave_chunk(group, lane):
        return fx.Int32(
            fx.get_scalar(
                fx.crd2idx(
                    (group, lane), fx.make_layout((UNBOUNDED_OUTER_EXTENT, 64), (64, 1))
                )
            )
        )

    def global_head(peer, head):
        return fx.Int32(
            fx.get_scalar(
                fx.crd2idx(
                    (peer, head), fx.make_layout((npes, heads_local), (heads_local, 1))
                )
            )
        )

    def global_token(seq):
        return fx.Int32(
            fx.get_scalar(
                fx.crd2idx(
                    (fx.Int32(rank), seq), fx.make_layout((npes, seq_len), (seq_len, 1))
                )
            )
        )

    def partial_offset(source_rank, head, quarter, channel):
        row = fx.Int32(
            fx.get_scalar(
                fx.crd2idx(
                    (source_rank, head),
                    fx.make_layout((npes, heads_local), (heads_local, 1)),
                )
            )
        )
        return _row_field_offset(
            row, _row_field_offset(fx.Int32(quarter), channel, 128), 256
        )

    def paired_partial_offset(sender, head, quarter, channel):
        pair_layout = fx.make_layout(
            (2, heads_local, 2, 128), (heads_local * 256, 256, 128, 1)
        )
        pair_base = partial_offset(rank // 2 * 2, 0, 0, 0)
        return pair_base + fx.Int32(
            fx.get_scalar(fx.crd2idx((sender, head, quarter, channel), pair_layout))
        )

    def amax_partial_offset(dest_pe, part):
        return 1 + fx.Int32(
            fx.get_scalar(
                fx.crd2idx(
                    (dest_pe, part),
                    fx.make_layout(
                        (npes, block_num * warp_num_per_block),
                        (block_num * warp_num_per_block, 1),
                    ),
                )
            )
        )

    def pc_partial_offset(dest_pe, part, parts, channels):
        row = fx.Int32(
            fx.get_scalar(
                fx.crd2idx((dest_pe, part), fx.make_layout((npes, parts), (parts, 1)))
            )
        )
        return _row_field_offset(row, fx.Int32(0), channels)

    def ready_offset(source_rank, head):
        return fx.Int32(
            fx.get_scalar(
                fx.crd2idx(
                    (source_rank, head),
                    fx.make_layout((npes, heads_local), (heads_local, 1)),
                )
            )
        )

    def k_fp6_tile_offset(head, tile):
        row = fx.Int32(
            fx.get_scalar(
                fx.crd2idx(
                    (head, tile),
                    fx.make_layout((heads_local, k_tiles), (k_tiles, 1)),
                )
            )
        )
        return _row_field_offset(row, fx.Int32(0), 4352)

    def v4_word_offset(head, seq, chunk, mode, codec):
        if codec == "mxfp8":
            return row_offset(seq, head, _row_field_offset(chunk, fx.Int32(0), 2), 32)
        if codec == "mxfp6" and mode == "k":
            coord = _k_sequence_coord(seq)
            return k_fp6_tile_offset(head, fx.Int32(fx.get_scalar(fx.get(coord, 0))))
        if codec == "mxfp6":
            pair_coord = fx.idx2crd(
                fx.Int32(chunk), fx.make_layout((4, 2, 2), (4, 2, 1))
            )
            word = fx.Int32(
                fx.get_scalar(
                    fx.crd2idx(pair_coord, fx.make_layout((4, 2, 2), (6, 0, 3)))
                )
            )
            return row_offset(seq, head, word, 24)
        if mode == "k":
            seq_coord = _k_sequence_coord(seq)
            tile = fx.Int32(fx.get_scalar(fx.get(seq_coord, 0)))
            token = fx.Int32(fx.get_scalar(fx.get(seq_coord, 1)))
            chunk_coord = fx.idx2crd(fx.Int32(chunk), fx.make_layout((4, 4), (1, 4)))
            row = fx.Int32(
                fx.get_scalar(
                    fx.crd2idx(
                        (head, tile),
                        fx.make_layout((heads_local, k_tiles), (k_tiles, 1)),
                    )
                )
            )
            field = fx.Int32(
                fx.get_scalar(
                    fx.crd2idx(
                        chunk_coord,
                        fx.composition(
                            fx.make_layout(2048, 1), fx.make_layout((4, 4), (1, 512))
                        ),
                    )
                )
            )
            return _row_field_offset(row, _row_field_offset(token, field, 4), 2048)
        return row_offset(seq, head, chunk, 16)

    # Source chunks remain bf16-sized even when the wire payload is quantized.
    elements_per_chunk = _TRANSPORT_CHUNK_BYTES // 2
    chunks_per_row = head_dim // elements_per_chunk
    chunk_words = _TRANSPORT_CHUNK_BYTES // 4
    total_chunks = heads * seq_len * chunks_per_row
    vec = 8
    hd = heads * head_dim
    shared_storage = fx.struct(
        type(
            "_SharedStorage",
            (),
            {
                "__annotations__": {
                    "p2p_bases_q": fx.Array[fx.Int64, npes, 16],
                    **(
                        {"fp6_words": fx.Array[fx.Int32, warp_num_per_block * 384, 16]}
                        if quant
                        and (
                            (codec == "mxfp6" and native_fp6 and v4_output in ("", "q"))
                            or codec == "mxfp6_p"
                        )
                        else {}
                    ),
                    **(
                        {"k_tail": fx.Array[fx.Int8, warp_num_per_block * 1024, 16]}
                        if quant
                        and codec == "mxfp6"
                        and native_fp6
                        and v4_output == "k"
                        else {}
                    ),
                    **(
                        {"v_scales": fx.Array[fx.Int32, 512, 16]}
                        if codec == "mxfp6_p"
                        else {}
                    ),
                    **(
                        {"v_amax": fx.Array[fx.Float32, 512, 16]}
                        if v4_output == "v" and codec in ("mxfp4", "mxfp6_p")
                        else {}
                    ),
                }
            },
        )
    )

    @flyc.kernel(known_block_size=[warp_num_per_block * 64, 1, 1])
    def attention_a2a_single_push(
        addr_input: fx.Int64,
        addr_p2p_output: fx.Int64,
        addr_p2p_scale: fx.Int64,
        addr_xdb_flag: fx.Int64,
        addr_p2p_partial: fx.Int64,
        addr_p2p_partial_ready: fx.Int64,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        warp_coord = fx.idx2crd(
            fx.Int32(tid), fx.make_layout((warp_num_per_block, 64), (64, 1))
        )
        warp = fx.Int32(fx.get_scalar(fx.get(warp_coord, 0)))
        lane = fx.Int32(fx.get_scalar(fx.get(warp_coord, 1)))
        global_warp_id = _row_field_offset(bid, warp, warp_num_per_block)
        global_warp_num = block_num * warp_num_per_block

        p2p_output_q = ptr_buf_tensor(addr_p2p_output, fx.Int64)

        if const_expr(split_v_exchange):
            xdb_flag = ptr_buf_tensor(addr_xdb_flag, fx.Int64, num_records_bytes=8)
            partial_generation = fx.Int64(buf_copy_load(xdb_flag, 0, fx.Int64))

        shared = fx.SharedAllocator().allocate(shared_storage).peek()
        p2p_bases_q = shared.p2p_bases_q.view(fx.make_layout(npes, 1))
        if lane < npes:
            peer_base_q = buf_copy_load(p2p_output_q, lane, fx.Int64)
            fx.memref_store(peer_base_q, p2p_bases_q, lane)
        fx.barrier()

        input_elem = (
            fx.BFloat16
            if v4_output in ("q", "k", "v")
            and (codec in ("int8", "e4m3") or v4_output == "v")
            else fx.Int32
        )
        input_width = 8 if input_elem == fx.BFloat16 else chunk_words
        input_q = ptr_buf_tensor(
            addr_input, input_elem, unit_elems=input_width, unit_stride=1
        )

        def transport(input_q, p2p_bases, addr_p2p_scale, codec, rotate=False, mode=""):
            wire_bytes = _transport_bytes(8, codec) if quant else 16
            wire_words = wire_bytes // 4
            peer_chunks = total_chunks // npes
            peer_group_count = (peer_chunks + 63) // 64
            peer_warp_num = global_warp_num // npes
            peer_coord = fx.idx2crd(
                fx.Int32(global_warp_id),
                fx.make_layout((UNBOUNDED_OUTER_EXTENT, npes), (npes, 1)),
            )
            peer_warp_id = fx.Int32(fx.get_scalar(fx.get(peer_coord, 0)))
            dest_pe = fx.Int32(fx.get_scalar(fx.get(peer_coord, 1)))
            peer_base = fx.Uint64(fx.memref_load(p2p_bases, dest_pe))
            uniform_peer_base = wave_uniform_i64(peer_base)
            dst_bytes_extent = (
                heads_local * k_tiles * (17408 if codec == "mxfp6" else 8192)
                if mode == "k" and codec != "mxfp8"
                else total_chunks * wire_bytes
            )
            if const_expr(quant and codec == "mxfp6"):
                dst_views = _fp6_output_views(uniform_peer_base, dst_bytes_extent)
                dst_words, dst_bytes = dst_views[0], dst_views[3]
            else:
                # One wire chunk per store: 4 dwords raw, 2 for FP8/INT8, 1 for FP4.
                dst_words = ptr_buf_tensor(
                    uniform_peer_base,
                    fx.Int32,
                    unit_elems=wire_words,
                    unit_stride=1,
                    num_records_bytes=dst_bytes_extent,
                )
            if const_expr(quant):
                p2p_scale = ptr_buf_tensor(addr_p2p_scale, fx.Int64)
                scale_base = fx.Uint64(buf_copy_load(p2p_scale, dest_pe, fx.Int64))
                uniform_scale_base = wave_uniform_i64(scale_base)
                scale_dst = ptr_buf_tensor(
                    uniform_scale_base,
                    fx.Int8,
                    num_records_bytes=total_chunks * elements_per_chunk // 32,
                )
            group_step = peer_warp_num * _PUSH_PIPELINE_DEPTH
            for group_base in range(peer_warp_id, peer_group_count, group_step):
                values = []
                destinations = []
                valid_values = []
                scales = []
                scale_destinations = []
                for batch_idx in range_constexpr(_PUSH_PIPELINE_DEPTH):
                    group_idx = fx.Int32(
                        fx.get_scalar(
                            fx.crd2idx(
                                (fx.Int32(batch_idx), group_base),
                                fx.make_layout(
                                    (_PUSH_PIPELINE_DEPTH, UNBOUNDED_OUTER_EXTENT),
                                    (peer_warp_num, 1),
                                ),
                            )
                        )
                    )
                    dest_chunk = wave_chunk(group_idx, lane)
                    valid = dest_chunk < peer_chunks
                    safe_dest_chunk = valid.select(dest_chunk, 0)
                    # Decode the owned peer slab before mapping the source/destination rows.
                    work_coord = fx.idx2crd(
                        fx.Int32(safe_dest_chunk),
                        fx.make_layout(
                            (UNBOUNDED_OUTER_EXTENT, seq_len, chunks_per_row),
                            (seq_len * chunks_per_row, chunks_per_row, 1),
                        ),
                    )
                    local_head = fx.Int32(fx.get_scalar(fx.get(work_coord, 0)))
                    seq = fx.Int32(fx.get_scalar(fx.get(work_coord, 1)))
                    row_chunk = fx.Int32(fx.get_scalar(fx.get(work_coord, 2)))
                    head = global_head(dest_pe, local_head)
                    src_chunk = input_offset(seq, head, row_chunk, elements_per_chunk)
                    raw = buf_copy_load(
                        input_q,
                        _row_field_offset(src_chunk, fx.Int32(0), chunk_words),
                        fx.Int32,
                        chunk_words,
                    )
                    if const_expr(quant):
                        decoded = fx.Vector(raw).bitcast(fx.BFloat16).to(fx.Float32)
                        if const_expr(rotate):
                            decoded = _hadamard_head(decoded, lane, head_dim)
                        if const_expr(mode and codec == "mxfp8"):
                            decoded = decoded.to(fx.BFloat16).to(fx.Float32)
                        if const_expr(mode):
                            multiplier = q_multiplier if mode == "q" else 1.0
                            decoded = (
                                (decoded * multiplier).to(fx.BFloat16).to(fx.Float32)
                            )
                        amax = fx.Float32(0.0)
                        for i in range_constexpr(elements_per_chunk):
                            amax = amax.maximumf(fmath.absf(decoded[i]))
                        # Tail lanes load a safe row, but must not contribute its max.
                        amax = fx.Float32(valid.select(amax, fx.Float32(0.0)))
                        # Each aligned four-lane group owns 32 contiguous head values.
                        for shift in (1, 2):
                            amax = amax.maximumf(amax.shuffle_xor(shift, 64))
                        scale = (
                            _v4_fp6_scale(amax)
                            if mode and codec == "mxfp6"
                            else (
                                _v4_fp8_scale(amax, fp8_fnuz)
                                if mode and codec == "mxfp8"
                                else _transport_scale(amax, codec, fp8_fnuz)
                            )
                        )
                        reciprocal = ((fx.Int32(254) - scale) << 23).bitcast(fx.Float32)
                        if mode and codec == "mxfp6":
                            values.append(_pack_v4_fp6(decoded, reciprocal))
                        else:
                            packed = []
                            for pair in range_constexpr(elements_per_chunk // 2):
                                packed.append(
                                    _pack_transport_pair(
                                        decoded[2 * pair] * reciprocal,
                                        decoded[2 * pair + 1] * reciprocal,
                                        codec,
                                    )
                                )
                            values.append(_pack_transport_words(packed, codec))
                        scales.append(scale.to(fx.Int8))
                    else:
                        values.append(raw)
                    dst_chunk = row_offset(
                        global_token(seq),
                        local_head,
                        row_chunk,
                        chunks_per_row,
                        head_major=True,
                    )
                    destinations.append(
                        v4_word_offset(
                            local_head, global_token(seq), row_chunk, mode, codec
                        )
                        if mode
                        else (
                            dst_chunk
                            if quant and codec == "mxfp6"
                            else _row_field_offset(dst_chunk, fx.Int32(0), wire_words)
                        )
                    )
                    # Explicit //4 scale addressing: layout forms add load drains inside the
                    # 16-load batch and serialize it.
                    scale_destinations.append(
                        row_offset(global_token(seq), local_head, row_chunk // 4, 4)
                        if mode
                        else dst_chunk // 4
                    )
                    valid_values.append(valid)
                for batch_idx in range_constexpr(_PUSH_PIPELINE_DEPTH):
                    if valid_values[batch_idx]:
                        if const_expr(mode == "k" and codec == "mxfp6"):
                            _store_v4_fp6_k(
                                values[batch_idx],
                                scales[batch_idx],
                                dst_words,
                                dst_bytes,
                                destinations[batch_idx],
                                scale_destinations[batch_idx] // (heads_local * 4),
                                lane,
                            )
                        elif const_expr(mode and codec == "mxfp6"):
                            _store_v4_fp6_q(
                                values[batch_idx],
                                dst_words,
                                destinations[batch_idx],
                                lane,
                            )
                        elif const_expr(quant and codec == "mxfp6"):
                            _store_fp6(
                                values[batch_idx], dst_words, destinations[batch_idx]
                            )
                        else:
                            buf_copy_store(
                                dst_words,
                                destinations[batch_idx],
                                values[batch_idx],
                                fx.Int32,
                                wire_words,
                            )
                        # Keep compile-time specialization outside the lane predicate.
                        if const_expr(quant):  # noqa: SIM102
                            if lane % 4 == 0:
                                buf_copy_store(
                                    scale_dst,
                                    scale_destinations[batch_idx],
                                    scales[batch_idx],
                                    fx.Int8,
                                )

        @flyc.jit
        def store_dense_fp6(
            words,
            output,
            first_word,
            last_word,
            wave_word,
            coalesced,
            staging,
            tiles,
            warp,
            lane,
        ):
            if coalesced:
                # One thread owns one compact six-word LDS record.
                producer = fx.make_view(
                    fx.get_iter(staging),
                    fx.make_layout((warp_num_per_block * 64, 6), (6, 1)),
                )
                for word in range_constexpr(6):
                    fx.memref_store(words[word], producer, (fx.Int32(tid), word))
                # Drain this wave's LDS writes before other lanes read the staging slab.
                wait_lds_wave()
                # Redistribute the six-word lane records into 64 + 32 four-word stores.
                wave_output = fx.slice(
                    fx.make_view(
                        fx.get_iter(output[2]),
                        fx.make_layout((UNBOUNDED_OUTER_EXTENT, 384), (1, 1)),
                    ),
                    (wave_word, None),
                )
                lines = fx.logical_divide(wave_output, fx.make_layout(4, 1))
                line_output = fx.make_view(
                    fx.get_iter(lines), fx.select(fx.get_layout(lines), [1, 0])
                )
                first = fx.memref_load_vec(fx.slice(tiles, (None, warp, lane)))
                buf_copy_store(line_output, lane, first, fx.Int32, 4)
                if lane < 32:
                    last = fx.memref_load_vec(fx.slice(tiles, (None, warp, 64 + lane)))
                    buf_copy_store(line_output, 64 + lane, last, fx.Int32, 4)
                # Drain this wave's LDS reads before the staging slab is overwritten.
                wait_lds_wave()
            else:
                buf_copy_store(
                    output[2],
                    first_word,
                    fx.Vector.from_elements([words[i] for i in range(4)], fx.Int32),
                    fx.Int32,
                    4,
                )
                buf_copy_store(
                    output[1],
                    last_word,
                    fx.Vector.from_elements([words[i] for i in range(4, 6)], fx.Int32),
                    fx.Int32,
                    2,
                )

        @flyc.jit
        def emit_fp6_group(
            shared,
            output,
            scale_output,
            group,
            peer_groups,
            dest_pe,
            head,
            seq,
            channel,
            valid,
            staged: fx.Constexpr[bool],
            scale_only: fx.Constexpr[bool] = False,
        ):
            groups_per_row = head_dim // 32
            source = input_offset(
                seq,
                global_head(dest_pe, head),
                _row_field_offset(channel, fx.Int32(0), 16),
                2,
            )
            source_row = fx.slice(
                fx.make_view(
                    fx.get_iter(input_q),
                    fx.make_layout((UNBOUNDED_OUTER_EXTENT, 16), (1, 1)),
                ),
                (source, None),
            )
            source_parts = fx.logical_divide(source_row, fx.make_layout(4, 1))
            source_fields = fx.make_view(
                fx.get_iter(source_parts),
                fx.select(fx.get_layout(source_parts), [1, 0]),
            )
            vals = []
            for part in range_constexpr(4):
                raw = buf_copy_load(source_fields, part, fx.Int32, 4)
                decoded = fx.Vector(raw).bitcast(fx.BFloat16).to(fx.Float32)
                vals.extend([decoded[i] for i in range(8)])
            if const_expr(hadamard):
                rotated = _hadamard_head(vals, lane, head_dim, 32)
                vals = [rotated[i] for i in range(32)]
            if const_expr(v4_output):
                multiplier = q_multiplier if v4_output == "q" else 1.0
                vals = [
                    (value * multiplier).to(fx.BFloat16).to(fx.Float32)
                    for value in vals
                ]
            maximum = fx.Float32(0.0)
            for value in vals:
                maximum = maximum.maximumf(fmath.absf(value))
            scale = (
                _v4_fp6_scale(maximum)
                if v4_output
                else _transport_scale(maximum, "mxfp6", fp8_fnuz)
            )
            if const_expr(scale_only):
                if valid:
                    buf_copy_store(
                        scale_output,
                        row_offset(global_token(seq), head, channel, groups_per_row),
                        scale.to(fx.Int8),
                        fx.Int8,
                    )
            else:
                reciprocal = ((fx.Int32(254) - scale) << 23).bitcast(fx.Float32)
                # Round the reciprocal product to FP32 before FP6 conversion, including overflow.
                vals = [value * reciprocal for value in vals]
                # Native FP6 pack: software is 68-94% slower for packed Q/K recipes at ws8.
                words = emit_f32_to_e2m3_native(
                    fx.Vector.from_elements(
                        vals[:16] if v4_output else vals[::2], fx.Float32
                    ),
                    fx.Vector.from_elements(
                        vals[16:] if v4_output else vals[1::2], fx.Float32
                    ),
                    fx.Float32(1.0),
                )
                full_seq = global_token(seq)
                destination = row_offset(
                    full_seq, head, channel, groups_per_row, head_major=not v4_output
                )
                first_word = _row_field_offset(destination, fx.Int32(0), 6)
                last_word = _row_field_offset(destination, fx.Int32(4), 6)
                if const_expr(v4_output == "k"):
                    seq_coord = _k_sequence_coord(full_seq)
                    tile_word = k_fp6_tile_offset(
                        head, fx.Int32(fx.get_scalar(fx.get(seq_coord, 0)))
                    )
                    token = fx.Int32(fx.get_scalar(fx.get(seq_coord, 1)))
                    band_a = _k_fp6_band_view(output[2], tile_word, token, channel, 4)
                    band_b = _k_fp6_band_view(output[1], tile_word, token, channel, 2)
                    first_word = fx.Int32(0)
                    last_word = fx.Int32(0)
                if valid:
                    if const_expr(v4_output in ("", "q")):
                        wave_start = wave_chunk(group, fx.Int32(0))
                        coalesced = wave_start + 63 < peer_groups
                        if const_expr(not v4_output):
                            # A head boundary skips the other source ranks' sequence slabs.
                            work_layout = fx.make_layout(
                                (UNBOUNDED_OUTER_EXTENT, seq_len, groups_per_row),
                                (seq_len * groups_per_row, groups_per_row, 1),
                            )
                            first_coord = fx.idx2crd(fx.Int32(wave_start), work_layout)
                            last_coord = fx.idx2crd(
                                fx.Int32(wave_start + 63), work_layout
                            )
                            coalesced = coalesced & (
                                fx.Int32(fx.get_scalar(fx.get(first_coord, 0)))
                                == fx.Int32(fx.get_scalar(fx.get(last_coord, 0)))
                            )
                        staging = shared.fp6_words.view(
                            fx.make_layout(warp_num_per_block * 384, 1)
                        )
                        tiles = shared.fp6_words.view(
                            fx.make_layout((4, warp_num_per_block, 96), (1, 384, 4))
                        )
                        wave_origin = fx.make_layout(
                            (UNBOUNDED_OUTER_EXTENT, 64), (6, -6)
                        )
                        wave_word = fx.Int32(
                            fx.get_scalar(fx.crd2idx((destination, lane), wave_origin))
                        )
                        store_dense_fp6(
                            words,
                            output,
                            first_word,
                            last_word,
                            wave_word,
                            coalesced,
                            staging,
                            tiles,
                            warp,
                            lane,
                        )
                    else:
                        band_a_wide = fx.make_view(
                            fx.get_iter(band_a), fx.make_layout((4, 4), (1, 1))
                        )
                        band_b_wide = fx.make_view(
                            fx.get_iter(band_b), fx.make_layout((2, 2), (1, 1))
                        )
                        buf_copy_store(
                            band_a_wide,
                            first_word,
                            fx.Vector.from_elements(
                                [words[i] for i in range(4)], fx.Int32
                            ),
                            fx.Int32,
                            4,
                        )
                        buf_copy_store(
                            band_b_wide,
                            last_word,
                            fx.Vector.from_elements(
                                [words[i] for i in range(4, 6)], fx.Int32
                            ),
                            fx.Int32,
                            2,
                        )
                    if const_expr(v4_output != "k"):
                        buf_copy_store(
                            scale_output, destination, scale.to(fx.Int8), fx.Int8
                        )
                    if const_expr(v4_output == "k"):
                        # B replicates the preceding channel/token, correcting a tile crossing.
                        tile_byte = _row_field_offset(tile_word, fx.Int32(0), 4)
                        image_a = _k_fp6_scale_image(output[3], tile_byte, 0)
                        image_b = _k_fp6_scale_image(output[3], tile_byte, 1)
                        scale_slot = _k_fp6_scale_slot(token)
                        previous_token, previous_channel = _k_fp6_predecessor(
                            full_seq, channel
                        )
                        previous_slot = _k_fp6_scale_slot(previous_token)
                        previous_tile = _row_field_offset(tile_word, fx.Int32(0), 4) - (
                            token == 0
                        ).select(fx.Int32(17408), fx.Int32(0))
                        if const_expr(staged):
                            tail_view = shared.k_tail.view(
                                fx.make_layout(
                                    (warp_num_per_block, 2, 512), (1024, 512, 1)
                                )
                            )
                            fx.memref_store(
                                scale.to(fx.Int8),
                                tail_view,
                                (warp, 0, _k_fp6_scale_field(scale_slot, channel)),
                            )
                            if channel > 0:
                                fx.memref_store(
                                    scale.to(fx.Int8),
                                    tail_view,
                                    (
                                        warp,
                                        1,
                                        _k_fp6_scale_field(
                                            scale_slot, previous_channel
                                        ),
                                    ),
                                )
                            elif token > 0:
                                fx.memref_store(
                                    scale.to(fx.Int8),
                                    tail_view,
                                    (
                                        warp,
                                        1,
                                        _k_fp6_scale_field(
                                            previous_slot, previous_channel
                                        ),
                                    ),
                                )
                            elif full_seq > 0:
                                # The previous tile's last B byte may belong to another wave.
                                buf_copy_store(
                                    _k_fp6_scale_image(output[3], previous_tile, 1),
                                    _k_fp6_scale_field(previous_slot, previous_channel),
                                    scale.to(fx.Int8),
                                    fx.Int8,
                                    1,
                                )
                        else:
                            buf_copy_store(
                                image_a,
                                _k_fp6_scale_field(scale_slot, channel),
                                scale.to(fx.Int8),
                                fx.Int8,
                                1,
                            )
                            if channel > 0:
                                buf_copy_store(
                                    image_b,
                                    _k_fp6_scale_field(scale_slot, previous_channel),
                                    scale.to(fx.Int8),
                                    fx.Int8,
                                    1,
                                )
                            elif full_seq > 0:
                                buf_copy_store(
                                    _k_fp6_scale_image(output[3], previous_tile, 1),
                                    _k_fp6_scale_field(previous_slot, previous_channel),
                                    scale.to(fx.Int8),
                                    fx.Int8,
                                    1,
                                )

        @flyc.jit
        def transport_fp6_native(shared):
            if const_expr(v4_output == "q"):
                # Layout decomposition adds 26 VGPRs to packed native FP6 Q.
                native_q_peer = fx.idx2crd(
                    fx.Int32(global_warp_id),
                    fx.make_layout((UNBOUNDED_OUTER_EXTENT, npes), (npes, 1)),
                )
                peer_warp = fx.Int32(fx.get_scalar(fx.get(native_q_peer, 0)))
                dest_pe = fx.Int32(fx.get_scalar(fx.get(native_q_peer, 1)))
            else:
                peer_coord = fx.idx2crd(
                    fx.Int32(global_warp_id),
                    fx.make_layout((UNBOUNDED_OUTER_EXTENT, npes), (npes, 1)),
                )
                peer_warp = fx.Int32(fx.get_scalar(fx.get(peer_coord, 0)))
                dest_pe = fx.Int32(fx.get_scalar(fx.get(peer_coord, 1)))
            peer_warps = global_warp_num // npes
            groups_per_row = head_dim // 32
            peer_groups = heads_local * seq_len * groups_per_row
            base = fx.Uint64(fx.memref_load(p2p_bases_q, dest_pe))
            output = _fp6_output_views(wave_uniform_i64(base))
            scale_table = ptr_buf_tensor(addr_p2p_scale, fx.Int64)
            scale_base = fx.Uint64(buf_copy_load(scale_table, dest_pe, fx.Int64))
            scale_output = ptr_buf_tensor(wave_uniform_i64(scale_base), fx.Int8)
            if const_expr(v4_output == "k" and k_own_tiles > 0):
                for item in range(peer_warp, heads_local * k_own_tiles, peer_warps):
                    native_tile_coord = fx.idx2crd(
                        fx.Int32(item),
                        fx.make_layout(
                            (UNBOUNDED_OUTER_EXTENT, k_own_tiles), (k_own_tiles, 1)
                        ),
                    )
                    head = fx.Int32(fx.get_scalar(fx.get(native_tile_coord, 0)))
                    tile = fx.Int32(fx.get_scalar(fx.get(native_tile_coord, 1)))
                    tile = k_first_tile + tile
                    tile_seq = (
                        _row_field_offset(tile, fx.Int32(0), 128) - rank * seq_len
                    )
                    for sub in range(fx.Int32(0), fx.Int32(8), fx.Int32(1)):
                        emit_fp6_group(
                            shared,
                            output,
                            scale_output,
                            fx.Int32(0),
                            peer_groups,
                            dest_pe,
                            head,
                            tile_seq
                            + fx.Int32(
                                fx.get_scalar(
                                    fx.crd2idx(
                                        (fx.Int32(sub), lane // 4),
                                        fx.make_layout((8, 16), (16, 1)),
                                    )
                                )
                            ),
                            lane % 4,
                            lane >= 0,
                            True,
                        )
                    # Drain this wave's LDS writes before other lanes read the staging slab.
                    wait_lds_wave()
                    tail_view = shared.k_tail.view(
                        fx.make_layout((warp_num_per_block, 2, 512), (1024, 512, 1))
                    )
                    lines = shared.k_tail.view(
                        fx.make_layout((16, warp_num_per_block, 64), (1, 1024, 16))
                    )
                    tile_word = k_fp6_tile_offset(head, tile)
                    # Lines 0-31 are image A, 32-62 the whole lines of B. B's last
                    # byte is written by the next tile's first token, so line 63
                    # keeps byte stores for its 15 bytes owned here.
                    if lane < 63:
                        packed = fx.Vector(
                            fx.memref_load_vec(fx.slice(lines, (None, warp, lane)))
                        ).bitcast(fx.Int32)
                        tail_tile = _field_view(output[2], tile_word, 4352)
                        tail_row = fx.make_view(
                            fx.add_offset(fx.get_iter(tail_tile), 4096),
                            fx.make_layout(256, 1),
                        )
                        tail_lines = fx.logical_divide(tail_row, fx.make_layout(4, 1))
                        tail_output = fx.make_view(
                            fx.get_iter(tail_lines),
                            fx.select(fx.get_layout(tail_lines), [1, 0]),
                        )
                        buf_copy_store(tail_output, lane, packed, fx.Int32, 4)
                    if lane < 15:
                        last_line = fx.slice(
                            fx.logical_divide(
                                fx.slice(tail_view, (warp, 1, None)),
                                fx.make_layout(16, 1),
                            ),
                            (None, 31),
                        )
                        byte = fx.memref_load(last_line, lane)
                        buf_copy_store(
                            _field_view(
                                _k_fp6_scale_image(
                                    output[3],
                                    _row_field_offset(tile_word, fx.Int32(0), 4),
                                    1,
                                ),
                                496,
                                16,
                            ),
                            lane,
                            byte,
                            fx.Int8,
                            1,
                        )
                    # Drain this wave's LDS reads before the staging slab is overwritten.
                    wait_lds_wave()
            for group in range(peer_warp, (peer_groups + 63) // 64, peer_warps):
                index = wave_chunk(group, lane)
                valid = index < peer_groups
                safe = valid.select(index, 0)
                if const_expr(v4_output == "q"):
                    # Token-major lanes make each wave's packed Q destination one
                    # contiguous slab, so the LDS-coalesced store applies.
                    native_q_coord = fx.idx2crd(
                        fx.Int32(safe),
                        fx.make_layout(
                            (UNBOUNDED_OUTER_EXTENT, heads_local, groups_per_row),
                            (heads_local * groups_per_row, groups_per_row, 1),
                        ),
                    )
                    seq = fx.Int32(fx.get_scalar(fx.get(native_q_coord, 0)))
                    head = fx.Int32(fx.get_scalar(fx.get(native_q_coord, 1)))
                    channel = fx.Int32(fx.get_scalar(fx.get(native_q_coord, 2)))
                else:
                    native_k_coord = fx.idx2crd(
                        fx.Int32(safe),
                        fx.make_layout(
                            (UNBOUNDED_OUTER_EXTENT, seq_len, groups_per_row),
                            (seq_len * groups_per_row, groups_per_row, 1),
                        ),
                    )
                    head = fx.Int32(fx.get_scalar(fx.get(native_k_coord, 0)))
                    seq = fx.Int32(fx.get_scalar(fx.get(native_k_coord, 1)))
                    channel = fx.Int32(fx.get_scalar(fx.get(native_k_coord, 2)))
                if const_expr(v4_output == "k" and k_own_tiles > 0):
                    own_lo = k_first_tile * 128 - rank * seq_len
                    own_hi = k_last_tile * 128 - rank * seq_len
                    # Whole tiles have a wave owner; this loop writes only boundary tokens.
                    valid = valid & ((seq < own_lo) | (seq >= own_hi))
                selected = fx.Int32(1) == 1
                if const_expr(v4_output == "k" and k_own_tiles > 0):
                    first = wave_chunk(group, fx.Int32(0))
                    last = first + 63
                    work_layout = fx.make_layout(
                        (UNBOUNDED_OUTER_EXTENT, seq_len, groups_per_row),
                        (seq_len * groups_per_row, groups_per_row, 1),
                    )
                    first_coord = fx.idx2crd(fx.Int32(first), work_layout)
                    last_coord = fx.idx2crd(fx.Int32(last), work_layout)
                    first_seq = fx.Int32(fx.get_scalar(fx.get(first_coord, 1)))
                    last_seq = fx.Int32(fx.get_scalar(fx.get(last_coord, 1)))
                    whole = (
                        (
                            fx.Int32(fx.get_scalar(fx.get(first_coord, 0)))
                            == fx.Int32(fx.get_scalar(fx.get(last_coord, 0)))
                        )
                        & (first_seq >= own_lo)
                        & (last_seq < own_hi)
                    )
                    selected = ~whole
                if selected:
                    emit_fp6_group(
                        shared,
                        output,
                        scale_output,
                        group,
                        peer_groups,
                        dest_pe,
                        head,
                        seq,
                        channel,
                        valid,
                        False,
                    )
            if const_expr(v4_output == "k"):
                # Token-major scales keep remote writes contiguous across heads.
                for group in range(peer_warp, (peer_groups + 63) // 64, peer_warps):
                    index = wave_chunk(group, lane)
                    valid = index < peer_groups
                    safe = valid.select(index, 0)
                    scale_coord = fx.idx2crd(
                        fx.Int32(safe),
                        fx.make_layout(
                            (UNBOUNDED_OUTER_EXTENT, heads_local, groups_per_row),
                            (heads_local * groups_per_row, groups_per_row, 1),
                        ),
                    )
                    emit_fp6_group(
                        shared,
                        output,
                        scale_output,
                        group,
                        peer_groups,
                        dest_pe,
                        fx.Int32(fx.get_scalar(fx.get(scale_coord, 1))),
                        fx.Int32(fx.get_scalar(fx.get(scale_coord, 0))),
                        fx.Int32(fx.get_scalar(fx.get(scale_coord, 2))),
                        valid,
                        False,
                        True,
                    )

        @flyc.jit
        def reduce_partial_amax(local_scales, dest_pe):
            maximum = fx.Float32(0.0)
            for part, state in range(
                lane, fx.Int32(global_warp_num), fx.Int32(64), init=[maximum]
            ):
                value = buf_copy_load(
                    local_scales,
                    amax_partial_offset(dest_pe, fx.Int32(part)),
                    fx.Float32,
                )
                result = yield [state[0].maximumf(value)]
            maximum = fx.Float32(result)
            for shift in (32, 16, 8, 4, 2, 1):
                maximum = maximum.maximumf(maximum.shuffle_xor(shift, 64))
            return maximum

        @flyc.jit
        def broadcast_partial_amax(
            maximum, scale_table, dest_pe, peer_warp, peer_warps
        ):
            for shift in (32, 16, 8, 4, 2, 1):
                maximum = maximum.maximumf(maximum.shuffle_xor(shift, 64))
            if lane == 0:
                part = fx.Int32(
                    fx.get_scalar(
                        fx.crd2idx(
                            (fx.Int32(rank), peer_warp),
                            fx.make_layout((npes, peer_warps), (peer_warps, 1)),
                        )
                    )
                )
                slot = amax_partial_offset(dest_pe, part)
                for peer in range_constexpr(npes):
                    base = buf_copy_load(scale_table, peer, fx.Int64)
                    buf_copy_store(
                        ptr_buf_tensor(base, fx.Float32), slot, maximum, fx.Float32
                    )

        @flyc.jit
        def transport_v4_per_tensor_qk():
            peer_coord = fx.idx2crd(
                fx.Int32(global_warp_id),
                fx.make_layout((UNBOUNDED_OUTER_EXTENT, npes), (npes, 1)),
            )
            peer_warp = fx.Int32(fx.get_scalar(fx.get(peer_coord, 0)))
            dest_pe = fx.Int32(fx.get_scalar(fx.get(peer_coord, 1)))
            peer_warps = global_warp_num // npes
            scale_table = ptr_buf_tensor(addr_p2p_scale, fx.Int64)
            local_scale_base = buf_copy_load(scale_table, rank, fx.Int64)
            local_scales = ptr_buf_tensor(local_scale_base, fx.Float32)
            scale = fx.Float32(1.0)
            if const_expr(not v4_amax):
                maximum = reduce_partial_amax(local_scales, dest_pe)
                if const_expr(codec == "int8"):
                    scale = maximum / fx.Float32(127.0)
                else:
                    # Match torch's amax / dtype_max: fp32 reciprocal, then one multiply.
                    scale = maximum * fx.Float32(1.0 / (240.0 if fp8_fnuz else 448.0))
                scale = (scale > 0.0).select(scale, fx.Float32(1.0))
                if (dest_pe == rank) & (peer_warp == 0) & (lane == 0):
                    buf_copy_store(local_scales, 0, scale, fx.Float32)

            base = fx.Uint64(fx.memref_load(p2p_bases_q, dest_pe))
            output = ptr_buf_tensor(
                wave_uniform_i64(base), fx.Int32, unit_elems=2, unit_stride=1
            )
            maximum = fx.Float32(0.0)
            # Each 16-lane group owns one (row, destination head) pair, so every lane loads
            # an owned head and the Hadamard lane groups stay head-aligned.
            groups = 64 * vec // head_dim
            rows = 8
            loads = rows * heads_local // groups
            lane_pair_coord = fx.idx2crd(
                fx.Int32(lane),
                fx.make_layout((groups, head_dim // vec), (head_dim // vec, 1)),
            )
            lane_pair = fx.Int32(fx.get_scalar(fx.get(lane_pair_coord, 0)))
            group_lane = fx.Int32(fx.get_scalar(fx.get(lane_pair_coord, 1)))
            pair_rows = []
            pair_heads = []
            for load in range_constexpr(loads):
                pair_row_coord = fx.idx2crd(
                    fx.Int32(lane_pair + load * groups),
                    fx.make_layout(
                        (UNBOUNDED_OUTER_EXTENT, heads_local), (heads_local, 1)
                    ),
                )
                pair_row = fx.Int32(fx.get_scalar(fx.get(pair_row_coord, 0)))
                pair_head = fx.Int32(fx.get_scalar(fx.get(pair_row_coord, 1)))
                pair_rows.append(pair_row)
                pair_heads.append(pair_head)
            for seq0, state in range(
                peer_warp,
                fx.Int32(seq_len),
                fx.Int32(peer_warps * rows),
                init=[maximum],
            ):
                seq0 = fx.Int32(seq0)
                seqs = []
                tiles = []
                # Preload the batch before compute to overlap load latency.
                for load in range_constexpr(loads):
                    seq = seq0 + pair_rows[load] * peer_warps
                    valid = seq < seq_len
                    seq = valid.select(seq, fx.Int32(seq_len - 1))
                    seqs.append((seq, valid))
                    tiles.append(
                        fx.Vector(
                            buf_copy_load(
                                input_q,
                                fx.Int32(
                                    fx.get_scalar(
                                        fx.crd2idx(
                                            (
                                                seq,
                                                global_head(dest_pe, pair_heads[load]),
                                                _row_field_offset(
                                                    group_lane, fx.Int32(0), vec
                                                ),
                                            ),
                                            fx.make_layout(
                                                (seq_len, heads, head_dim),
                                                (hd, head_dim, 1),
                                            ),
                                        )
                                    )
                                ),
                                elem=fx.BFloat16,
                                unit_elems=vec,
                            )
                        ).to(fx.Float32)
                    )
                current = state[0]
                for load in range_constexpr(loads):
                    seq, valid = seqs[load]
                    values = tiles[load]
                    if const_expr(codec == "e4m3" and hadamard):
                        # quantize_fp8_rotated materializes BF16 after normalized WHT;
                        # neither per-tensor recipe folds the MX Q multiplier.
                        values = (
                            _hadamard_head(values, lane, head_dim)
                            .to(fx.BFloat16)
                            .to(fx.Float32)
                        )
                    if const_expr(v4_amax):
                        # A clamped duplicate row cannot change the max.
                        for i in range_constexpr(vec):
                            current = current.maximumf(fmath.absf(values[i]))
                    else:
                        pairs = [
                            (
                                _pack_int8_pair(
                                    values[2 * i] / scale, values[2 * i + 1] / scale
                                )
                                if codec == "int8"
                                else _pack_transport_pair(
                                    values[2 * i] * (fx.Float32(1.0) / scale),
                                    values[2 * i + 1] * (fx.Float32(1.0) / scale),
                                    "e4m3",
                                )
                            )
                            for i in range(4)
                        ]
                        if valid:
                            destination = row_offset(
                                global_token(seq), pair_heads[load], group_lane, 16
                            )
                            buf_copy_store(
                                output,
                                _row_field_offset(destination, fx.Int32(0), 2),
                                _pack_transport_words(pairs, codec),
                                fx.Int32,
                                2,
                            )
                result = yield [current]
            if const_expr(v4_amax):
                broadcast_partial_amax(
                    fx.Float32(result), scale_table, dest_pe, peer_warp, peer_warps
                )

        @flyc.jit
        def transport_v4_fp8_pc_v():
            peer_coord = fx.idx2crd(
                fx.Int32(global_warp_id),
                fx.make_layout((UNBOUNDED_OUTER_EXTENT, npes), (npes, 1)),
            )
            peer_warp = fx.Int32(fx.get_scalar(fx.get(peer_coord, 0)))
            dest_pe = fx.Int32(fx.get_scalar(fx.get(peer_coord, 1)))
            peer_warps = global_warp_num // npes
            channels = heads_local * head_dim
            scale_table = ptr_buf_tensor(addr_p2p_scale, fx.Int64)
            local_scale_base = buf_copy_load(scale_table, rank, fx.Int64)
            local_scales = ptr_buf_tensor(local_scale_base, fx.Float32)
            descales = _field_view(local_scales, fx.Int32(0), channels)
            exchange = fx.make_view(
                fx.add_offset(fx.get_iter(local_scales), channels),
                fx.make_layout((npes, npes, channels), (npes * channels, channels, 1)),
            )
            base = fx.Uint64(fx.memref_load(p2p_bases_q, dest_pe))
            output = ptr_buf_tensor(
                wave_uniform_i64(base), fx.Int32, unit_elems=2, unit_stride=1
            )
            token_tile = V_PC_TOKEN_TILE
            token_tiles = ceildiv(seq_len, token_tile)
            channel_chunks = channels // 8
            partials = ptr_buf_tensor(addr_p2p_partial, fx.Float32)
            work_items = channel_chunks if pc_reduce else token_tiles * channel_chunks
            for item in range(
                wave_chunk(peer_warp, lane),
                fx.Int32(work_items),
                fx.Int32(peer_warps * 64),
            ):
                item = fx.Int32(item)
                # Work-item decomposition assigns one channel vector and token tile.
                pc_work_coord = fx.idx2crd(
                    fx.Int32(item),
                    fx.make_layout(
                        (channel_chunks, UNBOUNDED_OUTER_EXTENT), (1, channel_chunks)
                    ),
                )
                channel_chunk = fx.Int32(fx.get_scalar(fx.get(pc_work_coord, 0)))
                tile = fx.Int32(fx.get_scalar(fx.get(pc_work_coord, 1)))
                channel = _row_field_offset(channel_chunk, fx.Int32(0), 8)
                seq_begin = _row_field_offset(tile, fx.Int32(0), token_tile)
                seq_end = fx.min(seq_begin + token_tile, fx.Int32(seq_len))
                if const_expr(v4_amax):
                    maximum = fx.Vector.filled(8, 0.0, fx.Float32)
                    if const_expr(pc_reduce):
                        # Every source rank writes all npes slots of its channel,
                        # including zero maxima, so each slot has one writer.
                        for part, state in range(
                            fx.Int32(0),
                            fx.Int32(token_tiles),
                            fx.Int32(1),
                            init=[maximum],
                        ):
                            offset = pc_partial_offset(
                                dest_pe, fx.Int32(part), token_tiles, channels
                            )
                            partial_channels = _field_view(partials, offset, channels)
                            partial_vectors = fx.logical_divide(
                                partial_channels, fx.make_layout(8, 1)
                            )
                            partial_row = fx.slice(
                                partial_vectors, (None, channel_chunk)
                            )
                            combined = fx.Vector.from_elements(
                                [
                                    fx.Float32(
                                        buf_copy_load(partial_row, i, fx.Float32)
                                    )
                                    for i in range(8)
                                ],
                                fx.Float32,
                            )
                            result = yield [state[0].maximumf(combined)]
                        maximum = fx.Vector(result)
                        for peer in range_constexpr(npes):
                            remote = buf_copy_load(scale_table, peer, fx.Int64)
                            resource = ptr_buf_tensor(remote, fx.Float32)
                            remote_exchange = fx.make_view(
                                fx.add_offset(fx.get_iter(resource), channels),
                                fx.make_layout(
                                    (npes, npes, channels),
                                    (npes * channels, channels, 1),
                                ),
                            )
                            scale_channels = fx.slice(
                                remote_exchange, (dest_pe, rank, None)
                            )
                            scale_row = fx.slice(
                                fx.logical_divide(scale_channels, fx.make_layout(8, 1)),
                                (None, channel_chunk),
                            )
                            for i in range_constexpr(8):
                                buf_copy_store(scale_row, i, maximum[i], fx.Float32)
                    else:
                        # Partial producers never wait: each tile publishes one
                        # local maximum per channel before the cross-rank handshake.
                        for seq, state in range(
                            seq_begin,
                            seq_end,
                            fx.Int32(1),
                            init=[maximum],
                        ):
                            source = fx.Int32(
                                fx.get_scalar(
                                    fx.crd2idx(
                                        (
                                            fx.Int32(seq),
                                            global_head(dest_pe, fx.Int32(0)),
                                            channel,
                                        ),
                                        fx.make_layout(
                                            (seq_len, heads, head_dim),
                                            (hd, head_dim, 1),
                                        ),
                                    )
                                )
                            )
                            values = fx.Vector(
                                buf_copy_load(input_q, source, fx.BFloat16, 8)
                            ).to(fx.Float32)
                            result = yield [state[0].maximumf(fmath.absf(values))]
                        maximum = fx.Vector(result)
                        offset = pc_partial_offset(dest_pe, tile, token_tiles, channels)
                        partial_channels = _field_view(partials, offset, channels)
                        partial_row = fx.slice(
                            fx.logical_divide(partial_channels, fx.make_layout(8, 1)),
                            (None, channel_chunk),
                        )
                        for i in range_constexpr(8):
                            buf_copy_store(partial_row, i, maximum[i], fx.Float32)
                else:
                    scales = []
                    channel_scales = fx.slice(
                        fx.logical_divide(descales, fx.make_layout(8, 1)),
                        (None, channel_chunk),
                    )
                    for i in range_constexpr(8):
                        maximum = fx.Float32(0.0)
                        for peer in range_constexpr(npes):
                            peer_channels = fx.slice(exchange, (dest_pe, peer, None))
                            peer_scales = fx.slice(
                                fx.logical_divide(peer_channels, fx.make_layout(8, 1)),
                                (None, channel_chunk),
                            )
                            partial = buf_copy_load(peer_scales, i, fx.Float32)
                            maximum = maximum.maximumf(partial)
                        # Clamp amax like quantize_v_fp8 so zero channels get a positive descale.
                        maximum = maximum.maximumf(fx.Float32(1.0e-12))
                        # The per-channel descale uses a rounded FP32 reciprocal multiply.
                        scale = maximum * fx.Float32(
                            1.0 / (240.0 if fp8_fnuz else 448.0)
                        )
                        scales.append(scale)
                        if (dest_pe == rank) & (tile == 0):
                            buf_copy_store(channel_scales, i, scale, fx.Float32)
                    for seq in range(
                        seq_begin,
                        seq_end,
                        fx.Int32(1),
                    ):
                        source = input_offset(
                            fx.Int32(seq), global_head(dest_pe, fx.Int32(0)), channel
                        )
                        values = fx.Vector(
                            buf_copy_load(input_q, source, fx.BFloat16, 8)
                        ).to(fx.Float32)
                        pairs = [
                            _pack_transport_pair(
                                values[2 * i] / scales[2 * i],
                                values[2 * i + 1] / scales[2 * i + 1],
                                "e4m3",
                            )
                            for i in range(4)
                        ]
                        destination = row_offset(
                            global_token(fx.Int32(seq)),
                            0,
                            _row_field_offset(channel_chunk, fx.Int32(0), 2),
                            head_dim // 4,
                        )
                        buf_copy_store(
                            output,
                            destination,
                            _pack_transport_words(pairs, "e4m3"),
                            fx.Int32,
                            2,
                        )

        @flyc.jit
        def transport_v4_fp8_v():
            # Every source owns a token shard of every destination's tensor.
            # Replicate only warp amax metadata so quantization stays send-side.
            peer_coord = fx.idx2crd(
                fx.Int32(global_warp_id),
                fx.make_layout((UNBOUNDED_OUTER_EXTENT, npes), (npes, 1)),
            )
            peer_warp = fx.Int32(fx.get_scalar(fx.get(peer_coord, 0)))
            dest_pe = fx.Int32(fx.get_scalar(fx.get(peer_coord, 1)))
            peer_warps = global_warp_num // npes
            peer_chunks = total_chunks // npes
            scale_table = ptr_buf_tensor(addr_p2p_scale, fx.Int64)
            local_scale_base = buf_copy_load(scale_table, rank, fx.Int64)
            local_scales = ptr_buf_tensor(local_scale_base, fx.Float32)
            if const_expr(v4_amax):
                maximum = fx.Float32(0.0)
                for chunk, state in range(
                    wave_chunk(peer_warp, lane),
                    fx.Int32(peer_chunks),
                    fx.Int32(peer_warps * 64),
                    init=[maximum],
                ):
                    chunk = fx.Int32(chunk)
                    v_chunk_coord = fx.idx2crd(
                        fx.Int32(chunk),
                        fx.make_layout(
                            (UNBOUNDED_OUTER_EXTENT, heads_local * 16),
                            (heads_local * 16, 1),
                        ),
                    )
                    seq = fx.Int32(fx.get_scalar(fx.get(v_chunk_coord, 0)))
                    head_chunk = fx.Int32(fx.get_scalar(fx.get(v_chunk_coord, 1)))
                    source = input_offset(
                        seq, global_head(dest_pe, fx.Int32(0)), head_chunk, 8
                    )
                    values = fx.Vector(
                        buf_copy_load(
                            input_q,
                            _row_field_offset(source, fx.Int32(0), 8),
                            fx.BFloat16,
                            8,
                        )
                    ).to(fx.Float32)
                    current = state[0]
                    for i in range_constexpr(8):
                        current = current.maximumf(fmath.absf(values[i]))
                    result = yield [current]
                broadcast_partial_amax(
                    fx.Float32(result), scale_table, dest_pe, peer_warp, peer_warps
                )
            else:
                maximum = reduce_partial_amax(local_scales, dest_pe)
                scale = maximum / fx.Float32(240.0 if fp8_fnuz else 448.0)
                scale = (scale > 0.0).select(scale, fx.Float32(1.0))
                reciprocal = fx.Float32(1.0) / scale
                if (dest_pe == rank) & (peer_warp == 0) & (lane == 0):
                    buf_copy_store(local_scales, 0, scale, fx.Float32)
                base = fx.Uint64(fx.memref_load(p2p_bases_q, dest_pe))
                output = ptr_buf_tensor(
                    wave_uniform_i64(base), fx.Int32, unit_elems=2, unit_stride=1
                )
                for chunk in range(
                    wave_chunk(peer_warp, lane),
                    fx.Int32(peer_chunks),
                    fx.Int32(peer_warps * 64),
                ):
                    chunk = fx.Int32(chunk)
                    v_chunk_coord = fx.idx2crd(
                        fx.Int32(chunk),
                        fx.make_layout(
                            (UNBOUNDED_OUTER_EXTENT, heads_local * 16),
                            (heads_local * 16, 1),
                        ),
                    )
                    seq = fx.Int32(fx.get_scalar(fx.get(v_chunk_coord, 0)))
                    head_chunk = fx.Int32(fx.get_scalar(fx.get(v_chunk_coord, 1)))
                    source = input_offset(
                        seq, global_head(dest_pe, fx.Int32(0)), head_chunk, 8
                    )
                    values = fx.Vector(
                        buf_copy_load(
                            input_q,
                            _row_field_offset(source, fx.Int32(0), 8),
                            fx.BFloat16,
                            8,
                        )
                    ).to(fx.Float32)
                    pairs = [
                        _pack_transport_pair(
                            values[2 * i] * reciprocal,
                            values[2 * i + 1] * reciprocal,
                            "e4m3",
                        )
                        for i in range(4)
                    ]
                    destination = fx.Int32(
                        fx.get_scalar(
                            fx.crd2idx(
                                (rank, chunk),
                                fx.make_layout((npes, peer_chunks), (peer_chunks, 1)),
                            )
                        )
                    )
                    buf_copy_store(
                        output,
                        _row_field_offset(destination, fx.Int32(0), 2),
                        _pack_transport_words(pairs, "e4m3"),
                        fx.Int32,
                        2,
                    )

        @flyc.jit
        def wait_v_partial(dest_pe, local_head):
            ready_table = ptr_buf_tensor(addr_p2p_partial_ready, fx.Int64)
            ready_base = buf_copy_load(ready_table, dest_pe, fx.Int64)
            # Local and remote publishers finish in an earlier launch.
            if lane < 2:
                sender = fx.Int32(
                    fx.get_scalar(
                        fx.crd2idx(
                            (fx.Int32(rank // 2), lane),
                            fx.make_layout((npes // 2, 2), (2, 1)),
                        )
                    )
                )
                slot = ready_offset(sender, local_head)
                ready = fx.inttoptr(
                    fx.PointerType.get(
                        fx.Int64.ir_type,
                        address_space=fx.AddressSpace.Global,
                        alignment=8,
                    ),
                    ready_base,
                )
                spin_until_ge_i64(
                    fx.ptrtoint(fx.add_offset(ready, slot)), partial_generation
                )
                fx.memory_fence(
                    ordering=fx.AtomicOrdering.Acquire,
                    syncscope=fx.rocdl.SyncScope.OneAs,
                )
            fx.barrier()

        @flyc.jit
        def load_v4_v_amax(shared, head, global_start):
            scratch = shared.v_amax.view(fx.make_layout((4, 16, 8), (128, 8, 1)))
            v_amax_thread_coord = fx.idx2crd(
                fx.Int32(tid), fx.make_layout((warp_num_per_block * 4, 16), (16, 1))
            )
            token_lane = fx.Int32(fx.get_scalar(fx.get(v_amax_thread_coord, 0)))
            channel_chunk = fx.Int32(fx.get_scalar(fx.get(v_amax_thread_coord, 1)))
            values = []
            for half in range_constexpr(2):
                # Explicit half/token/channel arithmetic: layout forms turn the scalar (SCC)
                # loop backedge into a per-lane (VCC) branch.
                token = global_start + token_lane + half * 16
                if const_expr(v_pack == AttentionPack.V_FOR_FP6_P):
                    token = fp6p_source_token(token)
                valid = (token >= rank * seq_len) & (token < (rank + 1) * seq_len)
                seq = valid.select(token - rank * seq_len, 0)
                raw = fx.Vector(
                    buf_copy_load(
                        input_q,
                        input_offset(seq, head, channel_chunk * 8),
                        elem=fx.BFloat16,
                        unit_elems=8,
                    )
                ).to(fx.Float32)
                values.append(
                    fx.Vector.from_elements(
                        [valid.select(raw[i], fx.Float32(0.0)) for i in range(8)],
                        fx.Float32,
                    )
                )
            scratch_row = fx.slice(scratch, (warp, channel_chunk, None))
            for i in range_constexpr(8):
                amax = fmath.absf(values[0][i]).maximumf(fmath.absf(values[1][i]))
                for shift in (16, 32):
                    amax = amax.maximumf(amax.shuffle_xor(shift, 64))
                if lane < 16:
                    fx.memref_store(
                        amax,
                        scratch_row,
                        i,
                    )
            fx.barrier()
            maxima = []
            for i in range_constexpr(8):
                amax = fx.Float32(0.0)
                for source_wave in range_constexpr(4):
                    amax = amax.maximumf(
                        fx.memref_load(
                            fx.slice(scratch, (source_wave, channel_chunk, None)),
                            i,
                        )
                    )
                maxima.append(amax)
            return values, fx.Vector.from_elements(maxima, fx.Float32)

        @flyc.jit
        def publish_v_partial(shared):
            v_partial_thread_coord = fx.idx2crd(
                fx.Int32(tid), fx.make_layout((warp_num_per_block * 4, 16), (16, 1))
            )
            token_lane = fx.Int32(fx.get_scalar(fx.get(v_partial_thread_coord, 0)))
            channel_chunk = fx.Int32(fx.get_scalar(fx.get(v_partial_thread_coord, 1)))
            partial_table = ptr_buf_tensor(addr_p2p_partial, fx.Int64)
            for head in range(bid, fx.Int32(heads), fx.Int32(block_num)):
                v_partial_head_coord = fx.idx2crd(
                    fx.Int32(head),
                    fx.make_layout(
                        (UNBOUNDED_OUTER_EXTENT, heads_local), (heads_local, 1)
                    ),
                )
                dest_pe = fx.Int32(fx.get_scalar(fx.get(v_partial_head_coord, 0)))
                local_head = fx.Int32(fx.get_scalar(fx.get(v_partial_head_coord, 1)))
                partial_base = buf_copy_load(partial_table, dest_pe, fx.Int64)
                partials = ptr_buf_tensor(partial_base, fx.Float32)
                for quarter in range_constexpr(2):
                    _values, maxima = load_v4_v_amax(
                        shared, head, split_frame * 64 + quarter * 32
                    )
                    if token_lane == 0:
                        partial_row = fx.slice(
                            fx.make_view(
                                fx.get_iter(partials),
                                fx.make_layout((UNBOUNDED_OUTER_EXTENT, 8), (1, 1)),
                            ),
                            (
                                partial_offset(
                                    rank, local_head, quarter, channel_chunk * 8
                                ),
                                None,
                            ),
                        )
                        for i in range_constexpr(8):
                            buf_copy_store(
                                partial_row,
                                i,
                                maxima[i],
                                fx.Float32,
                            )
                    fx.barrier()
                fx.memory_fence(
                    ordering=fx.AtomicOrdering.Release,
                    syncscope=fx.rocdl.SyncScope.OneAs,
                )
                fx.barrier()
                if tid == 0:
                    ready_table = ptr_buf_tensor(addr_p2p_partial_ready, fx.Int64)
                    ready_base = buf_copy_load(ready_table, dest_pe, fx.Int64)
                    ready = fx.inttoptr(
                        fx.PointerType.get(
                            fx.Int64.ir_type,
                            address_space=fx.AddressSpace.Global,
                            alignment=8,
                        ),
                        ready_base,
                    )
                    fx.generic_store(
                        fx.add_offset(ready, ready_offset(rank, local_head)),
                        partial_generation,
                        memory_order=fx.AtomicOrdering.Release,
                        syncscope=fx.rocdl.SyncScope.OneAs,
                    )

        @flyc.jit
        def transport_v4_v(shared):
            # FP6-P groups share an amax across a 32-mod-64 sender boundary.
            first_tile = rank * seq_len // 128
            rank_tiles = ((rank + 1) * seq_len + 127) // 128 - first_tile
            scale_table = ptr_buf_tensor(addr_p2p_scale, fx.Int64)
            v_transport_thread_coord = fx.idx2crd(
                fx.Int32(tid), fx.make_layout((warp_num_per_block * 4, 16), (16, 1))
            )
            token_lane = fx.Int32(fx.get_scalar(fx.get(v_transport_thread_coord, 0)))
            channel_chunk = fx.Int32(fx.get_scalar(fx.get(v_transport_thread_coord, 1)))
            for phase in range_constexpr(2 if split_v_exchange else 1):
                if const_expr(split_v_exchange and phase == 1):
                    first_tile = split_frame // 2
                    rank_tiles = 1
                for work in range(
                    bid, fx.Int32(heads * rank_tiles), fx.Int32(block_num)
                ):
                    v_work_coord = fx.idx2crd(
                        fx.Int32(work),
                        fx.make_layout(
                            (UNBOUNDED_OUTER_EXTENT, rank_tiles), (rank_tiles, 1)
                        ),
                    )
                    head = fx.Int32(fx.get_scalar(fx.get(v_work_coord, 0)))
                    tile_id = first_tile + fx.Int32(
                        fx.get_scalar(fx.get(v_work_coord, 1))
                    )
                    v_head_coord = fx.idx2crd(
                        head,
                        fx.make_layout(
                            (UNBOUNDED_OUTER_EXTENT, heads_local), (heads_local, 1)
                        ),
                    )
                    dest_pe = fx.Int32(fx.get_scalar(fx.get(v_head_coord, 0)))
                    local_head = fx.Int32(fx.get_scalar(fx.get(v_head_coord, 1)))
                    peer_base = fx.Uint64(fx.memref_load(p2p_bases_q, dest_pe))
                    dst_words = ptr_buf_tensor(
                        wave_uniform_i64(peer_base),
                        fx.Int32,
                        num_records_bytes=heads_local * k_tiles * 8192 + 64,
                    )
                    scale_base = fx.Uint64(
                        buf_copy_load(scale_table, dest_pe, fx.Int64)
                    )
                    scale_dst = ptr_buf_tensor(
                        wave_uniform_i64(scale_base),
                        fx.Int8,
                        num_records_bytes=heads_local * k_tiles * 512,
                    )
                    if const_expr(split_v_exchange):
                        partial_table = ptr_buf_tensor(addr_p2p_partial, fx.Int64)
                        # Both senders publish to the destination; packers read its
                        # two slots only after the publication release/acquire.
                        partial_base = buf_copy_load(partial_table, dest_pe, fx.Int64)
                        partials = ptr_buf_tensor(partial_base, fx.Float32)
                    for quarter in range_constexpr(4):
                        global_start = _row_field_offset(
                            tile_id,
                            _row_field_offset(fx.Int32(quarter), fx.Int32(0), 32),
                            128,
                        )
                        owned = (global_start >= rank * seq_len) & (
                            global_start
                            < (
                                (rank + 1) * seq_len
                                if rank != npes - 1
                                else k_tiles * 128
                            )
                        )
                        if const_expr(split_v_exchange):
                            # Explicit //64*64: a layout form here turns the scalar (SCC) loop backedge
                            # into a per-lane (VCC) branch.
                            frame_start = global_start // 64 * 64
                            owned = (frame_start < (rank + 1) * seq_len) & (
                                frame_start + 64 > rank * seq_len
                            )
                            if const_expr(rank == npes - 1):
                                owned = owned | (frame_start >= seq_full)
                        if const_expr(split_v_exchange):
                            owned = owned & (
                                (frame_start != split_frame * 64)
                                if phase == 0
                                else (frame_start == split_frame * 64)
                            )
                        if owned:
                            if const_expr(split_v_exchange and phase == 1):
                                wait_v_partial(dest_pe, local_head)
                            values, maxima = load_v4_v_amax(shared, head, global_start)
                            reciprocals = []
                            if const_expr(split_v_exchange):
                                pair_row = _field_view(
                                    partials,
                                    paired_partial_offset(
                                        fx.Int32(0),
                                        local_head,
                                        quarter % 2,
                                        channel_chunk * 8,
                                    ),
                                    8,
                                )
                                peer_row = _field_view(
                                    partials,
                                    paired_partial_offset(
                                        fx.Int32(1),
                                        local_head,
                                        quarter % 2,
                                        channel_chunk * 8,
                                    ),
                                    8,
                                )
                            for i in range_constexpr(8):
                                amax = maxima[i]
                                if const_expr(split_v_exchange):  # noqa: SIM102
                                    # The paired sender is one source-rank slab after this slot.
                                    if frame_start == split_frame * 64:
                                        amax = buf_copy_load(
                                            pair_row, i, fx.Float32
                                        ).maximumf(
                                            buf_copy_load(
                                                peer_row,
                                                i,
                                                fx.Float32,
                                            )
                                        )
                                bits = amax.maximumf(fx.Float32(1.0e-12)).bitcast(
                                    fx.Int32
                                )
                                scale = (
                                    ((bits >> 23) & 255)
                                    - 2
                                    + ((bits & 0x7FFFFF) > 0x400000).to(fx.Int32)
                                )
                                scale = (scale < 0).select(fx.Int32(0), scale)
                                scale = (scale > 255).select(fx.Int32(255), scale)
                                reciprocals.append(
                                    ((fx.Int32(254) - scale) << 23).bitcast(fx.Float32)
                                )
                                write_scale = token_lane == 0
                                if const_expr(split_v_exchange and rank % 2 == 1):
                                    write_scale = write_scale & (
                                        frame_start != split_frame * 64
                                    )
                                if write_scale:
                                    channel = channel_chunk * 8 + i
                                    scale_row = fx.Int32(
                                        fx.get_scalar(
                                            fx.crd2idx(
                                                (local_head, tile_id),
                                                fx.make_layout(
                                                    (heads_local, k_tiles), (k_tiles, 1)
                                                ),
                                            )
                                        )
                                    )
                                    scale_tile = fx.slice(
                                        fx.make_view(
                                            fx.get_iter(scale_dst),
                                            fx.make_layout(
                                                (heads_local * k_tiles, 512), (512, 1)
                                            ),
                                        ),
                                        (scale_row, None),
                                    )
                                    scale_fields = fx.make_view(
                                        fx.get_iter(scale_tile),
                                        fx.make_layout(
                                            (4, (2, 16, 4)), (128, (4, 8, 1))
                                        ),
                                    )
                                    scale_offset = fx.Int32(
                                        fx.get_scalar(
                                            fx.crd2idx(
                                                (fx.Int32(quarter), channel),
                                                fx.get_layout(scale_fields),
                                            )
                                        )
                                    )
                                    buf_copy_store(
                                        scale_tile,
                                        scale_offset,
                                        scale.to(fx.Int8),
                                        fx.Int8,
                                    )
                            for half in range_constexpr(2):
                                pairs = [
                                    _pack_v4_v_pair(
                                        values[half][2 * pair] * reciprocals[2 * pair],
                                        values[half][2 * pair + 1]
                                        * reciprocals[2 * pair + 1],
                                    )
                                    for pair in range(4)
                                ]
                                token = token_lane + half * 16
                                tile_row = fx.Int32(
                                    fx.get_scalar(
                                        fx.crd2idx(
                                            (local_head, tile_id),
                                            fx.make_layout(
                                                (heads_local, k_tiles), (k_tiles, 1)
                                            ),
                                        )
                                    )
                                )
                                tile_words = fx.slice(
                                    fx.make_view(
                                        fx.get_iter(dst_words),
                                        fx.make_layout(
                                            (heads_local * k_tiles, 2048), (2048, 1)
                                        ),
                                    ),
                                    (tile_row, None),
                                )
                                payload = fx.make_view(
                                    fx.get_iter(tile_words),
                                    fx.make_layout(
                                        ((4, 4), (2, 2), (4, 2, 4)),
                                        ((1, 512), (128, 256), (4, 64, 16)),
                                    ),
                                )
                                word_offset = fx.Int32(
                                    fx.get_scalar(
                                        fx.crd2idx(
                                            (
                                                fx.Int32(channel_chunk),
                                                fx.Int32(quarter),
                                                fx.Int32(token),
                                            ),
                                            fx.get_layout(payload),
                                        )
                                    )
                                )
                                source_token = global_start + token
                                if const_expr(v_pack == AttentionPack.V_FOR_FP6_P):
                                    source_token = fp6p_source_token(source_token)
                                write_payload = (source_token >= rank * seq_len) & (
                                    source_token < (rank + 1) * seq_len
                                )
                                if const_expr(rank == npes - 1):
                                    write_payload = write_payload | (
                                        source_token >= seq_full
                                    )
                                if write_payload:
                                    buf_copy_store(
                                        tile_words,
                                        word_offset,
                                        _pack_transport_words(pairs, "mxfp4"),
                                        fx.Int32,
                                    )
                            # All waves finish reading the reduction before its next use.
                            fx.barrier()

        @flyc.jit
        def transport_v4_fp6_p_v(shared):
            first_tile = rank * seq_len // 128
            rank_tiles = ((rank + 1) * seq_len + 127) // 128 - first_tile
            staging = shared.fp6_words.view(fx.make_layout(warp_num_per_block * 384, 1))
            tiles = shared.fp6_words.view(
                fx.make_layout((4, warp_num_per_block, 96), (1, 384, 4))
            )
            scale_image = shared.v_scales.view(fx.make_layout((2, 64, 4), (256, 4, 1)))
            scale_table = ptr_buf_tensor(addr_p2p_scale, fx.Int64)
            fp6p_lane_coord = fx.idx2crd(
                fx.Int32(lane), fx.make_layout((2, 32), (32, 1))
            )
            half_wave = fx.Int32(fx.get_scalar(fx.get(fp6p_lane_coord, 0)))
            channel_lane = fx.Int32(fx.get_scalar(fx.get(fp6p_lane_coord, 1)))
            channel = fx.Int32(
                fx.get_scalar(
                    fx.crd2idx(
                        (warp, channel_lane),
                        fx.make_layout((warp_num_per_block, 32), (32, 1)),
                    )
                )
            )
            for phase in range_constexpr(2 if split_v_exchange else 1):
                if const_expr(split_v_exchange and phase == 1):
                    first_tile = split_frame // 2
                    rank_tiles = 1
                for work in range(
                    bid, fx.Int32(heads * rank_tiles), fx.Int32(block_num)
                ):
                    fp6p_work_coord = fx.idx2crd(
                        fx.Int32(work),
                        fx.make_layout(
                            (UNBOUNDED_OUTER_EXTENT, rank_tiles), (rank_tiles, 1)
                        ),
                    )
                    head = fx.Int32(fx.get_scalar(fx.get(fp6p_work_coord, 0)))
                    tile_id = first_tile + fx.Int32(
                        fx.get_scalar(fx.get(fp6p_work_coord, 1))
                    )
                    fp6p_head_coord = fx.idx2crd(
                        head,
                        fx.make_layout(
                            (UNBOUNDED_OUTER_EXTENT, heads_local), (heads_local, 1)
                        ),
                    )
                    dest_pe = fx.Int32(fx.get_scalar(fx.get(fp6p_head_coord, 0)))
                    local_head = fx.Int32(fx.get_scalar(fx.get(fp6p_head_coord, 1)))
                    peer_base = fx.Uint64(fx.memref_load(p2p_bases_q, dest_pe))
                    output = _fp6_output_views(
                        wave_uniform_i64(peer_base),
                        num_records_bytes=heads_local * k_tiles * 12288 + 256,
                    )
                    scale_base = fx.Uint64(
                        buf_copy_load(scale_table, dest_pe, fx.Int64)
                    )
                    scale_output = ptr_buf_tensor(
                        wave_uniform_i64(scale_base),
                        fx.Int32,
                        num_records_bytes=heads_local * k_tiles * 512,
                    )
                    if const_expr(split_v_exchange):
                        partial_table = ptr_buf_tensor(addr_p2p_partial, fx.Int64)
                        partials = ptr_buf_tensor(
                            buf_copy_load(partial_table, dest_pe, fx.Int64), fx.Float32
                        )
                    for k in range_constexpr(2):
                        frame_start = _row_field_offset(
                            tile_id,
                            _row_field_offset(fx.Int32(k), fx.Int32(0), 64),
                            128,
                        )
                        owned = (frame_start < (rank + 1) * seq_len) & (
                            frame_start + 64 > rank * seq_len
                        )
                        if const_expr(rank == npes - 1):
                            owned = owned | (frame_start >= seq_full)
                        if const_expr(split_v_exchange):
                            owned = owned & (
                                (frame_start != split_frame * 64)
                                if phase == 0
                                else (frame_start == split_frame * 64)
                            )
                        if owned:
                            if const_expr(split_v_exchange and phase == 1):
                                wait_v_partial(dest_pe, local_head)
                            values = []
                            amax = fx.Float32(0.0)
                            for field in range_constexpr(32):
                                # The physical pairing and source-token swaps cancel.
                                token = fx.Int32(
                                    fx.get_scalar(
                                        fx.crd2idx(
                                            (half_wave, field),
                                            fx.make_layout(
                                                (2, (4, 4, 2)),
                                                (4, (1, 8, 32)),
                                            ),
                                        )
                                    )
                                )
                                token = frame_start + token
                                # FP6-P padding repeats the final token through the packed tile.
                                token = (token < seq_full).select(
                                    token, fx.Int32(seq_full - 1)
                                )
                                valid = (token >= rank * seq_len) & (
                                    token < (rank + 1) * seq_len
                                )
                                seq = valid.select(token - rank * seq_len, fx.Int32(0))
                                raw = buf_copy_load(
                                    input_q,
                                    input_offset(seq, head, channel),
                                    fx.BFloat16,
                                ).to(fx.Float32)
                                value = valid.select(raw, fx.Float32(0.0))
                                values.append(value)
                                amax = amax.maximumf(fmath.absf(value))
                            if const_expr(split_v_exchange):  # noqa: SIM102
                                if frame_start == split_frame * 64:
                                    # The paired sender is one source-rank slab after this slot.
                                    slot = paired_partial_offset(
                                        fx.Int32(0), local_head, half_wave, channel
                                    )
                                    paired_slot = paired_partial_offset(
                                        fx.Int32(1), local_head, half_wave, channel
                                    )
                                    amax = buf_copy_load(
                                        partials, slot, fx.Float32
                                    ).maximumf(
                                        buf_copy_load(
                                            partials,
                                            paired_slot,
                                            fx.Float32,
                                        )
                                    )
                            scale = _v4_fp6_scale(amax)
                            reciprocal = ((fx.Int32(254) - scale) << 23).bitcast(
                                fx.Float32
                            )
                            values = [value * reciprocal for value in values]
                            if const_expr(native_fp6):
                                # Native FP6 pack: software FP6-P V alone is 0.45% slower at ws8.
                                words = emit_f32_to_e2m3_native(
                                    fx.Vector.from_elements(values[::2], fx.Float32),
                                    fx.Vector.from_elements(values[1::2], fx.Float32),
                                    fx.Float32(1.0),
                                )
                            else:
                                codes = [
                                    fx.Int32(
                                        emit_f32_to_e2m3(fx.Float32(value).ir_value())
                                    )
                                    for value in values
                                ]
                                packed = []
                                for word in range_constexpr(6):
                                    bits = fx.Int32(0)
                                    for field in range_constexpr(32):
                                        shift = field * 6 - word * 32
                                        if const_expr(-6 < shift < 32):
                                            bits = bits | (
                                                codes[field] << shift
                                                if shift >= 0
                                                else codes[field] >> -shift
                                            )
                                    packed.append(bits)
                                words = fx.Vector.from_elements(packed, fx.Int32)
                            tile_row = fx.Int32(
                                fx.get_scalar(
                                    fx.crd2idx(
                                        (local_head, tile_id),
                                        fx.make_layout(
                                            (heads_local, k_tiles), (k_tiles, 1)
                                        ),
                                    )
                                )
                            )
                            frame_row = fx.Int32(
                                fx.get_scalar(
                                    fx.crd2idx(
                                        (warp, k),
                                        fx.make_layout((warp_num_per_block, 2), (2, 1)),
                                    )
                                )
                            )
                            # Keep the six-word lane span compact for the LDS redistribution.
                            first_word = _row_field_offset(
                                tile_row,
                                _row_field_offset(
                                    frame_row,
                                    _row_field_offset(lane, fx.Int32(0), 6),
                                    384,
                                ),
                                3072,
                            )
                            split_payload = fx.Int32(0) == 1
                            if const_expr(split_v_exchange):
                                split_payload = frame_start == split_frame * 64
                            if split_payload:
                                # Each sender owns exactly three dwords, never a shared byte.
                                split_row = fx.slice(
                                    fx.make_view(
                                        fx.get_iter(output[0]),
                                        fx.make_layout(
                                            (UNBOUNDED_OUTER_EXTENT, 6), (1, 1)
                                        ),
                                    ),
                                    (first_word, None),
                                )
                                split_fields = fx.logical_divide(
                                    split_row, fx.make_layout(3, 1)
                                )
                                sender_words = fx.slice(
                                    split_fields, (None, fx.Int32(rank % 2))
                                )
                                for word in range_constexpr(3):
                                    buf_copy_store(
                                        sender_words,
                                        word,
                                        words[rank % 2 * 3 + word],
                                        fx.Int32,
                                        1,
                                    )
                            else:
                                wave_word = _row_field_offset(
                                    tile_row,
                                    _row_field_offset(frame_row, fx.Int32(0), 384),
                                    3072,
                                )
                                store_dense_fp6(
                                    words,
                                    output,
                                    first_word,
                                    fx.Int32(0),
                                    wave_word,
                                    True,
                                    staging,
                                    tiles,
                                    warp,
                                    lane,
                                )
                            fx.memref_store(scale & 255, scale_image, (k, lane, warp))
                            fx.barrier()
                            write_scale = tid < 64
                            if const_expr(split_v_exchange and rank % 2 == 1):
                                write_scale = write_scale & (
                                    frame_start != split_frame * 64
                                )
                            if write_scale:
                                scale_word = fx.Vector.from_elements(
                                    [
                                        fx.memref_load(scale_image, (k, tid, byte)).to(
                                            fx.Uint8
                                        )
                                        for byte in range_constexpr(4)
                                    ],
                                    fx.Uint8,
                                ).bitcast(fx.Int32)[0]
                                scale_row = fx.slice(
                                    fx.make_view(
                                        fx.get_iter(scale_output),
                                        fx.make_layout(
                                            (heads_local, k_tiles, 128),
                                            (k_tiles * 128, 128, 1),
                                        ),
                                    ),
                                    (local_head, tile_id, None),
                                )
                                scale_frames = fx.logical_divide(
                                    scale_row, fx.make_layout(64, 1)
                                )
                                scale_frame = fx.slice(
                                    scale_frames, (None, fx.Int32(k))
                                )
                                buf_copy_store(scale_frame, tid, scale_word, fx.Int32)
                            fx.barrier()

        if const_expr(partial_only):
            # Publication must finish before any block starts packing split frames.
            publish_v_partial(shared)
        else:
            if const_expr(v4_output in ("q", "k") and codec in ("int8", "e4m3")):
                transport_v4_per_tensor_qk()
            elif const_expr(v4_output == "v" and codec == "e4m3_pc"):
                transport_v4_fp8_pc_v()
            elif const_expr(v4_output == "v" and codec == "e4m3"):
                transport_v4_fp8_v()
            elif const_expr(v4_output == "v" and codec == "mxfp6_p"):
                transport_v4_fp6_p_v(shared)
            elif const_expr(v4_output == "v"):
                transport_v4_v(shared)
            elif const_expr(quant and codec == "mxfp6" and native_fp6):
                transport_fp6_native(shared)
            else:
                transport(
                    input_q,
                    p2p_bases_q,
                    addr_p2p_scale,
                    codec,
                    hadamard,
                    v4_output,
                )

    return attention_a2a_single_push


def make_attention_a2a_completion_kernel(*, rank, npes):
    @flyc.kernel(known_block_size=[64, 1, 1])
    def completion(
        addr_xdb_mem: fx.Int64,
        addr_p2p_xdb_mem: fx.Int64,
        addr_xdb_flag: fx.Int64,
        publish: fx.Constexpr[bool],
    ):
        tid = fx.thread_idx.x
        flags = ptr_buf_tensor(addr_xdb_flag, fx.Int64, num_records_bytes=8)
        generation = fx.Int64(buf_copy_load(flags, 0, fx.Int64))
        if const_expr(publish):
            # The stream boundary drains every payload block before publication.
            if tid < npes:
                peers = ptr_buf_tensor(
                    addr_p2p_xdb_mem, fx.Int64, num_records_bytes=npes * 8
                )
                remote = fx.inttoptr(
                    fx.PointerType.get(
                        fx.Int64.ir_type,
                        address_space=fx.AddressSpace.Global,
                        alignment=8,
                    ),
                    buf_copy_load(peers, tid, fx.Int64),
                )
                fx.generic_store(
                    fx.add_offset(remote, rank),
                    generation,
                    memory_order=fx.AtomicOrdering.Release,
                    syncscope=fx.rocdl.SyncScope.OneAs,
                )
            fx.barrier()
            if tid == 0:
                fx.atomic_add(
                    fx.inttoptr(
                        fx.PointerType.get(
                            fx.Int64.ir_type,
                            address_space=fx.AddressSpace.Global,
                            alignment=8,
                        ),
                        addr_xdb_flag,
                    ),
                    fx.Int64(1),
                    ordering=fx.AtomicOrdering.Monotonic,
                    syncscope=fx.rocdl.SyncScope.OneAs,
                )
        else:
            if tid < npes:
                ready = fx.inttoptr(
                    fx.PointerType.get(
                        fx.Int64.ir_type,
                        address_space=fx.AddressSpace.Global,
                        alignment=8,
                    ),
                    addr_xdb_mem,
                )
                spin_until_ge_i64(
                    fx.ptrtoint(fx.add_offset(ready, tid)), generation - fx.Int64(1)
                )
                fx.memory_fence(
                    ordering=fx.AtomicOrdering.Acquire,
                    syncscope=fx.rocdl.SyncScope.OneAs,
                )
            fx.barrier()

    return completion


def make_attention_a2a_jit(
    *,
    rank,
    npes,
    heads,
    seq_len,
    head_dim,
    block_num,
    warp_num_per_block,
    quant=False,
    codec="e4m3",
    hadamard=False,
    v4_output="",
    v_pack=AttentionPack.DEFAULT,
    q_multiplier=1.0,
    v4_amax=False,
    fp8_fnuz=False,
):
    split_v_exchange = needs_split_v_exchange(v_pack, seq_len)
    pc_partial = v4_output == "v" and codec == "e4m3_pc" and v4_amax
    common = {
        "rank": rank,
        "npes": npes,
        "heads": heads,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "block_num": block_num,
        "warp_num_per_block": warp_num_per_block,
        "quant": quant,
        "codec": codec,
        "hadamard": hadamard,
        "v4_output": v4_output,
        "v_pack": v_pack,
        "q_multiplier": q_multiplier,
        "v4_amax": v4_amax,
        "fp8_fnuz": fp8_fnuz,
    }
    # The partial launch precedes the payload launch so no payload block waits
    # on a block of its own launch.
    payload_kernels = []
    if split_v_exchange or pc_partial:
        payload_kernels.append(
            make_attention_a2a_kernel(**common, partial_only=split_v_exchange)
        )
    payload_kernels.append(make_attention_a2a_kernel(**common, pc_reduce=pc_partial))
    completion = make_attention_a2a_completion_kernel(rank=rank, npes=npes)
    key = (
        rank,
        npes,
        heads,
        seq_len,
        head_dim,
        block_num,
        warp_num_per_block,
        quant,
        codec,
        hadamard,
        v4_output,
        v_pack,
        q_multiplier,
        v4_amax,
        fp8_fnuz,
        current_target().arch,
        os.environ.get("AITER_A2A_FP6_FORCE_SOFTWARE", "0") == "1",
        _JIT_CACHE_TAG,
    )

    @flyc.jit
    def launch(
        addr_input: fx.Int64,
        addr_p2p_output: fx.Int64,
        addr_p2p_scale: fx.Int64,
        addr_xdb_mem: fx.Int64,
        addr_p2p_xdb_mem: fx.Int64,
        addr_xdb_flag: fx.Int64,
        addr_p2p_partial: fx.Int64,
        addr_p2p_partial_ready: fx.Int64,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        _ = key
        for index in range_constexpr(len(payload_kernels)):
            payload_kernels[index](
                addr_input,
                addr_p2p_output,
                addr_p2p_scale,
                addr_xdb_flag,
                addr_p2p_partial,
                addr_p2p_partial_ready,
            ).launch(
                grid=(block_num, 1, 1),
                block=(warp_num_per_block * 64, 1, 1),
                stream=stream,
            )
        for phase in range_constexpr(2):
            completion(
                addr_xdb_mem, addr_p2p_xdb_mem, addr_xdb_flag, phase == 0
            ).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    return launch


def make_attention_a2a_dequant_jit(*, numel, codec="e4m3", fp8_fnuz=False):
    block_threads = 256
    vec = 8
    codecs = (codec,) * 3 if isinstance(codec, str) else codec
    key = (numel, codecs, fp8_fnuz, _JIT_CACHE_TAG)

    @flyc.jit
    def dequant_one(
        addr_payload: fx.Int64,
        addr_scale: fx.Int64,
        addr_out: fx.Int64,
        role_codec: fx.Constexpr[str],
    ):
        packing = 2 if role_codec == "mxfp4" else 1
        payload = ptr_buf_tensor(
            addr_payload,
            fx.Int32,
            unit_elems=1 if packing == 2 else 2,
            unit_stride=1,
            num_records_bytes=_transport_bytes(numel, role_codec),
        )
        scales = ptr_buf_tensor(addr_scale, fx.Int32, num_records_bytes=numel // 32)
        output = ptr_buf_tensor(
            addr_out,
            fx.Int32,
            unit_elems=4,
            unit_stride=1,
            num_records_bytes=numel * 2,
        )
        thread = fx.Int32(
            fx.get_scalar(
                fx.crd2idx(
                    (fx.Int32(fx.gpu.block_id("x")), fx.Int32(fx.gpu.thread_id("x"))),
                    fx.make_layout(
                        (ceildiv(numel, block_threads * vec), block_threads),
                        (block_threads, 1),
                    ),
                )
            )
        )
        offset = _row_field_offset(thread, fx.Int32(0), vec)
        if offset < numel:
            words = (
                _load_fp6(payload, thread)
                if const_expr(role_codec == "mxfp6")
                else (
                    fx.Vector.from_elements(
                        [buf_copy_load(payload, thread)],
                        fx.Int32,
                    )
                    if const_expr(packing == 2)
                    else fx.Vector(
                        buf_copy_load(
                            payload,
                            _row_field_offset(thread, fx.Int32(0), 2),
                            fx.Int32,
                            2,
                        )
                    )
                )
            )
            # Four neighboring lanes share a scale; four scales fit in one dword.
            scale_coord = fx.idx2crd(
                thread, fx.make_layout((UNBOUNDED_OUTER_EXTENT, 4, 4), (16, 4, 1))
            )
            scale_index = fx.Int32(fx.get_scalar(fx.get(scale_coord, 1)))
            scale_word = fx.Uint32(
                buf_copy_load(scales, fx.Int32(fx.get_scalar(fx.get(scale_coord, 0))))
            )
            exponent = (scale_word >> ((scale_index % 4) * 8)) & 255
            scale_bits = (exponent == 0).select(fx.Uint32(0x00400000), exponent << 23)
            scale_bits = (exponent == 255).select(fx.Uint32(0x7FC00000), scale_bits)
            scale = scale_bits.bitcast(fx.Float32)
            values = []
            for pair_index in range_constexpr(vec // 2):
                pair = (
                    _unpack_fp6_pair(words, pair_index)
                    if const_expr(role_codec == "mxfp6")
                    else _unpack_transport_pair(
                        words[pair_index // (2 * packing)],
                        pair_index % 4 if packing == 2 else bool(pair_index % 2),
                        role_codec,
                    )
                )
                values.append(pair[0] * scale)
                values.append(pair[1] * scale)
            bf16 = fx.Vector.from_elements(values, fx.Float32).to(fx.BFloat16)
            buf_copy_store(
                output,
                _row_field_offset(thread, fx.Int32(0), 4),
                bf16.bitcast(fx.Int32),
                fx.Int32,
                4,
            )

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def attention_a2a_dequant(
        addr_q: fx.Int64,
        addr_k: fx.Int64,
        addr_v: fx.Int64,
        addr_scale_q: fx.Int64,
        addr_scale_k: fx.Int64,
        addr_scale_v: fx.Int64,
        addr_out_q: fx.Int64,
        addr_out_k: fx.Int64,
        addr_out_v: fx.Int64,
    ):
        tensor_id = fx.gpu.block_id("y")
        for role in range_constexpr(3):
            if tensor_id == role:
                dequant_one(
                    (addr_q, addr_k, addr_v)[role],
                    (addr_scale_q, addr_scale_k, addr_scale_v)[role],
                    (addr_out_q, addr_out_k, addr_out_v)[role],
                    codecs[role],
                )

    @flyc.jit
    def launch(
        addr_q: fx.Int64,
        addr_k: fx.Int64,
        addr_v: fx.Int64,
        addr_scale_q: fx.Int64,
        addr_scale_k: fx.Int64,
        addr_scale_v: fx.Int64,
        addr_out_q: fx.Int64,
        addr_out_k: fx.Int64,
        addr_out_v: fx.Int64,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        _ = key
        attention_a2a_dequant(
            addr_q,
            addr_k,
            addr_v,
            addr_scale_q,
            addr_scale_k,
            addr_scale_v,
            addr_out_q,
            addr_out_k,
            addr_out_v,
        ).launch(
            grid=((numel + block_threads * vec - 1) // (block_threads * vec), 3, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch
