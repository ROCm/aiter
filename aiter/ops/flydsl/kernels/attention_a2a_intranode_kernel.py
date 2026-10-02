# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL single-tensor intranode push all-to-all kernel."""

from __future__ import annotations

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.backends import current_target
from flydsl.expr import T, const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Stream

from aiter.ops.mha_v4 import AttentionPack
from aiter.utility.mx_types import MxDtypeInt, MxScaleRoundModeInt

from .buffer_ops import buffer_load, buffer_store, create_buffer_resource_from_addr
from .communication_ops_utils import (
    atomic_add_global_at,
    fence_system_acquire,
    spin_until_ge_i64,
    store_i64_global_system,
    wave_uniform_i64,
)
from .kernels_common import ceildiv
from .quant_utils import (
    emit_f32_to_e2m1,
    emit_f32_to_e2m3,
    emit_f32_to_e2m3_native,
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
                remote_slot = (
                    fx.Int64(buf_copy_load(p2p_ready, tid, fx.Int64))
                    + fx.Int64(rank) * 8
                )
                # Reaching this launch drains earlier consumers on this stream.
                store_i64_global_system(remote_slot, generation)
            else:
                spin_until_ge_i64(addr_ready_mem + fx.Int64(tid) * 8, generation)
                fence_system_acquire()
        fx.barrier()
        # Nested rather than combined: const_expr must gate the runtime test.
        if const_expr(not publish):  # noqa: SIM102
            if tid == 0:
                atomic_add_global_at(addr_ready_flag, fx.Int64(1))

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
    word = fx.rocdl.cvt_pk_fp8_f32(
        T.i32, first.ir_value(), second.ir_value(), fx.Int32(0).ir_value(), 0
    )
    return fx.Int32(word).to(fx.Int16)


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
def _store_v4_fp6_q(words, resource, offset, lane):
    if lane % 4 < 2:
        for i in range_constexpr(3):
            buffer_store(words[i], resource, offset + i)


@flyc.jit
def _store_v4_fp6_k(words, scale, resource, tile_word, seq, lane):
    group = lane % 16 // 4
    token = seq % 128
    if lane % 4 < 2:
        for i in range_constexpr(3):
            word = lane % 2 * 3 + i
            offset = (word < 4).select(
                token // 32 * 512 + group * 128 + token % 32 * 4 + word,
                2048 + token // 32 * 256 + group * 64 + token % 32 * 2 + word - 4,
            )
            buffer_store(words[i], resource, tile_word + offset)
    if lane % 4 == 0:
        scale_slot = token % 32 // 16 * 256 + (token % 16 * 4 + token // 32) * 4
        buffer_store(scale, resource, tile_word * 4 + 16384 + scale_slot + group)
        if group > 0:
            buffer_store(
                scale, resource, tile_word * 4 + 16896 + scale_slot + group - 1
            )
        elif seq > 0:
            previous = (token + 127) % 128
            previous_slot = (
                previous % 32 // 16 * 256 + (previous % 16 * 4 + previous // 32) * 4
            )
            previous_tile = tile_word * 4 - (token == 0).select(
                fx.Int32(17408), fx.Int32(0)
            )
            buffer_store(scale, resource, previous_tile + 16896 + previous_slot + 3)


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
def _store_fp6(words, resource, chunk):
    if chunk % 2 == 0:
        for i in range_constexpr(3):
            buffer_store(words[i], resource, chunk // 2 * 3 + i)


def _load_fp6(payload, chunk):
    words = buf_copy_load(payload, chunk // 2 * 3 + chunk % 2, fx.Int32, 2)
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
    return fx.Vector(fx.rocdl.cvt_pk_f32_fp8(T.f32x2, word, high))


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

    def v4_word_offset(head, seq, chunk, mode, codec):
        if codec == "mxfp8":
            return (seq * heads_local + head) * 32 + chunk * 2
        if codec == "mxfp6" and mode == "k":
            return (head * k_tiles + seq // 128) * 4352
        if codec == "mxfp6":
            return (seq * heads_local + head) * 24 + (chunk // 4) * 6 + (chunk % 2) * 3
        if mode == "k":
            return (
                (head * k_tiles + seq // 128) * 2048
                + (chunk // 4) * 512
                + (seq % 128) * 4
                + chunk % 4
            )
        return (seq * heads_local + head) * 16 + chunk

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
        lane = tid & 63
        warp = tid >> 6
        global_warp_id = bid * warp_num_per_block + warp
        global_warp_num = block_num * warp_num_per_block

        p2p_output_q = ptr_buf_tensor(addr_p2p_output, fx.Int64)

        if const_expr(split_v_exchange):
            rsrc_xdb_flag = create_buffer_resource_from_addr(addr_xdb_flag)
            partial_generation = buffer_load(rsrc_xdb_flag, 0, vec_width=1, dtype=T.i64)

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
            dest_pe = global_warp_id % npes
            peer_warp_id = global_warp_id // npes
            peer_base = fx.Uint64(fx.memref_load(p2p_bases, dest_pe))
            uniform_peer_base = wave_uniform_i64(peer_base)
            rsrc_dst = create_buffer_resource_from_addr(
                uniform_peer_base,
                num_records_bytes=(
                    heads_local * k_tiles * (17408 if codec == "mxfp6" else 8192)
                    if mode == "k" and codec != "mxfp8"
                    else total_chunks * wire_bytes
                ),
            )
            if const_expr(quant):
                p2p_scale = ptr_buf_tensor(addr_p2p_scale, fx.Int64)
                scale_base = fx.Uint64(buf_copy_load(p2p_scale, dest_pe, fx.Int64))
                uniform_scale_base = wave_uniform_i64(scale_base)
                rsrc_scale = create_buffer_resource_from_addr(
                    uniform_scale_base,
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
                    group_idx = group_base + batch_idx * peer_warp_num
                    dest_chunk = group_idx * 64 + lane
                    valid = dest_chunk < peer_chunks
                    safe_dest_chunk = valid.select(dest_chunk, 0)
                    local_head = safe_dest_chunk // (seq_len * chunks_per_row)
                    seq_chunk = safe_dest_chunk % (seq_len * chunks_per_row)
                    seq = seq_chunk // chunks_per_row
                    row_chunk = seq_chunk % chunks_per_row
                    head = dest_pe * heads_local + local_head
                    src_chunk = (seq * heads + head) * chunks_per_row + row_chunk
                    raw = buf_copy_load(
                        input_q, src_chunk * chunk_words, fx.Int32, chunk_words
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
                    dst_chunk = (
                        local_head * seq_full * chunks_per_row
                        + (rank * seq_len + seq) * chunks_per_row
                        + row_chunk
                    )
                    destinations.append(
                        v4_word_offset(
                            local_head, rank * seq_len + seq, row_chunk, mode, codec
                        )
                        if mode
                        else (
                            dst_chunk
                            if quant and codec == "mxfp6"
                            else dst_chunk * wire_words
                        )
                    )
                    scale_destinations.append(
                        ((rank * seq_len + seq) * heads_local + local_head) * 4
                        + row_chunk // 4
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
                                rsrc_dst,
                                destinations[batch_idx],
                                scale_destinations[batch_idx] // (heads_local * 4),
                                lane,
                            )
                        elif const_expr(mode and codec == "mxfp6"):
                            _store_v4_fp6_q(
                                values[batch_idx],
                                rsrc_dst,
                                destinations[batch_idx],
                                lane,
                            )
                        elif const_expr(quant and codec == "mxfp6"):
                            _store_fp6(
                                values[batch_idx], rsrc_dst, destinations[batch_idx]
                            )
                        else:
                            buffer_store(
                                values[batch_idx], rsrc_dst, destinations[batch_idx]
                            )
                        # Keep compile-time specialization outside the lane predicate.
                        if const_expr(quant):  # noqa: SIM102
                            if lane % 4 == 0:
                                buffer_store(
                                    scales[batch_idx],
                                    rsrc_scale,
                                    scale_destinations[batch_idx],
                                )

        @flyc.jit
        def store_dense_fp6(
            words, output, first_word, coalesced, staging, tiles, warp, lane
        ):
            if coalesced:
                # Each wave owns its staging slab; no cross-wave barrier is needed.
                for word in range_constexpr(6):
                    fx.memref_store(words[word], staging, warp * 384 + lane * 6 + word)
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                wave_word = first_word - lane * 6
                first = fx.memref_load_vec(fx.slice(tiles, (None, warp * 96 + lane)))
                buf_copy_store(output[2], wave_word + lane * 4, first, fx.Int32, 4)
                if lane < 32:
                    last = fx.memref_load_vec(
                        fx.slice(tiles, (None, warp * 96 + 64 + lane))
                    )
                    buf_copy_store(
                        output[2], wave_word + 256 + lane * 4, last, fx.Int32, 4
                    )
                fx.rocdl.s_waitcnt(lgkmcnt=0)
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
                    first_word + 4,
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
            source = (
                (seq * heads + dest_pe * heads_local + head) * head_dim + channel * 32
            ) // 2
            vals = []
            for part in range_constexpr(4):
                raw = buf_copy_load(input_q, source + part * 4, fx.Int32, 4)
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
                        (rank * seq_len + seq) * heads_local * groups_per_row
                        + head * groups_per_row
                        + channel,
                        scale.to(fx.Int8),
                        fx.Int8,
                    )
            else:
                reciprocal = ((fx.Int32(254) - scale) << 23).bitcast(fx.Float32)
                # Round the reciprocal product to FP32 before FP6 conversion, including overflow.
                vals = [value * reciprocal for value in vals]
                words = emit_f32_to_e2m3_native(
                    fx.Vector.from_elements(
                        vals[:16] if v4_output else vals[::2], fx.Float32
                    ),
                    fx.Vector.from_elements(
                        vals[16:] if v4_output else vals[1::2], fx.Float32
                    ),
                    fx.Float32(1.0),
                )
                full_seq = rank * seq_len + seq
                destination = (
                    (full_seq * heads_local + head) * groups_per_row + channel
                    if v4_output
                    else (head * seq_full + full_seq) * groups_per_row + channel
                )
                first_word = destination * 6
                last_word = first_word + 4
                if const_expr(v4_output == "k"):
                    tile_word = (head * k_tiles + full_seq // 128) * 4352
                    token = full_seq % 128
                    first_word = (
                        tile_word + token // 32 * 512 + channel * 128 + token % 32 * 4
                    )
                    last_word = (
                        tile_word
                        + 2048
                        + token // 32 * 256
                        + channel * 64
                        + token % 32 * 2
                    )
                if valid:
                    if const_expr(v4_output in ("", "q")):
                        wave_start = group * 64
                        coalesced = wave_start + 63 < peer_groups
                        if const_expr(not v4_output):
                            # A head boundary skips the other source ranks' sequence slabs.
                            coalesced = coalesced & (
                                wave_start // (seq_len * groups_per_row)
                                == (wave_start + 63) // (seq_len * groups_per_row)
                            )
                        staging = shared.fp6_words.view(
                            fx.make_layout(warp_num_per_block * 384, 1)
                        )
                        tiles = shared.fp6_words.view(
                            fx.make_layout((4, warp_num_per_block * 96), (1, 4))
                        )
                        store_dense_fp6(
                            words,
                            output,
                            first_word,
                            coalesced,
                            staging,
                            tiles,
                            warp,
                            lane,
                        )
                    else:
                        buf_copy_store(
                            output[2],
                            first_word,
                            fx.Vector.from_elements(
                                [words[i] for i in range(4)], fx.Int32
                            ),
                            fx.Int32,
                            4,
                        )
                        buf_copy_store(
                            output[1],
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
                        scale_slot = (
                            token % 32 // 16 * 256 + (token % 16 * 4 + token // 32) * 4
                        )
                        previous = (token + 127) % 128
                        previous_slot = (
                            previous % 32 // 16 * 256
                            + (previous % 16 * 4 + previous // 32) * 4
                        )
                        previous_tile = tile_word * 4 - (token == 0).select(
                            fx.Int32(17408), fx.Int32(0)
                        )
                        if const_expr(staged):
                            tail_view = shared.k_tail.view(
                                fx.make_layout(warp_num_per_block * 1024, 1)
                            )
                            tail_base = warp * 1024
                            fx.memref_store(
                                scale.to(fx.Int8),
                                tail_view,
                                tail_base + scale_slot + channel,
                            )
                            if channel > 0:
                                fx.memref_store(
                                    scale.to(fx.Int8),
                                    tail_view,
                                    tail_base + 512 + scale_slot + channel - 1,
                                )
                            elif token > 0:
                                fx.memref_store(
                                    scale.to(fx.Int8),
                                    tail_view,
                                    tail_base + 512 + previous_slot + 3,
                                )
                            elif full_seq > 0:
                                # The previous tile's last B byte may belong to another wave.
                                buf_copy_store(
                                    output[3],
                                    previous_tile + 16896 + previous_slot + 3,
                                    scale.to(fx.Int8),
                                    fx.Int8,
                                    1,
                                )
                        else:
                            buf_copy_store(
                                output[3],
                                tile_word * 4 + 16384 + scale_slot + channel,
                                scale.to(fx.Int8),
                                fx.Int8,
                                1,
                            )
                            if channel > 0:
                                buf_copy_store(
                                    output[3],
                                    tile_word * 4 + 16896 + scale_slot + channel - 1,
                                    scale.to(fx.Int8),
                                    fx.Int8,
                                    1,
                                )
                            elif full_seq > 0:
                                buf_copy_store(
                                    output[3],
                                    previous_tile + 16896 + previous_slot + 3,
                                    scale.to(fx.Int8),
                                    fx.Int8,
                                    1,
                                )

        @flyc.jit
        def transport_fp6_native(shared):
            dest_pe = global_warp_id % npes
            peer_warp = global_warp_id // npes
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
                    head = item // k_own_tiles
                    tile = k_first_tile + item % k_own_tiles
                    tile_seq = tile * 128 - rank * seq_len
                    for sub in range(fx.Int32(0), fx.Int32(8), fx.Int32(1)):
                        emit_fp6_group(
                            shared,
                            output,
                            scale_output,
                            fx.Int32(0),
                            peer_groups,
                            dest_pe,
                            head,
                            tile_seq + sub * 16 + lane // 4,
                            lane % 4,
                            lane >= 0,
                            True,
                        )
                    fx.rocdl.s_waitcnt(lgkmcnt=0)
                    tail_view = shared.k_tail.view(
                        fx.make_layout(warp_num_per_block * 1024, 1)
                    )
                    lines = shared.k_tail.view(
                        fx.make_layout((16, warp_num_per_block * 64), (1, 16))
                    )
                    tile_word = (head * k_tiles + tile) * 4352
                    # Lines 0-31 are image A, 32-62 the whole lines of B. B's last
                    # byte is written by the next tile's first token, so line 63
                    # keeps byte stores for its 15 bytes owned here.
                    if lane < 63:
                        packed = fx.Vector(
                            fx.memref_load_vec(
                                fx.slice(lines, (None, warp * 64 + lane))
                            )
                        ).bitcast(fx.Int32)
                        buf_copy_store(
                            output[2], tile_word + 4096 + lane * 4, packed, fx.Int32, 4
                        )
                    if lane < 15:
                        byte = fx.memref_load(tail_view, warp * 1024 + 1008 + lane)
                        buf_copy_store(
                            output[3], tile_word * 4 + 17392 + lane, byte, fx.Int8, 1
                        )
                    fx.rocdl.s_waitcnt(lgkmcnt=0)
            for group in range(peer_warp, (peer_groups + 63) // 64, peer_warps):
                index = group * 64 + lane
                valid = index < peer_groups
                safe = valid.select(index, 0)
                if const_expr(v4_output == "q"):
                    # Token-major lanes make each wave's packed Q destination one
                    # contiguous slab, so the LDS-coalesced store applies.
                    seq = safe // (heads_local * groups_per_row)
                    head = safe // groups_per_row % heads_local
                else:
                    head = safe // (seq_len * groups_per_row)
                    seq = safe // groups_per_row % seq_len
                channel = safe % groups_per_row
                if const_expr(v4_output == "k" and k_own_tiles > 0):
                    own_lo = k_first_tile * 128 - rank * seq_len
                    own_hi = k_last_tile * 128 - rank * seq_len
                    # Whole tiles have a wave owner; this loop writes only boundary tokens.
                    valid = valid & ((seq < own_lo) | (seq >= own_hi))
                selected = fx.Int32(1) == 1
                if const_expr(v4_output == "k" and k_own_tiles > 0):
                    first = group * 64
                    last = first + 63
                    first_seq = first // groups_per_row % seq_len
                    last_seq = last // groups_per_row % seq_len
                    whole = (
                        (
                            first // (seq_len * groups_per_row)
                            == last // (seq_len * groups_per_row)
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
                    index = group * 64 + lane
                    valid = index < peer_groups
                    safe = valid.select(index, 0)
                    emit_fp6_group(
                        shared,
                        output,
                        scale_output,
                        group,
                        peer_groups,
                        dest_pe,
                        safe // groups_per_row % heads_local,
                        safe // (heads_local * groups_per_row),
                        safe % groups_per_row,
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
                    1 + dest_pe * global_warp_num + fx.Int32(part),
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
                slot = 1 + dest_pe * global_warp_num + rank * peer_warps + peer_warp
                for peer in range_constexpr(npes):
                    base = buf_copy_load(scale_table, peer, fx.Int64)
                    buffer_store(maximum, create_buffer_resource_from_addr(base), slot)

        @flyc.jit
        def transport_v4_per_tensor_qk():
            dest_pe = global_warp_id % npes
            peer_warp = global_warp_id // npes
            peer_warps = global_warp_num // npes
            scale_table = ptr_buf_tensor(addr_p2p_scale, fx.Int64)
            local_scale_base = buf_copy_load(scale_table, rank, fx.Int64)
            local_scales = ptr_buf_tensor(local_scale_base, fx.Float32)
            scale = fx.Float32(1.0)
            if const_expr(not v4_amax):
                maximum = reduce_partial_amax(local_scales, dest_pe)
                scale = maximum / fx.Float32(
                    127.0 if codec == "int8" else (240.0 if fp8_fnuz else 448.0)
                )
                scale = (scale > 0.0).select(scale, fx.Float32(1.0))
                if (dest_pe == rank) & (peer_warp == 0) & (lane == 0):
                    buf_copy_store(local_scales, 0, scale, fx.Float32)

            base = fx.Uint64(fx.memref_load(p2p_bases_q, dest_pe))
            output = create_buffer_resource_from_addr(wave_uniform_i64(base))
            maximum = fx.Float32(0.0)
            # Each 16-lane group owns one (row, destination head) pair, so every lane loads
            # an owned head and the Hadamard lane groups stay head-aligned.
            groups = 64 * vec // head_dim
            rows = 8
            loads = rows * heads_local // groups
            group_lane = lane % (head_dim // vec)
            pair_rows = []
            pair_heads = []
            for load in range_constexpr(loads):
                pair = lane // (head_dim // vec) + load * groups
                pair_rows.append(pair // heads_local)
                pair_heads.append(pair % heads_local)
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
                # sched_barrier(0) keeps the compiler from sinking loads into compute.
                for load in range_constexpr(loads):
                    seq = seq0 + pair_rows[load] * peer_warps
                    valid = seq < seq_len
                    seq = valid.select(seq, fx.Int32(seq_len - 1))
                    seqs.append((seq, valid))
                    tiles.append(
                        fx.Vector(
                            buf_copy_load(
                                input_q,
                                seq * hd
                                + (dest_pe * heads_local + pair_heads[load]) * head_dim
                                + group_lane * vec,
                                elem=fx.BFloat16,
                                unit_elems=vec,
                            )
                        ).to(fx.Float32)
                    )
                fx.rocdl.sched_barrier(0)
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
                            destination = (
                                (rank * seq_len + seq) * heads_local + pair_heads[load]
                            ) * 16 + group_lane
                            buffer_store(
                                _pack_transport_words(pairs, codec),
                                output,
                                destination * 2,
                            )
                result = yield [current]
            if const_expr(v4_amax):
                broadcast_partial_amax(
                    fx.Float32(result), scale_table, dest_pe, peer_warp, peer_warps
                )

        @flyc.jit
        def transport_v4_fp8_pc_v():
            dest_pe = global_warp_id % npes
            peer_warp = global_warp_id // npes
            peer_warps = global_warp_num // npes
            channels = heads_local * head_dim
            scale_table = ptr_buf_tensor(addr_p2p_scale, fx.Int64)
            local_scale_base = buf_copy_load(scale_table, rank, fx.Int64)
            local_scales = ptr_buf_tensor(local_scale_base, fx.Float32)
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
                peer_warp * 64 + lane,
                fx.Int32(work_items),
                fx.Int32(peer_warps * 64),
            ):
                item = fx.Int32(item)
                channel = (item % channel_chunks) * 8
                tile = item // channel_chunks
                seq_begin = tile * token_tile
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
                            offset = (dest_pe * token_tiles + fx.Int32(part)) * channels
                            combined = fx.Vector.from_elements(
                                [
                                    fx.Float32(
                                        buf_copy_load(
                                            partials, offset + channel + i, fx.Float32
                                        )
                                    )
                                    for i in range(8)
                                ],
                                fx.Float32,
                            )
                            result = yield [state[0].maximumf(combined)]
                        maximum = fx.Vector(result)
                        slot = channels + (dest_pe * npes + rank) * channels + channel
                        for peer in range_constexpr(npes):
                            remote = buf_copy_load(scale_table, peer, fx.Int64)
                            resource = ptr_buf_tensor(remote, fx.Float32)
                            for i in range_constexpr(8):
                                buf_copy_store(
                                    resource, slot + i, maximum[i], fx.Float32
                                )
                    else:
                        # Partial producers never wait: each tile publishes one
                        # local maximum per channel before the cross-rank handshake.
                        for seq, state in range(
                            seq_begin,
                            seq_end,
                            fx.Int32(1),
                            init=[maximum],
                        ):
                            source = fx.Int32(seq) * hd + dest_pe * channels + channel
                            values = fx.Vector(
                                buf_copy_load(input_q, source, fx.BFloat16, 8)
                            ).to(fx.Float32)
                            result = yield [state[0].maximumf(fmath.absf(values))]
                        maximum = fx.Vector(result)
                        offset = (dest_pe * token_tiles + tile) * channels + channel
                        for i in range_constexpr(8):
                            buf_copy_store(partials, offset + i, maximum[i], fx.Float32)
                else:
                    scales = []
                    for i in range_constexpr(8):
                        maximum = fx.Float32(0.0)
                        for peer in range_constexpr(npes):
                            slot = channels + (dest_pe * npes + peer) * channels
                            partial = buf_copy_load(
                                local_scales, slot + channel + i, fx.Float32
                            )
                            maximum = maximum.maximumf(partial)
                        # The per-channel descale uses a rounded FP32 reciprocal multiply.
                        scale = maximum * fx.Float32(
                            1.0 / (240.0 if fp8_fnuz else 448.0)
                        )
                        scales.append(scale)
                        if (dest_pe == rank) & (tile == 0):
                            buf_copy_store(local_scales, channel + i, scale, fx.Float32)
                    for seq in range(
                        seq_begin,
                        seq_end,
                        fx.Int32(1),
                    ):
                        source = fx.Int32(seq) * hd + dest_pe * channels + channel
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
                        destination = (
                            rank * seq_len + fx.Int32(seq)
                        ) * channels + channel
                        buf_copy_store(
                            output,
                            destination // 4,
                            _pack_transport_words(pairs, "e4m3"),
                            fx.Int32,
                            2,
                        )

        @flyc.jit
        def transport_v4_fp8_v():
            # Every source owns a token shard of every destination's tensor.
            # Replicate only warp amax metadata so quantization stays send-side.
            dest_pe = global_warp_id % npes
            peer_warp = global_warp_id // npes
            peer_warps = global_warp_num // npes
            peer_chunks = total_chunks // npes
            scale_table = ptr_buf_tensor(addr_p2p_scale, fx.Int64)
            local_scale_base = buf_copy_load(scale_table, rank, fx.Int64)
            local_scales = ptr_buf_tensor(local_scale_base, fx.Float32)
            if const_expr(v4_amax):
                maximum = fx.Float32(0.0)
                for chunk, state in range(
                    peer_warp * 64 + lane,
                    fx.Int32(peer_chunks),
                    fx.Int32(peer_warps * 64),
                    init=[maximum],
                ):
                    chunk = fx.Int32(chunk)
                    seq = chunk // (heads_local * 16)
                    head_chunk = chunk % (heads_local * 16)
                    source = (seq * heads + dest_pe * heads_local) * 16 + head_chunk
                    values = fx.Vector(
                        buf_copy_load(input_q, source * 8, fx.BFloat16, 8)
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
                output = create_buffer_resource_from_addr(wave_uniform_i64(base))
                for chunk in range(
                    peer_warp * 64 + lane,
                    fx.Int32(peer_chunks),
                    fx.Int32(peer_warps * 64),
                ):
                    chunk = fx.Int32(chunk)
                    seq = chunk // (heads_local * 16)
                    head_chunk = chunk % (heads_local * 16)
                    source = (seq * heads + dest_pe * heads_local) * 16 + head_chunk
                    values = fx.Vector(
                        buf_copy_load(input_q, source * 8, fx.BFloat16, 8)
                    ).to(fx.Float32)
                    pairs = [
                        _pack_transport_pair(
                            values[2 * i] * reciprocal,
                            values[2 * i + 1] * reciprocal,
                            "e4m3",
                        )
                        for i in range(4)
                    ]
                    destination = rank * peer_chunks + chunk
                    buffer_store(
                        _pack_transport_words(pairs, "e4m3"), output, destination * 2
                    )

        @flyc.jit
        def wait_v_partial(dest_pe, local_head):
            ready_table = ptr_buf_tensor(addr_p2p_partial_ready, fx.Int64)
            ready_base = buf_copy_load(ready_table, dest_pe, fx.Int64)
            # Local and remote publishers finish in an earlier launch.
            if lane < 2:
                slot = (rank // 2 * 2 + lane) * heads_local + local_head
                spin_until_ge_i64(ready_base + fx.Int64(slot) * 8, partial_generation)
                fence_system_acquire()
            fx.barrier()

        @flyc.jit
        def load_v4_v_amax(shared, head, global_start):
            scratch = shared.v_amax.view(fx.make_layout(512, 1))
            channel_chunk = tid % 16
            token_lane = tid // 16
            values = []
            for half in range_constexpr(2):
                token = global_start + token_lane + half * 16
                if const_expr(v_pack == AttentionPack.V_FOR_FP6_P):
                    token = (
                        (token & ~0x24) | ((token & 0x04) << 3) | ((token & 0x20) >> 3)
                    )
                valid = (token >= rank * seq_len) & (token < (rank + 1) * seq_len)
                seq = valid.select(token - rank * seq_len, 0)
                raw = fx.Vector(
                    buf_copy_load(
                        input_q,
                        (seq * heads + head) * 128 + channel_chunk * 8,
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
            for i in range_constexpr(8):
                amax = fmath.absf(values[0][i]).maximumf(fmath.absf(values[1][i]))
                for shift in (16, 32):
                    amax = amax.maximumf(amax.shuffle_xor(shift, 64))
                if lane < 16:
                    fx.memref_store(
                        amax,
                        scratch,
                        warp * 128 + channel_chunk * 8 + i,
                    )
            fx.barrier()
            maxima = []
            for i in range_constexpr(8):
                amax = fx.Float32(0.0)
                for source_wave in range_constexpr(4):
                    amax = amax.maximumf(
                        fx.memref_load(
                            scratch,
                            source_wave * 128 + channel_chunk * 8 + i,
                        )
                    )
                maxima.append(amax)
            return values, fx.Vector.from_elements(maxima, fx.Float32)

        @flyc.jit
        def publish_v_partial(shared):
            channel_chunk = tid % 16
            token_lane = tid // 16
            partial_table = ptr_buf_tensor(addr_p2p_partial, fx.Int64)
            for head in range(bid, fx.Int32(heads), fx.Int32(block_num)):
                dest_pe = head // heads_local
                local_head = head % heads_local
                partial_base = buf_copy_load(partial_table, dest_pe, fx.Int64)
                partials = ptr_buf_tensor(partial_base, fx.Float32)
                for quarter in range_constexpr(2):
                    _values, maxima = load_v4_v_amax(
                        shared, head, split_frame * 64 + quarter * 32
                    )
                    if token_lane == 0:
                        for i in range_constexpr(8):
                            slot = (rank * heads_local + local_head) * 256
                            buf_copy_store(
                                partials,
                                slot + quarter * 128 + channel_chunk * 8 + i,
                                maxima[i],
                                fx.Float32,
                            )
                    fx.barrier()
                fx.rocdl.s_waitcnt(vmcnt=0)
                fx.barrier()
                if tid == 0:
                    ready_table = ptr_buf_tensor(addr_p2p_partial_ready, fx.Int64)
                    ready_base = buf_copy_load(ready_table, dest_pe, fx.Int64)
                    store_i64_global_system(
                        ready_base + fx.Int64(rank * heads_local + local_head) * 8,
                        partial_generation,
                    )

        @flyc.jit
        def transport_v4_v(shared):
            # FP6-P groups share an amax across a 32-mod-64 sender boundary.
            first_tile = rank * seq_len // 128
            rank_tiles = ((rank + 1) * seq_len + 127) // 128 - first_tile
            scale_table = ptr_buf_tensor(addr_p2p_scale, fx.Int64)
            channel_chunk = tid % 16
            token_lane = tid // 16
            for phase in range_constexpr(2 if split_v_exchange else 1):
                if const_expr(split_v_exchange and phase == 1):
                    first_tile = split_frame // 2
                    rank_tiles = 1
                for work in range(
                    bid, fx.Int32(heads * rank_tiles), fx.Int32(block_num)
                ):
                    head = work // rank_tiles
                    tile_id = first_tile + work % rank_tiles
                    dest_pe = head // heads_local
                    local_head = head % heads_local
                    peer_base = fx.Uint64(fx.memref_load(p2p_bases_q, dest_pe))
                    rsrc_dst = create_buffer_resource_from_addr(
                        wave_uniform_i64(peer_base),
                        num_records_bytes=heads_local * k_tiles * 8192 + 64,
                    )
                    scale_base = fx.Uint64(
                        buf_copy_load(scale_table, dest_pe, fx.Int64)
                    )
                    rsrc_scale = create_buffer_resource_from_addr(
                        wave_uniform_i64(scale_base),
                        num_records_bytes=heads_local * k_tiles * 512,
                    )
                    if const_expr(split_v_exchange):
                        partial_table = ptr_buf_tensor(addr_p2p_partial, fx.Int64)
                        # Both senders publish to the destination; packers read its
                        # two slots only after the publication release/acquire.
                        partial_base = buf_copy_load(partial_table, dest_pe, fx.Int64)
                        partials = ptr_buf_tensor(partial_base, fx.Float32)
                    for quarter in range_constexpr(4):
                        global_start = tile_id * 128 + quarter * 32
                        owned = (global_start >= rank * seq_len) & (
                            global_start
                            < (
                                (rank + 1) * seq_len
                                if rank != npes - 1
                                else k_tiles * 128
                            )
                        )
                        if const_expr(split_v_exchange):
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
                            for i in range_constexpr(8):
                                amax = maxima[i]
                                if const_expr(split_v_exchange):
                                    slot = (
                                        ((rank // 2 * 2) * heads_local + local_head)
                                        * 256
                                        + (quarter % 2) * 128
                                        + channel_chunk * 8
                                        + i
                                    )
                                    if frame_start == split_frame * 64:
                                        amax = buf_copy_load(
                                            partials, slot, fx.Float32
                                        ).maximumf(
                                            buf_copy_load(
                                                partials,
                                                slot + heads_local * 256,
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
                                    scale_offset = (
                                        local_head * k_tiles + tile_id
                                    ) * 512 + quarter * 128
                                    scale_offset = (
                                        scale_offset
                                        + (channel % 32 // 2) * 8
                                        + channel // 32
                                        + (channel % 2) * 4
                                    )
                                    buffer_store(
                                        scale.to(fx.Int8), rsrc_scale, scale_offset
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
                                column = (
                                    (quarter % 2) * 32
                                    + 4 * (token // 8)
                                    + 16 * ((token // 4) % 2)
                                    + token % 4
                                )
                                word_offset = (local_head * k_tiles + tile_id) * 2048
                                word_offset = (
                                    word_offset
                                    + (2 * (channel_chunk // 4) + quarter // 2) * 256
                                    + column * 4
                                    + channel_chunk % 4
                                )
                                source_token = global_start + token
                                if const_expr(v_pack == AttentionPack.V_FOR_FP6_P):
                                    source_token = (
                                        (source_token & ~0x24)
                                        | ((source_token & 0x04) << 3)
                                        | ((source_token & 0x20) >> 3)
                                    )
                                write_payload = (source_token >= rank * seq_len) & (
                                    source_token < (rank + 1) * seq_len
                                )
                                if const_expr(rank == npes - 1):
                                    write_payload = write_payload | (
                                        source_token >= seq_full
                                    )
                                if write_payload:
                                    buffer_store(
                                        _pack_transport_words(pairs, "mxfp4"),
                                        rsrc_dst,
                                        word_offset,
                                    )
                            # All waves finish reading the reduction before its next use.
                            fx.barrier()

        @flyc.jit
        def transport_v4_fp6_p_v(shared):
            first_tile = rank * seq_len // 128
            rank_tiles = ((rank + 1) * seq_len + 127) // 128 - first_tile
            staging = shared.fp6_words.view(fx.make_layout(warp_num_per_block * 384, 1))
            tiles = shared.fp6_words.view(
                fx.make_layout((4, warp_num_per_block * 96), (1, 4))
            )
            scale_image = shared.v_scales.view(fx.make_layout(512, 1))
            scale_table = ptr_buf_tensor(addr_p2p_scale, fx.Int64)
            channel = warp * 32 + lane % 32
            for phase in range_constexpr(2 if split_v_exchange else 1):
                if const_expr(split_v_exchange and phase == 1):
                    first_tile = split_frame // 2
                    rank_tiles = 1
                for work in range(
                    bid, fx.Int32(heads * rank_tiles), fx.Int32(block_num)
                ):
                    head = work // rank_tiles
                    tile_id = first_tile + work % rank_tiles
                    dest_pe = head // heads_local
                    local_head = head % heads_local
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
                        frame_start = tile_id * 128 + k * 64
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
                                physical = lane // 32 * 32 + field
                                paired = (
                                    (physical & 15)
                                    | ((physical & 16) << 1)
                                    | ((physical & 32) >> 1)
                                )
                                byte = paired % 32
                                token = (
                                    32 * (byte // 16)
                                    + 8 * ((byte % 16) // 4)
                                    + byte % 4
                                    + 4 * (paired // 32)
                                )
                                token = frame_start + (
                                    (token & ~0x24)
                                    | ((token & 4) << 3)
                                    | ((token & 32) >> 3)
                                )
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
                                    (seq * heads + head) * 128 + channel,
                                    fx.BFloat16,
                                ).to(fx.Float32)
                                value = valid.select(raw, fx.Float32(0.0))
                                values.append(value)
                                amax = amax.maximumf(fmath.absf(value))
                            if const_expr(split_v_exchange):  # noqa: SIM102
                                if frame_start == split_frame * 64:
                                    slot = (
                                        ((rank // 2 * 2) * heads_local + local_head)
                                        * 256
                                        + (lane // 32) * 128
                                        + channel
                                    )
                                    amax = buf_copy_load(
                                        partials, slot, fx.Float32
                                    ).maximumf(
                                        buf_copy_load(
                                            partials,
                                            slot + heads_local * 256,
                                            fx.Float32,
                                        )
                                    )
                            scale = _v4_fp6_scale(amax)
                            reciprocal = ((fx.Int32(254) - scale) << 23).bitcast(
                                fx.Float32
                            )
                            values = [value * reciprocal for value in values]
                            if const_expr(native_fp6):
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
                            first_word = (
                                (local_head * k_tiles + tile_id) * 3072
                                + (warp * 2 + k) * 384
                                + lane * 6
                            )
                            split_payload = fx.Int32(0) == 1
                            if const_expr(split_v_exchange):
                                split_payload = frame_start == split_frame * 64
                            if split_payload:
                                # Each sender owns exactly three dwords, never a shared byte.
                                for word in range_constexpr(3):
                                    buf_copy_store(
                                        output[0],
                                        first_word + rank % 2 * 3 + word,
                                        words[rank % 2 * 3 + word],
                                        fx.Int32,
                                        1,
                                    )
                            else:
                                store_dense_fp6(
                                    words,
                                    output,
                                    first_word,
                                    True,
                                    staging,
                                    tiles,
                                    warp,
                                    lane,
                                )
                            fx.memref_store(
                                scale & 255, scale_image, k * 256 + lane * 4 + warp
                            )
                            fx.barrier()
                            write_scale = tid < 64
                            if const_expr(split_v_exchange and rank % 2 == 1):
                                write_scale = write_scale & (
                                    frame_start != split_frame * 64
                                )
                            if write_scale:
                                scale_word = fx.Int32(0)
                                for byte in range_constexpr(4):
                                    scale_word = scale_word | (
                                        fx.memref_load(
                                            scale_image, k * 256 + tid * 4 + byte
                                        )
                                        << (byte * 8)
                                    )
                                buf_copy_store(
                                    scale_output,
                                    (local_head * k_tiles + tile_id) * 128
                                    + k * 64
                                    + tid,
                                    scale_word,
                                    fx.Int32,
                                )
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
                remote = fx.Int64(buf_copy_load(peers, tid, fx.Int64))
                store_i64_global_system(remote + fx.Int64(rank) * 8, generation)
            fx.barrier()
            if tid == 0:
                atomic_add_global_at(addr_xdb_flag, fx.Int64(1))
        else:
            if tid < npes:
                spin_until_ge_i64(
                    addr_xdb_mem + fx.Int64(tid) * 8, generation - fx.Int64(1)
                )
                fence_system_acquire()
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
        offset = fx.Int32(
            (fx.gpu.block_id("x") * block_threads + fx.gpu.thread_id("x")) * vec
        )
        if offset < numel:
            words = (
                _load_fp6(payload, offset // 8)
                if const_expr(role_codec == "mxfp6")
                else (
                    fx.Vector.from_elements(
                        [buf_copy_load(payload, offset // 8)],
                        fx.Int32,
                    )
                    if const_expr(packing == 2)
                    else fx.Vector(buf_copy_load(payload, offset // 4, fx.Int32, 2))
                )
            )
            # Four neighboring lanes share a scale; four scales fit in one dword.
            scale_index = offset // 32
            scale_word = fx.Uint32(buf_copy_load(scales, scale_index // 4))
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
            buf_copy_store(output, offset // 2, bf16.bitcast(fx.Int32), fx.Int32, 4)

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
