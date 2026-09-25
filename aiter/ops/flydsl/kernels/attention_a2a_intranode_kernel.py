# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL single-tensor intranode push all-to-all kernel."""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.ir.flydsl as mori_shmem
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
from .quant_utils import emit_f32_to_e2m1, emit_f32_to_e2m3, emit_mx_e8m0_scale
from .tensor_shim import buf_copy_load, buf_copy_store, ptr_buf_tensor

_JIT_CACHE_TAG = "attention_a2a_intranode"
_TRANSPORT_CHUNK_BYTES = 16
_PUSH_PIPELINE_DEPTH = 16


def make_attention_a2a_reuse_jit(*, rank, npes):
    """Gate parity reuse on stream-ordered consumer completion at every rank."""

    @flyc.kernel(known_block_size=[64, 1, 1])
    def attention_a2a_reuse_barrier(
        addr_ready_mem: fx.Int64,
        addr_p2p_ready_mem: fx.Int64,
        addr_ready_flag: fx.Int64,
    ):
        tid = fx.thread_idx.x
        rsrc_p2p_ready = create_buffer_resource_from_addr(
            addr_p2p_ready_mem, num_records_bytes=npes * 8
        )
        rsrc_ready_flag = create_buffer_resource_from_addr(
            addr_ready_flag, num_records_bytes=8
        )
        generation = buffer_load(rsrc_ready_flag, 0, vec_width=1, dtype=T.i64)
        if tid < npes:
            remote_slot = (
                buffer_load(rsrc_p2p_ready, tid, vec_width=1, dtype=T.i64)
                + fx.Int64(rank) * 8
            )
            # Reaching this kernel drains earlier consumers on this stream.
            store_i64_global_system(remote_slot, generation)
            spin_until_ge_i64(addr_ready_mem + fx.Int64(tid) * 8, generation)
            fence_system_acquire()
        fx.barrier()
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
        attention_a2a_reuse_barrier(
            addr_ready_mem, addr_p2p_ready_mem, addr_ready_flag
        ).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    return launch


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


def _hadamard_head(values, lane, head_dim):
    # Eight adjacent channels live in registers; the rest of the head is in
    # aligned lane groups. XOR never crosses a head boundary.
    result = [values[i] for i in range(8)]
    for shift in (1, 2, 4):
        result = [
            (
                result[i ^ shift] - result[i]
                if i & shift
                else result[i] + result[i ^ shift]
            )
            for i in range(8)
        ]
    for stage in range((head_dim // 8).bit_length() - 1):
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
):
    heads_local = heads // npes
    seq_full = seq_len * npes
    k_tiles = (seq_full + 127) // 128

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
                        {"v_amax": fx.Array[fx.Float32, 512, 16]}
                        if v4_output == "v" and codec == "mxfp4"
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
        addr_xdb_mem: fx.Int64,
        addr_p2p_xdb_mem: fx.Int64,
        addr_xdb_flag: fx.Int64,
        addr_grid_barrier: fx.Int64,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid & 63
        warp = tid >> 6
        global_warp_id = bid * warp_num_per_block + warp
        global_warp_num = block_num * warp_num_per_block
        grid_thread_id = bid * (warp_num_per_block * 64) + tid

        p2p_output_q = ptr_buf_tensor(addr_p2p_output, fx.Int64)
        rsrc_p2p_xdb = create_buffer_resource_from_addr(addr_p2p_xdb_mem)
        rsrc_xdb_flag = create_buffer_resource_from_addr(addr_xdb_flag)
        rsrc_grid_barrier = create_buffer_resource_from_addr(addr_grid_barrier)

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
        def transport_v4_v():
            # A rank boundary may cut a tile, but never a token-32 scale block.
            first_tile = rank * seq_len // 128
            rank_tiles = ((rank + 1) * seq_len + 127) // 128 - first_tile
            scratch = shared.v_amax.view(fx.make_layout(512, 1))
            scale_table = ptr_buf_tensor(addr_p2p_scale, fx.Int64)
            channel_chunk = tid % 16
            token_lane = tid // 16
            for work in range(bid, fx.Int32(heads * rank_tiles), fx.Int32(block_num)):
                head = work // rank_tiles
                tile_id = first_tile + work % rank_tiles
                dest_pe = head // heads_local
                local_head = head % heads_local
                peer_base = fx.Uint64(fx.memref_load(p2p_bases_q, dest_pe))
                rsrc_dst = create_buffer_resource_from_addr(
                    wave_uniform_i64(peer_base),
                    num_records_bytes=heads_local * k_tiles * 8192 + 64,
                )
                scale_base = fx.Uint64(buf_copy_load(scale_table, dest_pe, fx.Int64))
                rsrc_scale = create_buffer_resource_from_addr(
                    wave_uniform_i64(scale_base),
                    num_records_bytes=heads_local * k_tiles * 512,
                )
                for quarter in range_constexpr(4):
                    global_start = tile_id * 128 + quarter * 32
                    owned = (global_start >= rank * seq_len) & (
                        global_start
                        < ((rank + 1) * seq_len if rank != npes - 1 else k_tiles * 128)
                    )
                    if owned:
                        valid = global_start < seq_full
                        values = []
                        for half in range_constexpr(2):
                            seq = global_start - rank * seq_len + token_lane + half * 16
                            if const_expr(v_pack == AttentionPack.V_FOR_FP6_P):
                                # 64-aligned rank slices preserve the producer's
                                # half-tile token frame, before the scale reduction.
                                seq = (
                                    (seq & ~0x24)
                                    | ((seq & 0x04) << 3)
                                    | ((seq & 0x20) >> 3)
                                )
                            seq = valid.select(seq, 0)
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
                                    [
                                        valid.select(raw[i], fx.Float32(0.0))
                                        for i in range(8)
                                    ],
                                    fx.Float32,
                                )
                            )
                        for i in range_constexpr(8):
                            amax = fmath.absf(values[0][i]).maximumf(
                                fmath.absf(values[1][i])
                            )
                            for shift in (16, 32):
                                amax = amax.maximumf(amax.shuffle_xor(shift, 64))
                            if lane < 16:
                                fx.memref_store(
                                    amax, scratch, warp * 128 + channel_chunk * 8 + i
                                )
                        fx.barrier()
                        reciprocals = []
                        for i in range_constexpr(8):
                            amax = fx.Float32(0.0)
                            for source_wave in range_constexpr(4):
                                amax = amax.maximumf(
                                    fx.memref_load(
                                        scratch,
                                        source_wave * 128 + channel_chunk * 8 + i,
                                    )
                                )
                            bits = amax.maximumf(fx.Float32(1.0e-12)).bitcast(fx.Int32)
                            scale = (
                                ((bits >> 23) & 255)
                                - 2
                                + ((bits & 0x7FFFFF) > 0x400000).to(fx.Int32)
                            )
                            reciprocals.append(
                                ((fx.Int32(254) - scale) << 23).bitcast(fx.Float32)
                            )
                            if token_lane == 0:
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
                            buffer_store(
                                _pack_transport_words(pairs, "mxfp4"),
                                rsrc_dst,
                                word_offset,
                            )
                        # All waves finish reading the reduction before its next use.
                        fx.barrier()

        if const_expr(v4_output in ("q", "k") and codec in ("int8", "e4m3")):
            transport_v4_per_tensor_qk()
        elif const_expr(v4_output == "v" and codec == "e4m3"):
            transport_v4_fp8_v()
        elif const_expr(v4_output == "v"):
            transport_v4_v()
        else:
            transport(
                input_q,
                p2p_bases_q,
                addr_p2p_scale,
                codec,
                hadamard,
                v4_output,
            )
        # Snapshot before arrival so block 0 cannot advance a late block's epoch.
        xdb_cur_flag = buffer_load(rsrc_xdb_flag, 0, vec_width=1, dtype=T.i64)
        fx.rocdl.s_waitcnt(vmcnt=0)

        # All blocks must be resident: this is a grid-wide software barrier.
        fx.barrier()
        if tid == 0:
            atomic_add_global_at(addr_grid_barrier, 1)

        if grid_thread_id < npes:
            mori_shmem.int32_wait_until_equals(addr_grid_barrier, block_num)
            fence_system_acquire()
            buffer_store(fx.Int32(0), rsrc_grid_barrier, 0)
            xdb_remote_addr = (
                buffer_load(rsrc_p2p_xdb, grid_thread_id, vec_width=1, dtype=T.i64)
                + fx.Int64(rank) * 8
            )
            store_i64_global_system(xdb_remote_addr, xdb_cur_flag)

        if grid_thread_id == 0:
            atomic_add_global_at(addr_xdb_flag, fx.Int64(1))

        if tid < npes:
            peer_slot = addr_xdb_mem + fx.Int64(tid) * 8
            spin_until_ge_i64(peer_slot, xdb_cur_flag)
            fence_system_acquire()
        fx.barrier()

    return attention_a2a_single_push


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
    kernel = make_attention_a2a_kernel(
        rank=rank,
        npes=npes,
        heads=heads,
        seq_len=seq_len,
        head_dim=head_dim,
        block_num=block_num,
        warp_num_per_block=warp_num_per_block,
        quant=quant,
        codec=codec,
        hadamard=hadamard,
        v4_output=v4_output,
        v_pack=v_pack,
        q_multiplier=q_multiplier,
        v4_amax=v4_amax,
        fp8_fnuz=fp8_fnuz,
    )
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
        addr_grid_barrier: fx.Int64,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        _ = key
        kernel(
            addr_input,
            addr_p2p_output,
            addr_p2p_scale,
            addr_xdb_mem,
            addr_p2p_xdb_mem,
            addr_xdb_flag,
            addr_grid_barrier,
        ).launch(
            grid=(block_num, 1, 1),
            block=(warp_num_per_block * 64, 1, 1),
            stream=stream,
        )

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
